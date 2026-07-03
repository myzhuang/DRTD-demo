"""
将离散的事件流 HDF5 文件 + YOLO 标注转换为 RVT (Gen4/1Mpx) 预处理数据集格式。

输入(每个样本):
  {index}_{session}_{frametime}.hdf5   事件流, key='events', shape=(4,N)=[x,y,p(0/1),t_us]
  {index}_{session}_{frametime}.txt    YOLO 标注 [cls xc yc w h] (归一化, 相对已校准RGB 1280x720)
  {index}_{session}_{frametime}.jpg     已校准 RGB(参考帧, 仅用于可视化验证)

输出(RVT 序列目录结构), 一个序列 = 一个 index(≈60s 录制):
  DEST/<split>/<index>/
    event_representations_v2/stacked_histogram_dt=50_nbins=10/
        event_representations_ds2_nearest.h5   (N,20,360,640) uint8   [downsample_by_2=True]
        objframe_idx_2_repr_idx.npy            (M,) int64   带标注帧 -> repr 下标
        timestamps_us.npy                      (N,) int64   每个 repr 帧的全局时间戳
    labels_v2/
        labels.npz   {labels: BBOX_DTYPE(total,), objframe_idx_2_label_idx: (M,) int64}
        timestamps_us.npy                      (M,) int64   每个标注帧的全局时间戳

其中 N = 序列内全部表示帧数(每个 hdf5 一帧), M = 其中"有标注"的帧数(M<=N)。

分组规则: 按文件名第一段 index(如 '104')分组 = 一段 ≈60s 录制 = 一个序列(对齐 Gen4)。
序列内按文件名时间(第2、3段)排序, 帧时间戳取内部事件时钟并强制全局单调递增。
内存安全: 逐帧流式写入 h5(整段几百帧也不会爆内存)。

核心步骤: 事件累积(stacked histogram, 完全照搬 RVT) -> 逐通道几何校准(照搬 event_img_process 的
图像空间 undistort+裁剪+缩放) -> 降采样 2x -> 写盘; YOLO(中心点)->Prophesee(左上角)标注。
"""
import os
import glob
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# ----------------------------------------------------------------------------
# 1. 全局参数
# ----------------------------------------------------------------------------
HEIGHT_CAM, WIDTH_CAM = 720, 1280          # 采集原始分辨率
NBINS        = 10                          # 时间分箱数 (RVT 默认)
COUNT_CUTOFF = 10                          # 计数上限 (stacked_hist.yaml)
DT_MS        = 50                          # 事件累积时长 ms (RVT dt=50)
DURING_US    = DT_MS * 1000                # 50ms -> us
EV_REPR_NAME = f'stacked_histogram_dt={DT_MS}_nbins={NBINS}'
DOWNSAMPLE_BY_2 = True                     # gen4/1Mpx: 720x1280 -> 360x640

# YOLO 标注 .txt 根目录, 与 hdf5 路径完全无关, 可任意指定。
#   - None         : 退回"与 hdf5 同目录、同名 .txt"(向后兼容)
#   - r'D:\labels' : 在该目录下(含所有子文件夹, 递归)按文件名 stem 找同名 .txt
# 匹配规则: hdf5 的 stem(去掉 .hdf5)== txt 的 stem(去掉 .txt)。
LABEL_DIR = None
_LABEL_INDEX = None          # {stem: txt完整路径} 缓存, 首次使用时递归扫描 LABEL_DIR

# 事件相机内参 / 畸变 (event_camera_intrinsics.yaml)
MTX_EV = np.array([[1.74722570e+03, 0.0,            6.42214538e+02],
                   [0.0,            1.74604304e+03, 3.08360732e+02],
                   [0.0,            0.0,            1.0]], dtype=np.float64)
DIST_EV = np.array([[-8.00216024e-01, 1.48864224e+00, 4.81119812e-03,
                     -3.21381674e-03, -4.13338009e+00]], dtype=np.float64)

# 每个 scenario 的 cut_and_scale 参数 (cut_and_scale_data.yaml): (w_offset, h_offset, rgb_scale)
SCENARIO_CALIB = {
    1: (31.0, 62.0, 0.919000),
    2: (26.0, 60.0, 0.888000),
    3: (35.0, 63.0, 0.981000),
    4: (37.0, 60.0, 1.037000),
}

# index -> scenario 映射 (event_img_process.py 的 scene_map)
SCENE_MAP = {
    1: ['101','102','103','104','105','106','111','112','113','114','115','118','119','120','121',
        '122','123','124','125','126','127','128','129','130','131','132','133','134','135','136',
        '137','138','139','140','141'],
    2: ['232','233','234','235','236','237','238','239','240','241','242','243','244','245','246','247','248'],
    3: ['152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167',
        '168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','196','197','198'],
    4: ['283','284','285','286','287','288','289','290','291','292','293'],
}
INDEX2SCENARIO = {idx: s for s, lst in SCENE_MAP.items() for idx in lst}

# RVT Prophesee 标注 dtype (utils/evaluation/prophesee/io/box_loading.py)
BBOX_DTYPE = np.dtype({'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence'],
                       'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
                       'offsets': [0, 8, 12, 16, 20, 24, 28, 32], 'itemsize': 40})


# ----------------------------------------------------------------------------
# 2. Stacked Histogram (完全照搬 RVT data/utils/representations.py 的逻辑)
# ----------------------------------------------------------------------------
def build_stacked_histogram(x, y, pol, t, height=HEIGHT_CAM, width=WIDTH_CAM,
                            bins=NBINS, count_cutoff=COUNT_CUTOFF):
    """返回 (2*bins, H, W) uint8。通道排布: [pol0_bin0..pol0_bin9, pol1_bin0..pol1_bin9]。"""
    x = torch.from_numpy(np.asarray(x, dtype=np.int64))
    y = torch.from_numpy(np.asarray(y, dtype=np.int64))
    pol = torch.from_numpy(np.asarray(pol, dtype=np.int64))
    t = torch.from_numpy(np.asarray(t, dtype=np.int64))
    channels = 2
    representation = torch.zeros((channels, bins, height, width), dtype=torch.int16)
    if x.numel() == 0:
        return representation.reshape(-1, height, width).to(torch.uint8).numpy()
    assert pol.min() >= 0 and pol.max() <= 1
    # 时间 -> bin 索引 (假设 t 已排序)
    t0, t1 = t[0], t[-1]
    t_norm = (t - t0).float() / max(int(t1 - t0), 1) * bins
    t_idx = torch.clamp(t_norm.floor(), max=bins - 1).long()
    # 把负坐标/越界裁掉(校准前坐标应都合法, 这里保险处理)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, pol, t_idx = x[valid], y[valid], pol[valid], t_idx[valid]
    indices = x + width * y + height * width * t_idx + bins * height * width * pol
    values = torch.ones_like(indices, dtype=torch.int16)
    representation.put_(indices, values, accumulate=True)
    representation = torch.clamp(representation, min=0, max=count_cutoff).to(torch.uint8)
    return representation.reshape(-1, height, width).numpy()   # (20, H, W)


# ----------------------------------------------------------------------------
# 3. 几何校准 (照搬 event_img_process.py 的图像空间方法, 逐通道执行)
# ----------------------------------------------------------------------------
# 预计算 undistort 映射表 (只与相机有关, 复用)
_UNDIST_MAP = cv2.initUndistortRectifyMap(
    MTX_EV, DIST_EV, None, MTX_EV, (WIDTH_CAM, HEIGHT_CAM), cv2.CV_32FC1)


def _crop_box_for_scenario(scenario):
    """返回 (top, down, left, right) 像素裁剪边界, 复刻 event_img_process.py 的逻辑。"""
    w_off, h_off, scale = SCENARIO_CALIB[scenario]
    left = int(w_off + 0.5 * WIDTH_CAM * (1.0 - scale))
    if scenario in (1, 2):
        down  = int(0.5 * HEIGHT_CAM * (1.0 + scale) - h_off)
        right = int(0.5 * WIDTH_CAM * (1.0 + scale) + w_off)
        return 0, down, left, right                  # img[0:down, left:right]
    else:  # scenario 3, 4
        down  = int(h_off + 0.5 * HEIGHT_CAM * (1.0 - scale))
        return 0, HEIGHT_CAM - down, left, WIDTH_CAM  # img[0:-down, left:]


def calibrate_repr(repr_20chw, scenario):
    """对 (20,H,W) 的每个通道做 undistort -> 按场景裁剪 -> resize 回 (1280,720)。
    用 INTER_NEAREST 保持事件计数为整数, 几何上与 RGB 校准帧对齐。"""
    top, down, left, right = _crop_box_for_scenario(scenario)
    out = np.zeros_like(repr_20chw)
    for c in range(repr_20chw.shape[0]):
        ch = cv2.remap(repr_20chw[c], _UNDIST_MAP[0], _UNDIST_MAP[1],
                       interpolation=cv2.INTER_NEAREST)
        ch = ch[top:down, left:right]
        ch = cv2.resize(ch, (WIDTH_CAM, HEIGHT_CAM), interpolation=cv2.INTER_NEAREST)
        out[c] = ch
    return out


# ----------------------------------------------------------------------------
# 4. 降采样 2x (照搬 RVT downsample_ev_repr: nearest-exact)
# ----------------------------------------------------------------------------
def downsample_repr(repr_20chw):
    x = torch.from_numpy(repr_20chw).unsqueeze(0).float()   # (1,20,H,W)
    x = F.interpolate(x, scale_factor=0.5, mode='nearest-exact')
    return x[0].to(torch.uint8).numpy()                     # (20,360,640)


# ----------------------------------------------------------------------------
# 5. YOLO -> Prophesee BBOX 标注
# ----------------------------------------------------------------------------
def yolo_to_bbox(txt_path, t_us, img_w=WIDTH_CAM, img_h=HEIGHT_CAM):
    """YOLO [cls xc yc w h](归一化, 中心点) -> BBOX_DTYPE 结构数组(全分辨率像素, 左上角)。"""
    rows = []
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls, xc, yc, w, h = line.split()[:5]
                rows.append((int(float(cls)), float(xc), float(yc), float(w), float(h)))
    out = np.zeros((len(rows),), dtype=BBOX_DTYPE)
    for i, (cls, xc, yc, w, h) in enumerate(rows):
        bw = w * img_w
        bh = h * img_h
        x_tl = (xc - w / 2.0) * img_w          # 左上角 x
        y_tl = (yc - h / 2.0) * img_h          # 左上角 y
        out[i]['t'] = np.int64(t_us)
        out[i]['x'] = np.float32(max(x_tl, 0))
        out[i]['y'] = np.float32(max(y_tl, 0))
        out[i]['w'] = np.float32(bw)
        out[i]['h'] = np.float32(bh)
        out[i]['class_id'] = np.uint32(cls)
        out[i]['track_id'] = np.uint32(0)
        out[i]['class_confidence'] = np.float32(1.0)
    return out


# ----------------------------------------------------------------------------
# 6. blosc 压缩参数 (照搬 utils/preprocessing._blosc_opts), 不可用时回退 gzip
# ----------------------------------------------------------------------------
def _h5_dataset_kwargs():
    try:
        import hdf5plugin  # noqa: F401  注册 blosc 过滤器
        compressors = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
        complib = ['blosc:' + c for c in compressors].index('blosc:zstd')
        return {'compression': 32001,
                'compression_opts': (0, 0, 0, 0, 1, 1, complib),
                'shuffle': False}
    except Exception:
        return {'compression': 'gzip', 'compression_opts': 1}


# ----------------------------------------------------------------------------
# 7. 处理一个事件 HDF5 文件 -> 一帧校准后的 (20,Hd,Wd) + 标注
# ----------------------------------------------------------------------------
def process_file(hdf5_path, txt_path, during_us=DURING_US):
    basename = os.path.basename(hdf5_path)
    index = basename.split('_')[0]
    if index not in INDEX2SCENARIO:
        raise ValueError(f'index {index} 不在任何 scenario 中: {basename}')
    scenario = INDEX2SCENARIO[index]

    with h5py.File(hdf5_path, 'r') as hf:
        ev = np.array(hf['events'])          # (4, N)
    x, y, p, t = ev[0], ev[1], ev[2], ev[3]
    order = np.argsort(t, kind='stable')     # 保证时间有序
    x, y, p, t = x[order], y[order], p[order], t[order]

    # 取窗口 [t0, t0+dt] (与 event_img_process.py 取 t[0] 起一致, 对齐已校准RGB/标签)
    t0 = t[0]
    if during_us > 0:
        m = t <= (t0 + during_us)
        x, y, p, t = x[m], y[m], p[m], t[m]
    t_end = int(t[-1])                        # 该帧 representation 的时间戳(窗口末端)

    repr20 = build_stacked_histogram(x, y, p, t)        # (20,720,1280) 原始坐标系
    repr20 = calibrate_repr(repr20, scenario)           # (20,720,1280) 已对齐RGB
    repr_store = downsample_repr(repr20) if DOWNSAMPLE_BY_2 else repr20  # (20,360,640)

    bbox = yolo_to_bbox(txt_path, t_us=t_end)           # 全分辨率像素, 左上角
    return repr_store, bbox, t_end, scenario, repr20    # repr20 全分辨率, 留作可视化


# ----------------------------------------------------------------------------
# 8. 文件名时间排序键
# ----------------------------------------------------------------------------
def filename_sort_key(hdf5_path):
    """按文件名第2、3段(时间)排序, 在整段录制内得到正确的时间先后。"""
    parts = os.path.basename(hdf5_path).replace('.hdf5', '').split('_')
    try:
        return (int(parts[1]), int(parts[2]))
    except (IndexError, ValueError):
        return (0, 0)


def _build_label_index(label_dir):
    """递归扫描 label_dir 下所有 .txt, 建立 {stem: 完整路径} 映射(独立于 hdf5 路径)。"""
    index, dups = {}, 0
    for root, _, files in os.walk(label_dir):
        for fn in files:
            if fn.lower().endswith('.txt'):
                stem = fn[:-4]
                if stem in index:
                    dups += 1
                index[stem] = os.path.join(root, fn)
    print(f'  [标注索引] 在 {label_dir} 下递归找到 {len(index)} 个 .txt'
          + (f' (有 {dups} 个同名 stem, 后者覆盖前者)' if dups else ''))
    return index


def find_label_txt(hdf5_path):
    """按文件名 stem 定位 YOLO .txt, 路径与 hdf5 完全无关(由 LABEL_DIR 决定)。"""
    stem = os.path.basename(hdf5_path)[:-5]              # 去掉 '.hdf5'
    if LABEL_DIR:
        global _LABEL_INDEX
        if _LABEL_INDEX is None:
            _LABEL_INDEX = _build_label_index(LABEL_DIR)
        # 命中索引就用真实路径; 没命中返回一个 LABEL_DIR 下的路径(供存在性检查/报缺失)
        return _LABEL_INDEX.get(stem, os.path.join(LABEL_DIR, stem + '.txt'))
    return hdf5_path[:-5] + '.txt'                       # LABEL_DIR=None: 同目录同名


# ----------------------------------------------------------------------------
# 9. 主流程: 按 index 分组(≈60s 录制), 逐帧流式写入
# ----------------------------------------------------------------------------
def convert(src_dir, dest_root, split='train', label_dir=None):
    global LABEL_DIR, _LABEL_INDEX
    if label_dir is not None:                # 允许调用时直接传入独立的标注目录
        LABEL_DIR = label_dir
    _LABEL_INDEX = None                       # 每次转换重建标注索引(避免跨调用串缓存)
    hdf5_list = sorted(glob.glob(os.path.join(src_dir, '*.hdf5')))
    print(f'找到 {len(hdf5_list)} 个事件文件')
    print(f'标注来源: {LABEL_DIR if LABEL_DIR else "(与 hdf5 同目录同名)"}')

    # 按文件名"第一段 index"分组 (例: 104) = 一段 ≈60s 录制 = 一个序列
    sessions = {}
    for h in hdf5_list:
        index = os.path.basename(h).split('_')[0]
        sessions.setdefault(index, []).append(h)
    print(f'共 {len(sessions)} 个序列(index): {sorted(sessions.keys())}')

    # 输出张量形状(下采样与否)
    C = 2 * NBINS
    Hs = HEIGHT_CAM // 2 if DOWNSAMPLE_BY_2 else HEIGHT_CAM
    Ws = WIDTH_CAM // 2 if DOWNSAMPLE_BY_2 else WIDTH_CAM
    repr_shape = (C, Hs, Ws)
    ds_suffix = '_ds2_nearest' if DOWNSAMPLE_BY_2 else ''

    summary = []
    for session in sorted(sessions.keys()):
        files = sorted(sessions[session], key=filename_sort_key)   # 录制内按时间排序
        seq_dir = os.path.join(dest_root, split, session)
        ev_dir  = os.path.join(seq_dir, 'event_representations_v2', EV_REPR_NAME)
        lab_dir = os.path.join(seq_dir, 'labels_v2')
        os.makedirs(ev_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        ev_h5 = os.path.join(ev_dir, f'event_representations{ds_suffix}.h5')

        repr_ts = []                      # 每个 repr 帧的全局时间戳 (length N)
        objframe_idx_2_repr_idx = []      # 有标注帧 -> repr 下标       (length M)
        objframe_ts = []                  # 有标注帧的全局时间戳        (length M)
        labels_list = []
        objframe_idx_2_label_idx = []
        label_start = 0
        prev_ts = None
        n = 0                             # 已写入的 repr 帧计数
        n_txt_missing = 0                 # 缺失的 .txt 数
        n_txt_empty = 0                   # 找到但为空(0 框)的 .txt 数

        # ---- 逐帧流式写入事件表示 h5 (一次只在内存里放一帧) ----
        with h5py.File(ev_h5, 'w') as f:
            dset = f.create_dataset('data', shape=(0,) + repr_shape, dtype='uint8',
                                    chunks=(1,) + repr_shape, maxshape=(None,) + repr_shape,
                                    **_h5_dataset_kwargs())
            for h in files:
                txt = find_label_txt(h)
                if not os.path.exists(txt):
                    n_txt_missing += 1
                try:
                    repr_store, bbox, t_end, scenario, _ = process_file(h, txt)
                except Exception as e:
                    print(f'  [跳过] {os.path.basename(h)}: {e}')
                    continue
                if os.path.exists(txt) and len(bbox) == 0:
                    n_txt_empty += 1

                # 全局单调时间戳: 用内部事件时钟, 若非严格递增则 +1us 强制递增
                ts = int(t_end)
                if prev_ts is not None and ts <= prev_ts:
                    ts = prev_ts + 1
                prev_ts = ts

                # 写入该 repr 帧
                dset.resize(n + 1, axis=0)
                dset[n] = repr_store
                repr_ts.append(ts)

                # 有标注的帧才登记为 objframe(无标注帧仅作上下文/预热帧)
                if len(bbox) > 0:
                    bbox = bbox.copy()
                    bbox['t'] = np.int64(ts)              # 用全局时间戳覆盖
                    objframe_idx_2_repr_idx.append(n)
                    objframe_idx_2_label_idx.append(label_start)
                    objframe_ts.append(ts)
                    labels_list.append(bbox)
                    label_start += len(bbox)
                n += 1

        # ---- 元数据: 事件表示侧 ----
        np.save(os.path.join(ev_dir, 'timestamps_us.npy'),
                np.asarray(repr_ts, dtype=np.int64))                       # 长度 N
        np.save(os.path.join(ev_dir, 'objframe_idx_2_repr_idx.npy'),
                np.asarray(objframe_idx_2_repr_idx, dtype=np.int64))       # 长度 M

        # ---- 元数据: 标注侧 ----
        labels_v2 = np.concatenate(labels_list) if labels_list else np.zeros((0,), BBOX_DTYPE)
        np.savez(os.path.join(lab_dir, 'labels.npz'),
                 labels=labels_v2,
                 objframe_idx_2_label_idx=np.asarray(objframe_idx_2_label_idx, dtype=np.int64))
        np.save(os.path.join(lab_dir, 'timestamps_us.npy'),
                np.asarray(objframe_ts, dtype=np.int64))                   # 长度 M

        M = len(objframe_idx_2_repr_idx)
        max_seq_len = (objframe_idx_2_repr_idx[-1] + 1) if M > 0 else 0    # seq_len 上限提示
        warn = ''
        if M == 0:
            warn = '  <<< 警告: 0 个标注帧! 该序列无法训练'
            if n_txt_missing == n:
                warn += f' (全部 {n} 个 .txt 都没找到 -> 检查 LABEL_DIR / 命名)'
            elif n_txt_empty == n:
                warn += f' (找到 .txt 但全是空文件)'
        print(f'  序列 {session}: {n} 个repr帧, {M} 个标注帧 -> {seq_dir}  '
              f'(seq_len<={max_seq_len} 才有样本; txt缺失{n_txt_missing}/空{n_txt_empty}){warn}')
        summary.append((session, n, M, max_seq_len, n_txt_missing, n_txt_empty))

    print('\n=== 汇总(index, repr帧数N, 标注帧数M, 最大seq_len, txt缺失, txt空) ===')
    total_M = sum(s[2] for s in summary)
    for s in summary:
        print(f'  {s[0]}: N={s[1]}, M={s[2]}, max_seq_len={s[3]}, txt缺失={s[4]}, txt空={s[5]}')
    if total_M == 0:
        print('\n*** 严重: 所有序列标注帧总数 = 0, 数据集不可训练! ***')
        print('    最常见原因: YOLO .txt 不在脚本查找的位置。')
        print('    解决: 若标注在单独文件夹, 把顶部 LABEL_DIR 改成该文件夹路径;')
        print('         若与 hdf5 同目录, 确认 .txt 与 .hdf5 同名(仅扩展名不同)。')
    return summary


if __name__ == '__main__':
    # 三个路径互相独立, 按需修改:
    # src       = r'D:\20260614_sdata_drtd_r1_process\2_hdf5_val'              # hdf5 事件目录
    # label_dir = r'F:\dataset\DRTD\DRTD\aligned\labels\val'             # YOLO txt 目录(任意位置, 递归匹配同名)
    # dst       = r'D:\20260614_sdata_drtd_r1_process\rvt_steam_data\data\val'  # 输出根目录
    # convert(src, dst, split='val', label_dir=label_dir)

    # src       = r'D:\20260614_sdata_drtd_r1_process\2_hdf5_test'              # hdf5 事件目录
    # label_dir = r'F:\dataset\DRTD\DRTD\aligned\labels\test'             # YOLO txt 目录(任意位置, 递归匹配同名)
    # dst       = r'D:\20260614_sdata_drtd_r1_process\rvt_steam_data\data\test'  # 输出根目录
    # convert(src, dst, split='test', label_dir=label_dir)

    src       = r'D:\20260614_sdata_drtd_r1_process\2_hdf5_train'              # hdf5 事件目录
    label_dir = r'F:\dataset\DRTD\DRTD\aligned\labels\train'             # YOLO txt 目录(任意位置, 递归匹配同名)
    dst       = r'D:\20260614_sdata_drtd_r1_process\rvt_steam_data\data\train'  # 输出根目录
    convert(src, dst, split='train', label_dir=label_dir)



