"""
Convert discrete event stream HDF5 files + YOLO annotations into the RVT (Gen4/1Mpx) preprocessed dataset format.

Input (per sample):
  {index}_{session}_{frametime}.hdf5   Event stream, key='events', shape=(4,N)=[x,y,p(0/1),t_us]
  {index}_{session}_{frametime}.txt    YOLO annotation [cls xc yc w h] (normalized, relative to calibrated RGB 1280x720)
  {index}_{session}_{frametime}.jpg     Calibrated RGB (reference frame, for visual verification only)

Output (RVT sequence directory structure), one sequence = one index (~60s recording):
  DEST/<split>/<index>/
    event_representations_v2/stacked_histogram_dt=50_nbins=10/
        event_representations_ds2_nearest.h5   (N,20,360,640) uint8   [downsample_by_2=True]
        objframe_idx_2_repr_idx.npy            (M,) int64   annotated frame -> repr index
        timestamps_us.npy                      (N,) int64   global timestamp for each repr frame
    labels_v2/
        labels.npz   {labels: BBOX_DTYPE(total,), objframe_idx_2_label_idx: (M,) int64}
        timestamps_us.npy                      (M,) int64   global timestamp for each annotated frame

Where N = total number of representation frames in the sequence (one frame per hdf5), M = number of "annotated" frames (M<=N).

Grouping rule: group by the first segment of the filename (index, e.g. '104') = one ~60s recording = one sequence (aligned with Gen4).
Within a sequence, sort by the time in filename segments 2 and 3; frame timestamps are taken from the internal event clock and forced to be globally monotonically increasing.
Memory safety: stream-write frames to h5 one at a time (even hundreds of frames for one sequence won't blow up memory).

Core steps: event accumulation (stacked histogram, exactly as in RVT) -> per-channel geometric calibration (mirroring event_img_process's
image-space undistort+crop+scale) -> 2x downsampling -> write to disk; YOLO (center-point) -> Prophesee (top-left) annotation.
"""
import os
import glob
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# ----------------------------------------------------------------------------
# 1. Global parameters
# ----------------------------------------------------------------------------
HEIGHT_CAM, WIDTH_CAM = 720, 1280          # Native capture resolution
NBINS        = 10                          # Number of time bins (RVT default)
COUNT_CUTOFF = 10                          # Count upper limit (stacked_hist.yaml)
DT_MS        = 50                          # Event accumulation duration in ms (RVT dt=50)
DURING_US    = DT_MS * 1000                # 50ms -> us
EV_REPR_NAME = f'stacked_histogram_dt={DT_MS}_nbins={NBINS}'
DOWNSAMPLE_BY_2 = True                     # gen4/1Mpx: 720x1280 -> 360x640

# Root directory for YOLO annotation .txt files; completely independent of hdf5 paths, can be any path.
#   - None         : fall back to "same directory as hdf5, same basename .txt" (backward compatible)
#   - r'D:\labels' : look for the same-name .txt under this directory (including all subfolders, recursive)
# Matching rule: hdf5 stem (without .hdf5) == txt stem (without .txt).
LABEL_DIR = None
_LABEL_INDEX = None          # {stem: full path to txt} cache, recursively scans LABEL_DIR on first use

# Event camera intrinsics / distortion (event_camera_intrinsics.yaml)
MTX_EV = np.array([[1.74722570e+03, 0.0,            6.42214538e+02],
                   [0.0,            1.74604304e+03, 3.08360732e+02],
                   [0.0,            0.0,            1.0]], dtype=np.float64)
DIST_EV = np.array([[-8.00216024e-01, 1.48864224e+00, 4.81119812e-03,
                     -3.21381674e-03, -4.13338009e+00]], dtype=np.float64)

# Per-scenario cut_and_scale parameters (cut_and_scale_data.yaml): (w_offset, h_offset, rgb_scale)
SCENARIO_CALIB = {
    1: (31.0, 62.0, 0.919000),
    2: (26.0, 60.0, 0.888000),
    3: (35.0, 63.0, 0.981000),
    4: (37.0, 60.0, 1.037000),
}

# index -> scenario mapping (event_img_process.py's scene_map)
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

# RVT Prophesee annotation dtype (utils/evaluation/prophesee/io/box_loading.py)
BBOX_DTYPE = np.dtype({'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence'],
                       'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
                       'offsets': [0, 8, 12, 16, 20, 24, 28, 32], 'itemsize': 40})


# RVT Prophesee annotation dtype (utils/evaluation/prophesee/io/box_loading.py)
BBOX_DTYPE = np.dtype({'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence'],
                       'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
                       'offsets': [0, 8, 12, 16, 20, 24, 28, 32], 'itemsize': 40})


# ----------------------------------------------------------------------------
# 2. Stacked Histogram (logic exactly mirrored from RVT data/utils/representations.py)
# ----------------------------------------------------------------------------
def build_stacked_histogram(x, y, pol, t, height=HEIGHT_CAM, width=WIDTH_CAM,
                            bins=NBINS, count_cutoff=COUNT_CUTOFF):
    """Returns (2*bins, H, W) uint8. Channel layout: [pol0_bin0..pol0_bin9, pol1_bin0..pol1_bin9]."""
    x = torch.from_numpy(np.asarray(x, dtype=np.int64))
    y = torch.from_numpy(np.asarray(y, dtype=np.int64))
    pol = torch.from_numpy(np.asarray(pol, dtype=np.int64))
    t = torch.from_numpy(np.asarray(t, dtype=np.int64))
    channels = 2
    representation = torch.zeros((channels, bins, height, width), dtype=torch.int16)
    if x.numel() == 0:
        return representation.reshape(-1, height, width).to(torch.uint8).numpy()
    assert pol.min() >= 0 and pol.max() <= 1
    # time -> bin index (assuming t is sorted)
    t0, t1 = t[0], t[-1]
    t_norm = (t - t0).float() / max(int(t1 - t0), 1) * bins
    t_idx = torch.clamp(t_norm.floor(), max=bins - 1).long()
    # clip out-of-range / negative coordinates (pre-calibration coordinates should all be valid; defensive check here)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, pol, t_idx = x[valid], y[valid], pol[valid], t_idx[valid]
    indices = x + width * y + height * width * t_idx + bins * height * width * pol
    values = torch.ones_like(indices, dtype=torch.int16)
    representation.put_(indices, values, accumulate=True)
    representation = torch.clamp(representation, min=0, max=count_cutoff).to(torch.uint8)
    return representation.reshape(-1, height, width).numpy()   # (20, H, W)


# ----------------------------------------------------------------------------
# 3. Geometric calibration (mirrored from event_img_process.py's image-space method, applied per channel)
# ----------------------------------------------------------------------------
# Pre-compute undistort mapping table (depends only on the camera, reused)
_UNDIST_MAP = cv2.initUndistortRectifyMap(
    MTX_EV, DIST_EV, None, MTX_EV, (WIDTH_CAM, HEIGHT_CAM), cv2.CV_32FC1)


def _crop_box_for_scenario(scenario):
    """Returns (top, down, left, right) pixel crop boundaries, mirroring event_img_process.py."""
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
    """For each channel of (20,H,W): undistort -> crop by scenario -> resize back to (1280,720).
    Uses INTER_NEAREST to keep event counts as integers and geometrically align with calibrated RGB frames."""
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
# 4. 2x Downsampling (mirrored from RVT downsample_ev_repr: nearest-exact)
# ----------------------------------------------------------------------------
def downsample_repr(repr_20chw):
    x = torch.from_numpy(repr_20chw).unsqueeze(0).float()   # (1,20,H,W)
    x = F.interpolate(x, scale_factor=0.5, mode='nearest-exact')
    return x[0].to(torch.uint8).numpy()                     # (20,360,640)


# ----------------------------------------------------------------------------
# 5. YOLO -> Prophesee BBOX annotation
# ----------------------------------------------------------------------------
def yolo_to_bbox(txt_path, t_us, img_w=WIDTH_CAM, img_h=HEIGHT_CAM):
    """YOLO [cls xc yc w h] (normalized, center-point) -> BBOX_DTYPE structured array (full-resolution pixels, top-left corner)."""
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
        x_tl = (xc - w / 2.0) * img_w          # top-left x
        y_tl = (yc - h / 2.0) * img_h          # top-left y
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
# 6. blosc compression parameters (mirrored from utils/preprocessing._blosc_opts), falls back to gzip if unavailable
# ----------------------------------------------------------------------------
def _h5_dataset_kwargs():
    try:
        import hdf5plugin  # noqa: F401  register blosc filter
        compressors = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
        complib = ['blosc:' + c for c in compressors].index('blosc:zstd')
        return {'compression': 32001,
                'compression_opts': (0, 0, 0, 0, 1, 1, complib),
                'shuffle': False}
    except Exception:
        return {'compression': 'gzip', 'compression_opts': 1}


# ----------------------------------------------------------------------------
# 7. Process one event HDF5 file -> one frame of calibrated (20,Hd,Wd) + annotation
# ----------------------------------------------------------------------------
def process_file(hdf5_path, txt_path, during_us=DURING_US):
    basename = os.path.basename(hdf5_path)
    index = basename.split('_')[0]
    if index not in INDEX2SCENARIO:
        raise ValueError(f'index {index} not found in any scenario: {basename}')
    scenario = INDEX2SCENARIO[index]

    with h5py.File(hdf5_path, 'r') as hf:
        ev = np.array(hf['events'])          # (4, N)
    x, y, p, t = ev[0], ev[1], ev[2], ev[3]
    order = np.argsort(t, kind='stable')     # ensure time order
    x, y, p, t = x[order], y[order], p[order], t[order]

    # window [t0, t0+dt] (consistent with event_img_process.py starting from t[0], aligned with calibrated RGB/labels)
    t0 = t[0]
    if during_us > 0:
        m = t <= (t0 + during_us)
        x, y, p, t = x[m], y[m], p[m], t[m]
    t_end = int(t[-1])                        # timestamp of this frame's representation (window end)

    repr20 = build_stacked_histogram(x, y, p, t)        # (20,720,1280) raw coordinate system
    repr20 = calibrate_repr(repr20, scenario)           # (20,720,1280) aligned with RGB
    repr_store = downsample_repr(repr20) if DOWNSAMPLE_BY_2 else repr20  # (20,360,640)

    bbox = yolo_to_bbox(txt_path, t_us=t_end)           # full resolution pixels, top-left corner
    return repr_store, bbox, t_end, scenario, repr20    # repr20 full resolution, kept for visualization


# ----------------------------------------------------------------------------
# 8. Filename time-sort key
# ----------------------------------------------------------------------------
def filename_sort_key(hdf5_path):
    """Sort by segments 2 and 3 (time) of the filename to get correct temporal order within a full recording."""
    parts = os.path.basename(hdf5_path).replace('.hdf5', '').split('_')
    try:
        return (int(parts[1]), int(parts[2]))
    except (IndexError, ValueError):
        return (0, 0)


def _build_label_index(label_dir):
    """Recursively scan all .txt under label_dir, build {stem: full path} mapping (independent of hdf5 path)."""
    index, dups = {}, 0
    for root, _, files in os.walk(label_dir):
        for fn in files:
            if fn.lower().endswith('.txt'):
                stem = fn[:-4]
                if stem in index:
                    dups += 1
                index[stem] = os.path.join(root, fn)
    print(f'  [Label Index] Found {len(index)} .txt files recursively under {label_dir}'
          + (f' ({dups} duplicate stems, later one overrides earlier)' if dups else ''))
    return index


def find_label_txt(hdf5_path):
    """Locate YOLO .txt by filename stem; path is completely independent of hdf5 (determined by LABEL_DIR)."""
    stem = os.path.basename(hdf5_path)[:-5]              # strip '.hdf5'
    if LABEL_DIR:
        global _LABEL_INDEX
        if _LABEL_INDEX is None:
            _LABEL_INDEX = _build_label_index(LABEL_DIR)
        # hit index -> use the real path; miss -> return a path under LABEL_DIR (for existence check / missing report)
        return _LABEL_INDEX.get(stem, os.path.join(LABEL_DIR, stem + '.txt'))
    return hdf5_path[:-5] + '.txt'                       # LABEL_DIR=None: same directory, same name


# ----------------------------------------------------------------------------
# 9. Main flow: group by index (~60s recording), stream-write frames one at a time
# ----------------------------------------------------------------------------
def convert(src_dir, dest_root, split='train', label_dir=None):
    global LABEL_DIR, _LABEL_INDEX
    if label_dir is not None:                # allow passing an independent label directory at call time
        LABEL_DIR = label_dir
    _LABEL_INDEX = None                       # rebuild label index for each conversion (avoid cross-call cache contamination)
    hdf5_list = sorted(glob.glob(os.path.join(src_dir, '*.hdf5')))
    print(f'Found {len(hdf5_list)} event files')
    print(f'Label source: {LABEL_DIR if LABEL_DIR else "(same directory and name as hdf5)"}')

    # group by the first segment "index" of the filename (e.g. 104) = one ~60s recording = one sequence
    sessions = {}
    for h in hdf5_list:
        index = os.path.basename(h).split('_')[0]
        sessions.setdefault(index, []).append(h)
    print(f'Total {len(sessions)} sequences (indexes): {sorted(sessions.keys())}')

    # output tensor shape (depending on whether downsampling is enabled)
    C = 2 * NBINS
    Hs = HEIGHT_CAM // 2 if DOWNSAMPLE_BY_2 else HEIGHT_CAM
    Ws = WIDTH_CAM // 2 if DOWNSAMPLE_BY_2 else WIDTH_CAM
    repr_shape = (C, Hs, Ws)
    ds_suffix = '_ds2_nearest' if DOWNSAMPLE_BY_2 else ''

    summary = []
    for session in sorted(sessions.keys()):
        files = sorted(sessions[session], key=filename_sort_key)   # sort by time within the recording
        seq_dir = os.path.join(dest_root, split, session)
        ev_dir  = os.path.join(seq_dir, 'event_representations_v2', EV_REPR_NAME)
        lab_dir = os.path.join(seq_dir, 'labels_v2')
        os.makedirs(ev_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        ev_h5 = os.path.join(ev_dir, f'event_representations{ds_suffix}.h5')

        repr_ts = []                      # global timestamps for each repr frame (length N)
        objframe_idx_2_repr_idx = []      # annotated frame -> repr index       (length M)
        objframe_ts = []                  # global timestamps for annotated frames        (length M)
        labels_list = []
        objframe_idx_2_label_idx = []
        label_start = 0
        prev_ts = None
        n = 0                             # count of written repr frames
        n_txt_missing = 0                 # count of missing .txt files
        n_txt_empty = 0                   # count of .txt files that exist but are empty (0 boxes)

        # ---- Stream-write event representation h5 one frame at a time (only one frame in memory at a time) ----
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
                    print(f'  [Skipped] {os.path.basename(h)}: {e}')
                    continue
                if os.path.exists(txt) and len(bbox) == 0:
                    n_txt_empty += 1

                # global monotonically increasing timestamp: use internal event clock; if not strictly increasing, force +1us
                ts = int(t_end)
                if prev_ts is not None and ts <= prev_ts:
                    ts = prev_ts + 1
                prev_ts = ts

                # write this repr frame
                dset.resize(n + 1, axis=0)
                dset[n] = repr_store
                repr_ts.append(ts)

                # only register as objframe if the frame has annotations (frames without annotations serve only as context/warmup frames)
                if len(bbox) > 0:
                    bbox = bbox.copy()
                    bbox['t'] = np.int64(ts)              # overwrite with global timestamp
                    objframe_idx_2_repr_idx.append(n)
                    objframe_idx_2_label_idx.append(label_start)
                    objframe_ts.append(ts)
                    labels_list.append(bbox)
                    label_start += len(bbox)
                n += 1

        # ---- Metadata: event representation side ----
        np.save(os.path.join(ev_dir, 'timestamps_us.npy'),
                np.asarray(repr_ts, dtype=np.int64))                       # length N
        np.save(os.path.join(ev_dir, 'objframe_idx_2_repr_idx.npy'),
                np.asarray(objframe_idx_2_repr_idx, dtype=np.int64))       # length M

        # ---- Metadata: annotation side ----
        labels_v2 = np.concatenate(labels_list) if labels_list else np.zeros((0,), BBOX_DTYPE)
        np.savez(os.path.join(lab_dir, 'labels.npz'),
                 labels=labels_v2,
                 objframe_idx_2_label_idx=np.asarray(objframe_idx_2_label_idx, dtype=np.int64))
        np.save(os.path.join(lab_dir, 'timestamps_us.npy'),
                np.asarray(objframe_ts, dtype=np.int64))                   # length M

        M = len(objframe_idx_2_repr_idx)
        max_seq_len = (objframe_idx_2_repr_idx[-1] + 1) if M > 0 else 0    # seq_len upper bound hint
        warn = ''
        if M == 0:
            warn = '  <<< WARNING: 0 annotated frames! This sequence cannot be used for training'
            if n_txt_missing == n:
                warn += f' (all {n} .txt files not found -> check LABEL_DIR / naming)'
            elif n_txt_empty == n:
                warn += f' (found .txt but all are empty files)'
        print(f'  Sequence {session}: {n} repr frames, {M} annotated frames -> {seq_dir}  '
              f'(only seq_len<={max_seq_len} has samples; txt missing={n_txt_missing} / empty={n_txt_empty}){warn}')
        summary.append((session, n, M, max_seq_len, n_txt_missing, n_txt_empty))

    print('\n=== Summary (index, repr_frames N, annotated_frames M, max_seq_len, txt_missing, txt_empty) ===')
    total_M = sum(s[2] for s in summary)
    for s in summary:
        print(f'  {s[0]}: N={s[1]}, M={s[2]}, max_seq_len={s[3]}, txt_missing={s[4]}, txt_empty={s[5]}')
    if total_M == 0:
        print('\n*** CRITICAL: Total annotated frames across all sequences = 0, dataset is not trainable! ***')
        print('    Most common cause: YOLO .txt files are not in the locations the script searches.')
        print('    Fix: if annotations are in a separate folder, change LABEL_DIR at the top to that folder path;')
        print('         if they are in the same directory as hdf5, ensure .txt and .hdf5 share the same name (only extension differs).')
    return summary


if __name__ == '__main__':
    # Three paths are independent of each other, modify as needed:
    # src       = r'D:\20260614_sdata_drtd_r1_process\2_hdf5_val'              # hdf5 event directory
    # label_dir = r'F:\dataset\DRTD\DRTD\aligned\labels\val'             # YOLO txt directory (any location, recursive stem matching)
    # dst       = r'D:\20260614_sdata_drtd_r1_process\rvt_steam_data\data\val'  # output root directory
    # convert(src, dst, split='val', label_dir=label_dir)

    # src       = r'D:\20260614_sdata_drtd_r1_process\2_hdf5_test'              # hdf5 event directory
    # label_dir = r'F:\dataset\DRTD\DRTD\aligned\labels\test'             # YOLO txt directory (any location, recursive stem matching)
    # dst       = r'D:\20260614_sdata_drtd_r1_process\rvt_steam_data\data\test'  # output root directory
    # convert(src, dst, split='test', label_dir=label_dir)

    src       = r'D:\20260614_sdata_drtd_r1_process\2_hdf5_train'              # hdf5 event directory
    label_dir = r'F:\dataset\DRTD\DRTD\aligned\labels\train'             # YOLO txt directory (any location, recursive stem matching)
    dst       = r'D:\20260614_sdata_drtd_r1_process\rvt_steam_data\data\train'  # output root directory
    convert(src, dst, split='train', label_dir=label_dir)



