#!/usr/bin/env -S conda run -n mwi_xpcs_analysis python3
"""
inspect_xpcs.py: A fast and interactive detector frame and metadata viewer. Second window plots the TTCF up to slider position. 

Loads frames cropped to the qmap ROI into RAM, then displays them in a
PyQtGraph ImageView. Qmaps are auto-detected from results.hdf files.

The TTCF is computed either with a plain numpy matmul or with boost_corr's
TwotimeCorrelator (torch, GPU-capable); pick with --engine / the Engine combo
in the TTCF window, and the torch device with --device / the Device combo.
Both give the same matrix, so switching is a speed choice — changing either
combo recomputes the current ROI immediately.

boost_corr needs `pip install boost_corr torch` in the running environment. If
they are missing the Engine combo still lists boost_corr, greyed out, with the
reason in its tooltip.

The TTCF ROI persists between runs (stored in detector coordinates, so it lands
on the same patch of detector even though the qmap crop moves from scan to
scan). The viewer opens on the last frame and correlates that ROI immediately,
so the full map is on screen at startup.

To-do
    - other sources of metadata (txt files vs NDAttributes)?
    - add Qxy, Qz values to cursor positions
    - "discover" masks instead of using qmaps.
    - integrate bad pixel mask

"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np

DATA_DIR = Path('/gdata/dm/8ID/8IDE/2026-2/marks202606/data')
BOTH_DIR  = Path('/gdata/dm/8ID/8IDE/2026-2/marks202606/analysis/Both')
OVERFLOW = 2**32 - 1

# CLI
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('data', help='HDF5 data file (path or stem; searched in DATA_DIR)')
p.add_argument('--qmap', default=None, help='qmap HDF5 file')
p.add_argument('--results', default=None, help='results HDF with embedded qmap')
p.add_argument('--every', type=int, default=1, metavar='N', help='use every Nth frame (default 1)')
p.add_argument('--pad', type=int, default=10, metavar='PIX', help='padding around qmap bounding box (default 10)')
p.add_argument('--log', action='store_true', help='start in log scale')
p.add_argument('--cmap', default='inferno', help='colormap name (default: inferno)')
p.add_argument('--metadata', nargs='+', default=[], metavar='KEY', help='NDAttribute keys to plot alongside the frames; none by default (e.g. --metadata biologic_current biologic_voltage)')
p.add_argument('--dt', type=float, default=None, metavar='SEC', help='frame period in seconds; overrides the value read from the HDF')
p.add_argument('--engine', choices=['auto', 'numpy', 'boost'], default='auto', help='TTCF engine: numpy, boost (boost_corr/torch), or auto (boost if importable)')
p.add_argument('--device', default='auto', metavar='DEV', help='torch device for the boost engine: auto, cpu, cuda, cuda:0, mps (default auto -> cuda if available); also switchable from the Device combo')
args = p.parse_args()

# TTCF engine — availability is settled up front so a bad --engine fails before
# we spend minutes loading frames into RAM. The modules themselves are imported
# lazily: `import torch` costs seconds, and most runs never touch the boost path.
import importlib.util

BOOST_MODULES = ('torch', 'boost_corr')

def boost_missing():
    """Names of the boost dependencies that are not installed.

    find_spec only proves the module is *present*; a broken build still raises on
    real import. That case is caught at first use and reported in the TTCF window
    rather than here, so a bad torch cannot stop the viewer from opening."""
    missing = []
    for mod in BOOST_MODULES:
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError):
            missing.append(mod)
    return missing

BOOST_MISSING = boost_missing()
BOOST_OK = not BOOST_MISSING
BOOST_HINT = 'pip install ' + ' '.join(BOOST_MISSING or BOOST_MODULES)

if args.engine == 'boost' and not BOOST_OK:
    sys.exit(f'ERROR: --engine boost needs {", ".join(BOOST_MISSING)} ({BOOST_HINT})')

default_engine = 'boost' if (args.engine in ('auto', 'boost') and BOOST_OK) else 'numpy'

_boost_mods = None      # (torch, TwotimeCorrelator) once imported

def load_boost():
    """Import torch/boost_corr on first use. Raises with the real error if the
    packages are present but unusable (wrong CUDA build, partial install …)."""
    global _boost_mods
    if _boost_mods is None:
        import torch
        from boost_corr import TwotimeCorrelator
        _boost_mods = (torch, TwotimeCorrelator)
    return _boost_mods

def resolve_device(name):
    """Map 'auto' to cuda when present, else cpu.

    mps stays opt-in. Older torch could not run boost_corr's diagonal averaging
    there at all (torch.bincount was unimplemented on Metal; it works as of 2.13),
    and even where it runs it measured slower than plain numpy on an M-series
    laptop — the boost engine pays off on cuda, not on Metal."""
    if name != 'auto':
        return name
    torch, _ = load_boost()
    return 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'TTCF : engine={default_engine}' +
      (f'  device={args.device}' if BOOST_OK
       else f'  (boost unavailable: no {", ".join(BOOST_MISSING)} — {BOOST_HINT})'))

# Find datafile 
data_path = Path(args.data)
if data_path.is_file():
    pass
elif '/' not in args.data:
    candidates = sorted(c for c in DATA_DIR.glob(f'{args.data}*') if c.is_dir())
    if not candidates:
        sys.exit(f'ERROR: no data directory matching {args.data}* in {DATA_DIR}')
    stem = candidates[0].name
    data_path = candidates[0] / f'{stem}.h5'
elif data_path.is_dir():
    stem = data_path.name
    data_path = data_path / f'{stem}.h5'
if not data_path.exists():
    sys.exit(f'ERROR: data file not found: {data_path}')
print(f'Data : {data_path}')

# Find qmap from results.hdf
def read_roi_map(hdf_path, prefix):
    with h5py.File(hdf_path, 'r') as f:
        return f[f'{prefix}/dynamic_roi_map'][:]

if args.qmap:
    qmap_path = Path(args.qmap)
    if not qmap_path.exists():
        sys.exit(f'ERROR: qmap not found: {args.qmap}')
    roi_map = read_roi_map(qmap_path, 'qmap')
    print(f'Qmap : {qmap_path}')
elif args.results:
    rp = Path(args.results)
    if not rp.exists():
        matches = sorted(BOTH_DIR.glob(f'{args.results}*results*.hdf'))
        if not matches:
            sys.exit(f'ERROR: results HDF not found: {args.results}')
        rp = matches[-1]
    roi_map = read_roi_map(rp, 'xpcs/qmap')
    print(f'Qmap : embedded in {rp}')
else:
    stem = data_path.parent.name
    matches = sorted(BOTH_DIR.glob(f'{stem}*results*.hdf'))
    if matches:
        rp = matches[-1]
        roi_map = read_roi_map(rp, 'xpcs/qmap')
        print(f'Qmap : auto-detected {rp}')
    else:
        sys.exit('ERROR: no --qmap or --results given and no auto-detected results HDF.')

# Frame period, so the TTCF can be plotted against time rather than frame index
FRAME_TIME_KEYS = (
    'entry/instrument/detector_1/frame_time',
    'entry/instrument/detector/frame_time',
    'entry/instrument/detector_1/acquire_period',
    'entry/instrument/detector/acquire_period',
    'entry/instrument/detector_1/count_time',
    'entry/instrument/detector/count_time',
)

FRAME_TIME_HINTS = ('frame_time', 'acquire_period', 'acquire_time', 'count_time',
                    'exposure', 'frame_period', 'dwell')

def _scalar(dset):
    """First finite positive element of a dataset, or None."""
    try:
        val = float(np.ravel(dset[()])[0])
    except (TypeError, ValueError, IndexError):
        return None
    return val if np.isfinite(val) and val > 0 else None

def read_frame_time(path):
    """First positive frame-period dataset found, or (None, None).

    Tries the canonical NeXus locations, then falls back to a name scan — 8-ID
    writes this in different places depending on detector and acquisition mode.
    """
    with h5py.File(path, 'r') as f:
        for key in FRAME_TIME_KEYS:
            if key in f:
                val = _scalar(f[key])
                if val is not None:
                    return val, key

        found = []
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and any(h in name.lower() for h in FRAME_TIME_HINTS):
                found.append(name)
        f.visititems(visit)          # metadata walk only; does not read frames

        for name in found:
            val = _scalar(f[name])
            if val is not None:
                return val, name
        if found:
            print(f'WARN : time-like datasets found but unusable: {found}')
    return None, None

if args.dt is not None:
    frame_time, dt_source = args.dt, '--dt'
else:
    frame_time, dt_source = read_frame_time(data_path)

TIME_AXIS = frame_time is not None
# --every subsamples, so consecutive loaded frames are that much further apart
dt = frame_time * args.every if TIME_AXIS else 1.0
if TIME_AXIS:
    print(f'Time : frame_time={frame_time:g} s ({dt_source})  ->  TTCF step {dt:g} s')
else:
    print('Time : no frame period found in HDF; TTCF axes stay in frames (set --dt to override)')

rows_with = np.where(roi_map.any(axis=1))[0]
cols_with = np.where(roi_map.any(axis=0))[0]
r0 = max(0, int(rows_with[0]) - args.pad)
r1 = min(roi_map.shape[0], int(rows_with[-1]) + args.pad + 1)
c0 = max(0, int(cols_with[0]) - args.pad)
c1 = min(roi_map.shape[1], int(cols_with[-1]) + args.pad + 1)
print(f'ROI  : rows {r0}-{r1-1}, cols {c0}-{c1-1}  ({r1-r0}x{c1-c0} px)')

# Load all frames into RAM
with h5py.File(data_path, 'r') as f:
    dset = f.get('entry/data/data') or f.get('entry/instrument/detector/data')
    if dset is None:
        sys.exit('ERROR: cannot find data dataset')
    n_total = dset.shape[0]
    frame_indices = list(range(0, n_total, args.every))
    n_frames = len(frame_indices)
    print(f'Loading {n_frames}/{n_total} frames … ', end='', flush=True)
    frames = np.array([dset[i, r0:r1, c0:c1].astype(np.float32) for i in frame_indices])
    # frames = np.array([dset[i, :, :].astype(np.float32) for i in frame_indices]) # this plots the full detector image, expect delays 


frames[frames >= OVERFLOW] = np.nan
print(f'done  ({frames.nbytes / 1e6:.0f} MB)')

# Load requested metadata arrays (NDAttributes)
meta_arrays = {}   # key -> np.ndarray length n_frames
if args.metadata:
    with h5py.File(data_path, 'r') as f:
        nd = f.get('entry/instrument/NDAttributes', {})
        for key in args.metadata:
            if key in nd:
                arr = nd[key][:]
                meta_arrays[key] = arr[frame_indices]
                print(f'Meta : {key}  range [{np.nanmin(meta_arrays[key]):.4g}, '
                      f'{np.nanmax(meta_arrays[key]):.4g}]')
            else:
                available = list(nd.keys()) if nd else []
                print(f'WARN : NDAttribute "{key}" not found.'
                      f'Available: {available}')

# colors
flat = frames[np.isfinite(frames)]
flat_pos = flat[flat > 0]
vmin = float(np.percentile(flat_pos, 2))  if len(flat_pos) else 1.0
vmax = float(np.percentile(flat_pos, 95)) if len(flat_pos) else 100.0
print(f'Scale: vmin={vmin:.3g}  vmax={vmax:.3g}')

# PyQtGraph viewer
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
pg.setConfigOptions(imageAxisOrder='row-major', antialias=False)


class CloseButtonOverlay(QtCore.QObject):
    """Overlay a small ✕ button in the top-right corner of a widget.

    Clicking it calls on_close. The button follows the widget on resize/show.
    """
    def __init__(self, target, on_close):
        super().__init__(target)
        self.target = target
        self.btn = QtWidgets.QPushButton('✕', target)
        self.btn.setFixedSize(18, 18)
        self.btn.setToolTip('Remove this panel')
        self.btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn.setStyleSheet(
            'QPushButton { border: none; color: #ccc; font-weight: bold;'
            ' background: rgba(0, 0, 0, 120); border-radius: 9px; }'
            'QPushButton:hover { color: #fff; background: rgba(200, 40, 40, 220); }')
        self.btn.clicked.connect(on_close)
        target.installEventFilter(self)
        self._reposition()
        self.btn.raise_()

    def eventFilter(self, obj, event):
        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show):
            self._reposition()
        return False

    def _reposition(self):
        margin = 4
        self.btn.move(self.target.width() - self.btn.width() - margin, margin)
        self.btn.raise_()

# main window 
win = QtWidgets.QMainWindow()
win.setWindowTitle(f'{data_path.parent.name}  [{n_frames} frames, every {args.every}]')
win.resize(900, 860)

central = QtWidgets.QWidget()
win.setCentralWidget(central)
layout = QtWidgets.QVBoxLayout(central)
layout.setContentsMargins(4, 4, 4, 4)

# splitter: ImageView on top, metadata plots on bottom 
splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
layout.addWidget(splitter)

# ImageView (includes timeline scrubber + ROI + histogram)
iv = pg.ImageView()
splitter.addWidget(iv)

# ✕ on the integration-ROI plot: removing it un-toggles the ROI button
def close_roi_plot():
    if iv.ui.roiBtn.isChecked():
        iv.ui.roiBtn.click()   # hides roiPlot and removes iv.roi from the view
CloseButtonOverlay(iv.ui.roiPlot, close_roi_plot)

# metadata plot panel (one PlotWidget per key, stacked vertically) 
META_COLORS = ['#FFD700', '#00BFFF', '#FF6347', '#7CFC00', '#FF69B4']
meta_plots   = {} # key: PlotWidget
meta_lines   = {} # key: PlotDataItem (trail)
meta_cursors = {} # key: ScatterPlotItem (current point)
x_all = np.array(frame_indices, dtype=float)

if meta_arrays:
    plot_panel = QtWidgets.QWidget()
    plot_layout = QtWidgets.QVBoxLayout(plot_panel)
    plot_layout.setContentsMargins(2, 2, 2, 2)
    plot_layout.setSpacing(2)
    splitter.addWidget(plot_panel)
    splitter.setSizes([620, 200])

    def close_meta_plot(key):
        pw = meta_plots.pop(key, None)
        meta_lines.pop(key, None)
        meta_cursors.pop(key, None)
        meta_arrays.pop(key, None)   # stop update_label from touching it
        if pw is not None:
            plot_layout.removeWidget(pw)
            pw.setParent(None)
            pw.deleteLater()
        if not meta_plots:           # hide the whole panel once empty
            plot_panel.hide()

    for idx, (key, vals) in enumerate(list(meta_arrays.items())):
        color = META_COLORS[idx % len(META_COLORS)]
        pw = pg.PlotWidget(title=key)
        pw.setMaximumHeight(160)
        pw.setLabel('left', key)
        pw.setLabel('bottom', 'frame')
        pw.showGrid(x=True, y=True, alpha=0.3)
        pw.setXRange(x_all[0], x_all[-1], padding=0.02)
        pw.setYRange(float(np.nanmin(vals * -1)), float(np.nanmax(vals * -1)), padding=0.05)

        # faint full-range ghost so you can see where you are
        pw.plot(x_all, vals*-1, pen=pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DotLine)) 

        # trail up to current frame
        line = pw.plot([], [], pen=pg.mkPen(color, width=2))

        # current-point marker
        cursor = pg.ScatterPlotItem(size=9, pen=pg.mkPen('w', width=1),
                                    brush=pg.mkBrush(color))
        pw.addItem(cursor)

        plot_layout.addWidget(pw)
        meta_plots[key] = pw
        meta_lines[key] = line
        meta_cursors[key] = cursor
        CloseButtonOverlay(pw, lambda *_, k=key: close_meta_plot(k))

# controls row 
ctrl_widget = QtWidgets.QWidget()
ctrl = QtWidgets.QHBoxLayout(ctrl_widget)
layout.addWidget(ctrl_widget)

log_btn = QtWidgets.QCheckBox('Log scale')
log_btn.setChecked(args.log)
ctrl.addWidget(log_btn)

cmap_combo = QtWidgets.QComboBox()
cmaps = ['gray', 'inferno', 'viridis', 'plasma', 'magma', 'hot']
cmap_combo.addItems(cmaps)
if args.cmap in cmaps:
    cmap_combo.setCurrentText(args.cmap)
ctrl.addWidget(QtWidgets.QLabel('Colormap:'))
ctrl.addWidget(cmap_combo)

snap_int_btn = QtWidgets.QPushButton('Snap integration ROI')
ctrl.addWidget(snap_int_btn)

# ctrl.addStretch()

frame_label = QtWidgets.QLabel()
ctrl.addWidget(frame_label)

# colormap helper 
def apply_cmap(name):
    try:
        import matplotlib
        color_ttcf = (matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        iv.setColorMap(pg.ColorMap(pos=np.linspace(0, 1, 256), color=color_ttcf))
    except Exception:
        pass

apply_cmap(args.cmap)

# display data 
display = np.log10(np.clip(frames, vmin, None)) if args.log else frames
iv.setImage(display, autoRange=False, autoLevels=False,
            autoHistogramRange=False, xvals=np.array(frame_indices, dtype=float))
iv.setLevels(np.log10(vmin) if args.log else vmin,
             np.log10(vmax) if args.log else vmax)

def update_label(idx):
    i = int(np.clip(idx, 0, n_frames - 1))
    fi = frame_indices[i]
    stamp = f'   t = {fi * frame_time:.3f} s' if TIME_AXIS else ''
    frame_label.setText(f'Frame {fi + 1} / {n_total}  (index {i}){stamp}')
    for key, vals in meta_arrays.items():
        meta_lines[key].setData(x_all[:i + 1], vals[:i + 1] * -1)
        meta_cursors[key].setData([x_all[i]], [vals[i] * -1])

iv.sigTimeChanged.connect(lambda: update_label(iv.currentIndex))
update_label(0)

# log/cmap callbacks 
def toggle_log(state):
    use_log = bool(state)
    d = np.log10(np.clip(frames, vmin, None)) if use_log else frames
    iv.setImage(d, autoRange=False, autoLevels=False, autoHistogramRange=False,
                xvals=np.array(frame_indices, dtype=float))
    iv.setLevels(np.log10(vmin) if use_log else vmin,
                 np.log10(vmax) if use_log else vmax)

log_btn.stateChanged.connect(toggle_log)
cmap_combo.currentTextChanged.connect(apply_cmap)

# f panel 
ttcf_win = QtWidgets.QMainWindow(win)
ttcf_win.setWindowTitle('TTCF')
ttcf_win.resize(win.size())   # match the scattering window

ttcf_central = QtWidgets.QWidget()
ttcf_win.setCentralWidget(ttcf_central)
ttcf_layout = QtWidgets.QVBoxLayout(ttcf_central)
ttcf_layout.setContentsMargins(4, 4, 4, 4)

# compute button — TTCF is calculated on demand so the ROI can be moved freely
compute_row = QtWidgets.QWidget()
compute_layout = QtWidgets.QHBoxLayout(compute_row)
compute_layout.setContentsMargins(0, 0, 0, 0)

compute_btn = QtWidgets.QPushButton('Compute TTCF')
compute_btn.setToolTip('Calculate the TTCF for the current ROI position')
compute_layout.addWidget(compute_btn)

compute_layout.addWidget(QtWidgets.QLabel('Engine:'))
engine_combo = QtWidgets.QComboBox()
engine_combo.addItem('numpy', 'numpy')
# boost is always listed. When its dependencies are missing the entry is greyed
# out with the reason attached, rather than vanishing — an engine that silently
# is not there looks like a broken dropdown.
engine_combo.addItem('boost_corr', 'boost')
if not BOOST_OK:
    engine_combo.setItemData(1, f'needs {", ".join(BOOST_MISSING)}  —  {BOOST_HINT}',
                             QtCore.Qt.ItemDataRole.ToolTipRole)
    # disable just this row; Qt has no per-item setEnabled, so clear its
    # selectable/enabled flags through the underlying model
    engine_combo.model().item(1).setEnabled(False)
engine_combo.setCurrentIndex(1 if default_engine == 'boost' else 0)
engine_combo.setToolTip(
    'numpy: CPU matmul, always available.\n'
    'boost_corr: TwotimeCorrelator (torch), runs on GPU when one is present.\n'
    'Both compute the same matrix — this is purely a speed choice.'
    + ('' if BOOST_OK else f'\n\nboost_corr is greyed out: no {", ".join(BOOST_MISSING)}'
                           f' in this environment.\n{BOOST_HINT}'))
compute_layout.addWidget(engine_combo)

# device for the boost engine, switchable without restarting the viewer
device_combo = QtWidgets.QComboBox()
device_combo.addItems(['auto', 'cpu', 'cuda', 'mps'])
if args.device not in ('auto', 'cpu', 'cuda', 'mps'):
    device_combo.addItem(args.device)      # e.g. cuda:1 from the command line
device_combo.setCurrentText(args.device)
device_combo.setToolTip(
    'torch device for the boost engine.\n'
    'auto -> cuda when available, otherwise cpu.\n\n'
    'cuda is where this engine earns its keep. On cpu it lands within a factor of\n'
    'a couple of numpy, and mps (Apple GPU) measured slower than both — old torch\n'
    'could not run it at all there, as boost_corr averages diagonals with\n'
    'torch.bincount, unimplemented on Metal before 2.13.')
device_label = QtWidgets.QLabel('Device:')
compute_layout.addWidget(device_label)
compute_layout.addWidget(device_combo)

def sync_device_row():
    """The device only means anything to the boost engine."""
    on = engine_combo.currentData() == 'boost'
    device_label.setEnabled(on)
    device_combo.setEnabled(on)

sync_device_row()

static_cb = QtWidgets.QCheckBox('Static norm')
static_cb.setChecked(True)
static_cb.setToolTip('Divide each pixel by its time average before correlating.\n'
                     'Removes the static speckle pattern so the baseline sits at 1,\n'
                     'matching what boost_corr\'s sqmap smoothing does in the full pipeline.')
compute_layout.addWidget(static_cb)

# Window for that time average. "all" reproduces the whole-series behaviour;
# a finite window makes the normaliser track a drifting speckle pattern.
static_win = QtWidgets.QSpinBox()
static_win.setRange(0, n_frames)
static_win.setValue(0)
static_win.setSpecialValueText('all')
static_win.setPrefix('win ')
static_win.setSuffix(' fr')
static_win.setToolTip(
    'Frames in the per-pixel time average.\n\n'
    '"all" divides by the whole-series mean. If the sample drifts, that mean is a\n'
    'motion-blurred version of the speckle pattern, and the residual leaves a slow\n'
    'decaying shoulder in the TTCF that mimics real dynamics.\n\n'
    'A finite window lets the normaliser follow the drift. It also high-pass\n'
    'filters in time: dynamics slower than the window get suppressed along with\n'
    'the drift, so only use this when the two are well separated. Pick a window\n'
    'much longer than the correlation times you care about and much shorter than\n'
    'the drift; sweep it and watch how the artifact moves.')
compute_layout.addWidget(static_win)
static_cb.toggled.connect(static_win.setEnabled)

compute_layout.addStretch()
ttcf_layout.addWidget(compute_row)

# PlotWidget as the TTCF image container
ttcf_pw = pg.PlotWidget()
ttcf_pw.setAspectLocked(True)
ttcf_pw.setLabel('bottom', 't1 (s)' if TIME_AXIS else 'frame t1')
ttcf_pw.setLabel('left', 't2 (s)' if TIME_AXIS else 'frame t2')
ttcf_img = pg.ImageItem()
ttcf_pw.addItem(ttcf_img)

# crosshairs at current frame
ttcf_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen('w', width=1,
                              style=QtCore.Qt.PenStyle.DashLine))
ttcf_hline = pg.InfiniteLine(angle=0,  pen=pg.mkPen('w', width=1,
                              style=QtCore.Qt.PenStyle.DashLine))
ttcf_pw.addItem(ttcf_vline)
ttcf_pw.addItem(ttcf_hline)

# draggable colour scale: histogram of the TTCF values with a level region you
# can drag/resize, plus a gradient editor (right-click it for other colormaps)
ttcf_hist = pg.HistogramLUTWidget()
ttcf_hist.setImageItem(ttcf_img)
ttcf_hist.setFixedWidth(140)

img_row = QtWidgets.QWidget()
img_layout = QtWidgets.QHBoxLayout(img_row)
img_layout.setContentsMargins(0, 0, 0, 0)
img_layout.setSpacing(2)
img_layout.addWidget(ttcf_pw, 1)
img_layout.addWidget(ttcf_hist)

# map on top, extracted g2 curves below — a splitter rather than a fixed split
# so the g2 panel can be dragged shut when only the map matters
ttcf_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
ttcf_split.addWidget(img_row)
ttcf_layout.addWidget(ttcf_split, 1)

# g2 panel: one-time correlation from the diagonals of user-chosen sub-blocks
g2_panel = QtWidgets.QWidget()
g2_layout = QtWidgets.QVBoxLayout(g2_panel)
g2_layout.setContentsMargins(0, 0, 0, 0)
g2_layout.setSpacing(2)

g2_ctrl = QtWidgets.QWidget()
g2_ctrl_layout = QtWidgets.QHBoxLayout(g2_ctrl)
g2_ctrl_layout.setContentsMargins(0, 0, 0, 0)
g2_ctrl_layout.addWidget(QtWidgets.QLabel('g2 ranges:'))

g2_range_edit = QtWidgets.QLineEdit('0-25%, 0-50%, 0-75%, 0-100%')
g2_range_edit.setToolTip(
    'Comma-separated time ranges to extract g2 from. Each one is a square\n'
    'block on the TTCF diagonal, outlined on the map in the curve colour.\n\n'
    'Units by suffix:\n'
    '  0-25%      percent of the series\n'
    '  10-60s     seconds' + ('' if TIME_AXIS else ' (no frame period known: = frames)') + '\n'
    '  200-800    loaded-frame index\n'
    'An open end runs to the start/end of the series ("-50%", "300-").\n'
    'A bare value is taken as the end of a range starting at 0 ("50%").\n\n'
    'Ranges are how you handle a non-stationary sample: cut it into intervals\n'
    'over which the dynamics are steady and compare, rather than averaging a\n'
    'changing system into one curve.')
g2_ctrl_layout.addWidget(g2_range_edit, 1)

g2_btn = QtWidgets.QPushButton('Extract g2')
g2_btn.setToolTip('Re-extract g2 for the ranges above (also runs after every TTCF)')
g2_ctrl_layout.addWidget(g2_btn)

g2_logx = QtWidgets.QCheckBox('log τ')
g2_logx.setChecked(True)
g2_ctrl_layout.addWidget(g2_logx)

g2_save_btn = QtWidgets.QPushButton('Save CSV')
g2_save_btn.setToolTip('Write the extracted curves to a CSV (range, tau, g2, npts)')
g2_ctrl_layout.addWidget(g2_save_btn)

g2_layout.addWidget(g2_ctrl)

g2_pw = pg.PlotWidget()
g2_pw.setLabel('bottom', 'lag τ (s)' if TIME_AXIS else 'lag τ (frames)')
g2_pw.setLabel('left', 'g2')
g2_pw.showGrid(x=True, y=True, alpha=0.3)
g2_pw.setLogMode(x=True, y=False)
g2_pw.addItem(pg.InfiniteLine(pos=1.0, angle=0,
                              pen=pg.mkPen('#888', width=1,
                                           style=QtCore.Qt.PenStyle.DashLine)))
g2_legend = g2_pw.addLegend(offset=(-10, 10))
g2_layout.addWidget(g2_pw, 1)

g2_status = QtWidgets.QLabel('')
g2_status.setStyleSheet('font-family: monospace; padding: 2px')
g2_layout.addWidget(g2_status)

ttcf_split.addWidget(g2_panel)
ttcf_split.setSizes([560, 300])

ttcf_status = QtWidgets.QLabel('Position the ROI on the detector image, then click "Compute TTCF"')
ttcf_status.setStyleSheet('font-family: monospace; padding: 2px')
ttcf_layout.addWidget(ttcf_status)

# colour map for TTCF (blue=anticorrelated, red=correlated). Feed it to the
# histogram's gradient rather than the image directly — the gradient owns the
# LUT once setImageItem has been called.
try:
    import matplotlib
    color_ttcf = (matplotlib.colormaps['RdBu_r'](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    ttcf_hist.gradient.setColorMap(pg.ColorMap(pos=np.linspace(0, 1, 256), color=color_ttcf))
except Exception:
    pass

# state
ttcf_matrix = None   # full NxN matrix once computed

def running_mean(a, w, chunk=4096):
    """Centered running mean along axis 0 (time), window w frames.

    The window slides rather than shrinking at the ends, so every output is an
    average of exactly w real frames — the first and last w//2 entries just stop
    being centred. Chunked over pixels to bound the float64 cumsum.
    """
    n = a.shape[0]
    w = int(np.clip(w, 1, n))
    start = np.clip(np.arange(n) - w // 2, 0, n - w)
    stop = start + w
    out = np.empty_like(a)
    for s in range(0, a.shape[1], chunk):
        blk = a[:, s:s + chunk]
        c = np.zeros((n + 1, blk.shape[1]), dtype=np.float64)
        np.cumsum(blk, axis=0, out=c[1:])
        out[:, s:s + chunk] = ((c[stop] - c[start]) / w).astype(a.dtype)
    return out

def compute_ttcf(pixel_stack):
    """pixel_stack: (n_frames, n_pixels) float32, may contain nan."""
    # mean intensity per frame (nanmean over pixels)
    mean_i = np.nanmean(pixel_stack, axis=1, keepdims=True)
    mean_i = np.where(mean_i == 0, np.nan, mean_i)
    norm = pixel_stack / mean_i
    # C[i,j] = mean_pixels( norm[i] * norm[j] )
    valid  = np.isfinite(norm).all(axis=1)
    n = pixel_stack.shape[0]
    C = np.full((n, n), np.nan, dtype=np.float32)
    nv = norm[valid]
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return C
    block = (nv @ nv.T) / nv.shape[1]
    C[np.ix_(idx, idx)] = block
    return C

def compute_ttcf_boost(pixel_stack, device):
    """Same TTCF via boost_corr's TwotimeCorrelator (torch, GPU-capable).

    The correlator expects pixels grouped contiguously per q-bin; a rectangular
    ROI is a single bin spanning the whole stack, so dq/sq are one full slice.
    pixel_stack: (n_frames, n_pixels) float32, all finite.

    Verified against compute_ttcf() to float32 precision (max relative difference
    ~5e-6): boost normalises by the per-frame pixel sum where we divide by the
    per-frame mean, which is the same thing up to the n_pixels it multiplies back.

    Memory: calc_normal_twotime builds an n_frames x n_frames int64 index matrix
    for its diagonal averaging on top of the float32 c2, so it needs ~12 bytes per
    matrix element on the device — ~1.2 GB at 10k frames.
    """
    torch, TwotimeCorrelator = load_boost()

    n_fr, n_pix = pixel_stack.shape
    qinfo = {'dq_idx': [1], 'dq_slc': [slice(0, n_pix)],
             'sq_idx': [1], 'sq_slc': [slice(0, n_pix)]}
    corr = TwotimeCorrelator(qinfo, frame_num=n_fr, det_size=(1, n_pix),
                             device=device, method='normal')
    corr.process(torch.from_numpy(np.ascontiguousarray(pixel_stack)).to(device))
    # one q-bin -> the generator yields exactly one c2, upper triangle only
    tri = next(corr.calc_normal_twotime(num_partials=1))
    return (tri + tri.T - np.diag(np.diag(tri))).astype(np.float32)


def update_ttcf_display(i):
    global ttcf_matrix
    if ttcf_matrix is None:
        return
    # show upper-left (i+1)×(i+1) block; mask the rest
    n = ttcf_matrix.shape[0]
    disp = np.full((n, n), np.nan, dtype=np.float32)
    disp[:i + 1, :i + 1] = ttcf_matrix[:i + 1, :i + 1]
    ttcf_img.setImage(disp, autoLevels=False)
    # offset by half a step so pixel i is centred on t = i*dt, not on its left edge
    ttcf_img.setRect(QtCore.QRectF(-dt / 2, -dt / 2, n * dt, n * dt))
    ttcf_vline.setPos(i * dt)
    ttcf_hline.setPos(i * dt)

# g2 extraction
def extract_g2(matrix, i0=0, i1=None, max_frac=0.5):
    """One-time correlation g2(tau) from a square block on the TTCF diagonal.

    matrix : NxN two-time map. Averaging its k-th diagonal over t is exactly the
             usual g2 at lag k, but restricting that average to a block lets a
             non-stationary run be cut into intervals where it is stationary.
    i0, i1 : block bounds in loaded-frame units, i1 exclusive (i1=None -> end).
    max_frac : longest lag kept, as a fraction of the block length. The k-th
             diagonal holds only m-k points, so the far tail is an average over
             a handful of pixels-worth of noise and drags the axis around.

    Returns (tau, g2, npts) with tau in *frames* (multiply by dt for seconds),
    starting at 1 — the tau=0 diagonal carries the shot-noise self term and is
    not part of g2.
    """
    n = matrix.shape[0]
    i0 = int(np.clip(i0, 0, n - 1))
    i1 = n if i1 is None else int(np.clip(i1, i0 + 1, n))
    blk = matrix[i0:i1, i0:i1]
    m = blk.shape[0]
    tau, g2, npts = [], [], []
    for k in range(1, min(max(1, int(m * max_frac)), m - 1) + 1):
        d = np.diagonal(blk, offset=k)
        good = np.isfinite(d)
        c = int(good.sum())
        if c:
            tau.append(k)
            g2.append(float(d[good].mean()))
            npts.append(c)
    return np.array(tau), np.array(g2, dtype=np.float64), np.array(npts)

def parse_g2_ranges(text, n):
    """'0-25%, 10-60s, 200-800' -> ([(i0, i1, label), ...], [bad tokens]).

    Suffix picks the unit: % of the series, s for seconds, bare for loaded-frame
    index. Either end may be left off ('-50%', '300-'); a bare value is the end
    of a range starting at zero.
    """
    ranges, bad = [], []
    for tok in text.replace(';', ',').split(','):
        label = tok.strip()
        if not label:
            continue
        s = label.lower().replace(' ', '')
        unit = 'fr'
        if s.endswith('%'):
            unit, s = '%', s[:-1]
        elif s.endswith('s'):
            unit, s = 's', s[:-1]
        lo_s, hi_s = s.split('-', 1) if '-' in s else ('', s)

        def to_idx(v, default):
            if v == '':
                return default
            x = float(v)
            if unit == '%':
                return int(round(x / 100.0 * n))
            if unit == 's':
                return int(round(x / dt))
            return int(round(x))

        try:
            i0, i1 = to_idx(lo_s, 0), to_idx(hi_s, n)
        except ValueError:
            bad.append(label)
            continue
        i0 = int(np.clip(i0, 0, n - 1))
        i1 = int(np.clip(i1, i0 + 1, n))
        ranges.append((i0, i1, label))
    return ranges, bad

G2_COLORS = ['#FFD700', '#00BFFF', '#FF6347', '#7CFC00', '#FF69B4', '#FFFFFF']
g2_curves = []   # PlotDataItems in the g2 panel
g2_boxes  = []   # matching block outlines drawn on the TTCF map
g2_last   = []   # (label, tau_x, g2, npts) of the last extraction, for CSV

def update_g2():
    """Re-extract and redraw every range in the box against the current TTCF."""
    for it in g2_curves:
        g2_pw.removeItem(it)
    g2_curves.clear()
    for it in g2_boxes:
        ttcf_pw.removeItem(it)
    g2_boxes.clear()
    g2_legend.clear()
    g2_last.clear()

    if ttcf_matrix is None:
        g2_status.setText('no TTCF yet — click "Compute TTCF"')
        return
    n = ttcf_matrix.shape[0]
    ranges, bad = parse_g2_ranges(g2_range_edit.text(), n)
    notes = [f'unparsed: {b}' for b in bad]

    for k, (i0, i1, label) in enumerate(ranges):
        color = G2_COLORS[k % len(G2_COLORS)]
        tau, g2, npts = extract_g2(ttcf_matrix, i0, i1)
        if len(tau) == 0:
            notes.append(f'{label}: block too short')
            continue
        tau_x = tau * dt
        g2_curves.append(g2_pw.plot(tau_x, g2, pen=pg.mkPen(color, width=2),
                                    name=f'{label}  [{i0}–{i1 - 1}]'))
        # outline the block it came from, in the curve's colour
        lo, hi = i0 * dt - dt / 2, (i1 - 1) * dt + dt / 2
        g2_boxes.append(ttcf_pw.plot(
            [lo, hi, hi, lo, lo], [lo, lo, hi, hi, lo],
            pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.PenStyle.DashLine)))
        g2_last.append((label, tau_x, g2, npts))
        span = (f't {i0 * dt:.4g}–{(i1 - 1) * dt:.4g} s' if TIME_AXIS
                else f'fr {i0}–{i1 - 1}')
        notes.append(f'{label}: {i1 - i0} fr ({span}), {len(tau)} lags')

    g2_pw.enableAutoRange(axis='y')
    g2_status.setText('  |  '.join(notes) if notes else 'no valid ranges')

def save_g2_csv():
    if not g2_last:
        g2_status.setText('nothing to save — extract g2 first')
        return
    default = str(Path.cwd() / f'{data_path.parent.name}_g2.csv')
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        ttcf_win, 'Save g2 curves', default, 'CSV (*.csv)')
    if not path:
        return
    unit = 'tau_s' if TIME_AXIS else 'tau_frames'
    with open(path, 'w') as fh:
        fh.write(f'# {data_path}\n')
        fh.write(f'# ROI det rows {int(sel_roi.pos()[1]) + r0}+{int(sel_roi.size()[1])}, '
                 f'cols {int(sel_roi.pos()[0]) + c0}+{int(sel_roi.size()[0])}\n')
        fh.write(f'range,{unit},g2,npts\n')
        for label, tau_x, g2, npts in g2_last:
            for t, v, c in zip(tau_x, g2, npts):
                fh.write(f'{label},{t:.6g},{v:.6g},{c}\n')
    g2_status.setText(f'saved {sum(len(c[1]) for c in g2_last)} points -> {path}')

g2_btn.clicked.connect(update_g2)
g2_range_edit.returnPressed.connect(update_g2)
g2_logx.toggled.connect(lambda on: g2_pw.setLogMode(x=on, y=False))
g2_save_btn.clicked.connect(save_g2_csv)

# draggable rectangle ROI on the detector image
H, W = frames.shape[1], frames.shape[2]
sel_roi = pg.RectROI(
    pos  = [W // 4, H // 4],
    size = [max(4, W // 8), max(4, H // 8)],
    pen  = pg.mkPen('y', width=2),
)
iv.getView().addItem(sel_roi)

# ROI position persists across runs, so the same patch of detector can be
# followed from scan to scan
SETTINGS = QtCore.QSettings('8ID', 'inspect_xpcs')

# the g2 ranges are part of the same "keep looking at the same thing" state
saved_ranges = SETTINGS.value('g2_ranges')
if saved_ranges:
    g2_range_edit.setText(str(saved_ranges))
g2_range_edit.editingFinished.connect(
    lambda: SETTINGS.setValue('g2_ranges', g2_range_edit.text()))

def save_roi(*_):
    """Store the ROI in *detector* coordinates. Frames are cropped to the qmap
    bounding box, and that origin moves from scan to scan, so a position saved
    in crop-local coordinates would come back pointing somewhere else."""
    x, y = sel_roi.pos()
    rw, rh = sel_roi.size()
    SETTINGS.setValue('ttcf_roi', [float(x) + c0, float(y) + r0, float(rw), float(rh)])

def load_roi():
    """Restore the saved ROI into this scan's crop. False if there is nothing
    usable — no saved value, or it lands entirely off this crop."""
    saved = SETTINGS.value('ttcf_roi')
    if saved is None:
        return False
    try:
        dx, dy, rw, rh = (float(v) for v in saved)
    except (TypeError, ValueError):
        return False
    x, y = dx - c0, dy - r0
    rw, rh = min(max(4.0, rw), float(W)), min(max(4.0, rh), float(H))
    if x + rw <= 0 or y + rh <= 0 or x >= W or y >= H:
        return False
    sel_roi.setPos([float(np.clip(x, 0, W - rw)), float(np.clip(y, 0, H - rh))])
    sel_roi.setSize([rw, rh])
    return True

if load_roi():
    x, y = sel_roi.pos()
    rw, rh = sel_roi.size()
    print(f'ROI  : TTCF ROI restored at det ({y + r0:.0f}, {x + c0:.0f}), '
          f'{rw:.0f}x{rh:.0f} px')

# save on every drag as well as at exit, so a hard kill does not lose it
sel_roi.sigRegionChangeFinished.connect(save_roi)
app.aboutToQuit.connect(save_roi)

def on_roi_changed():
    global ttcf_matrix
    # extract all frames through the ROI at once
    region = sel_roi.getArrayRegion(frames, iv.getImageItem(), axes=(1, 2))
    # region shape: (n_frames, roi_h, roi_w)
    if region is None or region.size == 0:
        ttcf_status.setText('ROI too small')
        return
    pixel_stack = region.reshape(n_frames, -1).astype(np.float32)
    # drop masked/overflow pixels rather than whole frames — a single hot pixel
    # would otherwise wipe out every frame it appears in
    good = np.isfinite(pixel_stack).all(axis=0)
    norm_note = 'none'
    if static_cb.isChecked():
        # Divide each pixel by its own time average. boost_corr's pipeline does
        # the equivalent in compute_smooth_data(), dividing by the mean over each
        # *static* q bin — but sqmap bins are far finer than the dynamic ones, so
        # that flattens the static scattering pattern. Our ROI is a single bin, so
        # that step would cancel exactly against the per-frame normalisation and
        # do nothing. Without it the baseline sits at <I^2>_p/<I>_p^2 (tens, for an
        # ROI spanning an intensity gradient) instead of 1.
        pix_mean = np.zeros(pixel_stack.shape[1], dtype=np.float32)
        pix_mean[good] = pixel_stack[:, good].mean(axis=0)
        good &= pix_mean > 0
        sub = pixel_stack[:, good]
        win = static_win.value()
        if 0 < win < n_frames:
            # Running version: the "static" pattern is only static if the sample
            # holds still. Under drift the whole-series mean blurs it, and the
            # leftover structure shows up as slow correlation. A local mean
            # tracks the drift instead — at the cost of high-passing the data,
            # so anything slower than the window goes with it.
            pm = running_mean(sub, win)
            # Sparse-count guard: over a short window a dim pixel can average to
            # zero. Floor the local level at 5% of that pixel's own global mean
            # so the ratio cannot blow up. Well below where a real window lands.
            np.maximum(pm, 0.05 * pix_mean[good], out=pm)
            pixel_stack = sub / pm
            # counts per pixel per window — if this is O(1) the window is too
            # short to estimate a local mean and you are dividing by noise
            ph = win * float(pix_mean[good].mean())
            norm_note = f'running w={win} fr (~{ph:.1f} ph/px/win)'
        else:
            pixel_stack = sub / pix_mean[good]
            norm_note = 'whole-series'
    else:
        pixel_stack = pixel_stack[:, good]
    n_dropped = int((~good).sum())
    n_pix = pixel_stack.shape[1]
    if n_pix == 0:
        ttcf_status.setText('ROI contains no usable pixels')
        return

    use_boost = engine_combo.currentData() == 'boost'
    engine = f'boost:{device_combo.currentText()}' if use_boost else 'numpy'
    ttcf_status.setText(f'Computing TTCF over {n_pix} pixels ({engine}) …')
    QtWidgets.QApplication.processEvents()
    t0 = time.perf_counter()
    try:
        if use_boost:
            # resolve inside the try: the first boost run is also where a broken
            # torch install shows itself, and that belongs in the status line
            device = resolve_device(device_combo.currentText())
            engine = f'boost:{device}'
            ttcf_matrix = compute_ttcf_boost(pixel_stack, device)
        else:
            ttcf_matrix = compute_ttcf(pixel_stack)
    except Exception as exc:
        ttcf_status.setText(f'{engine} failed: {type(exc).__name__}: {exc}'
                            + ('  —  switch Engine to numpy to carry on' if use_boost else ''))
        traceback.print_exc()    # full detail on the console; status bar is one line
        ttcf_matrix = None
        update_g2()          # clear the stale curves along with the map
        return
    elapsed = time.perf_counter() - t0
    # set colour scale: symmetric around 1
    finite = ttcf_matrix[np.isfinite(ttcf_matrix)]
    if len(finite):
        lo = max(0, float(np.percentile(finite, 2)))
        hi = float(np.percentile(finite, 98))
        ctr = 1.0
        half = max(hi - ctr, ctr - lo, 0.05)
        lo_level, hi_level = ctr - half, ctr + half
        # span the histogram axis over the real data range so the region has
        # somewhere to be dragged to, then seat it on the auto levels
        ttcf_hist.setHistogramRange(float(finite.min()), float(finite.max()))
        ttcf_hist.setLevels(lo_level, hi_level)
    ttcf_pw.setXRange(-dt / 2, (n_frames - 0.5) * dt, padding=0)
    ttcf_pw.setYRange(-dt / 2, (n_frames - 0.5) * dt, padding=0)
    update_ttcf_display(iv.currentIndex)
    dropped = f'  |  {n_dropped} px masked' if n_dropped else ''
    ttcf_status.setText(
        f'TTCF  {n_frames}×{n_frames}  |  {n_pix} px in ROI{dropped}  |  '
        f'static norm: {norm_note}  |  {engine}  {elapsed:.2f} s')
    update_g2()   # curves follow the map, so a new ROI never leaves stale g2 on screen

# TTCF is computed on demand via the button, so the ROI can be moved freely
compute_btn.clicked.connect(on_roi_changed)

def on_engine_changed():
    """Switching engine recomputes straight away.

    The engine used to be read only inside on_roi_changed, so picking one did
    nothing until the next Compute click and the combo looked dead. Both engines
    give the same matrix, so an immediate recompute is also the honest way to
    compare their speed — the elapsed time lands in the status line.

    This recomputes unconditionally, including when the last attempt left
    ttcf_matrix as None. A failed boost run (no GPU memory, broken cuda build)
    clears the map, and switching back to numpy has to be enough to get it back —
    guarding on "there is already a matrix" would strand the viewer empty.
    """
    sync_device_row()
    on_roi_changed()

engine_combo.currentIndexChanged.connect(on_engine_changed)
# device only affects the boost path, so leave numpy alone
device_combo.currentIndexChanged.connect(
    lambda: on_roi_changed() if engine_combo.currentData() == 'boost' else None)

def snap_integration_roi():
    """Snap iv.roi (integration ROI) to the current yellow TTCF sel_roi."""
    iv.roi.setPos(sel_roi.pos())
    iv.roi.setSize(sel_roi.size())
    if not iv.ui.roiBtn.isChecked():
        iv.ui.roiBtn.click()

snap_int_btn.clicked.connect(snap_integration_roi)

# pixel coordinate/value readout in status bar 
status_bar = win.statusBar()
status_bar.setStyleSheet('QStatusBar { font-family: monospace; }')

def on_mouse_move(evt):
    pos = evt[0]
    view = iv.getView()
    if view.sceneBoundingRect().contains(pos):
        pt = view.mapSceneToView(pos)
        col = int(pt.x())
        row = int(pt.y())
        if 0 <= row < H and 0 <= col < W:
            val = frames[iv.currentIndex, row, col]
            det_col = col + c0
            det_row = row + r0
            if np.isnan(val):
                status_bar.showMessage(
                    f'det ({det_row}, {det_col})   local ({row}, {col})   value: masked')
            else:
                status_bar.showMessage(
                    f'det ({det_row}, {det_col})   local ({row}, {col})   value: {val:.4g}')
        else:
            status_bar.clearMessage()
    else:
        status_bar.clearMessage()

mouse_proxy = pg.SignalProxy(iv.scene.sigMouseMoved, rateLimit=60, slot=on_mouse_move)

# hook slider: TTCF crosshair update
orig_update = update_label
def update_label(idx):
    orig_update(idx)
    update_ttcf_display(int(np.clip(idx, 0, n_frames - 1)))
iv.sigTimeChanged.disconnect()
iv.sigTimeChanged.connect(lambda: update_label(iv.currentIndex))

# Open on the last frame so the first TTCF on screen is the complete map rather
# than a single corner pixel. setCurrentIndex sets currentIndex and repaints, but
# guards the timeline update with ignoreTimeLine, so sigTimeChanged never fires —
# the frame label and TTCF crosshair have to be driven by hand.
LAST = n_frames - 1
iv.setCurrentIndex(LAST)
update_label(LAST)

ttcf_win.show()
win.show()

# The ROI persists across runs, so there is already a meaningful one to correlate
# on — compute straight away instead of making the first click do it. Paint the
# windows first so the status line is visible while this runs.
QtWidgets.QApplication.processEvents()
try:
    on_roi_changed()
    # explicit index rather than iv.currentIndex, which older pyqtgraph leaves
    # stale after setCurrentIndex — a stale 0 would draw only the first pixel
    update_ttcf_display(LAST)
except Exception as exc:
    ttcf_status.setText(f'Startup TTCF failed: {exc}  —  '
                        'reposition the ROI and click "Compute TTCF"')

app.exec()
