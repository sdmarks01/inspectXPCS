#!/usr/bin/env -S conda run -n mwi_xpcs_analysis python3
"""
inspect_xpcs.py: A fast and interactive detector frame and metadata viewer. Second window plots the TTCF up to slider position. 

Loads frames cropped to the qmap ROI into RAM, then displays them in a
PyQtGraph ImageView. Qmaps are auto-detected from results.hdf files.

Usage — nothing depends on the working directory; name the experiment and scan:

    python inspect_xpcs_bc.py marks202606 L0188

Experiments are found by name under $XPCS_ROOTS (default /gdata/dm/8ID), searched
up to three levels down, so the station and cycle directories never have to be
typed and the command keeps working next cycle. data/ and analysis/Both/ hang off
the experiment root that lookup returns. Set XPCS_EXPERIMENT to skip the first
argument, pass a plain path to bypass the lookup, or --list to see what is there.
Running from inside an experiment (its data/ folder, say) picks that experiment,
so a bare scan name is enough there.

"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np

# Where experiments live. An experiment root is the directory holding data/ and
# analysis/, e.g. /gdata/dm/8ID/8IDE/2026-2/marks202606 — station and cycle are
# discovered rather than hard-coded, so the same command works next cycle and on
# a laptop with the folders copied somewhere else (set XPCS_ROOTS for that).
SEARCH_ROOTS = [Path(r).expanduser() for r in
                os.environ.get('XPCS_ROOTS', '/gdata/dm/8ID').split(os.pathsep) if r]
# Globbing GPFS three levels deep costs seconds, and the answer never changes for
# a given experiment, so remember it.
ROOT_CACHE = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'inspect_xpcs' / 'roots.json'
OVERFLOW = 2**32 - 1

# CLI
EPILOG = """
examples
    inspect_xpcs.py marks202606 L0188      experiment + scan, from any directory
    inspect_xpcs.py L0188                  same, with --exp or $XPCS_EXPERIMENT set,
                                           or from inside that experiment's folder
    inspect_xpcs.py marks202606            list the scans in an experiment
    inspect_xpcs.py kisiel202602 A0079 --run 2   a repeated scan's second run
    inspect_xpcs.py /path/to/L0188/L0188.h5    explicit file, no lookup

environment
    XPCS_EXPERIMENT   default experiment, so only the scan need be typed
    XPCS_ROOTS        ':'-separated dirs searched for experiments
                      (default /gdata/dm/8ID; experiments may be nested up to
                      three levels down, as <station>/<cycle>/<experiment>)
"""
p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('target', nargs='*', metavar='ARG',
               help='<experiment> <scan> (e.g. marks202606 L0188); a bare <scan> when '
                    '--exp or $XPCS_EXPERIMENT is set; a bare <experiment> to list its '
                    'scans; or a path to a scan dir or .h5 file')
p.add_argument('--exp', default=os.environ.get('XPCS_EXPERIMENT'), metavar='NAME',
               help='experiment to look scans up in (default $XPCS_EXPERIMENT)')
p.add_argument('--data-dir', default=None, metavar='DIR',
               help='override the scan directory (default <experiment>/data)')
p.add_argument('--analysis-dir', default=None, metavar='DIR',
               help='override where results HDFs are searched (default <experiment>/analysis/Both)')
p.add_argument('--run', default=None, metavar='N',
               help='which repeat of the scan to open when there are several '
                    '(e.g. 3, r00003, or any part of the run directory name)')
p.add_argument('--list', action='store_true',
               help='list matching scans (or experiments, with no arguments) and exit')
p.add_argument('--qmap', default=None, help='qmap HDF5 file')
p.add_argument('--results', default=None, help='results HDF with embedded qmap')
p.add_argument('--every', type=int, default=1, metavar='N', help='use every Nth frame (default 1)')
p.add_argument('--pad', type=int, default=10, metavar='PIX', help='padding around qmap bounding box (default 10)')
p.add_argument('--log', action='store_true', help='start in log scale')
p.add_argument('--cmap', default='inferno', help='colormap name (default: inferno)')
p.add_argument('--metadata', nargs='+', default=[], metavar='KEY', help='NDAttribute keys to plot alongside the frames; none by default (e.g. --metadata biologic_current biologic_voltage)')
p.add_argument('--dt', type=float, default=None, metavar='SEC', help='frame period in seconds; overrides the value read from the HDF')
p.add_argument('--norm', choices=['bins', 'series', 'running', 'none'], default='bins',
               help="static normalisation applied before correlating: bins (per-frame mean "
                    "in each fine static q-bin, what the 8-ID pipeline does; the default, and "
                    "the only one valid when the sample itself changes during the run), series "
                    "(per-pixel whole-series time average), running (per-pixel time average "
                    "over --norm-win frames), or none")
p.add_argument('--norm-win', type=int, default=0, metavar='N',
               help='window in frames for --norm running (default 0 = whole series)')
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

# Experiment / scan lookup 
# Nothing below depends on the working directory: an experiment name is resolved
# to its root under SEARCH_ROOTS, and data/ and analysis/ hang off that.
def _cache_load():
    try:
        return json.loads(ROOT_CACHE.read_text())
    except (OSError, ValueError):
        return {}

def _cache_store(name, path):
    data = _cache_load()
    data[name] = str(path)
    try:
        ROOT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ROOT_CACHE.write_text(json.dumps(data, indent=1, sort_keys=True))
    except OSError:
        pass   # a read-only or full home is no reason to fail the run

EXP_MARKERS = ('data', 'analysis')

def is_experiment(path):
    """An experiment root is a directory holding data/ and/or analysis/.

    Deliberately loose: the folder names below an experiment change between
    beamtimes, so anything stricter would need editing every cycle."""
    return path.is_dir() and any((path / m).is_dir() for m in EXP_MARKERS)

def find_experiment(name):
    """Experiment name (or path) -> its root directory, or None.

    Exact names win over prefixes and shallow hits over deep ones, so
    'marks202606' cannot be shadowed by 'marks202606_old' three levels down."""
    direct = Path(name).expanduser()
    if is_experiment(direct):
        return direct
    cached = _cache_load().get(name)
    if cached and is_experiment(Path(cached)):
        return Path(cached)
    for leaf in (name, f'{name}*'):
        for depth in ('', '*/', '*/*/', '*/*/*/'):
            hits = sorted({c for root in SEARCH_ROOTS if root.is_dir()
                           for c in root.glob(depth + leaf) if is_experiment(c)})
            if hits:
                if len(hits) > 1:
                    print(f'WARN : {len(hits)} experiments match {name!r}, using the last:')
                    for h in hits:
                        print(f'       {h}')
                _cache_store(name, hits[-1])
                return hits[-1]
    return None

def list_experiments(limit=60):
    found = sorted({c for root in SEARCH_ROOTS if root.is_dir()
                    for depth in ('*', '*/*', '*/*/*', '*/*/*/*')
                    for c in root.glob(depth) if is_experiment(c)})
    return found[:limit]

def scan_dirs(prefix=''):
    if DATA_DIR is None or not DATA_DIR.is_dir():
        return []
    return sorted(c for c in DATA_DIR.glob(f'{prefix}*') if c.is_dir())

def find_results(prefix):
    """Results HDFs for a scan stem, oldest first.

    analysis/Both is the 8-ID convention, but a reprocessed experiment can put
    them in a sibling folder, so fall back to a shallow sweep of analysis/
    rather than claiming there is no qmap."""
    pat = f'{prefix}*results*.hdf'
    hits = sorted(BOTH_DIR.glob(pat)) if BOTH_DIR is not None and BOTH_DIR.is_dir() else []
    if not hits and EXP_ROOT is not None:
        analysis = EXP_ROOT / 'analysis'
        if analysis.is_dir():
            hits = sorted(set(analysis.glob(pat)) | set(analysis.glob(f'*/{pat}'))
                          | set(analysis.glob(f'*/*/{pat}')))
    return hits

# --- work out what was asked for
targets = list(args.target)
if len(targets) > 2:
    p.error(f'expected at most <experiment> <scan>, got {len(targets)} arguments')
exp_name, scan_name = args.exp, None
if len(targets) == 2:
    exp_name, scan_name = targets
elif len(targets) == 1:
    only = targets[0]
    inside_exp = any(is_experiment(c) for c in [Path.cwd(), *Path.cwd().parents])
    if exp_name or inside_exp or '/' in only or Path(only).expanduser().exists():
        scan_name = only
    elif find_experiment(only) is not None:
        # a bare experiment name on its own: show what is in it
        exp_name, args.list = only, True
    else:
        scan_name = only

EXP_ROOT = None
if exp_name:
    EXP_ROOT = find_experiment(exp_name)
    if EXP_ROOT is None:
        avail = list_experiments()
        listing = ('\n  '.join(str(a) for a in avail) if avail
                   else f'(nothing found under {", ".join(str(r) for r in SEARCH_ROOTS)})')
        sys.exit(f'ERROR: no experiment matching {exp_name!r}. Known experiments:\n  {listing}\n'
                 'Set XPCS_ROOTS if the data lives somewhere else.')
    print(f'Exp  : {EXP_ROOT}')

if EXP_ROOT is None and not args.data_dir:
    # Standing inside an experiment is a statement of which one you mean, so
    # `inspect_xpcs L0188` works from the data folder with nothing else set.
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if is_experiment(cand):
            EXP_ROOT = cand
            print(f'Exp  : {EXP_ROOT}  (from the working directory)')
            break

DATA_DIR = (Path(args.data_dir).expanduser() if args.data_dir
            else (EXP_ROOT / 'data' if (EXP_ROOT / 'data').is_dir() else EXP_ROOT)
            if EXP_ROOT else None)
BOTH_DIR = (Path(args.analysis_dir).expanduser() if args.analysis_dir
            else EXP_ROOT / 'analysis' / 'Both' if EXP_ROOT else None)

if args.list:
    if EXP_ROOT is None and not args.data_dir:
        for e in list_experiments():
            print(e)
    else:
        for c in scan_dirs(scan_name or ''):
            print(c.name)
    sys.exit(0)

if scan_name is None:
    p.error('nothing to open: give <experiment> <scan>, a scan path, '
            'or --list to see what is available')

# --- scan -> data file
def scan_h5(scan_dir):
    """The raw .h5 inside a scan directory, whatever the nesting.

    Layouts vary between beamtimes — <scan>/<scan>.h5 for a single acquisition,
    <scan>/<scan>_rNNNNN/<scan>_rNNNNN.h5 once it is repeated, and no promise
    that the next cycle looks like either. So: search shallowest first, prefer a
    file named after its own directory (the 8-ID habit), and when several remain
    take the first and print the others rather than guessing silently. --run
    filters on any part of the path below the scan directory."""
    flat = scan_dir / f'{scan_dir.name}.h5'
    if flat.is_file():
        return flat
    cands = []
    for depth in ('*.h5', '*/*.h5', '*/*/*.h5', '*/*/*/*.h5'):
        found = sorted(c for c in scan_dir.glob(depth) if c.is_file())
        if not found:
            continue
        named = [c for c in found if c.stem == c.parent.name]
        cands = named or found
        break
    if not cands:
        sys.exit(f'ERROR: no .h5 anywhere under {scan_dir}')
    rel = lambda c: str(c.relative_to(scan_dir))
    # scan names run to 50 characters and the run folder repeats them, so collapse
    # the repeat to '*' when printing choices — only the tail distinguishes them
    show = lambda c: rel(c).replace(scan_dir.name, '*')
    if args.run:
        want = args.run
        if want.lstrip('r').isdigit():
            # A run number is matched only as _rNNNNN, never as a substring: the
            # '2' of a bare --run 2 also sits in 'VO2', and the '9' of --run 9 in
            # 'A0079', so falling back to substring would quietly open run 1.
            tag = f'r{int(want.lstrip("r")):05d}'    # '3' and 'r3' both mean _r00003
            picked = [c for c in cands if tag in rel(c)]
        else:
            picked = [c for c in cands if want in rel(c)]
        if not picked:
            listing = '\n  '.join(show(c) for c in cands)
            sys.exit(f'ERROR: nothing matching --run {want!r} in {scan_dir}\nFound:\n  {listing}')
        cands = picked
    if len(cands) > 1:
        others = ', '.join(show(c) for c in cands[1:6])
        print(f'WARN : {len(cands)} data files under {scan_dir.name}, using {show(cands[0])} '
              f'(also: {others}{" …" if len(cands) > 6 else ""}) — pick with --run')
    return cands[0]

def find_scan(name):
    given = Path(name).expanduser()
    if given.is_file():
        return given
    if given.is_dir():
        return scan_h5(given)
    if '/' in name:
        sys.exit(f'ERROR: no such scan path: {given}')
    if DATA_DIR is None:
        sys.exit(f'ERROR: {name!r} is not a path and no experiment was given — '
                 'use "<experiment> <scan>", --exp, or set $XPCS_EXPERIMENT')
    loose = DATA_DIR / f'{name}.h5'          # layouts that keep the h5 flat
    if loose.is_file():
        return loose
    hits = scan_dirs(name)
    exact = [c for c in hits if c.name == name]
    if exact:
        hit = exact[0]
    elif hits:
        hit = hits[0]
        if len(hits) > 1:
            others = ', '.join(c.name for c in hits[1:6])
            print(f'WARN : {len(hits)} scans match {name!r}, using {hit.name} (also: {others})')
    else:
        near = [c.name for c in scan_dirs()][:15]
        listing = ('\n  '.join(near) + ('\n  …' if len(near) == 15 else '')) if near else '(none)'
        sys.exit(f'ERROR: no scan matching {name}* in {DATA_DIR}\nScans there:\n  {listing}')
    return scan_h5(hit)

data_path = find_scan(scan_name)
if not data_path.exists():
    sys.exit(f'ERROR: data file not found: {data_path}')
print(f'Data : {data_path}')

if EXP_ROOT is None:
    # A scan given as a bare path still needs analysis/ for the auto-detected
    # qmap, so climb out of <experiment>/data/<scan>/<scan>.h5 to find it.
    for cand in data_path.parents:
        if is_experiment(cand):
            EXP_ROOT = cand
            DATA_DIR = DATA_DIR or EXP_ROOT / 'data'
            BOTH_DIR = BOTH_DIR or EXP_ROOT / 'analysis' / 'Both'
            print(f'Exp  : {EXP_ROOT}  (from the given path)')
            break

# Find qmap from results.hdf
def read_roi_map(hdf_path, prefix):
    with h5py.File(hdf_path, 'r') as f:
        return f[f'{prefix}/dynamic_roi_map'][:]

def read_static_map(hdf_path, prefix):
    """The fine static q-bin map from the qmap, or None if it has none.

    The full pipeline divides every frame by its own mean inside one of these
    bins before correlating (boost_corr's compute_smooth_data). They are much
    finer than the dynamic bins — hundreds of them across the detector — so that
    mean is a *per-frame* measure of the static scattering pattern, and it keeps
    working when the pattern itself changes during the run. See the Static norm
    combo for why that distinction decides whether the TTCF is dynamics or not.
    """
    with h5py.File(hdf_path, 'r') as f:
        d = f.get(f'{prefix}/static_roi_map')
        return None if d is None else d[:]

if args.qmap:
    qmap_path = Path(args.qmap)
    if not qmap_path.exists():
        sys.exit(f'ERROR: qmap not found: {args.qmap}')
    roi_map = read_roi_map(qmap_path, 'qmap')
    qmap_prefix = 'qmap'
    aux_path = qmap_path
    print(f'Qmap : {qmap_path}')
elif args.results:
    rp = Path(args.results)
    if not rp.exists():
        matches = find_results(args.results)
        if not matches:
            sys.exit(f'ERROR: results HDF not found: {args.results}')
        rp = matches[-1]
    roi_map = read_roi_map(rp, 'xpcs/qmap')
    qmap_prefix = 'xpcs/qmap'
    aux_path = rp
    print(f'Qmap : embedded in {rp}')
else:
    # <scan>_rNNNNN first, then the scan without the run suffix: analysis is
    # usually per run, but a repeated scan can be reduced once for the whole set
    stems = [data_path.parent.name]
    if data_path.parent.parent != DATA_DIR and data_path.parent.parent.name not in stems:
        stems.append(data_path.parent.parent.name)
    matches, stem = [], stems[0]
    for stem in stems:
        matches = find_results(stem)
        if matches:
            break
    if matches:
        rp = matches[-1]
        roi_map = read_roi_map(rp, 'xpcs/qmap')
        qmap_prefix = 'xpcs/qmap'
        aux_path = rp
        print(f'Qmap : auto-detected {rp}')
    else:
        searched = BOTH_DIR if BOTH_DIR is not None else '(no analysis dir known)'
        sys.exit(f'ERROR: no results HDF matching {" or ".join(s + "*results*.hdf" for s in stems)}\n'
                 f'       under {searched}. Pass --qmap or --results explicitly.')

static_map_full = read_static_map(aux_path, qmap_prefix)
if static_map_full is None:
    print(f'WARN : no {qmap_prefix}/static_roi_map in {aux_path.name} — the "static bins"\n'
          '       normalisation is unavailable; see the Static norm combo.')
elif static_map_full.shape != roi_map.shape:
    print(f'WARN : static_roi_map {static_map_full.shape} does not match '
          f'dynamic_roi_map {roi_map.shape}; ignoring it')
    static_map_full = None

# Frame period, so the TTCF can be plotted against time rather than frame index.
#
# Per-frame timestamps come first and the nominal exposure keys are only a
# fallback: FrameTime/CountTime say how long the detector integrates, not how
# often it is triggered, and at 8-ID those differ (0.3 s exposure on a 0.5 s
# cadence on the Cu111 runs). Using the exposure would compress every time axis
# and every g2 lag by the duty cycle.
TIMESTAMP_KEYS = (
    'entry/instrument/NDAttributes/NDArrayTimeStamp',
    'entry/instrument/detector_1/timestamp',
    'entry/instrument/detector/timestamp',
)
TIMESTAMP_HINTS = ('timestamp', 'frametimes', 'timeseries')

FRAME_TIME_KEYS = (
    'entry/instrument/detector_1/frame_time',
    'entry/instrument/detector/frame_time',
    # areaDetector NDAttributes — how the 8-ID Eiger/Lambda raw files write it
    'entry/instrument/NDAttributes/FrameTime',
    'entry/instrument/NDAttributes/AcquirePeriod',
    'entry/instrument/detector_1/acquire_period',
    'entry/instrument/detector/acquire_period',
    'entry/instrument/NDAttributes/CountTime',
    'entry/instrument/detector_1/count_time',
    'entry/instrument/detector/count_time',
)

# matched against the dataset name with '_' stripped, so 'frame_time' also
# catches the CamelCase NDAttribute spellings ('FrameTime', 'CountTime')
FRAME_TIME_HINTS = ('frametime', 'acquireperiod', 'acquiretime', 'counttime',
                    'exposure', 'frameperiod', 'dwell')

def _scalar(dset):
    """Representative finite positive value of a dataset, or None.

    NDAttributes are per-frame arrays, so take the median rather than the first
    element — a single glitched frame should not set the whole time axis.
    """
    try:
        arr = np.ravel(np.asarray(dset[()], dtype=float))
    except (TypeError, ValueError):
        return None
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if not arr.size:
        return None
    return float(np.median(arr))

def _period_from_timestamps(dset):
    """Frame period from a per-frame timestamp series, or None.

    Least-squares slope of t against frame index, not the median difference:
    the median is dragged around by per-frame jitter (0.5027 s vs a true
    0.5000 s on a 3600-frame Cu111 run), while the slope averages that jitter
    out over the whole series.
    """
    try:
        if dset.ndim != 1 or dset.shape[0] < 16:   # too short to fit a slope
            return None
        t = np.asarray(dset[()], dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    good = np.isfinite(t)
    if good.sum() < 16:
        return None
    t = t[good]
    if t[-1] <= t[0]:               # a timestamp series has to advance
        return None
    idx = np.arange(t.size, dtype=float)
    slope = float(np.polyfit(idx, t - t[0], 1)[0])
    if not np.isfinite(slope) or not 1e-9 < slope < 1e4:
        return None
    return slope

def read_frame_time(path):
    """Frame period in seconds and where it came from, or (None, None).

    Order: measured cadence from per-frame timestamps, then the canonical
    frame-period datasets, then a name scan — 8-ID writes the latter in
    different places depending on detector and acquisition mode.
    """
    with h5py.File(path, 'r') as f:
        # the nominal exposure, looked up first only so it can be compared
        # against the measured cadence in the report below
        nominal = None
        for key in FRAME_TIME_KEYS:
            if key in f:
                val = _scalar(f[key])
                if val is not None:
                    nominal = (val, key)
                    break

        ts_found = []
        def visit_ts(name, obj):
            flat = name.lower().replace('_', '')
            if isinstance(obj, h5py.Dataset) and any(h in flat for h in TIMESTAMP_HINTS):
                ts_found.append(name)
        f.visititems(visit_ts)       # metadata walk only; does not read frames

        ordered = [k for k in TIMESTAMP_KEYS if k in f]
        ordered += [n for n in ts_found if n not in ordered]
        for name in ordered:
            val = _period_from_timestamps(f[name])
            if val is None:
                continue
            if nominal is not None and abs(val - nominal[0]) > 0.02 * val:
                short = nominal[1].rsplit('/', 1)[-1]
                print(f'Time : {short}={nominal[0]:g} s is the exposure, not the '
                      f'cadence — using {val:g} s/frame measured from {name}')
            return val, f'{name}, measured'

        if nominal is not None:
            return nominal[0], nominal[1]

        found = []
        def visit(name, obj):
            flat = name.lower().replace('_', '')
            if isinstance(obj, h5py.Dataset) and any(h in flat for h in FRAME_TIME_HINTS):
                found.append(name)
        f.visititems(visit)

        for name in found:
            val = _scalar(f[name])
            if val is not None:
                return val, name
        if found or ts_found:
            print(f'WARN : time-like datasets found but unusable: {found + ts_found}')
        else:
            # nothing matched by name — dump the per-frame series that exist so
            # an unfamiliar file layout can be diagnosed without a second script
            series = []
            def visit_series(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 1 and obj.shape[0] >= 16:
                    series.append(f'{name} [{obj.shape[0]}]')
            f.visititems(visit_series)
            if series:
                print('WARN : no frame period found. Per-frame series in this file:')
                for name in series:
                    print(f'       {name}')
    return None, None

if args.dt is not None:
    frame_time, dt_source = args.dt, '--dt'
else:
    frame_time, dt_source = read_frame_time(data_path)
    if frame_time is None:
        # the analysis HDF carries detector_1/frame_time even when the raw file
        # keeps neither timestamps nor an exposure key
        frame_time, dt_source = read_frame_time(aux_path)
        if frame_time is not None:
            dt_source = f'{aux_path.name}:{dt_source}'

TIME_AXIS = frame_time is not None
# --every subsamples, so consecutive loaded frames are that much further apart
dt = frame_time * args.every if TIME_AXIS else 1.0
X_LABEL = 'time (s)' if TIME_AXIS else 'frame'
if TIME_AXIS:
    print(f'Time : frame_time={frame_time:g} s ({dt_source})  ->  TTCF step {dt:g} s')
else:
    print(f'Time : no frame period found in {data_path.name}; axes stay in frames '
          f'(pass --dt SEC to set it by hand)')

rows_with = np.where(roi_map.any(axis=1))[0]
cols_with = np.where(roi_map.any(axis=0))[0]
r0 = max(0, int(rows_with[0]) - args.pad)
r1 = min(roi_map.shape[0], int(rows_with[-1]) + args.pad + 1)
c0 = max(0, int(cols_with[0]) - args.pad)
c1 = min(roi_map.shape[1], int(cols_with[-1]) + args.pad + 1)
print(f'ROI  : rows {r0}-{r1-1}, cols {c0}-{c1-1}  ({r1-r0}x{c1-c0} px)')

# the static bin labels travel with the frames, cropped identically, so a box
# drawn on the displayed image can be mapped straight onto them
static_map = None if static_map_full is None else static_map_full[r0:r1, c0:c1]

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

    The button is drawn on top of the widget, so anything the widget itself
    puts in that corner becomes unclickable. On the timeline strip that corner
    is where the frame scrubber parks at the last frame — which is where the
    viewer opens — so for a PlotWidget the plot is inset by the width of the
    button and the button sits over the margin instead of over the data.
    """
    SIZE = 18
    MARGIN = 4

    def __init__(self, target, on_close):
        super().__init__(target)
        self.target = target
        self.btn = QtWidgets.QPushButton('✕', target)
        self.btn.setFixedSize(self.SIZE, self.SIZE)
        self.btn.setToolTip('Remove this panel')
        self.btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn.setStyleSheet(
            'QPushButton { border: none; color: #ccc; font-weight: bold;'
            ' background: rgba(0, 0, 0, 120); border-radius: 9px; }'
            'QPushButton:hover { color: #fff; background: rgba(200, 40, 40, 220); }')
        self.btn.clicked.connect(on_close)

        # keep the plot contents clear of the button's footprint
        get_item = getattr(target, 'getPlotItem', None)
        if get_item is not None:
            item = get_item()
            l, t, r, b = item.getContentsMargins()
            item.setContentsMargins(l, t, max(r, self.SIZE + 2 * self.MARGIN), b)

        target.installEventFilter(self)
        self._reposition()
        self.btn.raise_()

    def eventFilter(self, obj, event):
        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show):
            self._reposition()
        return False

    def _reposition(self):
        self.btn.move(self.target.width() - self.btn.width() - self.MARGIN,
                      self.MARGIN)
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

# the timeline strip doubles as the integration-ROI plot, and both take their x
# from the xvals handed to setImage below (seconds when the frame period is known)
iv.ui.roiPlot.setLabel('bottom', X_LABEL)
iv.ui.roiPlot.setLabel('left', 'ROI mean')

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
# x axis shared by the ImageView timeline (and so the integration-ROI plot)
# and the metadata plots: seconds when the frame period is known, else index
x_all = np.array(frame_indices, dtype=float) * (frame_time if TIME_AXIS else 1.0)

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
        pw.setLabel('bottom', X_LABEL)
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

fft_btn = QtWidgets.QPushButton('FFT of ROI intensity')
ctrl.addWidget(fft_btn)

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
            autoHistogramRange=False, xvals=x_all)
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
                xvals=x_all)
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

compute_layout.addWidget(QtWidgets.QLabel('Static norm:'))
norm_combo = QtWidgets.QComboBox()
norm_combo.addItem('static bins', 'bins')
norm_combo.addItem('whole series', 'series')
norm_combo.addItem('running win', 'running')
norm_combo.addItem('none', 'none')
if static_map is None:
    norm_combo.setItemData(0, 'this qmap carries no static_roi_map',
                           QtCore.Qt.ItemDataRole.ToolTipRole)
    norm_combo.model().item(0).setEnabled(False)
norm_combo.setCurrentIndex(
    {'bins': 0, 'series': 1, 'running': 2, 'none': 3}[
        'series' if (args.norm == 'bins' and static_map is None) else args.norm])
norm_combo.setToolTip(
    'What the pixel values are divided by before correlating, so that the TTCF\n'
    'baseline sits at 1 and what is left above it is dynamics.\n\n'
    'static bins: each frame by its own mean inside each fine static q-bin, which\n'
    '  is what boost_corr does in the full pipeline. The normaliser is measured\n'
    '  per frame, so it follows the static scattering pattern however fast that\n'
    '  pattern changes. This is the only choice that stays honest when the sample\n'
    '  itself changes during the run (plating, dissolution, a phase transition).\n\n'
    'whole series: each pixel by its average over every frame. Cheap, and correct\n'
    '  only while the static pattern holds still for the whole run. When it does\n'
    '  not, the residual is a state label: every pair of frames in the same state\n'
    '  shares the same leftover pattern and correlates, so the map fills with an\n'
    '  off-diagonal checkerboard of the cycle that has nothing to do with motion.\n\n'
    'running win: the same per-pixel average over a sliding window — a middle\n'
    '  ground that tracks slow drift but high-passes the dynamics with it.\n\n'
    'none: raw pixels. The baseline lands at <I^2>_p/<I>_p^2 for the ROI, tens if\n'
    '  it spans an intensity gradient, and swamps the contrast.')
compute_layout.addWidget(norm_combo)

# Window for the running average. "all" collapses it onto the whole-series case.
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
static_win.setValue(max(0, args.norm_win))
compute_layout.addWidget(static_win)

def sync_norm_row():
    """The window only means anything to the running average."""
    static_win.setEnabled(norm_combo.currentData() == 'running')

sync_norm_row()

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

ttcf_save_btn = QtWidgets.QPushButton('Save TTCF')
ttcf_save_btn.setToolTip('Write the whole two-time matrix to .npy — feeds\n'
                         'g2_osc_fit.py --ttcf, which tells a heterodyne beat\n'
                         '(stripes) apart from a pattern oscillation (checkerboard)')
g2_ctrl_layout.addWidget(ttcf_save_btn)

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

def clip_status(label):
    """Stop a one-line status label from setting the window's minimum width.

    An unwrapped QLabel demands room for its whole text, so a long message (a
    saved file path, several g2 notes) grew the TTCF window and then pinned it
    there — it could not be dragged narrower again. Ignoring the horizontal
    size hint lets the text clip instead; the full message goes in the tooltip.
    """
    label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                        QtWidgets.QSizePolicy.Policy.Preferred)
    set_text = label.setText
    def setText(text):
        set_text(text)
        label.setToolTip(text)
    label.setText = setText
    label.setToolTip(label.text())

clip_status(ttcf_status)
clip_status(g2_status)

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

def save_ttcf_npy():
    """Dump the raw two-time matrix so it can be analysed outside the GUI.

    The g2 CSV only carries the diagonal averages; the stripe-vs-checkerboard
    question needs the full map, because it turns on whether the correlation
    depends on t2 - t1 alone or on t1 and t2 separately.
    """
    if ttcf_matrix is None:
        g2_status.setText('no TTCF yet — click "Compute TTCF"')
        return
    default = str(Path.cwd() / f'{data_path.parent.name}_ttcf.npy')
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        ttcf_win, 'Save TTCF matrix', default, 'NumPy (*.npy)')
    if not path:
        return
    np.save(path, ttcf_matrix)
    n = ttcf_matrix.shape[0]
    g2_status.setText(f'saved {n}x{n} TTCF -> {path}')

g2_btn.clicked.connect(update_g2)
g2_range_edit.returnPressed.connect(update_g2)
g2_logx.toggled.connect(lambda on: g2_pw.setLogMode(x=on, y=False))
g2_save_btn.clicked.connect(save_g2_csv)
ttcf_save_btn.clicked.connect(save_ttcf_npy)

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

# A static bin sliced by the ROI down to fewer pixels than this normalises its
# pixels largely by themselves; below ~8 the self term is a visible bite out of
# the contrast. The real pipeline never sees this — it holds the whole bin.
MIN_BIN_PX = 8

def roi_static_labels(shape):
    """Static q-bin label per ROI pixel, ordered like the pixel stack.

    Sampled through the same ROI mapping as the frames so the two line up pixel
    for pixel, with order=0 so labels are picked rather than interpolated — the
    average of bin 7 and bin 9 would be a bin 8 that borders on neither.

    Returns None if the qmap has no static map or the sampling comes back a
    different shape than the frames did, in which case the caller falls back to
    a normalisation that needs no labels.
    """
    if static_map is None:
        return None
    lab = static_map[None, :, :].astype(np.float32)
    try:
        reg = sel_roi.getArrayRegion(lab, iv.getImageItem(), axes=(1, 2), order=0)
    except Exception as exc:
        print(f'static bins: label sampling failed: {type(exc).__name__}: {exc}')
        return None
    if reg is None or reg.shape[1:] != tuple(shape):
        print(f'static bins: labels came back {None if reg is None else reg.shape[1:]}, '
              f'frames {tuple(shape)} — falling back to whole-series')
        return None
    return np.rint(reg.reshape(-1)).astype(np.int64)

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
    mode = norm_combo.currentData()
    norm_note = 'none'
    if mode == 'bins':
        # What the full pipeline does (boost_corr's compute_smooth_data): divide
        # every frame by its own mean inside each *static* q bin. The static bins
        # are far finer than the dynamic ones — tens of them across this ROI — so
        # that mean is a per-frame measure of the static scattering pattern.
        #
        # The per-pixel time average below cannot do that job here. It assumes the
        # static pattern is the same in every frame, and in a plating run it is
        # not: the pattern during plating is not the pattern at rest. Divide by
        # the average of the two and every frame keeps a residue that says which
        # state it was in — positive where that state is bright, negative where it
        # is dim. Two frames in the same state then carry the *same* residue and
        # correlate strongly however far apart in time they are, which paints the
        # off-diagonal checkerboard of the electrochemical cycle over a map that
        # is supposed to show motion.
        labels = roi_static_labels(region.shape[1:])
        if labels is None:
            norm_combo.setCurrentIndex(1)      # falls back and recomputes
            return
        good &= labels > 0         # 0 is "no static bin here" in these maps
        uniq, inv = np.unique(labels[good], return_inverse=True)
        cnt = np.bincount(inv, minlength=uniq.size)
        # A static bin clipped by the ROI down to a handful of pixels normalises
        # a pixel largely by itself, which pins it near 1 in every frame and eats
        # the contrast. Cheaper to drop those pixels than to explain them.
        keep = cnt[inv] >= MIN_BIN_PX
        thin = int((~keep).sum())
        good[np.where(good)[0][~keep]] = False
        if not good.any():
            ttcf_status.setText(
                f'every static bin in this ROI has < {MIN_BIN_PX} px — '
                'enlarge the ROI or switch Static norm to "whole series"')
            return
        uniq, inv = np.unique(labels[good], return_inverse=True)
        cnt = np.bincount(inv, minlength=uniq.size).astype(np.float32)
        sub = pixel_stack[:, good]
        # per-frame sum in each bin: sort pixels by bin, then one reduceat
        order = np.argsort(inv, kind='stable')
        starts = np.searchsorted(inv[order], np.arange(uniq.size))
        bmean = np.add.reduceat(sub[:, order], starts, axis=1) / cnt
        # a dim bin can read a flat zero in a single frame; floor it at 5% of its
        # own series level so the ratio cannot blow up, as in the running case
        np.maximum(bmean, np.maximum(0.05 * bmean.mean(axis=0), 1e-6), out=bmean)
        pixel_stack = sub / bmean[:, inv]
        norm_note = (f'{uniq.size} static bins (~{cnt.mean():.0f} px/bin)'
                     + (f', {thin} px in thin bins dropped' if thin else ''))
    elif mode in ('series', 'running'):
        # Divide each pixel by its own time average. Correct only while the
        # static pattern holds still for the whole run — see the "bins" branch.
        # Without any of this the baseline sits at <I^2>_p/<I>_p^2 (tens, for an
        # ROI spanning an intensity gradient) instead of 1.
        pix_mean = np.zeros(pixel_stack.shape[1], dtype=np.float32)
        pix_mean[good] = pixel_stack[:, good].mean(axis=0)
        good &= pix_mean > 0
        sub = pixel_stack[:, good]
        win = static_win.value() if mode == 'running' else 0
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
# the normalisation decides what the map means, so it recomputes immediately too
norm_combo.currentIndexChanged.connect(lambda: (sync_norm_row(), on_roi_changed()))
static_win.editingFinished.connect(
    lambda: on_roi_changed() if norm_combo.currentData() == 'running' else None)
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

# FFT of the integration-ROI intensity trace 
# The bottom strip of the main window shows <I> inside iv.roi versus frame; this
# turns that same trace into an amplitude spectrum, so a periodic drive (pump,
# stirring, a 60 Hz pickup) can be read off directly instead of eyeballed.
fft_win = QtWidgets.QMainWindow(win)
fft_win.setWindowTitle('FFT of integrated ROI intensity')
fft_win.resize(820, 520)

fft_central = QtWidgets.QWidget()
fft_win.setCentralWidget(fft_central)
fft_layout = QtWidgets.QVBoxLayout(fft_central)
fft_layout.setContentsMargins(4, 4, 4, 4)

fft_ctrl_widget = QtWidgets.QWidget()
fft_ctrl = QtWidgets.QHBoxLayout(fft_ctrl_widget)

fft_ctrl.addWidget(QtWidgets.QLabel('Detrend:'))
detrend_combo = QtWidgets.QComboBox()
detrend_combo.addItem('mean', 'mean')       # kill DC only
detrend_combo.addItem('linear', 'linear')   # also kill beam decay / drift
detrend_combo.addItem('none', 'none')
fft_ctrl.addWidget(detrend_combo)

hann_cb = QtWidgets.QCheckBox('Hann window')
hann_cb.setChecked(True)
hann_cb.setToolTip('Taper the ends so a non-integer number of cycles does not '
                   'smear into a broad skirt')
fft_ctrl.addWidget(hann_cb)

fft_logx = QtWidgets.QCheckBox('log f')
fft_logy = QtWidgets.QCheckBox('log amp')
fft_logy.setChecked(True)
fft_ctrl.addWidget(fft_logx)
fft_ctrl.addWidget(fft_logy)

fft_refresh_btn = QtWidgets.QPushButton('Recompute')
fft_ctrl.addWidget(fft_refresh_btn)
fft_save_btn = QtWidgets.QPushButton('Save CSV')
fft_ctrl.addWidget(fft_save_btn)
fft_ctrl.addStretch()
fft_layout.addWidget(fft_ctrl_widget)

fft_pw = pg.PlotWidget()
fft_pw.setLabel('bottom', 'frequency (Hz)' if TIME_AXIS else 'frequency (1/frame)')
fft_pw.setLabel('left', 'amplitude (counts/px)')
fft_pw.showGrid(x=True, y=True, alpha=0.3)
fft_curve = fft_pw.plot([], [], pen=pg.mkPen('#00BFFF', width=1))
fft_peak = pg.ScatterPlotItem(size=9, pen=pg.mkPen('w', width=1),
                              brush=pg.mkBrush('#FFD700'))
fft_pw.addItem(fft_peak)
fft_layout.addWidget(fft_pw, 1)

# the trace itself, so what was transformed can be checked by eye
trace_pw = pg.PlotWidget()
trace_pw.setMaximumHeight(170)
trace_pw.setLabel('bottom', 't (s)' if TIME_AXIS else 'frame')
trace_pw.setLabel('left', 'mean I in ROI')
trace_pw.showGrid(x=True, y=True, alpha=0.3)
trace_curve = trace_pw.plot([], [], pen=pg.mkPen('w', width=1))
fft_layout.addWidget(trace_pw)

fft_status = QtWidgets.QLabel('')
fft_status.setStyleSheet('QLabel { font-family: monospace; }')
fft_layout.addWidget(fft_status)
clip_status(fft_status)

fft_last = None   # (freqs, amps, trace) of the most recent transform

def roi_intensity_trace():
    """Mean counts per pixel inside iv.roi, per frame, from the raw frames.

    Deliberately not iv.roiCurve.getData(): that curve follows the *display*
    array, which is log10 under the Log scale checkbox, and the FFT of a log
    signal is not the spectrum of the intensity."""
    # the ROI is only parented into the view while the roi button is on, and
    # getArrayRegion maps through that parent — off means no usable mapping
    if not iv.ui.roiBtn.isChecked():
        iv.ui.roiBtn.click()
    try:
        region = iv.roi.getArrayRegion(frames, iv.getImageItem(), axes=(1, 2))
    except Exception as exc:
        print(f'ROI FFT: getArrayRegion failed: {type(exc).__name__}: {exc}')
        return None
    if region is None or region.size == 0:
        return None
    stack = region.reshape(n_frames, -1).astype(np.float64)
    good = np.isfinite(stack).all(axis=0)   # drop masked/hot pixels, not frames
    if not good.any():
        return None
    return stack[:, good].mean(axis=1)

def compute_fft():
    global fft_last
    trace = roi_intensity_trace()
    if trace is None:
        fft_status.setText('integration ROI covers no usable pixels — '
                           'move it onto the detector image')
        fft_curve.setData([], [])
        fft_peak.setData([], [])
        trace_curve.setData([], [])
        fft_last = None
        return
    n = len(trace)
    if n < 4:
        fft_status.setText(f'only {n} frames — nothing to transform')
        return

    t = np.arange(n) * dt
    trace_curve.setData(t, trace)

    mode = detrend_combo.currentData()
    sig = trace.copy()
    if mode == 'mean':
        sig -= sig.mean()
    elif mode == 'linear':
        # a slow decay (beam current, bleaching) otherwise leaks a 1/f ramp
        # across the whole spectrum and buries everything real under it
        sig -= np.polyval(np.polyfit(t, sig, 1), t)

    if hann_cb.isChecked():
        w = np.hanning(n)
        # coherent gain: keep amplitudes comparable to the un-windowed case
        sig = sig * w / (w.mean() or 1.0)

    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=dt)
    amps = np.abs(spec) * (2.0 / n)   # single-sided amplitude
    if len(amps):
        amps[0] = np.abs(spec[0]) / n           # DC has no mirror image
    if n % 2 == 0 and len(amps) > 1:
        amps[-1] = np.abs(spec[-1]) / n         # nor does Nyquist

    # skip the DC bin: it is either zero by construction (detrended) or so large
    # it flattens everything else
    f_plot, a_plot = freqs[1:], amps[1:]
    fft_curve.setData(f_plot, a_plot)
    fft_last = (freqs, amps, trace)

    unit = 'Hz' if TIME_AXIS else '1/frame'
    if len(a_plot):
        k = int(np.argmax(a_plot))
        fpk, apk = float(f_plot[k]), float(a_plot[k])
        fft_peak.setData([fpk], [apk])
        period = (f'  |  period {1.0 / fpk:.4g} '
                  + ('s' if TIME_AXIS else 'frames')) if fpk > 0 else ''
        peak_note = f'peak {fpk:.5g} {unit}{period}  (amp {apk:.4g})'
    else:
        fft_peak.setData([], [])
        peak_note = 'no non-DC bins'
    x, y = iv.roi.pos()
    rw, rh = iv.roi.size()
    fft_status.setText(
        f'{n} frames  |  df {freqs[1]:.4g} {unit}  |  Nyquist {freqs[-1]:.4g} {unit}'
        f'  |  mean I {trace.mean():.4g}  |  {peak_note}\n'
        f'ROI det rows {int(y) + r0}+{int(rh)}, cols {int(x) + c0}+{int(rw)}'
        f'  |  detrend: {mode}, window: {"hann" if hann_cb.isChecked() else "none"}')

def save_fft_csv():
    if fft_last is None:
        fft_status.setText('nothing to save — compute a spectrum first')
        return
    default = str(Path.cwd() / f'{data_path.parent.name}_roi_fft.csv')
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        fft_win, 'Save ROI FFT', default, 'CSV (*.csv)')
    if not path:
        return
    freqs, amps, trace = fft_last
    x, y = iv.roi.pos()
    rw, rh = iv.roi.size()
    unit = 'freq_hz' if TIME_AXIS else 'freq_per_frame'
    with open(path, 'w') as fh:
        fh.write(f'# {data_path}\n')
        fh.write(f'# integration ROI det rows {int(y) + r0}+{int(rh)}, '
                 f'cols {int(x) + c0}+{int(rw)}\n')
        fh.write(f'# detrend={detrend_combo.currentData()} '
                 f'window={"hann" if hann_cb.isChecked() else "none"} '
                 f'dt={dt:g}{" s" if TIME_AXIS else " frames"}\n')
        fh.write(f'{unit},amplitude\n')
        for f_, a_ in zip(freqs, amps):
            fh.write(f'{f_:.8g},{a_:.8g}\n')
    fft_status.setText(f'saved {len(freqs)} bins -> {path}')

def show_fft():
    """Open the FFT window on the current integration ROI.

    Also switches the ROI on if it is off: without it the strip at the bottom of
    the main window is hidden and it is not obvious which box the spectrum
    belongs to."""
    if not iv.ui.roiBtn.isChecked():
        iv.ui.roiBtn.click()
    compute_fft()
    fft_win.show()
    fft_win.raise_()
    fft_win.activateWindow()

fft_btn.clicked.connect(show_fft)
fft_refresh_btn.clicked.connect(compute_fft)
fft_save_btn.clicked.connect(save_fft_csv)
detrend_combo.currentIndexChanged.connect(lambda: compute_fft() if fft_win.isVisible() else None)
hann_cb.toggled.connect(lambda: compute_fft() if fft_win.isVisible() else None)
fft_logx.toggled.connect(lambda on: fft_pw.setLogMode(x=on, y=fft_logy.isChecked()))
fft_logy.toggled.connect(lambda on: fft_pw.setLogMode(x=fft_logx.isChecked(), y=on))
fft_pw.setLogMode(x=False, y=True)
# moving/resizing the integration ROI re-transforms, but only while the window
# is up — otherwise every drag pays for a full-stack reduction for nothing
iv.roi.sigRegionChangeFinished.connect(
    lambda *_: compute_fft() if fft_win.isVisible() else None)

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
