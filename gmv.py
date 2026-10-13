#!/usr/bin/env python3
"""
Ground Motion Visualization (GMV) for SmartSolo Geophone Arrays.

Creates animated visualizations of seismic ground motion recorded by
dense arrays of SmartSolo geophones. Inspired by the IRIS/EarthScope GMV
data product (Trabant et al., 2012).

Author: Michail Henry (michail.henry@protonmail.com)
"""

import argparse
import glob
import os
import sys
from collections import OrderedDict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import obspy
import yaml
from matplotlib.animation import FuncAnimation, PillowWriter
from obspy import UTCDateTime, read, read_inventory
from obspy.core.inventory import (
    Inventory, Network, Station, Channel, Site, Response,
)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_station_coordinates(csv_path, channel="DPZ"):
    """
    Load station coordinates from geonat_mseed.csv.

    Returns dict: {station_code: (lon, lat, elevation, serial_number)}
    """
    stations = {}
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        idx = {col: i for i, col in enumerate(header)}
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < len(header):
                continue
            chan = parts[idx["channel"]]
            if chan != channel:
                continue
            sta = parts[idx["station"]]
            if sta in stations:
                continue
            stations[sta] = (
                float(parts[idx["longitude"]]),
                float(parts[idx["latitude"]]),
                float(parts[idx["elevation"]]),
                parts[idx["serial_number"]],
            )
    print(f"Loaded coordinates for {len(stations)} stations (channel={channel})")
    return stations


def find_mseed_files(mseed_dirs, station_codes, channel, event_date):
    """
    Find miniSEED files matching station codes, channel, and date.

    Parameters
    ----------
    mseed_dirs : list of str
        Directories containing miniSEED files.
    station_codes : set of str
        Station codes to search for.
    channel : str
        Channel code (e.g., DPZ).
    event_date : str
        Date string in format YYYY.MM.DD.

    Returns
    -------
    dict : {station_code: file_path}
    """
    found = {}
    for mseed_dir in mseed_dirs:
        pattern = os.path.join(mseed_dir, f"SS.*.SW.{channel}.*.{event_date}.*.miniseed")
        for filepath in glob.glob(pattern):
            fname = os.path.basename(filepath)
            parts = fname.split(".")
            sta = parts[1]  # Station code from filename
            if sta in station_codes and sta not in found:
                found[sta] = filepath
    print(f"Found {len(found)} miniSEED files for {event_date} (channel={channel})")
    return found


def get_resp_file(serial_number, resp_files):
    """
    Determine which RESP file to use based on serial number prefix.

    Parameters
    ----------
    serial_number : str
        Full serial number of the sensor.
    resp_files : dict
        Mapping of serial number prefix to RESP file path.

    Returns
    -------
    str or None
        Path to RESP file, or None if not found.
    """
    for prefix, resp_path in resp_files.items():
        if serial_number.startswith(prefix):
            return resp_path
    return None


def parse_pz_file(pz_file):
    """
    Parse a SAC pole-zero (.pz) file.

    Format:
        ZEROS n
        real imag
        ...
        POLES n
        real imag
        ...
        CONSTANT value

    Returns a PAZ dictionary compatible with ObsPy's simulate().
    """
    zeros = []
    poles = []
    constant = 1.0
    mode = None

    with open(pz_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("*"):
                continue
            if line.startswith("ZEROS"):
                mode = "zeros"
                continue
            elif line.startswith("POLES"):
                mode = "poles"
                continue
            elif line.startswith("CONSTANT"):
                constant = float(line.split()[1])
                mode = None
                continue

            if mode in ("zeros", "poles"):
                parts = line.split()
                if len(parts) >= 2:
                    c = complex(float(parts[0]), float(parts[1]))
                    if mode == "zeros":
                        zeros.append(c)
                    else:
                        poles.append(c)

    # For PZ files, CONSTANT = A0 * sensitivity
    # ObsPy's simulate expects separate gain and sensitivity
    paz = {
        "poles": poles,
        "zeros": zeros,
        "gain": 1.0,
        "sensitivity": constant,
    }
    return paz


def parse_resp_file(resp_file):
    """
    Parse a RESP file and extract poles, zeros, gain, and sensitivity.

    Returns a PAZ dictionary compatible with ObsPy's simulate().
    """
    zeros = []
    poles = []
    gain = 1.0
    sensitivity = 1.0

    with open(resp_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("B053F10-13"):
                # Complex zero
                parts = line.split()
                zeros.append(complex(float(parts[2]), float(parts[3])))
            elif line.startswith("B053F15-18"):
                # Complex pole
                parts = line.split()
                poles.append(complex(float(parts[2]), float(parts[3])))
            elif line.startswith("B053F07"):
                # A0 normalization factor
                gain = float(line.split()[-1])
            elif line.startswith("B058F04"):
                # Stage gain or overall sensitivity (last one = overall)
                sensitivity = float(line.split()[-1])

    paz = {
        "poles": poles,
        "zeros": zeros,
        "gain": gain,
        "sensitivity": sensitivity,
    }
    return paz


def parse_response_file(filepath):
    """
    Auto-detect and parse a response file (RESP or PZ format).

    Returns a PAZ dictionary.
    """
    if filepath.endswith(".pz"):
        return parse_pz_file(filepath)
    elif filepath.endswith(".RESP"):
        return parse_resp_file(filepath)
    else:
        # Try to detect from content
        with open(filepath, "r") as f:
            content = f.read(200)
        if "B053F" in content or "B058F" in content:
            return parse_resp_file(filepath)
        elif "ZEROS" in content or "POLES" in content:
            return parse_pz_file(filepath)
        else:
            # Default to RESP format
            return parse_resp_file(filepath)


# Cache parsed RESP files
_paz_cache = {}


def remove_response_with_paz(trace, resp_file, output_units, pre_filter):
    """
    Remove instrument response using poles and zeros parsed from a response file.

    Supports both RESP and PZ formats. This avoids the evalresp
    station/network matching issue by directly applying PAZ response removal.
    """
    if resp_file not in _paz_cache:
        _paz_cache[resp_file] = parse_response_file(resp_file)
    paz = _paz_cache[resp_file]

    trace.simulate(
        paz_remove=paz,
        paz_simulate=None,
        pre_filt=pre_filter,
    )
    return trace


def cascaded_decimate(trace, factor):
    """
    Decimate a trace by cascading smaller decimation steps.

    ObsPy's decimate() fails for factors > 16. This function breaks
    large factors into smaller steps.
    """
    remaining = factor
    # Factor into steps of max 16
    while remaining > 1:
        step = min(remaining, 16)
        # Find largest factor of remaining that is <= 16
        for s in range(step, 0, -1):
            if remaining % s == 0:
                step = s
                break
        trace.decimate(factor=step, no_filter=False)
        remaining //= step
    return trace


def load_and_process_waveforms(config, station_coords):
    """
    Load miniSEED data, remove instrument response, filter, and prepare for animation.

    Returns
    -------
    stream : obspy.Stream
        Processed waveform data.
    station_info : dict
        {station_code: (lon, lat)} for stations with valid data.
    """
    eq = config["earthquake"]
    tw = config["time_window"]
    proc = config["processing"]
    data = config["data"]

    origin = UTCDateTime(eq["origin_time"])

    # Auto-compute time window from phase arrivals if requested
    if tw.get("auto", False):
        from math import radians, degrees, sin, cos, atan2, sqrt, asin
        array_csv = data["station_csv"]
        # Quick array center from first few stations
        with open(array_csv) as _f:
            _hdr = _f.readline().strip().split(",")
            _idx = {c: i for i, c in enumerate(_hdr)}
            _lons, _lats = [], []
            for _line in _f:
                _p = _line.strip().split(",")
                if len(_p) >= len(_hdr):
                    _lons.append(float(_p[_idx["longitude"]]))
                    _lats.append(float(_p[_idx["latitude"]]))
        _alat, _alon = np.mean(_lats), np.mean(_lons)
        _dist, _ = compute_distance_azimuth(_alat, _alon, eq["latitude"], eq["longitude"])
        _tt = compute_travel_times(_dist, eq["depth_km"])
        p_time = _tt.get("P", _tt.get("PKP", 600))
        r1_time = _tt.get("R1", _dist * 28.5)
        pre_p = tw.get("before_P", 10)
        post_r1 = tw.get("after_R1", 20)
        t_start = origin + p_time - pre_p
        t_end = origin + r1_time + post_r1
        print(f"Auto time window: P-{pre_p}s ({p_time:.0f}s) to R1+{post_r1}s ({r1_time:.0f}s) = {t_end - t_start:.0f}s")
    else:
        t_start = origin - tw["pre_event_seconds"]
        t_end = origin + tw["post_event_seconds"]

    # Date string for file matching
    event_date = origin.strftime("%Y.%m.%d")

    channel = proc["channel"]
    station_codes = set(station_coords.keys())

    # Find miniSEED files
    mseed_files = find_mseed_files(data["mseed_dirs"], station_codes, channel, event_date)

    if not mseed_files:
        print("ERROR: No miniSEED files found. Check paths and date.")
        sys.exit(1)

    # Load and process waveforms
    stream = obspy.Stream()
    raw_stream = obspy.Stream()  # Unfiltered traces for reference display
    filt_stream = obspy.Stream()  # Filtered but unnormalized traces for reference
    station_info = OrderedDict()
    resp_files = data.get("resp_files", {})
    pre_filter = tuple(proc.get("pre_filter", [0.005, 0.01, 0.4, 0.5]))
    output_units = proc.get("output_units", "VEL")

    skipped_no_resp = 0
    skipped_error = 0

    # Add buffer for response removal / filtering edge effects
    buffer = 120  # seconds
    read_start = t_start - buffer
    read_end = t_end + buffer

    for i, (sta, fpath) in enumerate(sorted(mseed_files.items())):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing station {i+1}/{len(mseed_files)}...")
        try:
            tr = read(fpath, starttime=read_start, endtime=read_end)[0]

            if tr.stats.npts < 10:
                continue

            # Remove instrument response if RESP file available
            serial = station_coords[sta][3]
            resp_file = get_resp_file(serial, resp_files)

            tr.detrend("demean")
            tr.detrend("linear")
            tr.taper(max_percentage=0.05)

            if resp_file:
                try:
                    remove_response_with_paz(tr, resp_file, output_units, pre_filter)
                except Exception as e:
                    print(f"  Warning: Response removal failed for {sta}: {e}")
            else:
                skipped_no_resp += 1

            # Decimate if requested
            decimate = proc.get("decimate_factor", 1)
            if decimate > 1:
                cascaded_decimate(tr, decimate)

            # Save unfiltered copy (response-removed, decimated, but not bandpass filtered)
            tr_raw = tr.copy()
            tr_raw.trim(t_start, t_end)
            raw_stream += tr_raw

            # Bandpass filter
            filt = proc["filter"]
            if filt["type"] == "bandpass":
                tr.filter(
                    "bandpass",
                    freqmin=filt["freqmin"],
                    freqmax=filt["freqmax"],
                    corners=filt["corners"],
                    zerophase=filt["zerophase"],
                )
            elif filt["type"] == "lowpass":
                tr.filter("lowpass", freq=filt["freqmax"],
                          corners=filt["corners"], zerophase=filt["zerophase"])
            elif filt["type"] == "highpass":
                tr.filter("highpass", freq=filt["freqmin"],
                          corners=filt["corners"], zerophase=filt["zerophase"])

            # Trim to exact window (remove buffer)
            tr.trim(t_start, t_end)

            # Save filtered but unnormalized copy for reference trace
            filt_stream += tr.copy()

            # Normalize (for map display)
            if proc.get("normalize", True):
                tr.normalize()

            stream += tr
            lon, lat = station_coords[sta][0], station_coords[sta][1]
            station_info[sta] = (lon, lat)

        except Exception as e:
            skipped_error += 1
            print(f"  Warning: Failed to process {sta}: {e}")
            continue

    if skipped_no_resp > 0:
        print(f"  {skipped_no_resp} stations processed without instrument response removal")
    if skipped_error > 0:
        print(f"  {skipped_error} stations skipped due to errors")
    print(f"Successfully processed {len(stream)} traces")

    return stream, raw_stream, filt_stream, station_info


def compute_distance_azimuth(lat1, lon1, lat2, lon2):
    """Compute great-circle distance (degrees) and azimuth between two points."""
    from math import radians, degrees, sin, cos, atan2, sqrt, asin
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    dist_rad = 2 * asin(sqrt(a))
    dist_deg = degrees(dist_rad)

    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    az = (degrees(atan2(y, x)) + 360) % 360
    return dist_deg, az


def compute_travel_times(dist_deg, depth_km):
    """
    Compute travel times for major seismic phases using ObsPy TauPy (iasp91).

    Returns dict: {phase_name: travel_time_seconds}
    """
    tt = {}
    try:
        from obspy.taup import TauPyModel
        model = TauPyModel(model="iasp91")
        arrivals = model.get_travel_times(
            source_depth_in_km=depth_km,
            distance_in_degree=dist_deg,
            phase_list=["P", "pP", "S", "sS", "SS", "ScS", "PKP", "PKIKP",
                         "SKS", "PP", "PcP"],
        )
        for arr in arrivals:
            if arr.name not in tt:  # Keep first arrival per phase
                tt[arr.name] = arr.time
    except Exception as e:
        print(f"  Warning: TauPy travel time computation failed: {e}")

    # Rayleigh wave R1 estimated at ~4 km/s surface wave speed
    # This is a group velocity approximation, not from TauPy
    r_slowness = 28.5  # seconds per degree (matches IRIS GMV default)
    tt["R1"] = dist_deg * r_slowness

    return tt


def _load_coastlines():
    """
    Load Natural Earth country polygons for globe rendering.
    Uses bundled GeoJSON file (data/ne_110m_countries.geojson).
    Returns list of (lons_array, lats_array) polygon segments.
    """
    import json

    def _extract_geojson_segments(geojson):
        segments = []
        for feature in geojson.get("features", []):
            geom = feature.get("geometry", {})
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])
            if gtype == "Polygon":
                rings = [coords[0]]  # exterior ring only
            elif gtype == "MultiPolygon":
                rings = [poly[0] for poly in coords]
            else:
                continue
            for ring in rings:
                arr = np.array(ring)
                segments.append((arr[:, 0], arr[:, 1]))
        return segments

    # Load bundled GeoJSON file
    geojson_path = os.path.join(os.path.dirname(__file__), "data", "ne_110m_countries.geojson")
    try:
        with open(geojson_path, "r") as f:
            data = json.load(f)
        segs = _extract_geojson_segments(data)
        if segs:
            return segs
    except Exception as e:
        print(f"Warning: Could not load {geojson_path}: {e}")

    print("Warning: No Natural Earth data found, globe will have no land masses")
    return []


_coastline_cache = None


def _get_coastlines():
    """Get cached coastline data."""
    global _coastline_cache
    if _coastline_cache is None:
        _coastline_cache = _load_coastlines()
    return _coastline_cache


def add_basemap(ax, extent, zoom="auto"):
    """Add ArcGIS World Topo tile basemap using contextily."""
    try:
        import contextily as ctx
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ctx.add_basemap(
            ax, crs="EPSG:4326",
            source=ctx.providers.Esri.WorldTopoMap,
            zoom=zoom if zoom != "auto" else "auto",
            attribution=False,
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    except ImportError:
        print("Warning: contextily not available, no basemap tiles")
    except Exception as e:
        print(f"Warning: Failed to add basemap tiles: {e}")


def add_globe_inset(fig, array_lon, array_lat, eq_lon, eq_lat):
    """
    Add a small orthographic globe inset with Wikipedia-style filled land,
    country borders, the network as an inverted triangle, the epicenter
    as a star, and the great-circle path.
    """
    from math import radians, cos, sin, acos, degrees, atan2, sqrt

    mid_lon = (array_lon + eq_lon) / 2
    mid_lat = (array_lat + eq_lat) / 2
    lo0, la0 = radians(mid_lon), radians(mid_lat)

    def ortho(lon, lat):
        """Orthographic projection. Returns (x, y, cos_c)."""
        lo, la = radians(lon), radians(lat)
        x = cos(la) * sin(lo - lo0)
        y = cos(la0) * sin(la) - sin(la0) * cos(la) * cos(lo - lo0)
        cos_c = sin(la0) * sin(la) + cos(la0) * cos(la) * cos(lo - lo0)
        return x, y, cos_c

    def _clip_polygon_to_hemisphere(lons_arr, lats_arr):
        """Clip a polygon to the visible hemisphere with edge interpolation."""
        n = len(lons_arr)
        if n < 3:
            return []
        projected = [ortho(lons_arr[i], lats_arr[i]) for i in range(n)]
        clipped = []
        for i in range(n):
            x1, y1, c1 = projected[i]
            x2, y2, c2 = projected[(i + 1) % n]
            if c1 > 0:
                clipped.append((x1, y1))
            if (c1 > 0) != (c2 > 0):
                t = c1 / (c1 - c2) if abs(c1 - c2) > 1e-10 else 0.5
                xi = x1 + t * (x2 - x1)
                yi = y1 + t * (y2 - y1)
                r = sqrt(xi**2 + yi**2)
                if r > 0:
                    xi, yi = xi / r * 0.9999, yi / r * 0.9999
                clipped.append((xi, yi))
        return clipped

    ax_g = fig.add_axes([0.02, 0.72, 0.18, 0.18])
    ax_g.set_xlim(-1.05, 1.05)
    ax_g.set_ylim(-1.05, 1.05)
    ax_g.set_aspect("equal")
    ax_g.axis("off")

    # Globe circle (ocean)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_g.fill(np.cos(theta), np.sin(theta), color="#D6EAF8", zorder=0)
    ax_g.plot(np.cos(theta), np.sin(theta), color="black", linewidth=0.8, zorder=5)

    # Land masses with proper clipping
    coastlines = _get_coastlines()
    for lons_seg, lats_seg in coastlines:
        clipped = _clip_polygon_to_hemisphere(lons_seg, lats_seg)
        if len(clipped) < 3:
            continue
        xs, ys = zip(*clipped)
        ax_g.fill(xs, ys, color="#C8D6B9", edgecolor="#555555",
                  linewidth=0.3, zorder=1)

    # Great-circle path
    lat1r, lon1r = radians(array_lat), radians(array_lon)
    lat2r, lon2r = radians(eq_lat), radians(eq_lon)
    d = acos(min(1, max(-1,
        sin(lat1r) * sin(lat2r) + cos(lat1r) * cos(lat2r) * cos(lon2r - lon1r))))
    if d > 0.001:
        gc_xs, gc_ys = [], []
        for f in np.linspace(0, 1, 100):
            A = sin((1 - f) * d) / sin(d)
            B = sin(f * d) / sin(d)
            x = A * cos(lat1r) * cos(lon1r) + B * cos(lat2r) * cos(lon2r)
            y = A * cos(lat1r) * sin(lon1r) + B * cos(lat2r) * sin(lon2r)
            z = A * sin(lat1r) + B * sin(lat2r)
            lat_i = degrees(atan2(z, sqrt(x**2 + y**2)))
            lon_i = degrees(atan2(y, x))
            px, py, cos_c = ortho(lon_i, lat_i)
            if cos_c > 0:
                gc_xs.append(px)
                gc_ys.append(py)
        if gc_xs:
            ax_g.plot(gc_xs, gc_ys, color="black", linewidth=1.0, zorder=3)

    # Network (inverted triangle) and epicenter (star)
    nx, ny, nc = ortho(array_lon, array_lat)
    if nc > 0:
        ax_g.plot(nx, ny, marker="v", markersize=8, color="lime",
                  markeredgecolor="black", markeredgewidth=0.8, zorder=4)
    ex, ey, ec = ortho(eq_lon, eq_lat)
    if ec > 0:
        ax_g.plot(ex, ey, marker="*", markersize=10, color="red",
                  markeredgecolor="black", markeredgewidth=0.5, zorder=4)

    return ax_g


def create_animation(stream, raw_stream, filt_stream, station_info, config):
    """
    Create a two-panel GMV animation:
    - Top: Map with colored station markers and great-circle path
    - Bottom: Reference trace waveform with moving red time cursor

    Parameters
    ----------
    stream : obspy.Stream
        Processed waveform data (normalized, for map colors).
    raw_stream : obspy.Stream
        Unfiltered (but response-removed) traces for grey background.
    filt_stream : obspy.Stream
        Filtered but unnormalized traces for reference trace display.
    station_info : dict
        {station_code: (lon, lat)}.
    config : dict
        Full configuration dictionary.
    """
    anim_cfg = config["animation"]
    eq = config["earthquake"]
    proc = config["processing"]
    origin = UTCDateTime(eq["origin_time"])
    tw = config["time_window"]

    if tw.get("auto", False):
        # Recompute auto time window from phase arrivals
        array_csv = config["data"]["station_csv"]
        with open(array_csv) as _f:
            _hdr = _f.readline().strip().split(",")
            _idx = {c: i for i, c in enumerate(_hdr)}
            _lons, _lats = [], []
            for _line in _f:
                _p = _line.strip().split(",")
                if len(_p) >= len(_hdr):
                    _lons.append(float(_p[_idx["longitude"]]))
                    _lats.append(float(_p[_idx["latitude"]]))
        _alat, _alon = np.mean(_lats), np.mean(_lons)
        _dist, _ = compute_distance_azimuth(_alat, _alon, eq["latitude"], eq["longitude"])
        _tt = compute_travel_times(_dist, eq["depth_km"])
        p_time = _tt.get("P", _tt.get("PKP", 600))
        r1_time = _tt.get("R1", _dist * 28.5)
        pre_p = tw.get("before_P", 10)
        post_r1 = tw.get("after_R1", 20)
        t_start = origin + p_time - pre_p
        t_end = origin + r1_time + post_r1
    else:
        t_start = origin - tw["pre_event_seconds"]
        t_end = origin + tw["post_event_seconds"]

    # Compute time_step from realtime_factor and fps
    fps = anim_cfg.get("fps", 15)
    realtime_factor = anim_cfg.get("realtime_factor", 10)
    time_step = anim_cfg.get("time_step", realtime_factor / fps)

    # Build arrays for plotting
    stations = list(station_info.keys())
    lons = np.array([station_info[s][0] for s in stations])
    lats = np.array([station_info[s][1] for s in stations])

    # Array center
    array_lat = np.mean(lats)
    array_lon = np.mean(lons)

    # Compute distance and azimuth to earthquake
    dist_deg, az = compute_distance_azimuth(
        array_lat, array_lon, eq["latitude"], eq["longitude"]
    )
    baz = (az + 180) % 360

    # Time samples for animation
    n_frames = int((t_end - t_start) / time_step)
    times = [t_start + i * time_step for i in range(n_frames)]

    # Select reference station (center of array, highest SNR)
    ref_cfg = anim_cfg.get("reference_trace", {})
    ref_station = ref_cfg.get("station", None)
    if ref_station is None:
        # Pick station closest to array center
        dists_to_center = np.sqrt((lons - array_lon)**2 + (lats - array_lat)**2)
        ref_station = stations[np.argmin(dists_to_center)]

    ref_traces = filt_stream.select(station=ref_station)
    ref_trace = ref_traces[0] if len(ref_traces) > 0 else filt_stream[0]
    ref_station = ref_trace.stats.station

    # Map extent
    map_cfg = anim_cfg["map"]
    pad = map_cfg.get("extent_pad", 0.05)
    extent = [
        lons.min() - pad, lons.max() + pad,
        lats.min() - pad, lats.max() + pad,
    ]

    # Determine channel band label
    filt = proc["filter"]
    freq_band = f"{filt['freqmin']}-{filt['freqmax']} Hz"

    # Gain info
    gain_text = f"Gain: normalized"

    # ===================== FIGURE SETUP =====================
    fig = plt.figure(figsize=(12, 10), dpi=anim_cfg.get("dpi", 150))

    # Top panel: Map (55% of figure height)
    ax_map = fig.add_axes([0.05, 0.35, 0.80, 0.55])

    # Bottom panel: Reference trace (25% of figure height)
    ax_trace = fig.add_axes([0.1, 0.06, 0.8, 0.22])

    # ===================== MAP PANEL =====================
    ax_map.set_xlim(extent[0], extent[1])
    ax_map.set_ylim(extent[2], extent[3])
    ax_map.set_xlabel("Longitude (°)", fontsize=9)
    ax_map.set_ylabel("Latitude (°)", fontsize=9)
    ax_map.set_aspect("equal")
    ax_map.grid(True, linewidth=0.3, alpha=0.5)

    # ArcGIS basemap tiles
    tile_zoom = map_cfg.get("tile_zoom", "auto")
    add_basemap(ax_map, extent, zoom=tile_zoom)

    # Globe inset (top-left corner)
    add_globe_inset(fig, array_lon, array_lat, eq["longitude"], eq["latitude"])

    # Great-circle path from earthquake to array center
    ax_map.plot(
        [eq["longitude"], array_lon],
        [eq["latitude"], array_lat],
        color="black", linewidth=1.5, linestyle="-",
        zorder=3,
        label=f"Great Circle Path (~ {int(dist_deg * 111.195)} km)",
    )

    # Colormap — discretized like original GMV (32 levels)
    mk_cfg = anim_cfg["marker"]
    color_levels = 32
    cmap = plt.get_cmap(mk_cfg.get("cmap", "bwr"), color_levels)
    vmin = mk_cfg.get("vmin", -1)
    vmax = mk_cfg.get("vmax", 1)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    marker_size_pt = mk_cfg.get("size", 40) * 1.5

    def value_to_color(value):
        """Convert amplitude to RGBA, clamping to [vmin, vmax]."""
        value = max(vmin, min(vmax, value))
        return cmap(norm(value))

    # Highlight reference station
    ref_lon, ref_lat = station_info[ref_station]
    ax_map.plot(
        ref_lon, ref_lat,
        marker="v", markersize=9, color="lime",
        markeredgecolor="black", markeredgewidth=1,
        zorder=8,
        label="Reference Station",
    )

    # Static colorbar (drawn once, never updated — like original GMV)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.87, 0.40, 0.02, 0.40])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Normalized Amplitude", fontsize=9)

    # Earthquake epicenter marker (for legend — plotted off-map if outside extent)
    ax_map.plot(
        eq["longitude"], eq["latitude"],
        marker="*", markersize=10, color="red",
        markeredgecolor="black", markeredgewidth=0.5,
        linestyle="None", zorder=6,
        label="Epicenter",
    )

    # Legend on the right side
    ax_map.legend(
        loc="upper right", fontsize=8, framealpha=0.9,
        bbox_to_anchor=(1.0, 1.0),
    )

    # Time label on map
    time_text = ax_map.text(
        0.02, 0.02, "", transform=ax_map.transAxes,
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        zorder=20,
    )

    # Main title
    origin_dt = origin.strftime("%Y-%m-%d")
    event_desc = eq.get("description", "")
    title_line2 = f"{origin_dt} — M{eq['magnitude']}"
    if event_desc:
        title_line2 += f" — {event_desc}"
    fig.suptitle(
        f"Ground Motion Visualization\n"
        f"{title_line2}\n"
        f"Origin Time (OT) = {eq['origin_time']} UTC | "
        f"Lat {eq['latitude']:.2f}° Lon {eq['longitude']:.2f}° "
        f"Depth {eq['depth_km']:.0f} km",
        fontsize=11, fontweight="bold", y=0.99,
    )

    # ===================== TRACE PANEL =====================
    # Get reference traces (unfiltered + filtered)
    ref_raw_traces = raw_stream.select(station=ref_station)
    ref_raw = ref_raw_traces[0] if len(ref_raw_traces) > 0 else None

    # Convert to um/s (micro m/s) if data is in m/s after response removal
    scale_factor = 1e6  # m/s -> um/s
    unit_label = "Velocity (μm/s)"

    # Plot unfiltered trace in grey
    if ref_raw is not None:
        raw_times_s = np.arange(ref_raw.stats.npts) / ref_raw.stats.sampling_rate
        raw_times_rel = raw_times_s + (ref_raw.stats.starttime - origin)
        raw_data = ref_raw.data * scale_factor
        ax_trace.plot(raw_times_rel, raw_data, color="gray", linewidth=0.4,
                      label="Unfiltered", zorder=1)

    # Plot filtered trace in black
    ref_times_s = np.arange(ref_trace.stats.npts) / ref_trace.stats.sampling_rate
    ref_times_rel = ref_times_s + (ref_trace.stats.starttime - origin)
    # For filtered trace, don't re-normalize - use same scale
    filt_data = ref_trace.data * scale_factor
    ax_trace.plot(ref_times_rel, filt_data, color="black", linewidth=0.5,
                  label=f"Filtered ({freq_band})", zorder=2)

    ax_trace.set_xlim(t_start - origin, t_end - origin)
    ax_trace.set_xlabel("Time relative to OT (s)", fontsize=9)
    ax_trace.set_ylabel(unit_label, fontsize=9)
    ax_trace.axhline(y=0, color="gray", linewidth=0.3)
    ax_trace.axvline(x=0, color="green", linewidth=0.8, linestyle="--",
                     alpha=0.7)

    # Compute theoretical phase arrivals (used for auto time window)
    travel_times = compute_travel_times(dist_deg, eq["depth_km"])
    max_amp = max(abs(filt_data.max()), abs(filt_data.min())) if len(filt_data) > 0 else 1

    # Title and legend
    ax_trace.set_title("Vertical Component", fontsize=9, loc="left")
    ax_trace.tick_params(labelsize=8)
    ax_trace.legend(loc="upper right", fontsize=7, framealpha=0.8)

    # Red cursor line (will be updated)
    cursor_line = ax_trace.axvline(x=0, color="red", linewidth=1.5, zorder=10)

    # ===================== PRE-COMPUTE =====================
    # Like original GMV: pre-compute amplitude per station per frame,
    # convert to RGBA color, and group stations by color.
    print("Pre-computing animation frames...")
    marker_sqrt = np.sqrt(marker_size_pt)  # matplotlib markersize is in points

    # Pre-compute frames: for each frame, group stations by color
    # frames_data[i] = {color_rgba: {'x': [...], 'y': [...]}}
    frames_data = []
    all_colors = []  # ordered list of unique RGBA tuples

    for i, t in enumerate(times):
        color_groups = {}
        for j, sta in enumerate(stations):
            traces = stream.select(station=sta)
            if len(traces) == 0:
                continue
            tr = traces[0]
            idx = int((t - tr.stats.starttime) * tr.stats.sampling_rate)
            if 0 <= idx < tr.stats.npts:
                amp = tr.data[idx]
            else:
                amp = 0.0
            rgba = value_to_color(amp)
            if rgba not in color_groups:
                color_groups[rgba] = {'x': [], 'y': []}
            color_groups[rgba]['x'].append(lons[j])
            color_groups[rgba]['y'].append(lats[j])
            if rgba not in all_colors:
                all_colors.append(rgba)
        frames_data.append(color_groups)

    # Initialize one plot handle per unique color (like original GMV's set_marker)
    marker_handles = {}
    for rgba in all_colors:
        handle, = ax_map.plot([], [], marker='o', color=rgba,
                              markersize=marker_sqrt, markeredgewidth=0.3,
                              markeredgecolor=(0, 0, 0, 1),
                              linestyle='None', zorder=5)
        marker_handles[rgba] = handle

    print(f"  {len(all_colors)} unique colors across {n_frames} frames")

    # Add time cursor label on trace panel
    cursor_label = ax_trace.text(
        0, max_amp * 0.85, "",
        size=6, color="#FF0000",
        bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=0),
        zorder=11,
    )

    def update(frame_idx):
        """Update function for animation — only moves markers, never changes colors."""
        t = times[frame_idx]
        elapsed = t - origin

        # Update marker positions per color group (like original GMV)
        frame = frames_data[frame_idx]
        for rgba, handle in marker_handles.items():
            if rgba in frame:
                handle.set_data(frame[rgba]['x'], frame[rgba]['y'])
            else:
                handle.set_data([], [])

        # Format time as HH:MM:SS like original GMV
        abs_elapsed = abs(elapsed)
        hours = int(abs_elapsed // 3600)
        minutes = int((abs_elapsed % 3600) // 60)
        seconds = int(abs_elapsed % 60)
        sign = "+" if elapsed >= 0 else "-"
        ot_str = f"OT {sign} {hours}:{minutes:02d}:{seconds:02d}"

        # Update time label on map
        time_text.set_text(
            f"{t.strftime('%Y-%m-%d %H:%M:%S')} UTC ({ot_str})"
        )

        # Update red cursor on trace
        cursor_line.set_xdata([elapsed, elapsed])
        cursor_label.set_position((elapsed, max_amp * 0.85))
        cursor_label.set_text(ot_str)

        # Progress bar
        pct = (frame_idx + 1) / n_frames * 100
        bar_len = 40
        filled = int(bar_len * (frame_idx + 1) / n_frames)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:5.1f}% ({frame_idx+1}/{n_frames}) {ot_str}",
              end="", flush=True)

        return list(marker_handles.values()) + [time_text, cursor_line, cursor_label]

    print(f"Creating animation with {n_frames} frames...")
    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 / anim_cfg.get("fps", 15),
        blit=True,
    )

    # Save
    output_path = anim_cfg.get("output_file", "output/gmv_animation")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fmt = anim_cfg.get("format", "gif")

    if fmt == "mp4":
        try:
            # Try to find ffmpeg from imageio-ffmpeg if system ffmpeg not available
            import shutil
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is None:
                try:
                    import imageio_ffmpeg
                    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
                except ImportError:
                    pass

            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(
                fps=anim_cfg.get("fps", 15),
                codec="h264",
                bitrate=2000,
                metadata=dict(title="Ground Motion Visualization"),
            )
            output_file = f"{output_path}.mp4"
            anim.save(output_file, writer=writer)
        except Exception as e:
            print(f"MP4 export failed ({e}), falling back to GIF...")
            fmt = "gif"

    if fmt == "gif":
        writer = PillowWriter(fps=anim_cfg.get("fps", 15))
        output_file = f"{output_path}.gif"
        anim.save(output_file, writer=writer)

    print()  # Newline after progress bar
    plt.close(fig)
    print(f"Animation saved to: {output_file}")
    return output_file


def create_snapshot(stream, station_info, config, time_offset_s):
    """
    Create a single-frame snapshot at a given time offset from origin.

    Parameters
    ----------
    time_offset_s : float
        Seconds relative to earthquake origin time.

    Returns
    -------
    str : Path to saved PNG.
    """
    eq = config["earthquake"]
    anim_cfg = config["animation"]
    origin = UTCDateTime(eq["origin_time"])
    t = origin + time_offset_s

    stations = list(station_info.keys())
    lons = np.array([station_info[s][0] for s in stations])
    lats = np.array([station_info[s][1] for s in stations])

    amplitudes = np.zeros(len(stations))
    for j, sta in enumerate(stations):
        traces = stream.select(station=sta)
        if len(traces) == 0:
            continue
        tr = traces[0]
        idx = int((t - tr.stats.starttime) * tr.stats.sampling_rate)
        if 0 <= idx < tr.stats.npts:
            amplitudes[j] = tr.data[idx]

    map_cfg = anim_cfg["map"]
    mk_cfg = anim_cfg["marker"]
    pad = map_cfg.get("extent_pad", 0.05)
    extent = [lons.min() - pad, lons.max() + pad, lats.min() - pad, lats.max() + pad]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitude (°)", fontsize=9)
    ax.set_ylabel("Latitude (°)", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # ArcGIS basemap tiles and globe inset
    add_basemap(ax, extent)
    add_globe_inset(fig, np.mean(lons), np.mean(lats), eq["longitude"], eq["latitude"])

    cmap = plt.get_cmap(mk_cfg.get("cmap", "bwr"))
    norm = mcolors.Normalize(vmin=mk_cfg.get("vmin", -1), vmax=mk_cfg.get("vmax", 1))

    scatter = ax.scatter(
        lons, lats, c=amplitudes, cmap=cmap, norm=norm,
        s=mk_cfg.get("size", 40), alpha=mk_cfg.get("alpha", 0.85),
        edgecolors="k", linewidths=0.3,
        zorder=5,
    )

    plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02, label="Normalized Amplitude")

    epi_cfg = anim_cfg.get("epicenter", {})
    if epi_cfg.get("show", True):
        ax.plot(
            eq["longitude"], eq["latitude"],
            marker=epi_cfg.get("marker", "*"),
            markersize=epi_cfg.get("size", 200) / 15,
            color=epi_cfg.get("color", "gold"),
            markeredgecolor=epi_cfg.get("edgecolor", "black"),
            markeredgewidth=1,
            zorder=10,
        )

    elapsed = time_offset_s
    sign = "+" if elapsed >= 0 else ""
    ax.set_title(
        f"{map_cfg.get('title', 'GMV')}\n"
        f"M{eq['magnitude']} | {t.strftime('%Y-%m-%d %H:%M:%S')} UTC (T{sign}{elapsed:.0f}s)",
        fontsize=12, fontweight="bold",
    )

    output_path = anim_cfg.get("output_file", "output/gmv_snapshot")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_path}_T{sign}{int(elapsed)}s.png"
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Snapshot saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Ground Motion Visualization for SmartSolo geophone arrays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create animation from config file
  python gmv.py config/example_config.yaml

  # Create a single snapshot at T+600s
  python gmv.py config/example_config.yaml --snapshot 600

  # Override output format
  python gmv.py config/example_config.yaml --format mp4
        """,
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--snapshot", type=float, default=None,
        help="Create a single snapshot at this time offset (seconds from origin) instead of animation",
    )
    parser.add_argument(
        "--format", choices=["gif", "mp4"], default=None,
        help="Override output format",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override output file path (without extension)",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Apply command-line overrides
    if args.format:
        config["animation"]["format"] = args.format
    if args.output:
        config["animation"]["output_file"] = args.output

    # Load station coordinates
    station_coords = load_station_coordinates(
        config["data"]["station_csv"],
        channel=config["processing"]["channel"],
    )

    # Load and process waveforms
    print("Loading and processing waveforms...")
    stream, raw_stream, filt_stream, station_info = load_and_process_waveforms(config, station_coords)

    if len(stream) == 0:
        print("ERROR: No valid traces after processing.")
        sys.exit(1)

    print(f"Ready: {len(stream)} stations with data")

    # Create output
    if args.snapshot is not None:
        create_snapshot(stream, station_info, config, args.snapshot)
    else:
        create_animation(stream, raw_stream, filt_stream, station_info, config)

    print("Done.")


if __name__ == "__main__":
    main()
