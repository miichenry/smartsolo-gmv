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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
    proj = getattr(ccrs, map_cfg.get("projection", "PlateCarree"))()
    fig = plt.figure(figsize=(12, 10), dpi=anim_cfg.get("dpi", 150))

    # Top panel: Map (55% of figure height)
    ax_map = fig.add_axes([0.05, 0.35, 0.80, 0.55], projection=proj)
    ax_map.set_extent(extent, crs=ccrs.PlateCarree())

    # Bottom panel: Reference trace (25% of figure height)
    ax_trace = fig.add_axes([0.1, 0.06, 0.8, 0.22])

    # ===================== MAP PANEL =====================
    bg = map_cfg.get("background", "terrain")
    if bg == "terrain":
        try:
            ax_map.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
            ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax_map.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.5)
        except Exception:
            pass
    # Gridlines: only bottom (lat) and left (lon) labels, no top axis
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Great-circle path from earthquake to array center
    ax_map.plot(
        [eq["longitude"], array_lon],
        [eq["latitude"], array_lat],
        color="black", linewidth=1.5, linestyle="-",
        transform=ccrs.Geodetic(), zorder=3,
        label=f"GC path ({dist_deg:.1f}°)",
    )

    # Colormap
    mk_cfg = anim_cfg["marker"]
    cmap = plt.get_cmap(mk_cfg.get("cmap", "bwr"))
    norm = mcolors.Normalize(vmin=mk_cfg.get("vmin", -1), vmax=mk_cfg.get("vmax", 1))

    # Station markers (1.5x bigger)
    marker_size = mk_cfg.get("size", 40) * 1.5
    scatter = ax_map.scatter(
        lons, lats,
        c=np.zeros(len(stations)),
        cmap=cmap, norm=norm,
        s=marker_size,
        alpha=mk_cfg.get("alpha", 0.85),
        edgecolors="k", linewidths=0.3,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # Highlight reference station
    ref_lon, ref_lat = station_info[ref_station]
    ax_map.plot(
        ref_lon, ref_lat,
        marker="v", markersize=9, color="lime",
        markeredgecolor="black", markeredgewidth=1,
        transform=ccrs.PlateCarree(), zorder=8,
        label=f"Ref: SS.{ref_station}",
    )

    # Colorbar on a separate axis to the right (not overlapping map)
    cbar_ax = fig.add_axes([0.87, 0.40, 0.02, 0.40])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Normalized Amplitude", fontsize=9)

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
        ax_trace.plot(raw_times_rel, raw_data, color="lightgray", linewidth=0.4,
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
                     label="OT", alpha=0.7)

    # Compute and plot theoretical phase arrivals (blue, like original GMV)
    travel_times = compute_travel_times(dist_deg, eq["depth_km"])
    max_amp = max(abs(filt_data.max()), abs(filt_data.min())) if len(filt_data) > 0 else 1
    phase_spacing = 200  # Minimum seconds between phase labels
    last_label_time = -999
    for phase, tt in sorted(travel_times.items(), key=lambda x: x[1]):
        if t_start - origin <= tt <= t_end - origin:
            ax_trace.axvline(x=tt, color="blue", linewidth=0.5, zorder=3)
            # Enforce spacing between labels (except R1 always shown)
            if phase == "R1" or (tt - last_label_time) > phase_spacing:
                ax_trace.text(
                    tt, max_amp * 0.95, phase,
                    horizontalalignment="center", verticalalignment="center",
                    size=6, color="blue",
                    bbox=dict(facecolor="white", alpha=0.99, lw=0, pad=0),
                    zorder=4,
                )
                last_label_time = tt

    # Title and legend
    ax_trace.set_title("Vertical Component", fontsize=9, loc="left")
    ax_trace.tick_params(labelsize=8)
    ax_trace.legend(loc="upper right", fontsize=7, framealpha=0.8)

    # Red cursor line (will be updated)
    cursor_line = ax_trace.axvline(x=0, color="red", linewidth=1.5, zorder=10)

    # ===================== PRE-COMPUTE =====================
    print("Pre-computing animation frames...")
    amplitude_frames = np.zeros((n_frames, len(stations)))

    for i, t in enumerate(times):
        for j, sta in enumerate(stations):
            traces = stream.select(station=sta)
            if len(traces) == 0:
                continue
            tr = traces[0]
            idx = int((t - tr.stats.starttime) * tr.stats.sampling_rate)
            if 0 <= idx < tr.stats.npts:
                amplitude_frames[i, j] = tr.data[idx]

    # Add time cursor label on trace panel
    cursor_label = ax_trace.text(
        0, max_amp * 0.85, "",
        size=6, color="#FF0000",
        bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=0),
        zorder=11,
    )

    def update(frame_idx):
        """Update function for animation."""
        t = times[frame_idx]
        elapsed = t - origin

        # Update scatter colors
        scatter.set_array(amplitude_frames[frame_idx])

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

        return scatter, time_text, cursor_line, cursor_label

    print(f"Creating animation with {n_frames} frames...")
    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 / anim_cfg.get("fps", 15),
        blit=False,
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

    proj = getattr(ccrs, map_cfg.get("projection", "PlateCarree"))()
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": proj}, dpi=150)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    try:
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    except Exception:
        pass
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    cmap = plt.get_cmap(mk_cfg.get("cmap", "bwr"))
    norm = mcolors.Normalize(vmin=mk_cfg.get("vmin", -1), vmax=mk_cfg.get("vmax", 1))

    scatter = ax.scatter(
        lons, lats, c=amplitudes, cmap=cmap, norm=norm,
        s=mk_cfg.get("size", 40), alpha=mk_cfg.get("alpha", 0.85),
        edgecolors="k", linewidths=0.3,
        transform=ccrs.PlateCarree(), zorder=5,
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
            transform=ccrs.PlateCarree(), zorder=10,
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
