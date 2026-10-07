#!/usr/bin/env python3
"""
Earthquake catalog tools for GMV visualization.

Queries USGS FDSN earthquake catalog, computes distances to the array,
and generates configuration files for GMV visualization.

Author: Michail Henry (michail.henry@protonmail.com)
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime
    from obspy.geodetics import gps2dist_azimuth
except ImportError:
    print("ERROR: ObsPy is required. Install with: pip install obspy")
    sys.exit(1)


def load_array_center(station_csv):
    """Compute the center of the geophone array from station coordinates."""
    lons, lats = [], []
    seen = set()
    with open(station_csv, "r") as f:
        header = f.readline().strip().split(",")
        idx = {col: i for i, col in enumerate(header)}
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < len(header):
                continue
            serial = parts[idx["serial_number"]]
            if serial in seen:
                continue
            seen.add(serial)
            lons.append(float(parts[idx["longitude"]]))
            lats.append(float(parts[idx["latitude"]]))
    center_lon = np.mean(lons)
    center_lat = np.mean(lats)
    return center_lat, center_lon


def query_earthquakes(starttime, endtime, minmag=5.5, maxmag=None,
                      array_lat=None, array_lon=None):
    """
    Query USGS earthquake catalog via FDSN.

    Returns list of dicts with event info and distance to array.
    """
    client = Client("USGS")
    kwargs = {
        "starttime": UTCDateTime(starttime),
        "endtime": UTCDateTime(endtime),
        "minmagnitude": minmag,
    }
    if maxmag:
        kwargs["maxmagnitude"] = maxmag

    print(f"Querying USGS catalog: {starttime} to {endtime}, M >= {minmag}...")
    catalog = client.get_events(**kwargs)

    events = []
    for event in catalog:
        origin = event.preferred_origin()
        mag = event.preferred_magnitude()
        if not origin or not mag:
            continue

        ev = {
            "time": str(origin.time),
            "latitude": origin.latitude,
            "longitude": origin.longitude,
            "depth_km": origin.depth / 1000.0 if origin.depth else 0,
            "magnitude": mag.mag,
            "mag_type": mag.magnitude_type,
            "description": str(event.event_descriptions[0].text)
                if event.event_descriptions else "",
            "event_id": str(event.resource_id).split("/")[-1],
        }

        if array_lat is not None and array_lon is not None:
            dist_m, az, baz = gps2dist_azimuth(
                array_lat, array_lon,
                origin.latitude, origin.longitude,
            )
            ev["distance_km"] = dist_m / 1000.0
            ev["distance_deg"] = dist_m / 1000.0 / 111.195
            ev["azimuth"] = az
            ev["backazimuth"] = baz

            # Approximate P-wave travel time (IASP91-like)
            dist_deg = ev["distance_deg"]
            if dist_deg < 30:
                ev["p_travel_time_s"] = dist_deg * 13.0
            elif dist_deg < 100:
                ev["p_travel_time_s"] = dist_deg * 9.5 + 100
            else:
                ev["p_travel_time_s"] = dist_deg * 8.5 + 200

        events.append(ev)

    # Sort by magnitude descending
    events.sort(key=lambda x: x["magnitude"], reverse=True)
    print(f"Found {len(events)} events")
    return events


def print_event_table(events):
    """Print a formatted table of events."""
    print(f"\n{'#':>3} {'Mag':>4} {'Date/Time (UTC)':>22} {'Depth':>7} "
          f"{'Dist(km)':>10} {'Dist(°)':>8} {'Baz(°)':>7} {'P~(s)':>7}  Description")
    print("-" * 120)
    for i, ev in enumerate(events):
        dist_km = ev.get("distance_km", 0)
        dist_deg = ev.get("distance_deg", 0)
        baz = ev.get("backazimuth", 0)
        p_tt = ev.get("p_travel_time_s", 0)
        print(f"{i+1:3d} {ev['magnitude']:4.1f} {ev['time'][:22]:>22} "
              f"{ev['depth_km']:6.1f}  {dist_km:9.0f} {dist_deg:7.1f} "
              f"{baz:6.1f} {p_tt:6.0f}  {ev['description'][:50]}")


def plot_event_map(events, array_lat, array_lon, output_file="output/event_map.png"):
    """Plot global map of events with array location and great-circle paths."""
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=array_lon))
    ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.gridlines(linewidth=0.3, alpha=0.5)

    # Plot array location
    ax.plot(
        array_lon, array_lat,
        marker="^", markersize=12, color="green",
        markeredgecolor="black", markeredgewidth=1,
        transform=ccrs.PlateCarree(), zorder=10,
        label="GEONAT Array",
    )

    # Plot events
    for i, ev in enumerate(events):
        size = (ev["magnitude"] - 4) ** 2 * 20
        color = "red" if ev["magnitude"] >= 6.0 else "orange" if ev["magnitude"] >= 5.5 else "yellow"

        ax.plot(
            ev["longitude"], ev["latitude"],
            marker="o", markersize=max(4, size / 10),
            color=color, markeredgecolor="black", markeredgewidth=0.5,
            transform=ccrs.PlateCarree(), zorder=5,
        )

        # Great circle path
        ax.plot(
            [array_lon, ev["longitude"]],
            [array_lat, ev["latitude"]],
            color="gray", linewidth=0.5, alpha=0.5,
            transform=ccrs.Geodetic(), zorder=3,
        )

        # Label
        ax.text(
            ev["longitude"] + 2, ev["latitude"] + 2,
            f"M{ev['magnitude']:.1f}",
            fontsize=7, transform=ccrs.PlateCarree(), zorder=6,
        )

    ax.set_title(
        f"Earthquakes During GEONAT Deployment\n"
        f"{events[-1]['time'][:10]} to {events[0]['time'][:10]} | M >= {min(e['magnitude'] for e in events):.1f}",
        fontsize=13, fontweight="bold",
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="green",
               markeredgecolor="black", markersize=10, label="GEONAT Array"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=8, label="M >= 6.0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="orange",
               markeredgecolor="black", markersize=6, label="M 5.5-6.0"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nEvent map saved to: {output_file}")


def generate_config(event, template_path=None, output_dir="config"):
    """Generate a YAML config file for a specific event."""
    ev_date = event["time"][:10].replace("-", "")
    magnitude = event["magnitude"]
    dist_deg = event.get("distance_deg", 50)

    # Choose filter parameters based on distance and magnitude
    if dist_deg > 60:  # Teleseismic
        freqmin, freqmax = 0.5, 2.0
        post_event = 2400
        time_step = 1.0
        pre_filter = [0.1, 0.5, 5.0, 10.0]
    elif dist_deg > 20:  # Regional
        freqmin, freqmax = 0.5, 5.0
        post_event = 1200
        time_step = 0.5
        pre_filter = [0.1, 0.5, 10.0, 20.0]
    else:  # Local/near-regional
        freqmin, freqmax = 1.0, 10.0
        post_event = 300
        time_step = 0.2
        pre_filter = [0.5, 1.0, 20.0, 50.0]

    config = {
        "earthquake": {
            "latitude": round(event["latitude"], 4),
            "longitude": round(event["longitude"], 4),
            "depth_km": round(event["depth_km"], 1),
            "magnitude": round(event["magnitude"], 1),
            "origin_time": event["time"][:19],
        },
        "time_window": {
            "pre_event_seconds": 60,
            "post_event_seconds": post_event,
        },
        "data": {
            "mseed_dirs": [
                "/srv/beegfs/scratch/users/h/henrymi/project/GEONAT/miniseed_5Hz",
                "/srv/beegfs/scratch/users/h/henrymi/project/GEONAT/miniseed_2Hz",
            ],
            "station_csv": "/srv/beegfs/scratch/users/h/henrymi/project/GEONAT/geonat_mseed.csv",
            "resp_files": {
                "4530": "/srv/beegfs/scratch/users/h/henrymi/project/GEONAT/RESP/smartsolo.5Hz.pz",
                "4236": "/srv/beegfs/scratch/users/h/henrymi/project/GEONAT/RESP/smartsolo.2Hz.pz",
            },
        },
        "processing": {
            "channel": "DPZ",
            "output_units": "VEL",
            "filter": {
                "type": "bandpass",
                "freqmin": freqmin,
                "freqmax": freqmax,
                "corners": 4,
                "zerophase": True,
            },
            "decimate_factor": 50,
            "normalize": True,
            "pre_filter": pre_filter,
        },
        "animation": {
            "time_step": time_step,
            "fps": 15,
            "dpi": 150,
            "format": "gif",
            "output_file": f"output/gmv_{ev_date}_M{magnitude}",
            "map": {
                "projection": "PlateCarree",
                "extent_pad": 0.02,
                "background": "terrain",
                "title": f"GEONAT Array - M{magnitude} {event['description'][:40]}",
            },
            "marker": {
                "size": 30,
                "alpha": 0.85,
                "cmap": "bwr",
                "vmin": -1.0,
                "vmax": 1.0,
            },
            "epicenter": {
                "show": False,
            },
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/event_{ev_date}_M{magnitude}.yaml"
    with open(filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config generated: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Query earthquake catalog and generate GMV configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List events during GEONAT deployment
  python catalog.py --start 2026-08-29 --end 2026-09-17 --minmag 5.5

  # Generate configs for all M6+ events
  python catalog.py --start 2026-08-29 --end 2026-09-17 --minmag 6.0 --generate-configs

  # Plot event map
  python catalog.py --start 2026-08-29 --end 2026-09-17 --minmag 5.5 --plot
        """,
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--minmag", type=float, default=5.5, help="Minimum magnitude")
    parser.add_argument(
        "--station-csv",
        default="/srv/beegfs/scratch/users/h/henrymi/project/GEONAT/geonat_mseed.csv",
        help="Path to station CSV file",
    )
    parser.add_argument("--plot", action="store_true", help="Plot event map")
    parser.add_argument("--plot-output", default="output/event_map.png",
                        help="Output path for event map")
    parser.add_argument("--generate-configs", action="store_true",
                        help="Generate GMV config files for each event")
    args = parser.parse_args()

    # Get array center
    array_lat, array_lon = load_array_center(args.station_csv)
    print(f"Array center: {array_lat:.4f}°N, {array_lon:.4f}°E")

    # Query catalog
    events = query_earthquakes(
        args.start, args.end, args.minmag,
        array_lat=array_lat, array_lon=array_lon,
    )

    if not events:
        print("No events found.")
        return

    # Print table
    print_event_table(events)

    # Plot
    if args.plot:
        plot_event_map(events, array_lat, array_lon, args.plot_output)

    # Generate configs
    if args.generate_configs:
        print("\nGenerating configuration files...")
        for ev in events:
            generate_config(ev)


if __name__ == "__main__":
    main()
