# Ground Motion Visualization (GMV) for SmartSolo Geophone Arrays

Animated visualization of seismic ground motion recorded by dense arrays of SmartSolo IGU-16HR geophones. Produces two-panel video animations showing:
- **Top panel**: Map with color-coded station markers (red = ground up, blue = ground down), great-circle path to the earthquake, and reference station marker.
- **Bottom panel**: Vertical-component waveform at the reference station (grey = unfiltered, black = filtered), with theoretical phase arrivals (P, S, R1, ...) and a moving red time cursor.

Inspired by and adapted from the [IRIS/EarthScope GMV data product](https://ds.iris.edu/ds/products/gmv/) ([Trabant et al., 2012](https://doi.org/10.1785/gssrl.83.5.846)).

## Setup

```bash
pip install -r requirements.txt
pip install imageio-ffmpeg    # for MP4 output
```

### Dependencies

- [ObsPy](https://obspy.org/) >= 1.4.0 (waveform I/O, instrument response, travel times)
- [Cartopy](https://scitools.github.io/cartopy/) >= 0.20.0 (map projections)
- NumPy, Matplotlib, PyYAML
- imageio-ffmpeg (for MP4 export, bundles its own ffmpeg binary)

## Usage

### 1. Query earthquake catalog

```bash
# List M5.5+ events during the deployment, plot map, and generate configs
python catalog.py --start 2026-08-29 --end 2026-09-26 --minmag 5.5 --plot --generate-configs
```

This queries the USGS FDSN catalog, computes distances/azimuths to the array, generates per-event YAML configs, and optionally plots a global event map.

### 2. Create GMV animation

```bash
# Create MP4 animation
python gmv.py config/event_20260911_M6.5.yaml --format mp4

# Create GIF animation
python gmv.py config/event_20260911_M6.5.yaml --format gif

# Create a single snapshot at T+800 seconds
python gmv.py config/event_20260911_M6.5.yaml --snapshot 800

# Custom output path
python gmv.py config/event_20260911_M6.5.yaml --format mp4 --output output/my_event
```

### Configuration

Copy and edit the example configuration:

```bash
cp config/example_config.yaml config/my_event.yaml
```

Key sections:
- **earthquake**: Event location, depth, magnitude, origin time
- **data**: Paths to miniSEED directories, station CSV, and instrument response (PZ or RESP files)
- **processing**: Channel, bandpass filter, decimation, response removal
- **animation**: Output format, playback speed (`realtime_factor`), map settings, marker style

### Data Format

- **MiniSEED files**: Named as `SS.{station}.SW.{channel}.{serial}.{date}.*.miniseed`
- **Station coordinates**: CSV with columns `serial_number, longitude, latitude, elevation, channel, location, station, network`
- **Instrument response**: SAC pole-zero (`.pz`) or RESP format files, mapped by serial number prefix

### Instrument Response

SmartSolo IGU-16HR geophones come in 2 Hz and 5 Hz natural frequency variants. Each requires its own response file:

```yaml
resp_files:
  "4530": /path/to/smartsolo.5Hz.pz    # 5 Hz geophones (serial prefix 4530)
  "4236": /path/to/smartsolo.2Hz.pz    # 2 Hz geophones (serial prefix 4236)
```

Both SAC PZ and RESP formats are supported; format is auto-detected from content.

## Processing Pipeline

1. Load miniSEED waveforms for the event time window (with buffer for edge effects)
2. Remove instrument response using poles & zeros (velocity output)
3. Cascaded decimation (handles factors > 16 automatically)
4. Bandpass filter (configurable, zero-phase)
5. Normalize each trace to its absolute maximum (for map color display)
6. Compute theoretical phase arrivals (P, S, SS, R1) via ObsPy TauPy
7. Animate two-panel figure: map with colored markers + reference trace with cursor

## References

- Trabant, C., A. R. Hutko, M. Bahavar, R. Karstens, T. Ahern, and R. Aster (2012), Data Products at the IRIS DMC: Stepping Stones for Research and Other Applications, *Seismological Research Letters*, 83(5), 846-854, doi:10.1785/gssrl.83.5.846.
- IRIS GMV code: https://github.com/iris-edu/gmv
- EarthScope GMV product: https://ds.iris.edu/ds/products/gmv/

## Author

Michail Henry (michail.henry@protonmail.com)

## License

MIT
