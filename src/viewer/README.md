# Viewers

[Rerun](https://rerun.io/) based viewers.

**Dataset (ground truth only)** — take a sequence ID as a positional argument:

- `nhot3d_viewer.py` : N-HOT3D (Synthetic)
- `eehr_viewer.py` : EEH-R (Real)

**Prediction vs ground truth** — run the same inference pipeline as the matching
`test` script and log both hands into one 3D view. They take a `--config`, so the
checkpoints come from the config the model was evaluated with:

| | Synthetic (N-HOT3D) | Real (EEH-R) |
|---|---|---|
| v1 | `src/v1/vis_nhot3d.py` | `src/v1/vis_eehr.py` |
| v2 | `src/v2/vis.py --config .../config_nhot3d.yaml` | `src/v2/vis.py --config .../config_eehr.yaml` |

**Serving** — `serve_rrd.sh` serves saved `.rrd` recordings over HTTP.

Every viewer supports the same three output modes below. The scripts always run
**where the dataset is** (normally the server); only the display side differs.

## Simplest usage (dataset on the same machine as the display)

Start the Rerun viewer in one terminal, then feed it data from another:

```bash
rerun                                               # terminal 1
python src/viewer/nhot3d_viewer.py P0001_15c4300c   # terminal 2
python src/viewer/eehr_viewer.py P04_01 --fps 120   # terminal 2

# prediction viewers
python src/v1/vis_nhot3d.py --config src/v1/config/config_test_nhot3d.yaml \
    --sequence-ids P0002_016222d1
python src/v2/vis.py --config src/v2/config/config_eehr.yaml --sequence-ids P04_01
```

The script connects to `127.0.0.1:9876` by default; change it with `--ip` / `--port`.

## Dataset on a server, native viewer on your local machine

The script runs on the server and has to reach the Rerun viewer running on your
local machine. Open a **reverse** tunnel (`-R`) from your local machine, so that
`127.0.0.1:9876` on the server points back at your local Rerun viewer:

```bash
# local machine
rerun                                       # terminal 1
ssh -N -R 9876:localhost:9876 <user>@<server>   # terminal 2
```

```bash
# server (through a normal ssh session)
python src/viewer/eehr_viewer.py P04_01 --fps 120   # --ip/--port defaults are correct
```

With the tunnel the defaults already point at the right place, so no `--ip` is
needed. If the server can reach your machine directly (same LAN, no firewall),
you can skip the tunnel and pass your local address instead:

```bash
python src/viewer/eehr_viewer.py P04_01 --fps 120 --ip 192.168.x.x
```

## Browser instead of the native viewer

`--web-viewer` hosts the web viewer from the script itself, so nothing has to be
installed locally:

```bash
# server
python src/viewer/eehr_viewer.py P04_01 --fps 120 --web-viewer
```

Forward both ports from your local machine and open `http://localhost:9090`:

```bash
# local machine
ssh -N -L 9090:localhost:9090 -L 9877:localhost:9877 <user>@<server>
```

`<user>@<server>` is the remote side, and the `localhost` in the middle of `-L`
is resolved **on the server**. If the Rerun process runs on a different node than
the one you SSH into (e.g. a compute node behind a login node), put that node's
name there instead: `-L 9090:<node>:9090 -L 9877:<node>:9877`.

Note that the script has to stay running for the page to keep working.

## Save to `.rrd` and serve it later

`--recording` writes a recording file instead of streaming, so the heavy loading
runs once and can be replayed any time:

```bash
python src/viewer/nhot3d_viewer.py P0001_15c4300c --recording data/save/rrd/P0001_15c4300c.rrd
python src/viewer/eehr_viewer.py P04_01 --fps 120 --recording data/save/rrd/P04_01.rrd

python src/v1/vis_nhot3d.py --config src/v1/config/config_test_nhot3d.yaml \
    --sequence-ids P0002_016222d1 --recording data/save/rrd/v1_P0002_016222d1.rrd
```

Copy the file to your machine and open it with `rerun P04_01.rrd`, or host it
from the server:

```bash
# server
./src/viewer/serve_rrd.sh data/save/rrd/*.rrd
```

The script prints the URL to open and the two ports to forward, e.g.

```
rerun      : 0.22.1
Web viewer : http://localhost:9090?url=ws%3A%2F%2Flocalhost%3A9877
(forward ports 9090 and 9877 to this machine first)
```

Forward both ports from your local machine, then open the printed URL:

```bash
# local machine
ssh -N -L 9090:localhost:9090 -L 9877:localhost:9877 <user>@<server>
```

Keep the local and remote port numbers identical: the web page hardcodes the data
port into its connect URL. Ports and the buffer size can be overridden with
`WEB_PORT`, `DATA_PORT` and `MEMORY_LIMIT` environment variables.

`rerun < 0.23` streams to the web viewer over WebSocket and `>= 0.23` over gRPC;
`serve_rrd.sh` reads `rerun --version` and picks the right flag and URL, so it
works with the pinned version as well as newer ones.

## Prediction viewers

`src/v1/vis_nhot3d.py`, `src/v1/vis_eehr.py` and `src/v2/vis.py` reproduce the
inference pipeline of the matching `test` script, so they read the dataset and
checkpoint paths from the same config file.

```bash
python src/v1/vis_nhot3d.py --config src/v1/config/config_test_nhot3d.yaml \
    --sequence-ids P0002_016222d1 --num-samples 200
```

Common options: `--sequence-ids` (one or more sequences; all frames if omitted),
`--num-samples` (stop after N frames), `--checkpoint` (override
`test.checkpoint_path`), plus the output options above. `vis_nhot3d.py` adds
`--project-2d` (draw the predicted joints on the event frame with a pinhole
approximation that ignores the Aria distortion model) and `vis_eehr.py` adds
`--grayscale` (also load the reference images).

Entity layout, identical across the three scripts:

```
2D/event_frame, 2D/lnes, 2D/segmentation | 2D/yolo_mask, ...
3D/gt/{left,right}/{mesh,joints,skeleton}     saturated red / blue
3D/pred/{left,right}/{mesh,joints,skeleton}   pastel pink / blue
```

Both hands live in one 3D view, so use the entity tree on the left to hide
`3D/gt` or `3D/pred`. Hands the ground truth marks invalid, and hands YOLO did not
detect in v2, are cleared for that frame instead of being left over from the
previous one.

Frames are processed one at a time: `MANORegressor` feeds `betas[0]` to the MANO
layer, so a larger batch would render every hand in the batch with the first
sample's shape.

EEH-R annotations carry no mesh topology, so the ground-truth mesh borrows the
MANO faces from the model output. With `annotation_source: mocap` the ground
truth is 16 joints with no vertices, and only the skeleton is drawn.

## Dataset paths

The dataset roots are constants at the top of each script
(`BASE_DIR` / `RGB_BASE_DIR` in `nhot3d_viewer.py`, `DATASET_ROOT` /
`YOLO_MASK_ROOT` in `eehr_viewer.py`). Edit them to match your machine.
The prediction viewers take their paths from the config file instead.
