# WAYCON Local Run Guide

This guide starts the current Goal 3 + Goal 4 `person_creation` implementation
locally.

In every command below, replace `<REPO_ROOT>` with the path to your local
WAYCON repository.

Example repository paths:

- WSL:
  `/mnt/c/Users/<YOUR_WINDOWS_USERNAME>/Documents/GitHub/WAYCON`
- Windows:
  `C:\Users\<YOUR_WINDOWS_USERNAME>\Documents\GitHub\WAYCON`

## 1. What This Starts

Run these three parts in separate terminals:

1. The `face_engine` service performs face detection and face embedding. It
   listens at <http://127.0.0.1:5010>.
2. The `person_creation` backend runs the processing workflow and API. It
   listens at <http://127.0.0.1:5009>.
3. The React/Vite frontend provides the local user interface. It listens at
   <http://localhost:5175>.

The backend uses `face_engine` when
`PERSON_CREATION_USE_FACE_ENGINE=1`. Local fallback behavior is controlled by
`FACE_ENGINE_FALLBACK_LOCAL`.

## 2. Requirements

Before starting, make sure the machine has:

- WSL
- Python 3
- CUDA-enabled PyTorch
- Node.js and npm
- An NVIDIA GPU
- A local clone of the WAYCON repository

## 3. Terminal 1 — Start face_engine

Open a WSL terminal and run:

```bash
cd <REPO_ROOT>
PERSON_CREATION_DEVICE=cuda python3 -m forensics.face_engine.service
```

Expected logs include:

```text
FaceDetector loaded on cuda
FaceEmbedder loaded on cuda
Running on http://127.0.0.1:5010
```

From another WSL terminal, check service health:

```bash
curl http://127.0.0.1:5010/health
```

From PowerShell, use either:

```powershell
curl.exe http://127.0.0.1:5010/health
```

or:

```powershell
Invoke-RestMethod http://127.0.0.1:5010/health
```

## 4. Terminal 2 — Start person_creation Backend

Open another WSL terminal and run:

```bash
cd <REPO_ROOT>

export PERSON_CREATION_USE_FACE_ENGINE=1
export FACE_ENGINE_URL=http://127.0.0.1:5010
export FACE_ENGINE_FALLBACK_LOCAL=0
export PERSON_CREATION_DEVICE=cuda

python3 -m forensics.person_creation.service
```

Expected output includes the selected CUDA device and backend address:

```text
selected_device cuda
http://127.0.0.1:5009
```

Fallback behavior:

- `FACE_ENGINE_FALLBACK_LOCAL=0` fails clearly if `face_engine` is unavailable.
- `FACE_ENGINE_FALLBACK_LOCAL=1` logs a warning and falls back to local models.

## 5. Terminal 3 — Start Frontend

Open a third terminal and run:

```bash
cd <REPO_ROOT>/forensics/person_creation/frontend
npm run dev
```

Open the frontend at:

<http://localhost:5175>

## 6. Manual Test From Frontend

1. Open <http://localhost:5175>.
2. Choose a local video path and replace `<VIDEO_PATH>` with that path.
3. Enter:

   ```text
   <VIDEO_PATH>
   ```

   WSL video example:

   ```text
   /mnt/c/Users/<YOUR_WINDOWS_USERNAME>/Downloads/kn1.mp4
   ```

   Repo-relative video example:

   ```text
   forensics/person_creation/videos/test.mp4
   ```

4. Use a fresh output directory for every run, for example:

   ```text
   forensics/person_db/runs/manual_test_01
   ```

   On the next run, use another directory:

   ```text
   forensics/person_db/runs/manual_test_02
   ```

Do not reuse the same output directory.

## 7. RTSP Camera URI

For an optional live-camera test, use one of these URIs.

Main stream:

```text
rtsp://admin:Waycon2026@10.0.0.104:554/Streaming/Channels/101
```

Sub stream:

```text
rtsp://admin:Waycon2026@10.0.0.104:554/Streaming/Channels/102
```

Use channel `101` for main-stream quality. Use channel `102` for lighter
testing.

## 8. Expected Successful Logs

A successful run should include messages similar to:

```text
[process_video] using face_engine for face detection
[embed_all_faces] embedded X / X quality faces (face_engine)
[cluster_identities] N identity cluster(s)
[assign_bodies_to_clusters] clusters=N
[finalize] profile saved -> ...
[finalize] global memory registered ...
```

The values of `X`, `N`, and the saved paths depend on the input.

## 9. Global Memory Check

From WSL, run:

```bash
cd <REPO_ROOT>
python3 -m forensics.person_creation.tools.inspect_global_memory
```

This displays registered persons, appearances, profile runs, and suggestions.

## 10. Troubleshooting

### A. Port 5010 is already in use

Find the process:

```bash
ss -ltnp | grep 5010
```

Stop the existing `face_engine` process:

```bash
pkill -f "forensics.face_engine.service"
```

Then start Terminal 1 again.

### B. face_engine is down

Check its health:

```bash
curl http://127.0.0.1:5010/health
```

If the request fails, restart the service using the Terminal 1 commands.

### C. Backend cannot connect to face_engine

Check that:

- Terminal 1 is still running.
- `FACE_ENGINE_URL` is
  `http://127.0.0.1:5010`.
- The health check succeeds.

### D. PowerShell curl warning

PowerShell may map `curl` to another command. Use:

```powershell
curl.exe http://127.0.0.1:5010/health
```

### E. CUDA is not detected

Run:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Confirm that CUDA is available and an NVIDIA GPU name is displayed.

### F. Service is stale after changing service.py

Restart `face_engine` manually so the running process loads the current code.

### G. Output from an earlier run is being reused

Do not reuse an output directory. Create a new folder for every test run, such
as:

```text
forensics/person_db/runs/manual_test_03
```

## 11. Stop Services

Press `Ctrl+C` in each running terminal.

Alternatively, stop the backend services from WSL:

```bash
pkill -f "forensics.face_engine.service"
pkill -f "forensics.person_creation.service"
```

## 12. Current Goal Status

- Goal 3 — Global Memory: working
- Goal 4 — Standalone Face Engine: working
- Goal 5 — HITL removal: separate future work
- Goal 6 — Docker: separate future work
