# WAYCON Local Run Guide

This guide explains how to run the current **Goal 3 + Goal 4 `person_creation` implementation** locally.

In every command below, replace `<REPO_ROOT>` with the path to your local WAYCON repository.

Example repository paths:

```text
WSL:
/mnt/c/Users/<YOUR_WINDOWS_USERNAME>/Documents/GitHub/WAYCON

Windows:
C:\Users\<YOUR_WINDOWS_USERNAME>\Documents\GitHub\WAYCON
1. What This Starts

Run these three parts in separate terminals:

Part	Purpose	URL
face_engine service	Performs face detection and face embedding	http://127.0.0.1:5010
person_creation backend	Runs the processing workflow and API	http://127.0.0.1:5009
React/Vite frontend	Provides the local user interface	http://localhost:5175

The backend uses face_engine when:

PERSON_CREATION_USE_FACE_ENGINE=1

Local fallback behavior is controlled by:

FACE_ENGINE_FALLBACK_LOCAL=0
2. Requirements

Before starting, make sure the machine has:

WSL
Python 3
CUDA-enabled PyTorch
Node.js and npm
NVIDIA GPU
A local clone of the WAYCON repository
3. Terminal 1 — Start face_engine

Open a WSL terminal and run:

cd <REPO_ROOT>
PERSON_CREATION_DEVICE=cuda python3 -m forensics.face_engine.service

Expected logs include:

FaceDetector loaded on cuda
FaceEmbedder loaded on cuda
Running on http://127.0.0.1:5010

From another WSL terminal, check service health:

curl http://127.0.0.1:5010/health

From PowerShell, use either:

curl.exe http://127.0.0.1:5010/health

or:

Invoke-RestMethod http://127.0.0.1:5010/health

Expected health response includes:

{
  "ok": true,
  "selected_device": "cuda",
  "cuda_available": true,
  "models_loaded": {
    "detector": true,
    "embedder": true
  }
}
4. Terminal 2 — Start person_creation Backend

Open another WSL terminal and run:

cd <REPO_ROOT>

export PERSON_CREATION_USE_FACE_ENGINE=1
export FACE_ENGINE_URL=http://127.0.0.1:5010
export FACE_ENGINE_FALLBACK_LOCAL=0
export PERSON_CREATION_DEVICE=cuda

python3 -m forensics.person_creation.service

Expected output includes the selected CUDA device and backend address:

selected_device cuda
http://127.0.0.1:5009

Fallback behavior:

FACE_ENGINE_FALLBACK_LOCAL=0
    Fail clearly if face_engine is unavailable.

FACE_ENGINE_FALLBACK_LOCAL=1
    Log a warning and fall back to local models.
5. Terminal 3 — Start Frontend

Open a third terminal and run:

cd <REPO_ROOT>/forensics/person_creation/frontend
npm run dev

Open the frontend at:

http://localhost:5175
6. Manual Test From Frontend

Open:

http://localhost:5175

Choose a local video path and replace <VIDEO_PATH> with that path.

Example:

<VIDEO_PATH>

WSL video example:

/mnt/c/Users/<YOUR_WINDOWS_USERNAME>/Downloads/kn1.mp4

Repo-relative video example:

forensics/person_creation/videos/test.mp4

Use a fresh output directory for every run.

Example:

forensics/person_db/runs/manual_test_01

Next run:

forensics/person_db/runs/manual_test_02

Do not reuse the same output directory.

7. Optional RTSP Camera URI

For an optional live-camera test, use one of these URI formats.

Main stream:

rtsp://<USERNAME>:<PASSWORD>@<CAMERA_IP>:554/Streaming/Channels/101

Sub stream:

rtsp://<USERNAME>:<PASSWORD>@<CAMERA_IP>:554/Streaming/Channels/102

Use channel 101 for main-stream quality.
Use channel 102 for lighter testing.

8. Expected Successful Logs

A successful run should include messages similar to:

[process_video] using face_engine for face detection
[embed_all_faces] embedded X / X quality faces (face_engine)
[cluster_identities] N identity cluster(s)
[assign_bodies_to_clusters] clusters=N
[finalize] profile saved -> ...
[finalize] global memory registered ...

The values of X, N, and the saved paths depend on the input video.

9. Global Memory Check

From WSL, run:

cd <REPO_ROOT>
python3 -m forensics.person_creation.tools.inspect_global_memory

This displays registered persons, appearances, profile runs, and suggestions.

10. Troubleshooting
A. Port 5010 is already in use

Find the process:

ss -ltnp | grep 5010

Stop the existing face_engine process:

pkill -f "forensics.face_engine.service"

Then start Terminal 1 again.

B. face_engine is down

Check health:

curl http://127.0.0.1:5010/health

If the request fails, restart the service using the Terminal 1 commands.

C. Backend cannot connect to face_engine

Check that:

Terminal 1 is still running.
FACE_ENGINE_URL is http://127.0.0.1:5010.
The health check succeeds.
D. PowerShell curl warning

PowerShell may map curl to another command.

Use:

curl.exe http://127.0.0.1:5010/health

or:

Invoke-RestMethod http://127.0.0.1:5010/health
E. CUDA is not detected

Run:

python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

Confirm that CUDA is available and an NVIDIA GPU name is displayed.

F. Service is stale after changing service.py

The Flask service does not hot-reload in normal mode.

Restart face_engine manually so the running process loads the current code.

G. Output from an earlier run is being reused

Do not reuse an output directory.

Create a new folder for every test run, such as:

forensics/person_db/runs/manual_test_03
11. Stop Services

Press Ctrl+C in each running terminal.

Alternatively, stop the backend services from WSL:

pkill -f "forensics.face_engine.service"
pkill -f "forensics.person_creation.service"
12. Current Goal Status
Goal	Status
Goal 3 — Global Memory	Working
Goal 4 — Standalone Face Engine	Working
Goal 5 — HITL removal	Separate future work
Goal 6 — Docker	Separate future work
