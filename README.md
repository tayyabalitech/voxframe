# 🌌 VoxFrame — Multimodal Narrative Synthesis Engine

```
 ___  __  ___  __  ___  __  ___  __  ___  __  ___  __ 
|   \/  \|   \/  \|   \/  \|   \/  \|   \/  \|   \/  |
|            V O X F R A M E   E N G I N E           |
|___/\__/|___/\__/|___/\__/|___/\__/|___/\__/|___/\__|
```

**VoxFrame** is a video captioning pipeline for **AMD Developer Hackathon: ACT II (Track 2)**. It downloads each clip, extracts adaptive keyframes, optionally transcribes speech, grounds the scene with a vision model, generates four styled captions, audits them locally, and refines weak styles with a vision-based self-grader.

---

## 🛠️ Pipeline Overview

```
sample_inputs/tasks.json
        │
        ▼
Video download + adaptive keyframes (6/8/10) + Groq Whisper (optional)
        │
        ▼
Stage A   Scene JSON grounding (minimax-m3)
        │
        ▼
Stage A.2 Scene verification against frames
        │
        ▼
Stage B   Vision-grounded caption JSON (minimax-m3 + frames)
        │
        ▼
Stage C   Rule audit + targeted repair pass
        │
        ▼
Stage D   Self-grader vs keyframes (minimax-m3)
        │
        ▼
Stage E   Multi-candidate refinement for weak styles
        │
        ▼
output/results.json
```

Each task returns four captions: `formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`.

---

## 📂 Project Layout

| Path | Role |
|------|------|
| `run.py` | Container entrypoint — reads `/input/tasks.json`, writes `/output/results.json` |
| `voxframe/config/cfg.py` | API keys, models, timeouts, refinement threshold |
| `voxframe/config/defs.py` | Pydantic schemas for captions and grader scores |
| `voxframe/media_processing/media_utils.py` | Download, keyframes, scene-change sampling, ASR |
| `voxframe/engines/author.py` | Stages A–E caption pipeline |
| `voxframe/engines/grader.py` | Vision-based caption quality scoring |
| `sample_inputs/tasks.json` | Local validation tasks |
| `web_dashboard/` | Optional local UI on port 7860 |

---

## ⚙️ Configuration

Copy `.env.example` to `.env` for local runs outside Docker:

```env
FIREWORKS_API_KEY=your_key
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
FIREWORKS_VISION_MODEL=accounts/fireworks/models/minimax-m3
FIREWORKS_TEXT_MODEL=accounts/fireworks/models/deepseek-v4-pro

GROQ_API_KEY=your_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_WHISPER_MODEL=whisper-large-v3

PER_CLIP_TIMEOUT_S=300
MAX_CONCURRENT_CLIPS=1
JSON_RETRY_ATTEMPTS=3
SCORE_THRESHOLD=0.85
WEAK_STYLE_CANDIDATES=3
CAPTION_MIN_WORDS=8
CAPTION_MAX_WORDS=70
```

**Default models**

- **Vision (`minimax-m3`)** — scene grounding, caption generation, grader (requires image input)
- **Text (`deepseek-v4-pro`)** — reserved for text-only use; not used for image grading

---

## 🚀 Run Locally (Docker)

### Build

```powershell
docker build -t voxframe-app .
```

### Run evaluation engine (Track 2 / lablab.ai)

```powershell
docker run -it --rm -p 7860:7860 `
  -v "${PWD}\sample_inputs:/input" `
  -v "${PWD}\output:/output" `
  --name my-voxframe-app voxframe-app
```

Linux / macOS:

```bash
docker run -it --rm -p 7860:7860 \
  -v "$(pwd)/sample_inputs:/input" \
  -v "$(pwd)/output:/output" \
  --name my-voxframe-app voxframe-app
```

Results are written to `output/results.json`.

### Save image archive (optional)

```powershell
docker save -o voxframe-image.tar voxframe-app
```

---

## 📦 Push Source to GitHub

From the project root, after reviewing `git status`:

```powershell
cd "D:\Track 2 project"

git status
git add .
git commit -m "feat: minimax-m3 pipeline with grader refinement and JSON retry reliability"
git push -u origin main
```

If the remote branch already exists and you only need to upload new commits:

```powershell
git add .
git commit -m "feat: minimax-m3 pipeline with grader refinement and JSON retry reliability"
git push origin main
```

Repository: **https://github.com/tayyabalitech/voxframe**

---

## 📦 Build & Push Docker Image to GHCR

Use a GitHub PAT with `write:packages` and `repo` scopes.

### 1. Log in to GHCR

```powershell
docker login ghcr.io -u tayyabalitech -p YOUR_GITHUB_PAT
```

### 2. Build and tag for GHCR

```powershell
cd "D:\Track 2 project"

docker build -t ghcr.io/tayyabalitech/voxframe-app:latest .
```

### 3. Push to GitHub Container Registry

```powershell
docker push ghcr.io/tayyabalitech/voxframe-app:latest
```

### 4. (Optional) Versioned tag

```powershell
docker tag ghcr.io/tayyabalitech/voxframe-app:latest ghcr.io/tayyabalitech/voxframe-app:v2.0.0
docker push ghcr.io/tayyabalitech/voxframe-app:v2.0.0
```

Package page: **https://github.com/tayyabalitech/voxframe/pkgs/container/voxframe-app**

### 5. Make the package public (first time only)

GitHub → **Packages** → `voxframe-app` → **Package settings** → **Change visibility** → **Public**

Required for lablab.ai judges to pull without auth.

---

## 🧪 Pull & Run from GHCR

```powershell
docker login ghcr.io -u tayyabalitech -p YOUR_GITHUB_PAT
docker pull ghcr.io/tayyabalitech/voxframe-app:latest

docker run -it --rm -p 7860:7860 `
  -v "${PWD}\sample_inputs:/input" `
  -v "${PWD}\output:/output" `
  ghcr.io/tayyabalitech/voxframe-app:latest
```

---

## 🏁 lablab.ai Submission Checklist

- [ ] Image is public on GHCR: `ghcr.io/tayyabalitech/voxframe-app:latest`
- [ ] Container starts with `python run.py` (no manual setup)
- [ ] `/output/results.json` is created for all tasks
- [ ] Every task has all four caption styles
- [ ] Full run completes within the hackathon timeout
- [ ] Do not resubmit repeatedly to move up the queue (FAQ guidance)

**Submit this image reference:**

```text
ghcr.io/tayyabalitech/voxframe-app:latest
```

---

## 🎯 Design Notes

- **JSON retry logic** handles intermittent vision-model prose/truncation on Stage A.
- **Stage C repair** re-prompts only failing styles instead of failing the whole task.
- **Stage E refinement** keeps the best candidate per weak style using the self-grader.
- **Concurrency defaults to 1** for stable judge runs.

---

## 📄 License

See repository license. Hackathon submission for AMD Developer Hackathon ACT II — Track 2.
