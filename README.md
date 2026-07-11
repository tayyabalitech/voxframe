# 🌌 VoxFrame — Multimodal Narrative Synthesis Engine

```
 ___  __  ___  __  ___  __  ___  __  ___  __  ___  __ 
|   \/  \|   \/  \|   \/  \|   \/  \|   \/  \|   \/  |
|            V O X F R A M E   E N G I N E           |
|___/\__/|___/\__/|___/\__/|___/\__/|___/\__/|___/\__|
```

**VoxFrame** is a state-of-the-art cognitive video analysis and narrative generation system designed for the **AMD Developer Hackathon: ACT II (Track 2)**. By integrating zero-latency ffmpeg frame capture pipelines, whisper-based speech-to-text, and low-temperature scene verification loops, VoxFrame translates raw video streams into highly aligned styled narratives across four distinct semantic registers.

---

## 🛠️ System Architecture & Data Flow

Unlike naive one-shot caption models that suffer from hallucinations and style drift, VoxFrame uses a rigorous **Comprehend-Verify-Compose-Audit** execution pattern to secure accuracy and stylistic separation.

```
                  +-----------------------------------+
                  |      sample_inputs/tasks.json     |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Video Download & Segmenting    |
                  +-----------------------------------+
                                    |
            +-----------------------+-----------------------+
            | (Visual Stream)                               | (Audio Track)
            v                                               v
+-----------------------+                       +-----------------------+
|  Adaptive Keyframes   |                       |   PCM 16kHz WAV ASR   |
|     (6/8/10 sampling) |                       |   (Groq Whisper v3)   |
+-----------------------+                       +-----------------------+
            |                                               |
            +-----------------------+-----------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |     Stage A: Scene Comprehension  |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Stage A.2: Grounding Audit     |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |     Stage B: Multitone Synthesis  |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Rule-Based Post-Check Audit    |
                  +-----------------------------------+
                                    |  (Auditing violations detected)
                                    v  --> [Targeted Re-Generation Pass]
                  +-----------------------------------+
                  |   Dimension Grader & Self-Refine  |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |        output/results.json        |
                  +-----------------------------------+
```

---

## 📂 Modular Anatomy

The implementation is structured into a clean, decoupled layout to decouple IO acquisition from inference and grading loops:

* 📡 `voxframe.config`
  * `cfg.py`: Loads `.env` secrets, initializes runtime limits, timeouts, and thresholds.
  * `defs.py`: Strong data contracts (Pydantic V2) mapping input, scene comprehension schema, and evaluation scores.
* 🎥 `voxframe.media_processing`
  * `clip.py`: Downloads media, dynamically scales frame extraction buffers based on track duration, and transcribes spoken audio.
* 🧠 `voxframe.engines`
  * `author.py`: Coordinates the multi-stage visual composer, audits style outputs using fast vocabulary check algorithms, and manages targeted candidate improvements.
  * `grader.py`: Emulates the official evaluation judge to grade candidate descriptions.

---

## 🎯 Strategic Optimizations

### 1. Verification-Guided Grounding (Comprehend -> Audit -> Compose)
Instead of prompting the model to generate creative captions directly from raw frames, VoxFrame implements a **Stage A.1 Scene Comprehension** step to extract subject, setting, and motion details in a structured JSON layout. **Stage A.2 Verification** then re-evaluates the output against the visual frames to remove hallucinations. Finally, **Stage B** translates the verified grounding context into styled captions.

### 2. Multi-Candidate Tone Refinement
If the self-grader evaluates any tone below a critical score (`0.65` threshold), the refinement engine initiates candidate generation. It compiles three distinct variations for the flagging style and updates the output with the version scoring the highest cumulative quality.

### 3. Punctuation & Style Guard (Audit Pass)
Rule-based scripts run immediately after text generation to check target lengths and style markers:
* **Formal**: Strict news tone, third-person active verbs, absolute ban on exclamation marks `!`.
* **Sarcastic**: Irony checks, dry understatement, avoids simplistic slang.
* **Humorous Tech**: Compares text against a vocabulary vector of 70+ software and hardware terms (e.g., `pipeline`, `I/O`, `latency`, `stack`, `GPU`).
* **Humorous Non-Tech**: Filters and flags hard engineering concepts to keep the caption accessible.

---

## ⚙️ Configuration Setup

Save your credentials inside a `.env` file at the root of the workspace:

```env
FIREWORKS_API_KEY=your_key
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
FIREWORKS_VISION_MODEL=accounts/fireworks/models/minimax-m3
FIREWORKS_FALLBACK_VISION_MODEL=accounts/fireworks/models/qwen3p7-plus

GROQ_API_KEY=your_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_WHISPER_MODEL=whisper-large-v3

# Parameters
PER_CLIP_TIMEOUT_S=300       # Per-clip timeout (seconds)
SCORE_THRESHOLD=0.65         # Minimum target quality score
WEAK_STYLE_CANDIDATES=3      # Multi-candidate budget
```

---

## 🚀 Running VoxFrame (Docker)

## 🚀 Running VoxFrame (Docker)

To ensure a seamless experience across all environments, VoxFrame is entirely containerized using Docker. The necessary environment variables are pre-configured into the container, so no local setup is required.

### 1. Build the Docker Image
First, build the Docker container using the provided `Dockerfile`.
```powershell
docker build -t voxframe-app .
```

### 2. Run the Visual Web Dashboard
Start the container and expose port `7860` to access the premium drag-and-drop web interface. We mount local input and output directories so your data persists.

**For Windows (PowerShell):**
```powershell
docker run -it --rm -p 7860:7860 -v "${PWD}\sample_inputs:/input" -v "${PWD}\output:/output" --name my-voxframe-app voxframe-app
```

**For Linux / macOS:**
```bash
docker run -it --rm -p 7860:7860 -v "$(pwd)/sample_inputs:/input" -v "$(pwd)/output:/output" --name my-voxframe-app voxframe-app
```

Then navigate your browser to **`http://127.0.0.1:7860`** to access the application.

### 3. Distribute the Project
To share the container with your team or upload it to lablab.ai, save the image as a `.tar` archive:
```bash
docker save -o voxframe-image.tar voxframe-app
```

### 4. Load from Archive (For Evaluators/Users)
If you are receiving the `.tar` file, load it into your Docker engine and run the dashboard directly without needing the source code:

**Load Image:**
```bash
docker load -i voxframe-image.tar
```

**Run Container (Windows PowerShell):**
```powershell
docker run -it --rm -p 7860:7860 -v "${PWD}\sample_inputs:/input" -v "${PWD}\output:/output" --name my-voxframe-app voxframe-app
```

**Run Container (Linux / macOS):**
```bash
docker run -it --rm -p 7860:7860 -v "$(pwd)/sample_inputs:/input" -v "$(pwd)/output:/output" --name my-voxframe-app voxframe-app
```
