# TamilVision 156

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=nextdotjs&logoColor=white"/>
  <img src="https://img.shields.io/badge/Three.js-0.183-black?style=for-the-badge&logo=threedotjs&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-11.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
</p>

> **Real-time Tamil handwritten character recognition in the browser.**  
> Draw a character on the canvas, upload a scan, or drop a photo — TamilVision identifies it from **156 Unicode Tamil characters** in under 50 ms.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🖊️ **Canvas drawing** | Draw directly in the browser with an adjustable-width brush on a touch & mouse-friendly canvas |
| 📁 **Image upload** | Drag-and-drop or browse for PNG, JPG, or BMP — including transparent PNGs and scanned documents |
| ⚡ **Real-time prediction** | Top-3 predictions returned with confidence scores and animated progress bars |
| 🔀 **Universal preprocessing** | Handles white-on-black canvas art *and* black-on-white scans through the same robust OpenCV pipeline |
| 🧠 **156-class coverage** | Vowels (உயிர்), pure consonants (மெய்), base consonants, and all six vowel-marker series |
| 📊 **Confidence colouring** | Green ≥ 70 %, amber 40–70 %, red < 40 % — instant visual feedback on prediction quality |
| 🌄 **Animated background** | GLSL Perlin-noise hills rendered in real-time with Three.js WebGL — zero impact on prediction latency |
| 🪟 **Glassmorphism UI** | Panels use `backdrop-blur` + translucent fills so the WebGL background shows through; sky-blue accent throughout |

---

## 🗂️ Project Structure

```
TamilVision/
├── frontend/                   # React / Next.js 16 frontend (shadcn + Tailwind v4)
│   ├── app/
│   │   ├── layout.tsx          # Root layout — dark theme, Mukta Malar font
│   │   ├── page.tsx            # Full TamilVision UI (canvas, upload, results)
│   │   └── globals.css         # Tailwind v4 + shadcn CSS variables
│   ├── components/
│   │   └── ui/
│   │       └── glsl-hills.tsx  # Three.js GLSL Perlin-noise hills background
│   ├── lib/utils.ts            # shadcn utility helpers
│   ├── package.json            # React 19, Next.js 16, Three.js 0.183
│   └── index_vanilla.html      # Original vanilla JS/HTML backup
│
├── app/
│   └── main.py                 # FastAPI server & /predict endpoint
│
├── src/
│   ├── config.py               # 156 Tamil class labels & hyperparameters
│   ├── model.py                # TamilVision architecture (MobileNetV3-Small)
│   ├── preprocess.py           # OpenCV inference pipeline + torchvision train transforms
│   ├── dataset.py              # PyTorch Dataset with shared-memory RAM cache
│   └── train.py                # Full training loop (AMP, AdamW, cosine LR)
│
├── models/
│   └── best_model.pth          # Trained checkpoint (~19 MB)
│
├── scripts/                    # Utility scripts (visualize, validate, sanity-check, auto-tune)
├── data/                       # Dataset root (gitignored — see Dataset section)
├── requirements.txt            # Python backend dependencies
└── .gitignore
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Model** | PyTorch 2.1 · MobileNetV3-Small · ImageNet pretrained weights |
| **Preprocessing** | OpenCV 4.8 · NumPy · Pillow |
| **Backend** | FastAPI 0.104 · Uvicorn · python-multipart |
| **Frontend** | Next.js 16 · React 19 · TypeScript · Tailwind CSS v4 · shadcn/ui |
| **3-D Background** | Three.js 0.183 · GLSL Perlin-noise vertex shader (`GLSLHills`) |
| **Training** | Mixed-precision AMP · AdamW · CosineAnnealingLR · Label smoothing |
| **Hardware** | NVIDIA GTX 1650 · CUDA 11.8 |

---

## 🌄 GLSLHills — Animated Background

The animated wireframe hills are a self-contained React component (`components/ui/glsl-hills.tsx`) that runs entirely on the GPU via Three.js + custom GLSL shaders. It is zero-dependency beyond Three.js and adds **no overhead** to the prediction pipeline.

### How it works

| Step | Detail |
|---|---|
| **Geometry** | `PlaneGeometry(256, 256, 256, 256)` — 256×256 subdivided flat plane |
| **Vertex shader** | Rotates the plane to face the camera, then displaces each vertex vertically using **3-octave Classic Perlin Noise** (`cnoise`). The noise input drifts along the Z axis over `time`, creating the flowing hills illusion. |
| **Fragment shader** | Solid grey (`vec3(0.6)`) with opacity that fades out with distance — edges dissolve naturally. |
| **Animation** | A `requestAnimationFrame` loop advances `uniforms.time` each frame. The loop is cancelled on React unmount to prevent memory leaks. |
| **Resize** | A `window.resize` listener keeps the camera aspect ratio and renderer size in sync. |

### Props

| Prop | Type | Default | Description |
|---|---|---|---|
| `width` | `string` | `"100vw"` | Container width |
| `height` | `string` | `"100vh"` | Container height |
| `cameraZ` | `number` | `125` | Camera Z distance (zoom) |
| `planeSize` | `number` | `256` | Plane subdivisions & size |
| `speed` | `number` | `0.5` | Animation speed multiplier |

---

### 1 — Model Architecture

`TamilVision` is a fine-tuned **MobileNetV3-Small** with two surgical modifications:

1. **Grayscale input adapter** — the first 3-channel RGB conv layer is replaced with a single-channel layer. The pretrained RGB weights are *summed* across the channel axis, preserving all learned edge and texture detectors without discarding any prior knowledge.
2. **New classifier head** — the final `Linear(1024 → 1000)` ImageNet head is replaced with `Linear(1024 → 156)`, initialised with Kaiming-uniform weights.

The model accepts tensors of shape `[B, 1, 128, 128]` normalised to `[-1, 1]` and outputs `[B, 156]` logits.

---

### 2 — OpenCV Inference Preprocessing Pipeline

The biggest challenge this project solves is the **domain gap**: the training data is *black ink on white*, but the browser canvas produces *white ink on black* with large empty margins. The 12-step pipeline in `src/preprocess.py` bridges that gap for every possible input type:

```
Canvas PNG             Uploaded JPG/PNG        Transparent PNG
(white on black)       (black on white)        (BGRA)
       │                      │                     │
       └──────────────────────┴─────────────────────┘
                              ▼
          ┌─────────────────────────────────────────┐
          │  Step 1 · cv2.IMREAD_UNCHANGED           │
          │  4-channel BGRA → alpha-composite onto   │
          │  solid white → convert to grayscale      │
          └───────────────────┬─────────────────────┘
                              ▼
          ┌─────────────────────────────────────────┐
          │  Step 2 · Otsu's Binary Threshold        │
          │  Forces every pixel to pure 0 or 255,   │
          │  eliminating JPEG compression noise      │
          └───────────────────┬─────────────────────┘
                              ▼
          ┌─────────────────────────────────────────┐
          │  Step 3 · Smart Background Inversion     │
          │  np.mean(img) > 127 → light background  │
          │  detected → cv2.bitwise_not to flip      │
          │  Result: white character on black bg     │
          └───────────────────┬─────────────────────┘
                              ▼
          ┌─────────────────────────────────────────┐
          │  Step 4 · Median Blur (3×3)              │
          │  Removes single-pixel JPG artefacts      │
          │  without blurring stroke edges           │
          └───────────────────┬─────────────────────┘
                              ▼
          ┌─────────────────────────────────────────┐
          │  Step 5 · Bounding-Box Crop              │
          │  cv2.findNonZero + cv2.boundingRect      │
          │  Discards all surrounding black margin   │
          └───────────────────┬─────────────────────┘
                              ▼
          ┌─────────────────────────────────────────┐
          │  Step 6 · Aspect-Ratio Square Pad        │
          │  Pads the shorter axis with black pixels │
          │  → perfect square, no stretching         │
          └───────────────────┬─────────────────────┘
                              ▼
          ┌─────────────────────────────────────────┐
          │  Step 7 · Uniform Border Padding (15 px) │
          │  cv2.copyMakeBorder, value=0             │
          │  Glyph never touches the image edge      │
          └───────────────────┬─────────────────────┘
                              ▼
          ┌─────────────────────────────────────────┐
          │  Step 8 · Final Invert                   │
          │  cv2.bitwise_not → black ink on white    │
          │  Matches training data format exactly    │
          └───────────────────┬─────────────────────┘
                              ▼
          ┌─────────────────────────────────────────┐
          │  Step 9 · Gaussian Blur (3×3, σ=1.5)    │
          │  Softens hard digital edges to match     │
          │  the scanned-ink texture of the dataset  │
          └───────────────────┬─────────────────────┘
                              ▼
          ┌─────────────────────────────────────────┐
          │  Step 10 · Resize to 128×128 (Lanczos)  │
          └───────────────────┬─────────────────────┘
                              ▼
          ┌─────────────────────────────────────────┐
          │  Step 11 · Normalise → [-1, 1]           │
          │  tensor = (pixel/255 − 0.5) / 0.5        │
          │  Shape: [1, 1, 128, 128] float32         │
          └───────────────────┬─────────────────────┘
                              ▼
                       TamilVision Model
                              ▼
                    Softmax → Top-3 results
```

---

### 3 — Training Details

| Setting | Value |
|---|---|
| Dataset | uTHCD (80/20 split) |
| Classes | 156 Tamil Unicode characters |
| Input size | 128 × 128 grayscale |
| Batch size | 256 |
| Epochs | 30 |
| Optimiser | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | 3-epoch linear warm-up → CosineAnnealingLR |
| Loss | CrossEntropyLoss with label smoothing = 0.1 |
| Augmentation | RandomRotation ±20°, RandomAffine, RandomPerspective, ElasticTransform |
| Precision | Mixed (AMP / FP16) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ and npm (for the React frontend)
- NVIDIA GPU with CUDA 11.8 recommended (CPU also works — see note below)
- Git

### 1 — Clone the repository

```bash
git clone https://github.com/JaiivantArvind/TamilVision.git
cd TamilVision
```

### 2 — Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **CPU-only machines:** Edit the first two lines of `requirements.txt` before installing:
> ```
> --index-url https://download.pytorch.org/whl/cpu
> torch==2.1.0
> torchvision==0.16.0
> ```

### 4 — Start the API server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
[TamilVision] Model loaded — device: cuda | best val acc: XX.XX%
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5 — Start the React frontend

Open a **second terminal** tab and run:

```bash
cd frontend
npm install        # first run only — installs Next.js, Three.js, shadcn, etc.
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000) in your browser.

The animated GLSL wireframe hills render in the background while the prediction panels load on top.  
The status dot in the top-right corner turns **green** when the frontend is connected to the FastAPI server.

> **Vanilla fallback:** `frontend/index_vanilla.html` is the original single-file UI and still works  
> without any build step if you open it directly in the browser.

---

## 🌐 Deploy Live (Recommended: Render + Vercel)

### Backend (Render)

1. Push this repo to GitHub.
2. In Render, create a **Web Service** from the repo.
3. Render auto-detects `render.yaml`, so build/start are preconfigured:
   - **Build:** `pip install -r requirements.deploy.txt`
   - **Start:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Set environment variable:
   - `CORS_ORIGINS=https://<your-vercel-domain>`
5. Deploy and copy the backend URL, e.g. `https://tamilvision-api.onrender.com`.

> `requirements.deploy.txt` uses CPU PyTorch wheels so deployment works on standard cloud instances.

### Frontend (Vercel)

1. Import the same repo into Vercel.
2. Set the project root to `frontend/`.
3. Add environment variable:
   - `NEXT_PUBLIC_API_URL=https://<your-render-backend-domain>` (base URL only, no `/predict`)
4. Deploy.

After both deploys, open the Vercel URL and confirm the status indicator turns green.

Environment templates are included as `.env.example` (backend) and `frontend/.env.example` (frontend).

---

## 🔌 API Reference

### `GET /`
Health check — returns model status and best validation accuracy.

```json
{
  "status": "TamilVision API Online",
  "accuracy": "97.43%",
  "device": "cuda",
  "classes": 156
}
```

### `POST /predict`
Accepts a Tamil character image as `multipart/form-data`.

| Field | Type | Description |
|---|---|---|
| `file` | `UploadFile` | PNG, JPG, or BMP image of a Tamil character |

**Response**
```json
{
  "predictions": [
    { "predicted_character": "க",  "confidence": 0.973214, "label_id": 36 },
    { "predicted_character": "கி", "confidence": 0.018432, "label_id": 59 },
    { "predicted_character": "கீ", "confidence": 0.005102, "label_id": 82 }
  ]
}
```

---

## 📚 Dataset

This project uses the **uTHCD** (University of Tamil Nadu Handwritten Character Dataset).

1. Download from [Mendeley Data](https://data.mendeley.com/datasets/p36fh3jgbm/1)
2. Extract to `data/raw/`
3. Expected layout:

```
data/raw/uTHCD_b(80-20-split)/80-20-split/train-test-classwise/
    train/
        அ/   0001_0.bmp  ...
        ஆ/   ...
    test/
        ...
```

---

## 🏋️ Training Your Own Model

```bash
# Edit the data split path in src/train.py if needed, then:
python src/train.py
```

The best checkpoint is saved automatically to `models/best_model.pth` whenever validation Top-1 accuracy improves.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ for Tamil Language Preservation</p>
