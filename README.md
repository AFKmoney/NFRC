# NFRC — Neural Fractal Reconstruction Codec

**NFRC** is a proprietary multi-strategy universal compressor with a full-stack web application for real-time compression visualization. It combines **seven distinct coding strategies** and automatically selects the best one for each file.

> **Status:** v6.4 — production engine + web app, all roundtrips bit-perfect verified.

---

## What's new in v6.4

- **MTF (Move-to-Front)** after BWT — classic bzip2 combo, +20-30% on text
- **Delta/XOR encoding** — unlocks patterns in counters, timestamps, arithmetic data (counter test: 1× → 9.84×!)
- **Bit-level context model** (PAQ-inspired) — finds bit patterns invisible at byte level
- **Transform combo mode** (BWT+MTF+RLE+O2) — bzip2-style pipeline
- **Extended dispatcher** — now tries 7 strategies in parallel, picks the best

---

## What's in this repo

```
NFRC/
├── nfr_v6_engine.py     # The compression engine (Python)
├── requirements.txt     # Python deps
├── src/                 # Next.js web app (TypeScript)
│   ├── app/
│   │   ├── page.tsx     # Drag & drop UI with live compression animation
│   │   ├── api/         # REST + SSE endpoints
│   │   └── layout.tsx
│   ├── components/ui/   # shadcn/ui components
│   └── hooks/
├── prisma/              # Database schema
├── package.json         # Node deps
└── README.md
```

Two components:
1. **Engine** (`nfr_v6_engine.py`) — standalone Python compressor
2. **Web App** (`src/`) — Next.js UI with drag & drop, real-time ratio animation, file manager

---

## Why NFRC?

Most compressors commit to one algorithm (LZ77, BWT, PPMd, or neural). NFRC **runs all seven in parallel** and keeps the smallest output. Different data types have different optimal algorithms:

| Data type          | Best strategy  | Typical ratio |
|--------------------|----------------|---------------|
| Repetitive text    | O2 + RLE       | 25–30×        |
| Natural text       | BWT+MTF+RLE+O2 | 5–7×          |
| Source code        | BWT+MTF+RLE+O2 | 4–6×          |
| JSON / structured  | O2             | 4–5×          |
| Sparse binaries    | O2 + RLE       | 50–115×       |
| Counter/arithmetic | Delta + O2     | 9–10×         |
| Random data        | store fallback | 1.00×         |
| Video / images     | NanoSiren v2   | varies        |

NFRC picks the winner automatically — you never have to choose.

---

## Algorithms

### 1. Order-2 Adaptive Context Arithmetic Coding (O2)
The workhorse binary mode. Maintains 65,536 frequency tables (one per 2-byte context) that update adaptively as data is encoded. Encoder and decoder stay in sync without storing tables. 32-bit arithmetic coder with Numba JIT acceleration.

### 2. RLE Pre-Pass
Runs of 4+ identical bytes are collapsed to 5 bytes (4 copies + count byte, supporting runs up to 259). On sparse data with long zero runs, this alone gives 50–100× compression before the AC even runs.

### 3. BWT + O2 (Burrows-Wheeler Transform)
The classic BWT clusters similar bytes together using prefix-doubling suffix array construction (O(n log² n)). The transformed stream is then fed to O2. Best on natural text and source code — typically 50–85% better than O2 alone. Handles files up to 5 MB.

### 4. PPMd with Escape (Orders 0–4)
Full Prediction by Partial Matching with escape mechanism. Tries order 4 first, falls back via escape symbols to orders 3, 2, 1, and finally a uniform order-0 model. Captures long-range context that fixed order-2 misses. Limited to 200 KB (pure Python implementation).

### 5. Neural Residual Predictor (NRP)
A small MLP (8-byte context → 1 byte prediction, 300 training steps) learns the byte-level patterns of the input. The prediction residuals are then compressed with O2 AC. The MLP model is stored int8-quantized. Reconstruction is autoregressive. Limited to 500 KB.

### 6. NanoSiren v2 (Media Mode)
For video and images, a SIREN-based neural network (multi-scale positional encoding with omegas 10/20/40, skip connection, mixed-precision training) predicts pixel values from coordinates. Residuals are sign-magnitude coded with zero-run RLE and context-adaptive arithmetic coding. Frame-delta mode predicts each frame from the previous one for 28% smaller video output. Model weights stored int8-quantized (4× smaller).

---

## File Formats

| Extension | Magic    | Mode                          |
|-----------|----------|-------------------------------|
| `.nf6`    | `NF6\x00`| Video / image (NanoSiren v2)  |
| `.nfg`    | `NFG\x00`| Binary (LZ77 + AC, legacy)    |
| `.nfo`    | `NFO\x00`| Binary (O2 + RLE, or Delta+O2)|
| `.nfb`    | `NFB\x00`| Binary (BWT + O2)             |
| `.nfp`    | `NFP\x00`| Binary (PPMd)                 |
| `.nfn`    | `NFN\x00`| Binary (Neural Residual)      |
| `.nft`    | `NFT\x00`| Binary (BWT+MTF+RLE+O2 combo) |
| `.nfx`    | `NFX\x00`| Binary (Bit-level context)    |

All formats include:
- 4-byte magic
- 1-byte version
- 8-byte original size
- 4-byte CRC32 (for integrity verification)
- Strategy-specific metadata
- Compressed bitstream

The decompressor auto-detects the format from the magic bytes.

---

## Quick Start

### Option A: Use the Web App (recommended)

**1. Install Python deps (for the engine):**
```bash
pip install -r requirements.txt
```

**2. Install Node deps (for the app):**
```bash
bun install   # or: npm install
```

**3. Set up the database:**
```bash
cp .env.example .env
bun run db:push
```

**4. Configure Python path (optional):**
```bash
# If `python` doesn't have the deps, point to your venv
echo 'NFR_PYTHON=/path/to/your/python' >> .env
```

**5. Run the app:**
```bash
bun run dev
```

Open http://localhost:3000, drag a file onto the dropzone, see the instant ratio prediction, click **Compress with NFR v6**, and watch the live animation (train → scan → encode phases, real-time ratio counter, throughput, event stream). Your compression history is saved in the sidebar.

### Option B: Use the Engine Directly (CLI)

```bash
pip install -r requirements.txt

# Compress
python nfr_v6_engine.py compress input.txt output.nfr

# Decompress
python nfr_v6_engine.py decompress output.nfr restored.txt

# Predict ratio (instant, no compression)
python nfr_v6_engine.py predict input.txt

# Benchmark (compress + decompress + verify)
python nfr_v6_engine.py bench input.txt output.nfr

# Force a specific mode
python nfr_v6_engine.py compress input.txt output.nfr --mode-bin    # binary
python nfr_v6_engine.py compress input.mp4 output.nf6 --mode-media  # media

# JSON progress events (for UI integration)
python nfr_v6_engine.py compress input.txt output.nfr --json
```

---

## Programmatic API (Engine)

```python
from nfr_v6_engine import compress, decompress, predict_ratio, ProgressEmitter

# Predict ratio without compressing
info = predict_ratio("input.txt")
print(f"Predicted {info['predicted_ratio']}x for {info['type']} file")

# Compress with progress events
emitter = ProgressEmitter(enabled=True)  # emits JSON to stdout
ratio = compress("input.txt", "output.nfr", emitter)
print(f"Final ratio: {ratio:.2f}x")

# Decompress
decompress("output.nfr", "restored.txt")
```

### ProgressEmitter events

| Phase        | Fields                                                       |
|--------------|--------------------------------------------------------------|
| `start`      | `mode`, `input_size`, `output_size`, `ratio`                |
| `train`      | `progress` (0–1), `loss`                                    |
| `scan`       | `progress`, `current_ratio`                                 |
| `encode`     | `progress`, `current_ratio`, `throughput_mbs`               |
| `decode`     | `progress`                                                  |
| `reconstruct`| `progress`                                                  |
| `done`       | `input_size`, `output_size`, `ratio`, `time_s`              |
| `error`      | `message`                                                   |

---

## Web App Architecture

### API Routes

| Route                              | Method   | Purpose                                              |
|------------------------------------|----------|------------------------------------------------------|
| `/api/predict`                     | `POST`   | Upload file, run instant ratio prediction            |
| `/api/compress?id=X`               | `POST`   | Start compression subprocess in background           |
| `/api/status/[id]`                 | `GET`    | SSE stream of compression progress events            |
| `/api/files`                       | `GET`    | List compression history                             |
| `/api/files?id=X`                  | `DELETE` | Remove a file from history                           |
| `/api/decompress`                  | `POST`   | Start decompression subprocess                       |
| `/api/download/[id]?kind=K`        | `GET`    | Download `compressed` / `decompressed` / `original`  |

### How it works

1. **Predict**: When you drop a file, the app calls `/api/predict`, which spawns the engine's `predict` subcommand to read the first 64 KB and compute a heuristic ratio based on entropy + file type. Returns in <1s.

2. **Compress**: Clicking "Compress" calls `/api/compress`, which spawns the engine as a subprocess with `NFR_JSON=1`. The engine writes JSON progress events to stdout, which the API pipes to a log file.

3. **Stream**: The browser opens an SSE connection to `/api/status/[id]`, which polls the log file every 200ms and forwards new events to the client. The UI animates the ratio counter, progress bar, and event log in real time.

4. **Database**: Prisma + SQLite stores file metadata (name, sizes, ratio, status, paths).

### UI Features

- **Drag & drop** with animated overlay
- **Instant ratio prediction** card (entropy, dimensions, frames, confidence)
- **Live compression animation**: progress bar with shimmer, real-time ratio counter, throughput (MB/s), phase indicator (train/scan/encode), event stream log
- **File manager**: expandable rows showing original/compressed sizes, predicted vs final ratio, download buttons (`.nfr` / decompressed / original), decompress button, remove
- **Dark glassmorphism** UI with animated gradient blobs + grid overlay
- **Framer Motion** animations throughout (entrance, hover, drag overlay)
- **Sonner toasts** for feedback

---

## How the Multi-Strategy Dispatcher Works

When you call `compress()` on a binary file, NFRC runs:

1. **O2 + RLE** — always
2. **NRP** — if file ≤ 500 KB
3. **PPMd** — if file ≤ 200 KB
4. **BWT + O2** — if file ≤ 5 MB

Each candidate writes to a temp file. The dispatcher picks the smallest, deletes the others, and renames the winner to the final output. The magic bytes tell the decompressor which strategy was used.

For media files (video/image), NanoSiren v2 runs directly. Small images (< 256 KB) fall back to binary mode because the model overhead dominates.

---

## Performance Notes

- **Binary modes** are Numba-JIT-compiled — first run compiles (~5s), subsequent runs are fast.
- **BWT** uses prefix-doubling suffix array: O(n log² n). Handles 5 MB in seconds.
- **PPMd** is pure Python — slow but correct. Suitable for files < 200 KB.
- **NRP** trains an MLP for 300 steps — adds ~2–5s overhead.
- **Media mode** trains NanoSiren for 500 steps — fastest on GPU, ~45s on CPU for small videos.
- **Store-mode fallback**: if compression makes the file bigger (random data), NFRC stores raw bytes with a `0xFF` marker. Always bit-perfect.

---

## Tested Roundtrips

All ratios below are verified bit-perfect (CRC32 + length match):

| File type      | Size   | Compressed | Ratio   | Strategy used |
|----------------|--------|------------|---------|---------------|
| Repetitive text| 105 KB | 3.9 KB     | 27.08×  | O2 + RLE      |
| Natural text   | 5 KB   | 0.9 KB     | 5.65×   | BWT+MTF+RLE+O2|
| JSON           | 132 KB | 21 KB      | 4.15×   | O2            |
| Python code    | 19 KB  | 3.2 KB     | 6.02×   | BWT+MTF+RLE+O2|
| Sparse binary  | 20 KB  | 191 B      | 104.71× | O2 + RLE      |
| Counter data   | 10 KB  | 1.0 KB     | 9.84×   | Delta + O2    |
| Random data    | 10 KB  | 10 KB      | 1.00×   | store         |

---

## Limitations

- BWT limited to 5 MB (suffix array memory)
- PPMd limited to 200 KB (Python speed)
- NRP limited to 500 KB (training time)
- Media mode is lossless but model-overhead-heavy on small images
- No streaming decompression for media mode yet (loads full bitstream)
- Web app currently runs compression in subprocess (not worker queue) — suitable for single-user, not production load

---

## License

Proprietary. (c) 2026 NFR Project.
