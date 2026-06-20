# NFRC — Neural Fractal Reconstruction Codec

**NFRC** is a proprietary multi-strategy universal compressor that combines four distinct coding strategies and automatically selects the best one for each file. It achieves high compression ratios on text, code, structured data, sparse binaries, and media through a unified pipeline.

> **Status:** v6.3 — production engine, all roundtrips bit-perfect verified.

---

## Why NFRC?

Most compressors commit to one algorithm (LZ77, BWT, PPMd, or neural). NFRC **runs all four in parallel** and keeps the smallest output. Different data types have different optimal algorithms:

| Data type          | Best strategy  | Typical ratio |
|--------------------|----------------|---------------|
| Repetitive text    | O2 + RLE       | 25–30×        |
| Natural text       | BWT + O2       | 4–7×          |
| Source code        | BWT + O2       | 4–6×          |
| JSON / structured  | O2             | 4–5×          |
| Sparse binaries    | O2 + RLE       | 50–115×       |
| Random data        | store fallback | 1.00×         |
| Video / images     | NanoSiren v2   | varies        |

NFRC picks the winner automatically — you never have to choose.

---

## Algorithms

### 1. Order-2 Adaptive Context Arithmetic Coding (O2)
The workhorse binary mode. Maintains 65,536 frequency tables (one per 2-byte context) that update adaptively as data is encoded. Encoder and decoder stay in sync without storing tables. Uses a 32-bit arithmetic coder with Numba JIT acceleration.

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
| `.nfo`    | `NFO\x00`| Binary (O2 + RLE)             |
| `.nfb`    | `NFB\x00`| Binary (BWT + O2)             |
| `.nfp`    | `NFP\x00`| Binary (PPMd)                 |
| `.nfn`    | `NFN\x00`| Binary (Neural Residual)      |

All formats include:
- 4-byte magic
- 1-byte version
- 8-byte original size
- 4-byte CRC32 (for integrity verification)
- Strategy-specific metadata
- Compressed bitstream

The decompressor auto-detects the format from the magic bytes.

---

## Installation

```bash
pip install torch numba numpy opencv-python-headless
```

> For GPU acceleration (media mode only), install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org).

---

## Usage

### Compress
```bash
python nfr_v6_engine.py compress input.txt output.nfr
```

### Decompress
```bash
python nfr_v6_engine.py decompress output.nfr restored.txt
```

### Predict ratio (instant, no compression)
```bash
python nfr_v6_engine.py predict input.txt
# → {"type": "binary", "original_size": 12345, "predicted_ratio": 5.0, ...}
```

### Benchmark (compress + decompress + verify)
```bash
python nfr_v6_engine.py bench input.txt output.nfr
```

### Force a specific mode
```bash
python nfr_v6_engine.py compress input.txt output.nfr --mode-bin    # force binary
python nfr_v6_engine.py compress input.mp4 output.nf6 --mode-media  # force media
```

### JSON progress events (for UI integration)
```bash
python nfr_v6_engine.py compress input.txt output.nfr --json
# → {"phase":"start",...}
#   {"phase":"scan","progress":0.5,...}
#   {"phase":"encode","progress":1.0,"current_ratio":4.32,...}
#   {"phase":"done","ratio":4.32,"time_s":1.2}
```

Set `NFR_JSON=1` env var to enable JSON mode without the `--json` flag.

---

## Programmatic API

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

| File type      | Size   | Compressed | Ratio   |
|----------------|--------|------------|---------|
| Repetitive text| 105 KB | 3.9 KB     | 27.08×  |
| Natural text   | 26 KB  | 3.5 KB     | 7.41×   |
| JSON           | 132 KB | 21 KB      | 4.15×   |
| Python code    | 19 KB  | 3.2 KB     | 6.02×   |
| Sparse binary  | 50 KB  | 432 B      | 115.74× |
| Random data    | 30 KB  | 30 KB      | 1.00×   |

---

## Limitations

- BWT limited to 5 MB (suffix array memory)
- PPMd limited to 200 KB (Python speed)
- NRP limited to 500 KB (training time)
- Media mode is lossless but model-overhead-heavy on small images
- No streaming decompression for media mode yet (loads full bitstream)

---

## License

Proprietary. (c) 2026 NFR Project.
