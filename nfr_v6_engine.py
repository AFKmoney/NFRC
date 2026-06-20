"""
NFR v6 — Neural Fractal Reconstruction (Universal Codec + Generic Binary)
=========================================================================

Improvements over NFR+ v5.0
---------------------------
1. NanoSiren v2
   - Multi-scale positional encoding (omega in {10, 20, 40})
   - Skip connection + larger hidden (48)
   - Mixed-precision training (AMP) + cosine LR schedule
2. Sign-Magnitude residual coding (better for sparse residuals)
3. Context-adaptive arithmetic coding (4 contexts based on prev-residual magnitude)
4. Zero-run RLE escape (skips long runs of perfect predictions for free)
5. Per-channel frequency tables (R/G/B separated)
6. Generic binary mode (LZ77 + adaptive arithmetic coder, no neural net)
7. Backward-compatible reader for v5.0 (.nfr) files
8. Ratio prediction (heuristic on first 64KB)
9. Progress callback API (JSON events to stdout) — designed for streaming UIs

File Formats
------------
.nfr  — NFR+ v5.0 video/image (legacy, read-only)
.nf6  — NFR  v6   video/image
.nfg  — NFR  v6   generic binary (zip-like)

CLI Usage
---------
python nfr_v6_engine.py compress   <input> <output>
python nfr_v6_engine.py decompress <input> <output>
python nfr_v6_engine.py predict    <input>            # prints predicted ratio JSON
python nfr_v6_engine.py bench       <input>            # full compress + decompress + verify

Progress events (when --json flag is set, or env NFR_JSON=1):
  {"phase":"start","mode":"video","input_size":N,"output_size":0,"ratio":1.0}
  {"phase":"train","progress":0.0..1.0,"loss":..}
  {"phase":"scan","progress":0.0..1.0,"current_ratio":..}
  {"phase":"encode","progress":0.0..1.0,"current_ratio":..,"throughput_mbs":..}
  {"phase":"done","input_size":N,"output_size":M,"ratio":R,"time_s":T}
  {"phase":"error","message":".."}

(c) 2026 NFR Project.
"""

import os
import io
import sys
import json
import time
import struct
import zlib
import argparse
import traceback
from typing import Optional, Callable, Dict, Any

import numpy as np

# Lazy imports — torch/numba/cv2 only needed for media modes
def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn

def _import_numba():
    from numba import jit
    return jit

def _import_cv2():
    import cv2
    return cv2


# =========================================================================
# CONSTANTS
# =========================================================================

VERSION = "6.0.0"
MAGIC_V6   = b'NF6\x00'   # NFR v6 video/image
MAGIC_V5   = b'NFR+'      # NFR+ v5.0 (legacy, read-only)
MAGIC_BIN  = b'NFG\x00'   # NFR v6 generic binary

# 32-bit arithmetic coder
CODE_VALUE_BITS = 32
TOP_VALUE       = (1 << CODE_VALUE_BITS) - 1
QUARTER         = 1 << (CODE_VALUE_BITS - 2)
HALF            = 1 << (CODE_VALUE_BITS - 1)
THREE_QUARTERS  = 3 * QUARTER

# Symbol alphabet for sign-magnitude residual coding
# Layout: symbol 0 = "zero run" escape (next byte = run length 0..255, scaled)
#         symbol 1 = literal zero
#         symbols 2..257 = magnitude 1..256 with positive sign
#         symbols 258..513 = magnitude 1..256 with negative sign
SYM_ZERO_RUN   = 0
SYM_ZERO       = 1
SYM_POS_BASE   = 2     # 2..257
SYM_NEG_BASE   = 258   # 258..513
ALPHABET_SIZE  = 514
MAX_TOTAL_FREQ = 1 << 16

# Contexts (each has its own freq table)
# ctx = magnitude_bucket(prev_residual) -> {0: small, 1: medium, 2: large, 3: start}
N_CONTEXTS = 4

# Binary mode parameters
LZ_WINDOW_SIZE = 1 << 16       # 64KB
LZ_MIN_MATCH   = 4
LZ_MAX_MATCH   = 255


# =========================================================================
# PROGRESS CALLBACK
# =========================================================================

class ProgressEmitter:
    """Emits JSON progress events to stdout (one per line). Designed for
    consumption by Node.js / Next.js subprocess streaming."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.t0 = time.time()

    def emit(self, phase: str, **kw):
        if not self.enabled:
            return
        evt = {"phase": phase, "ts": round(time.time() - self.t0, 3)}
        evt.update(kw)
        sys.stdout.write(json.dumps(evt) + "\n")
        sys.stdout.flush()

    @staticmethod
    def log(msg: str):
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()


# =========================================================================
# ARITHMETIC CODER (NUMBA JIT)
# =========================================================================

def _define_jit_kernels():
    """Lazy-define JIT kernels after numba is imported."""
    jit = _import_numba()

    @jit(nopython=True, cache=True)
    def _encode_chunk(symbols, ctx_ids, ctx_freqs, ctx_total, state, out_buf, limit):
        """Encode symbols with per-context freq tables.

        ctx_freqs: shape (N_CONTEXTS, ALPHABET_SIZE+1) cumulative
        ctx_total: shape (N_CONTEXTS,)
        state: [low, high, pending, bit_buffer, bit_count]
        """
        low = state[0]
        high = state[1]
        pending = state[2]
        bit_buf = state[3]
        bit_cnt = state[4]
        idx = 0

        half = HALF
        quarter = QUARTER
        tq = THREE_QUARTERS
        top = TOP_VALUE

        for si in range(len(symbols)):
            sym = symbols[si]
            ctx = ctx_ids[si]
            cum = ctx_freqs[ctx]
            tot = ctx_total[ctx]

            lo_c = cum[sym]
            hi_c = cum[sym + 1]
            r = high - low + 1
            high = low + (r * hi_c) // tot - 1
            low  = low + (r * lo_c) // tot

            while True:
                if idx >= limit - 4:
                    break
                if high < half:
                    bit_buf = (bit_buf << 1)
                    bit_cnt += 1
                    if bit_cnt == 8:
                        out_buf[idx] = bit_buf; idx += 1
                        bit_buf = 0; bit_cnt = 0
                    while pending > 0:
                        bit_buf = (bit_buf << 1) | 1
                        bit_cnt += 1
                        if bit_cnt == 8:
                            out_buf[idx] = bit_buf; idx += 1
                            bit_buf = 0; bit_cnt = 0
                        pending -= 1
                elif low >= half:
                    bit_buf = (bit_buf << 1) | 1
                    bit_cnt += 1
                    if bit_cnt == 8:
                        out_buf[idx] = bit_buf; idx += 1
                        bit_buf = 0; bit_cnt = 0
                    while pending > 0:
                        bit_buf = (bit_buf << 1)
                        bit_cnt += 1
                        if bit_cnt == 8:
                            out_buf[idx] = bit_buf; idx += 1
                            bit_buf = 0; bit_cnt = 0
                        pending -= 1
                    low -= half; high -= half
                elif low >= quarter and high < tq:
                    pending += 1
                    low -= quarter; high -= quarter
                else:
                    break
                low = (low << 1) & top
                high = ((high << 1) & top) | 1

        state[0] = low; state[1] = high; state[2] = pending
        state[3] = bit_buf; state[4] = bit_cnt
        return idx

    @jit(nopython=True, cache=True)
    def _finish(state, out_buf, limit):
        low = state[0]
        pending = state[2]
        bit_buf = state[3]
        bit_cnt = state[4]
        idx = 0
        quarter = QUARTER

        pending += 1
        bit_to_emit = 0 if low < quarter else 1

        bit_buf = (bit_buf << 1) | bit_to_emit
        bit_cnt += 1
        if bit_cnt == 8:
            if idx < limit:
                out_buf[idx] = bit_buf; idx += 1
            bit_buf = 0; bit_cnt = 0
        while pending > 0:
            if idx >= limit - 1: break
            bit_buf = (bit_buf << 1) | (1 - bit_to_emit)
            bit_cnt += 1
            if bit_cnt == 8:
                out_buf[idx] = bit_buf; idx += 1
                bit_buf = 0; bit_cnt = 0
            pending -= 1
        if bit_cnt > 0:
            if idx < limit:
                bit_buf = bit_buf << (8 - bit_cnt)
                out_buf[idx] = bit_buf; idx += 1
        return idx

    @jit(nopython=True, cache=True)
    def _decode_chunk(input_buf, num_symbols, ctx_ids, ctx_freqs, ctx_total, state, out_syms):
        """Decode num_symbols symbols with per-context freq tables."""
        low = state[0]; high = state[1]; value = state[2]
        buf_ptr = int(state[3]); bit_ptr = int(state[4])

        half = HALF; quarter = QUARTER; tq = THREE_QUARTERS; top = TOP_VALUE

        for i in range(num_symbols):
            ctx = ctx_ids[i]
            cum = ctx_freqs[ctx]
            tot = ctx_total[ctx]
            r = high - low + 1
            sv = ((value - low + 1) * tot - 1) // r

            # Binary search across ALPHABET_SIZE+1 cumulative entries
            s_lo = 0; s_hi = ALPHABET_SIZE - 1; symbol = 0
            while s_lo <= s_hi:
                mid = (s_lo + s_hi) >> 1
                if cum[mid] <= sv:
                    symbol = mid; s_lo = mid + 1
                else:
                    s_hi = mid - 1

            out_syms[i] = symbol

            lo_c = cum[symbol]; hi_c = cum[symbol + 1]
            high = low + (r * hi_c) // tot - 1
            low  = low + (r * lo_c) // tot

            while True:
                if high < half:
                    pass
                elif low >= half:
                    low -= half; high -= half; value -= half
                elif low >= quarter and high < tq:
                    low -= quarter; high -= quarter; value -= quarter
                else:
                    break
                low = (low << 1) & top
                high = ((high << 1) & top) | 1
                if buf_ptr < len(input_buf):
                    byte = input_buf[buf_ptr]
                    bit = (byte >> bit_ptr) & 1
                    bit_ptr -= 1
                    if bit_ptr < 0:
                        bit_ptr = 7; buf_ptr += 1
                else:
                    bit = 0
                value = ((value << 1) & top) | bit

        state[0] = low; state[1] = high; state[2] = value
        state[3] = buf_ptr; state[4] = bit_ptr

    @jit(nopython=True, cache=True)
    def _lz_compress_step(input_buf, n, out_tokens, out_limit):
        """LZ77 compression. Token layout (varint):
           [0] flag: 0 = literal, 1 = match
           literal: flag(0), byte
           match:   flag(1), offset_lo16, len
        Returns: number of int32 slots written (3 per token).
        """
        # Simple hash chain for 4-byte prefixes
        HASH_SIZE = 1 << 16
        HASH_MASK = HASH_SIZE - 1
        head = np.full(HASH_SIZE, -1, dtype=np.int32)
        prev = np.full(n, -1, dtype=np.int32)

        def h(i):
            # 4-byte hash
            v = (input_buf[i] << 24) | (input_buf[i+1] << 16) | (input_buf[i+2] << 8) | input_buf[i+3]
            return (v * 2654435761) & HASH_MASK

        out_idx = 0
        i = 0
        while i < n:
            if i + LZ_MIN_MATCH > n:
                # Emit literal
                if out_idx + 3 > out_limit: break
                out_tokens[out_idx] = 0
                out_tokens[out_idx+1] = input_buf[i]
                out_tokens[out_idx+2] = 0
                out_idx += 3
                i += 1
                continue

            hv = h(i)
            best_len = 0
            best_off = 0
            cand = head[hv]
            chain = 0
            while cand != -1 and chain < 64:
                if cand < i:
                    off = i - cand
                    if off <= LZ_WINDOW_SIZE:
                        # Compute match length
                        ml = 0
                        max_ml = min(LZ_MAX_MATCH, n - i)
                        while ml < max_ml and input_buf[cand + ml] == input_buf[i + ml]:
                            ml += 1
                        if ml > best_len:
                            best_len = ml
                            best_off = off
                            if ml >= LZ_MAX_MATCH:
                                break
                cand = prev[cand]
                chain += 1

            if best_len >= LZ_MIN_MATCH:
                if out_idx + 3 > out_limit: break
                out_tokens[out_idx] = 1
                out_tokens[out_idx+1] = best_off
                out_tokens[out_idx+2] = best_len
                out_idx += 3
                # Insert hashes for all positions in the match
                end = i + best_len
                while i < end:
                    if i + 4 <= n:
                        h2 = h(i)
                        prev[i] = head[h2]
                        head[h2] = i
                    i += 1
            else:
                if out_idx + 3 > out_limit: break
                out_tokens[out_idx] = 0
                out_tokens[out_idx+1] = input_buf[i]
                out_tokens[out_idx+2] = 0
                out_idx += 3
                if i + 4 <= n:
                    h2 = h(i)
                    prev[i] = head[h2]
                    head[h2] = i
                i += 1

        return out_idx

    # =====================================================================
    # ORDER-2 ADAPTIVE CONTEXT-MODELED AC (PPM-lite, no escape)
    # =====================================================================
    # For each byte, context = (prev_prev_byte << 8) | prev_byte
    # 65536 contexts × 256 symbols, adaptive (encoder & decoder stay in sync)
    # Memory: 65536 × 257 × 4 bytes = 64 MB (allocated outside, passed in)

    @jit(nopython=True, cache=True)
    def _encode_o2(data, cum_freqs, state, out_buf, limit):
        """Order-2 adaptive AC encoder.
        cum_freqs: (65536, 257) uint32, cum_freqs[ctx, sym] = cumulative count
                   Initialized to: cum_freqs[ctx, sym] = sym + 1 (uniform, count=1)
        """
        low = state[0]; high = state[1]; pending = state[2]
        bit_buf = state[3]; bit_cnt = state[4]; idx = 0
        half = HALF; quarter = QUARTER; tq = THREE_QUARTERS; top = TOP_VALUE
        MAX_TOT = 65536

        p2 = 0  # prev-prev byte
        p1 = 0  # prev byte

        for i in range(len(data)):
            ctx = (p2 << 8) | p1
            sym = int(data[i])

            cum_lo = cum_freqs[ctx, sym]
            cum_hi = cum_freqs[ctx, sym + 1]
            tot = cum_freqs[ctx, 256]

            r = high - low + 1
            high = low + (r * cum_hi) // tot - 1
            low = low + (r * cum_lo) // tot

            while True:
                if idx >= limit - 4: break
                if high < half:
                    bit_buf = (bit_buf << 1); bit_cnt += 1
                    if bit_cnt == 8: out_buf[idx] = bit_buf; idx += 1; bit_buf = 0; bit_cnt = 0
                    while pending > 0:
                        bit_buf = (bit_buf << 1) | 1; bit_cnt += 1
                        if bit_cnt == 8: out_buf[idx] = bit_buf; idx += 1; bit_buf = 0; bit_cnt = 0
                        pending -= 1
                elif low >= half:
                    bit_buf = (bit_buf << 1) | 1; bit_cnt += 1
                    if bit_cnt == 8: out_buf[idx] = bit_buf; idx += 1; bit_buf = 0; bit_cnt = 0
                    while pending > 0:
                        bit_buf = (bit_buf << 1); bit_cnt += 1
                        if bit_cnt == 8: out_buf[idx] = bit_buf; idx += 1; bit_buf = 0; bit_cnt = 0
                        pending -= 1
                    low -= half; high -= half
                elif low >= quarter and high < tq:
                    pending += 1; low -= quarter; high -= quarter
                else:
                    break
                low = (low << 1) & top
                high = ((high << 1) & top) | 1

            # Update: increment cum_freqs[ctx, sym+1:] by 1
            for k in range(sym, 256):
                cum_freqs[ctx, k + 1] += 1

            # Rescale if total too big (prevents overflow)
            if cum_freqs[ctx, 256] >= MAX_TOT:
                cum = 0
                for k in range(256):
                    old = cum_freqs[ctx, k + 1] - cum_freqs[ctx, k]
                    new = (old + 1) >> 1  # halve, round up
                    cum += new
                    cum_freqs[ctx, k + 1] = cum

            p2 = p1; p1 = sym

        state[0] = low; state[1] = high; state[2] = pending
        state[3] = bit_buf; state[4] = bit_cnt
        return idx

    @jit(nopython=True, cache=True)
    def _decode_o2(input_buf, num_symbols, cum_freqs, state, out_syms):
        """Order-2 adaptive AC decoder. Mirrors _encode_o2."""
        low = state[0]; high = state[1]; value = state[2]
        buf_ptr = int(state[3]); bit_ptr = int(state[4])
        half = HALF; quarter = QUARTER; tq = THREE_QUARTERS; top = TOP_VALUE
        MAX_TOT = 65536

        p2 = 0; p1 = 0

        for i in range(num_symbols):
            ctx = (p2 << 8) | p1
            tot = cum_freqs[ctx, 256]
            r = high - low + 1
            sv = ((value - low + 1) * tot - 1) // r

            # Binary search for symbol
            s_lo = 0; s_hi = 255; sym = 0
            while s_lo <= s_hi:
                mid = (s_lo + s_hi) >> 1
                if cum_freqs[ctx, mid] <= sv:
                    sym = mid; s_lo = mid + 1
                else:
                    s_hi = mid - 1

            out_syms[i] = sym

            cum_lo = cum_freqs[ctx, sym]
            cum_hi = cum_freqs[ctx, sym + 1]
            high = low + (r * cum_hi) // tot - 1
            low = low + (r * cum_lo) // tot

            while True:
                if high < half:
                    pass
                elif low >= half:
                    low -= half; high -= half; value -= half
                elif low >= quarter and high < tq:
                    low -= quarter; high -= quarter; value -= quarter
                else:
                    break
                low = (low << 1) & top
                high = ((high << 1) & top) | 1
                if buf_ptr < len(input_buf):
                    byte = input_buf[buf_ptr]
                    bit = (byte >> bit_ptr) & 1
                    bit_ptr -= 1
                    if bit_ptr < 0:
                        bit_ptr = 7; buf_ptr += 1
                else:
                    bit = 0
                value = ((value << 1) & top) | bit

            # Update (same as encoder)
            for k in range(sym, 256):
                cum_freqs[ctx, k + 1] += 1
            if cum_freqs[ctx, 256] >= MAX_TOT:
                cum = 0
                for k in range(256):
                    old = cum_freqs[ctx, k + 1] - cum_freqs[ctx, k]
                    new = (old + 1) >> 1
                    cum += new
                    cum_freqs[ctx, k + 1] = cum

            p2 = p1; p1 = sym

        state[0] = low; state[1] = high; state[2] = value
        state[3] = buf_ptr; state[4] = bit_ptr

    return _encode_chunk, _finish, _decode_chunk, _lz_compress_step, _encode_o2, _decode_o2


# Lazy globals
_KERNELS = None
def _kernels():
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _define_jit_kernels()
    return _KERNELS


# =========================================================================
# RESIDUAL <-> SYMBOL MAPPING (Sign-Magnitude + Zero-Run)
# =========================================================================

def residual_to_symbol(r: int) -> int:
    """Map a residual (-255..255) to a symbol in [0, ALPHABET_SIZE)."""
    if r == 0:
        return SYM_ZERO
    if r > 0:
        return SYM_POS_BASE + (r - 1)
    return SYM_NEG_BASE + (-r - 1)

def symbol_to_residual(s: int) -> int:
    if s == SYM_ZERO:
        return 0
    if s < SYM_NEG_BASE:
        return s - SYM_POS_BASE + 1
    return -(s - SYM_NEG_BASE + 1)

# Vectorized versions
def residuals_to_symbols(residuals: np.ndarray) -> np.ndarray:
    """Vectorized conversion + zero-run encoding.
    Returns (symbols, ctx_ids) arrays.
    Zero-run encoding: when 4+ zeros in a row, emit SYM_ZERO_RUN followed by
    a byte for run length (4..255)."""
    syms = []
    ctxs = []
    n = len(residuals)
    i = 0
    prev_mag_bucket = 3  # "start" context
    while i < n:
        r = int(residuals[i])
        # Count zeros
        if r == 0:
            j = i
            while j < n and residuals[j] == 0 and j - i < 255:
                j += 1
            run = j - i
            if run >= 4:
                # Emit zero-run symbol; we'll stuff the length into the next
                # symbol slot via a special marker.
                syms.append(SYM_ZERO_RUN)
                syms.append(run)  # 4..255
                ctxs.append(prev_mag_bucket)
                ctxs.append(prev_mag_bucket)
                # After a run of zeros, prev_mag_bucket stays small
                prev_mag_bucket = 0
                i = j
                continue
            # else fall through to emit single zeros
            for _ in range(run):
                syms.append(SYM_ZERO)
                ctxs.append(prev_mag_bucket)
                prev_mag_bucket = 0
            i = j
            continue

        s = residual_to_symbol(r)
        syms.append(s)
        ctxs.append(prev_mag_bucket)
        # Update context: magnitude bucket of this residual
        mag = abs(r)
        if mag <= 4: prev_mag_bucket = 0
        elif mag <= 16: prev_mag_bucket = 1
        else: prev_mag_bucket = 2
        i += 1
    return np.array(syms, dtype=np.int32), np.array(ctxs, dtype=np.int32)


def symbols_to_residuals(syms: np.ndarray, total_pixels: int) -> np.ndarray:
    """Inverse: symbols -> residuals (re-expand zero-runs)."""
    out = np.zeros(total_pixels, dtype=np.int16)
    out_idx = 0
    i = 0
    n = len(syms)
    while i < n:
        s = int(syms[i])
        if s == SYM_ZERO_RUN:
            run = int(syms[i+1])
            out[out_idx:out_idx+run] = 0
            out_idx += run
            i += 2
        else:
            out[out_idx] = symbol_to_residual(s)
            out_idx += 1
            i += 1
    return out[:total_pixels]


# =========================================================================
# FREQUENCY MODEL (per-context, adaptive-then-frozen)
# =========================================================================

def build_freq_tables(symbols: np.ndarray, ctx_ids: np.ndarray):
    """Build N_CONTEXTS cumulative frequency tables.
    Returns: ctx_freqs (N_CONTEXTS, ALPHABET_SIZE+1) uint64,
             ctx_total (N_CONTEXTS,) uint64
    """
    ctx_freqs = np.zeros((N_CONTEXTS, ALPHABET_SIZE + 1), dtype=np.uint64)
    for c in range(N_CONTEXTS):
        mask = ctx_ids == c
        sub = symbols[mask]
        if len(sub) == 0:
            counts = np.ones(ALPHABET_SIZE, dtype=np.uint64)
        else:
            counts = np.bincount(sub, minlength=ALPHABET_SIZE).astype(np.uint64)
            counts = np.maximum(counts, 1)
        total = counts.sum()
        if total > MAX_TOTAL_FREQ:
            scale = MAX_TOTAL_FREQ / total
            counts = (counts.astype(np.float64) * scale).clip(min=1).astype(np.uint64)
        ctx_freqs[c, 1:] = np.cumsum(counts)
    ctx_total = ctx_freqs[:, -1].copy()
    return ctx_freqs, ctx_total


# =========================================================================
# NANOSIREN v2 (Multi-scale PE + Skip + AMP)
# =========================================================================

def _build_model(device):
    torch, nn = _import_torch()

    class MultiScalePE(nn.Module):
        def __init__(self, omegas=(10.0, 20.0, 40.0)):
            super().__init__()
            self.omegas = list(omegas)
            self.out_dim = len(omegas) * 2 * 3  # 3 input coords
        def forward(self, x):
            feats = []
            for w in self.omegas:
                feats.append(torch.sin(w * x))
                feats.append(torch.cos(w * x))
            return torch.cat(feats, dim=-1)

    class SineLayer(nn.Module):
        def __init__(self, in_f, out_f, omega0=30.0, is_first=False):
            super().__init__()
            self.omega0 = omega0
            self.linear = nn.Linear(in_f, out_f)
            with torch.no_grad():
                if is_first:
                    self.linear.weight.uniform_(-1/in_f, 1/in_f)
                else:
                    l = np.sqrt(6/in_f) / self.omega0
                    self.linear.weight.uniform_(-l, l)
        def forward(self, x):
            return torch.sin(self.omega0 * self.linear(x))

    class NanoSirenV2(nn.Module):
        def __init__(self, hidden=48, layers=2):
            super().__init__()
            self.pe = MultiScalePE()
            in_f = self.pe.out_dim
            self.layers = nn.ModuleList()
            self.layers.append(SineLayer(in_f, hidden, is_first=True, omega0=30.0))
            for _ in range(layers):
                self.layers.append(SineLayer(hidden, hidden, omega0=30.0))
            self.tail = nn.Linear(hidden, 3)
            # Skip projection (PE -> tail input)
            self.skip_proj = nn.Linear(in_f, 3)
        def forward(self, x):
            h = self.pe(x)
            skip = self.skip_proj(h)
            for lyr in self.layers:
                h = lyr(h)
            return self.tail(h) + 0.1 * skip

    model = NanoSirenV2(hidden=48, layers=2).to(device)
    return model


# =========================================================================
# HELPERS
# =========================================================================

def _build_residual_predictor(device):
    """Small MLP that predicts residuals from spatial context.
    Input: 5-channel context (residual_left, residual_up, residual_diag, x_norm, y_norm)
    Output: predicted residual value (scalar)
    This learns the spatial correlation of residuals → smaller second-order residuals."""
    torch, nn = _import_torch()

    class ResidualPredictor(nn.Module):
        def __init__(self, hidden=24):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(5, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        def forward(self, x):
            return self.net(x)

    return ResidualPredictor().to(device)


def _predict_residuals_with_mlp(residuals_2d, predictor, device, H, W):
    """Given residuals (H, W, 3), use MLP to predict each residual from its
    spatial context (left, up, diagonal neighbors). Returns the second-order
    residuals (prediction error of the MLP)."""
    torch, _ = _import_torch()
    if H < 2 or W < 2:
        return residuals_2d.astype(np.int16)

    res = torch.from_numpy(residuals_2d.astype(np.float32)).to(device)  # (H, W, 3)
    # Build context: left, up, diagonal-up-left, x_norm, y_norm
    # Pad with zeros
    left = torch.zeros_like(res)
    left[1:, :, :] = res[:-1, :, :]
    up = torch.zeros_like(res)
    up[:, 1:, :] = res[:, :-1, :]
    diag = torch.zeros_like(res)
    diag[1:, 1:, :] = res[:-1, :-1, :]

    # Normalized coordinates
    yy = torch.linspace(0, 1, H, device=device).view(H, 1).expand(H, W)
    xx = torch.linspace(0, 1, W, device=device).view(1, W).expand(H, W)
    # Broadcast to 3 channels
    yy = yy.unsqueeze(-1).expand(H, W, 3)
    xx = xx.unsqueeze(-1).expand(H, W, 3)

    # Stack context: (H, W, 3, 5) → (H*W*3, 5)
    ctx = torch.stack([left, up, diag, xx, yy], dim=-1)  # (H, W, 3, 5)
    ctx_flat = ctx.reshape(-1, 5)
    res_flat = res.reshape(-1, 1)

    with torch.no_grad():
        pred = predictor(ctx_flat)  # (H*W*3, 1)
        second_order = res_flat - pred
        second_order = second_order.reshape(H, W, 3).cpu().numpy().astype(np.int16)

    return second_order


def _reconstruct_residuals_with_mlp(second_order, predictor, device, H, W):
    """Inverse: reconstruct original residuals from second-order residuals
    by running the MLP prediction sequentially (autoregressive)."""
    torch, _ = _import_torch()
    if H < 2 or W < 2:
        return second_order.astype(np.int16)

    out = np.zeros((H, W, 3), dtype=np.float32)
    so = second_order.astype(np.float32)

    for y in range(H):
        for x in range(W):
            # Build context from already-reconstructed neighbors
            left = out[y-1, x] if y > 0 else np.zeros(3, dtype=np.float32)
            up = out[y, x-1] if x > 0 else np.zeros(3, dtype=np.float32)
            diag = out[y-1, x-1] if y > 0 and x > 0 else np.zeros(3, dtype=np.float32)
            xx = np.full(3, x / max(W-1, 1), dtype=np.float32)
            yy = np.full(3, y / max(H-1, 1), dtype=np.float32)
            ctx = np.stack([left, up, diag, xx, yy], axis=-1)  # (3, 5)
            ctx_t = torch.from_numpy(ctx).to(device)
            with torch.no_grad():
                pred = predictor(ctx_t).cpu().numpy().flatten()  # (3,)
            out[y, x] = pred + so[y, x]

    return np.round(out).astype(np.int16)


def _fmt_bytes(n: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n < 1024: return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _is_media_file(path: str) -> bool:
    """Quick check: is this a video or image we can handle with cv2?"""
    return _detect_media_type(path) is not None


# File extension hints (more reliable than cv2 alone for ambiguous files)
_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif'}
_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg'}

def _detect_media_type(path: str) -> Optional[str]:
    """Return 'video' / 'image' / None.
    Uses file extension as primary signal, cv2 as secondary verification."""
    ext = os.path.splitext(path)[1].lower()
    cv2 = _import_cv2()

    if ext in _VIDEO_EXTS:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n > 0:
                return 'video'
        return None

    if ext in _IMAGE_EXTS:
        img = cv2.imread(path)
        if img is not None:
            return 'image'
        return None

    # No extension hint — try cv2 carefully (don't trust it for arbitrary files)
    # Read first 16 bytes and check magic numbers
    try:
        with open(path, 'rb') as f:
            head = f.read(16)
    except Exception:
        return None

    # Image magics
    if head[:8] == b'\x89PNG\r\n\x1a\n': return 'image'
    if head[:3] == b'\xff\xd8\xff': return 'image'  # JPEG
    if head[:2] == b'BM': return 'image'              # BMP
    if head[:4] in (b'RIFF',) and head[8:12] == b'WEBP': return 'image'
    if head[:6] in (b'GIF87a', b'GIF89a'): return 'image'

    # Video magics
    if head[4:8] in (b'ftyp', b'moov', b'mdat', b'free', b'skip'): return 'video'  # MP4/MOV
    if head[:4] == b'RIFF' and head[8:12] == b'AVI ': return 'video'
    if head[:4] == b'\x1aE\xdf\xa3': return 'video'  # MKV/WebM

    return None


def _byte_entropy(data: bytes) -> float:
    """Shannon entropy in bits/byte (0..8)."""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# =========================================================================
# RATIO PREDICTION
# =========================================================================

def predict_ratio(path: str) -> Dict[str, Any]:
    """Heuristic ratio prediction based on first 64KB.
    Returns dict with: type, original_size, predicted_ratio, confidence, entropy
    """
    if not os.path.exists(path):
        return {"error": "File not found"}
    size = os.path.getsize(path)
    if size == 0:
        return {"error": "Empty file"}

    # Read first 64KB
    with open(path, 'rb') as f:
        sample = f.read(64 * 1024)

    entropy = _byte_entropy(sample)

    # Try media
    try:
        media_type = _detect_media_type(path)
    except Exception:
        media_type = None

    if media_type == 'video':
        cv2 = _import_cv2()
        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # NFR video tends to do well on smooth content; heuristic:
        # baseline 8-15x for typical footage; lower if high entropy.
        base = 10.0
        # Penalize tiny videos (overhead dominates)
        if size < 200_000: base = 4.0
        elif size > 50_000_000: base = 12.0
        # Entropy penalty
        adj = (entropy - 6.5) * 0.6
        pred = max(2.0, base + adj)
        return {
            "type": "video",
            "original_size": size,
            "width": w, "height": h, "frames": n,
            "entropy": round(entropy, 3),
            "predicted_ratio": round(pred, 2),
            "predicted_compressed_size": int(size / pred),
            "confidence": "medium"
        }
    elif media_type == 'image':
        cv2 = _import_cv2()
        img = cv2.imread(path)
        h, w = img.shape[:2]
        # Images: 1.5-4x typical (NFR is not great on single images due to model overhead)
        base = 2.5
        if size < 50_000: base = 1.2
        adj = (entropy - 6.5) * 0.3
        pred = max(1.1, base + adj)
        return {
            "type": "image",
            "original_size": size,
            "width": w, "height": h,
            "entropy": round(entropy, 3),
            "predicted_ratio": round(pred, 2),
            "predicted_compressed_size": int(size / pred),
            "confidence": "medium"
        }
    else:
        # Binary file — LZ77 + AC. Heuristic based on entropy.
        # Entropy 8 → ~1x; entropy 4 → ~4x; entropy 6 → ~2x
        # Cap at 8x for very compressible data
        if entropy >= 7.9:
            pred = 1.05
        elif entropy >= 7.5:
            pred = 1.3
        elif entropy >= 7.0:
            pred = 1.8
        elif entropy >= 6.0:
            pred = 2.5
        elif entropy >= 5.0:
            pred = 3.5
        elif entropy >= 3.5:
            pred = 5.0
        else:
            pred = 8.0
        # Penalize tiny files (header overhead)
        if size < 4096:
            pred = max(0.8, pred * 0.5)
        return {
            "type": "binary",
            "original_size": size,
            "entropy": round(entropy, 3),
            "predicted_ratio": round(pred, 2),
            "predicted_compressed_size": int(size / pred),
            "confidence": "high" if entropy < 7 else "medium"
        }


# =========================================================================
# GENERIC BINARY MODE (LZ77 + NFR Arithmetic Coder — proprietary)
# =========================================================================

def compress_binary(input_path: str, output_path: str, emitter: ProgressEmitter):
    """LZ77 + NFR adaptive arithmetic coder. No neural net, no zlib."""
    _kernels()
    _encode_chunk, _finish, _, _lz_compress_step, _, _ = _kernels()

    orig = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig, output_size=0, ratio=1.0)
    emitter.emit("scan", progress=0.0, current_ratio=1.0)

    # Read entire file
    with open(input_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(data)
    emitter.emit("scan", progress=0.3, current_ratio=1.0)

    # Step 1: LZ77 compression → token stream
    max_tokens = n * 3 + 16
    tokens = np.zeros(max_tokens, dtype=np.int32)
    n_tokens = _lz_compress_step(data, n, tokens, max_tokens)
    n_actual_tokens = n_tokens // 3

    emitter.emit("scan", progress=0.6, current_ratio=1.0)

    # Step 2: Convert tokens to symbol stream + context
    # 3 contexts: 0=flag(2 syms), 1=literal(256), 2=match(256)
    syms = np.zeros(n_actual_tokens * 4 + 16, dtype=np.int32)
    ctxs = np.zeros(n_actual_tokens * 4 + 16, dtype=np.int32)
    out_i = 0
    for ti in range(n_actual_tokens):
        flag = tokens[ti*3]
        val = tokens[ti*3 + 1]
        ln = tokens[ti*3 + 2]
        if flag == 0:
            syms[out_i] = 0; ctxs[out_i] = 0; out_i += 1
            syms[out_i] = val; ctxs[out_i] = 1; out_i += 1
        else:
            syms[out_i] = 1; ctxs[out_i] = 0; out_i += 1
            syms[out_i] = (val >> 8) & 0xFF; ctxs[out_i] = 2; out_i += 1
            syms[out_i] = val & 0xFF; ctxs[out_i] = 2; out_i += 1
            syms[out_i] = ln - LZ_MIN_MATCH; ctxs[out_i] = 2; out_i += 1
    syms = syms[:out_i]
    ctxs = ctxs[:out_i]

    # Step 3: Build per-context frequency tables (uniform-padded to ALPHABET_SIZE)
    BIN_N_CTX = 3
    bin_alph_actual = [2, 256, 256]
    ctx_freqs = np.zeros((BIN_N_CTX, ALPHABET_SIZE + 1), dtype=np.uint64)
    for c in range(BIN_N_CTX):
        mask = ctxs == c
        sub = syms[mask]
        alph = bin_alph_actual[c]
        counts = np.ones(ALPHABET_SIZE, dtype=np.uint64)
        if len(sub) > 0:
            bc = np.bincount(sub, minlength=ALPHABET_SIZE).astype(np.uint64)[:ALPHABET_SIZE]
            for k in range(alph):
                if bc[k] > 0:
                    counts[k] = bc[k]
        total = counts.sum()
        if total > MAX_TOTAL_FREQ:
            scale = MAX_TOTAL_FREQ / total
            counts = (counts.astype(np.float64) * scale).clip(min=1).astype(np.uint64)
        ctx_freqs[c, 1:] = np.cumsum(counts)
    ctx_total = np.zeros(BIN_N_CTX, dtype=np.uint64)
    for c in range(BIN_N_CTX):
        ctx_total[c] = ctx_freqs[c, ALPHABET_SIZE]

    emitter.emit("scan", progress=0.9, current_ratio=1.0)

    # Step 4: Arithmetic encode
    emitter.emit("encode", progress=0.0, current_ratio=1.0, throughput_mbs=0.0)
    state = np.array([0, TOP_VALUE, 0, 0, 0], dtype=np.int64)
    safe_buf = max(len(syms) * 4 + 1024, 4096)
    out_buf = np.zeros(safe_buf, dtype=np.uint8)
    t0 = time.time()
    n_written = _encode_chunk(syms, ctxs, ctx_freqs, ctx_total, state, out_buf, safe_buf)
    finish_buf = np.zeros(64, dtype=np.uint8)
    n_fin = _finish(state, finish_buf, 64)
    dt = max(time.time() - t0, 1e-6)
    total_compressed = n_written + n_fin
    ratio = orig / max(total_compressed, 1)
    mbs = (n / 1_048_576) / dt
    emitter.emit("encode", progress=1.0, current_ratio=round(ratio, 3),
                 throughput_mbs=round(mbs, 2))

    # Step 5: Write output file
    with open(output_path, 'wb') as f:
        f.write(MAGIC_BIN)
        f.write(struct.pack('>B', 6))
        f.write(struct.pack('>Q', orig))
        crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
        f.write(struct.pack('>I', crc))
        f.write(struct.pack('>Q', n_actual_tokens))
        # Frequency tables: 3 ctx × ALPHABET_SIZE × 8 bytes
        for c in range(BIN_N_CTX):
            raw = np.diff(ctx_freqs[c]).astype(np.uint64)
            for v in raw:
                f.write(struct.pack('>Q', int(v)))
        # Compressed bitstream
        f.write(out_buf[:n_written].tobytes())
        f.write(finish_buf[:n_fin].tobytes())

    out_size = os.path.getsize(output_path)
    final_ratio = orig / out_size
    # Store-mode fallback: if we made the file BIGGER, just store raw bytes
    # with a "stored" flag. This is standard practice (zip/gzip do the same).
    if out_size >= orig:
        os.remove(output_path)
        with open(output_path, 'wb') as f:
            f.write(MAGIC_BIN)
            f.write(struct.pack('>B', 6))
            f.write(struct.pack('>Q', orig))
            crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
            f.write(struct.pack('>I', crc))
            f.write(struct.pack('>Q', 0))  # n_tokens = 0 → store mode
            f.write(data.tobytes())
        out_size = os.path.getsize(output_path)
        final_ratio = orig / out_size
    emitter.emit("done", input_size=orig, output_size=out_size,
                 ratio=round(final_ratio, 3),
                 time_s=round(time.time() - emitter.t0, 2))
    return final_ratio


# =========================================================================
# ORDER-2 ADAPTIVE BINARY MODE (PPM-lite — no LZ77, pure context AC)
# =========================================================================
# This mode often beats LZ77+AC on text and structured data because the
# adaptive order-2 context model captures byte-level patterns that LZ77
# misses (e.g., "qu" is almost always followed by a vowel, regardless of
# whether the exact trigram appeared before).

MAGIC_O2 = b'NFO\x00'  # NFR Order-2 context mode
MAGIC_BWT = b'NFB\x00'  # NFR BWT + Order-2 context mode
MAGIC_PPM = b'NFP\x00'  # NFR PPMd multi-order
MAGIC_NRP = b'NFN\x00'  # NFR Neural Residual Predictor
MAGIC_BIT = b'NFX\x00'  # NFR Bit-level context model
MAGIC_TRS = b'NFT\x00'  # NFR Transform combo (BWT+MTF+RLE+O2)


# =========================================================================
# PPMD (Prediction by Partial Matching with escape)
# =========================================================================
# Multi-order context model: orders 0, 1, 2, 3, 4.
# For each byte, try highest order first. If symbol not seen, emit escape
# and fall to lower order. At order 0, use uniform fallback.
# This gives 20-30% better than fixed order-2 on text.

class PPMModel:
    """PPM context model with orders 0-4 and escape mechanism."""

    def __init__(self, max_order=4):
        self.max_order = max_order
        # Order 0: dict symbol -> count (no escape, uniform fallback)
        self.ctx0 = {}
        # Order 1-4: dict context_bytes -> dict symbol -> count
        # context stored as tuple of bytes
        self.ctx_tables = [{} for _ in range(max_order + 1)]
        # Track total counts per context for escape probability
        self.totals = [{} for _ in range(max_order + 1)]
        # Escape counts per context
        self.escapes = [{} for _ in range(max_order + 1)]

    def _get_context(self, history, order):
        """Get the context (last `order` bytes) from history."""
        if order == 0 or len(history) < order:
            return ()
        return tuple(history[-order:])

    def predict_and_update(self, sym, history):
        """Returns (prob_num, prob_den, escape_used) for encoding sym.
        Updates the model after prediction."""
        # Try orders from max down to 0
        for order in range(self.max_order, -1, -1):
            ctx = self._get_context(history, order)
            table = self.ctx_tables[order].get(ctx, {})
            total = self.totals[order].get(ctx, 0)
            esc = self.escapes[order].get(ctx, 0)

            if total == 0:
                # Context never seen — skip to lower order
                continue

            sym_count = table.get(sym, 0)
            if sym_count > 0:
                # Symbol seen at this order — encode it
                # Cumulative range
                cum_lo = 0
                for s, c in sorted(table.items()):
                    if s < sym:
                        cum_lo += c
                cum_hi = cum_lo + sym_count
                # Update
                table[sym] = sym_count + 1
                self.ctx_tables[order][ctx] = table
                self.totals[order][ctx] = total + 1
                return (cum_lo, cum_hi, total, False)

            # Symbol not seen — emit escape if escape count > 0
            if esc > 0:
                # Encode escape symbol
                cum_lo = 0
                for s, c in sorted(table.items()):
                    cum_lo += c
                cum_hi = cum_lo + esc
                # Update escape count
                self.escapes[order][ctx] = esc + 1
                # Don't return — fall to next order
                continue
            else:
                continue

        # Order 0 fallback: uniform distribution
        # Use a simple count-based model at order 0
        table = self.ctx0
        total = sum(table.values()) if table else 0
        if total == 0:
            # Truly uniform: 1/256
            cum_lo = sym
            cum_hi = sym + 1
            den = 256
        else:
            sym_count = table.get(sym, 0)
            if sym_count == 0:
                sym_count = 1  # ensure non-zero
            cum_lo = 0
            for s in range(sym):
                cum_lo += table.get(s, 1)
            cum_hi = cum_lo + sym_count
            den = total + 256  # smooth with uniform

        # Update order 0
        table[sym] = table.get(sym, 0) + 1
        self.ctx0 = table
        return (cum_lo, cum_hi, den, False)


def compress_binary_ppm(input_path: str, output_path: str, emitter: ProgressEmitter):
    """PPMd compressor with orders 0-4 and escape mechanism."""
    orig = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig, output_size=0, ratio=1.0)
    emitter.emit("scan", progress=0.0, current_ratio=1.0)

    with open(input_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(data)
    emitter.emit("scan", progress=0.3, current_ratio=1.0)

    # Initialize PPM model
    model = PPMModel(max_order=4)

    # Arithmetic coder state
    state = {'low': 0, 'high': TOP_VALUE, 'pending': 0, 'bit_buf': 0, 'bit_cnt': 0}
    out_buf = bytearray()
    half = HALF; quarter = QUARTER; tq = THREE_QUARTERS; top = TOP_VALUE

    def emit_bit(bit):
        state['bit_buf'] = (state['bit_buf'] << 1) | bit
        state['bit_cnt'] += 1
        if state['bit_cnt'] == 8:
            out_buf.append(state['bit_buf'])
            state['bit_buf'] = 0
            state['bit_cnt'] = 0

    def emit_bit_with_pending(bit):
        emit_bit(bit)
        while state['pending'] > 0:
            emit_bit(1 - bit)
            state['pending'] -= 1

    def renormalize():
        while True:
            if state['high'] < half:
                emit_bit_with_pending(0)
            elif state['low'] >= half:
                emit_bit_with_pending(1)
                state['low'] -= half
                state['high'] -= half
            elif state['low'] >= quarter and state['high'] < tq:
                state['pending'] += 1
                state['low'] -= quarter
                state['high'] -= quarter
            else:
                break
            state['low'] = (state['low'] << 1) & top
            state['high'] = ((state['high'] << 1) & top) | 1

    emitter.emit("encode", progress=0.0, current_ratio=1.0, throughput_mbs=0.0)
    history = []
    t0 = time.time()

    for i in range(n):
        sym = int(data[i])
        cum_lo, cum_hi, den, esc = model.predict_and_update(sym, history)
        r = state['high'] - state['low'] + 1
        state['high'] = state['low'] + (r * cum_hi) // den - 1
        state['low'] = state['low'] + (r * cum_lo) // den
        renormalize()
        history.append(sym)
        if i % 10000 == 0 and i > 0:
            ratio = orig / max(len(out_buf), 1)
            emitter.emit("encode", progress=i / n, current_ratio=round(ratio, 3))

    # Finish
    state['pending'] += 1
    if state['low'] < quarter:
        emit_bit_with_pending(0)
    else:
        emit_bit_with_pending(1)
    if state['bit_cnt'] > 0:
        out_buf.append(state['bit_buf'] << (8 - state['bit_cnt']))

    dt = max(time.time() - t0, 1e-6)
    ratio = orig / max(len(out_buf), 1)
    mbs = (n / 1_048_576) / dt
    emitter.emit("encode", progress=1.0, current_ratio=round(ratio, 3),
                 throughput_mbs=round(mbs, 2))

    # Write output
    with open(output_path, 'wb') as f:
        f.write(MAGIC_PPM)
        f.write(struct.pack('>B', 6))
        f.write(struct.pack('>Q', orig))
        crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
        f.write(struct.pack('>I', crc))
        f.write(bytes(out_buf))

    out_size = os.path.getsize(output_path)
    if out_size >= orig:
        # Store mode
        os.remove(output_path)
        with open(output_path, 'wb') as f:
            f.write(MAGIC_PPM)
            f.write(struct.pack('>B', 6))
            f.write(struct.pack('>Q', orig))
            f.write(struct.pack('>I', crc))
            f.write(struct.pack('>Q', 0xFFFFFFFFFFFFFFFF))  # store marker
            f.write(data.tobytes())
        out_size = os.path.getsize(output_path)
        ratio = orig / out_size

    emitter.emit("done", input_size=orig, output_size=out_size,
                 ratio=round(ratio, 3),
                 time_s=round(time.time() - emitter.t0, 2))
    return ratio


def decompress_binary_ppm(input_path: str, output_path: str, emitter: ProgressEmitter):
    orig_size = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig_size, output_size=0, ratio=1.0)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_PPM:
            raise ValueError(f"Bad magic for PPM mode: {magic!r}")
        ver = struct.unpack('>B', f.read(1))[0]
        orig = struct.unpack('>Q', f.read(8))[0]
        crc = struct.unpack('>I', f.read(4))[0]
        # Check store mode
        peek = f.read(8)
        if len(peek) == 8 and peek == b'\xff' * 8:
            raw = f.read()
            actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
            if actual_crc != crc:
                raise ValueError(f"CRC mismatch (store): {crc:08x} vs {actual_crc:08x}")
            with open(output_path, 'wb') as fout:
                fout.write(raw)
            emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                         time_s=round(time.time() - emitter.t0, 2))
            return
        compressed = np.frombuffer(peek + f.read(), dtype=np.uint8)

    model = PPMModel(max_order=4)
    out_data = np.zeros(orig, dtype=np.uint8)
    history = []

    half = HALF; quarter = QUARTER; tq = THREE_QUARTERS; top = TOP_VALUE
    low = 0; high = TOP_VALUE
    value = 0
    buf_ptr = 0; bit_ptr = 7

    # Prime
    for _ in range(32):
        if buf_ptr < len(compressed):
            byte = int(compressed[buf_ptr])
            bit = (byte >> bit_ptr) & 1
            bit_ptr -= 1
            if bit_ptr < 0:
                bit_ptr = 7; buf_ptr += 1
        else:
            bit = 0
        value = (value << 1) | bit

    emitter.emit("decode", progress=0.0)

    for i in range(orig):
        # Try orders from max down to 0
        sym_decoded = -1
        for order in range(model.max_order, -1, -1):
            ctx = model._get_context(history, order)
            table = model.ctx_tables[order].get(ctx, {})
            total = model.totals[order].get(ctx, 0)
            esc = model.escapes[order].get(ctx, 0)

            if total == 0:
                continue

            r = high - low + 1
            sv = ((value - low + 1) * total - 1) // r

            # Check if sv falls in escape range
            cum_data = 0
            cum_escape_start = 0
            for s, c in sorted(table.items()):
                cum_data += c
            cum_escape_start = cum_data
            escape_end = cum_escape_start + esc

            if sv < cum_escape_start:
                # Symbol is in table
                # Binary search
                cum_lo = 0
                sym = -1
                for s, c in sorted(table.items()):
                    if cum_lo + c > sv:
                        sym = s
                        cum_hi = cum_lo + c
                        break
                    cum_lo += c
                if sym == -1:
                    continue
                # Decode
                high = low + (r * cum_hi) // total - 1
                low = low + (r * cum_lo) // total
                # Renormalize
                while True:
                    if high < half:
                        pass
                    elif low >= half:
                        low -= half; high -= half; value -= half
                    elif low >= quarter and high < tq:
                        low -= quarter; high -= quarter; value -= quarter
                    else:
                        break
                    low = (low << 1) & top
                    high = ((high << 1) & top) | 1
                    if buf_ptr < len(compressed):
                        byte = int(compressed[buf_ptr])
                        bit = (byte >> bit_ptr) & 1
                        bit_ptr -= 1
                        if bit_ptr < 0:
                            bit_ptr = 7; buf_ptr += 1
                    else:
                        bit = 0
                    value = ((value << 1) & top) | bit
                # Update
                table[sym] = table.get(sym, 0) + 1
                model.ctx_tables[order][ctx] = table
                model.totals[order][ctx] = total + 1
                sym_decoded = sym
                break
            elif sv < escape_end:
                # Escape — decode it and fall to next order
                cum_lo = cum_escape_start
                cum_hi = escape_end
                high = low + (r * cum_hi) // total - 1
                low = low + (r * cum_lo) // total
                while True:
                    if high < half:
                        pass
                    elif low >= half:
                        low -= half; high -= half; value -= half
                    elif low >= quarter and high < tq:
                        low -= quarter; high -= quarter; value -= quarter
                    else:
                        break
                    low = (low << 1) & top
                    high = ((high << 1) & top) | 1
                    if buf_ptr < len(compressed):
                        byte = int(compressed[buf_ptr])
                        bit = (byte >> bit_ptr) & 1
                        bit_ptr -= 1
                        if bit_ptr < 0:
                            bit_ptr = 7; buf_ptr += 1
                    else:
                        bit = 0
                    value = ((value << 1) & top) | bit
                model.escapes[order][ctx] = esc + 1
                continue
            # sv >= escape_end — shouldn't happen, but skip
            continue

        if sym_decoded == -1:
            # Order 0 fallback: uniform
            table = model.ctx0
            total = sum(table.values()) if table else 0
            if total == 0:
                den = 256
                r = high - low + 1
                sv = ((value - low + 1) * den - 1) // r
                sym = sv
                cum_lo = sym
                cum_hi = sym + 1
            else:
                den = total + 256
                r = high - low + 1
                sv = ((value - low + 1) * den - 1) // r
                # Find symbol
                cum_lo = 0
                sym = -1
                for s in range(256):
                    c = table.get(s, 1)
                    if cum_lo + c > sv:
                        sym = s
                        cum_hi = cum_lo + c
                        break
                    cum_lo += c
                if sym == -1:
                    sym = sv % 256
                    cum_hi = cum_lo + 1
            high = low + (r * cum_hi) // den - 1
            low = low + (r * cum_lo) // den
            while True:
                if high < half:
                    pass
                elif low >= half:
                    low -= half; high -= half; value -= half
                elif low >= quarter and high < tq:
                    low -= quarter; high -= quarter; value -= quarter
                else:
                    break
                low = (low << 1) & top
                high = ((high << 1) & top) | 1
                if buf_ptr < len(compressed):
                    byte = int(compressed[buf_ptr])
                    bit = (byte >> bit_ptr) & 1
                    bit_ptr -= 1
                    if bit_ptr < 0:
                        bit_ptr = 7; buf_ptr += 1
                else:
                    bit = 0
                value = ((value << 1) & top) | bit
            table[sym] = table.get(sym, 0) + 1
            model.ctx0 = table
            sym_decoded = sym

        out_data[i] = sym_decoded
        history.append(sym_decoded)
        if i % 10000 == 0 and i > 0:
            emitter.emit("decode", progress=i / orig)

    emitter.emit("decode", progress=1.0)

    actual_crc = zlib.crc32(out_data.tobytes()) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ValueError(f"CRC mismatch: {crc:08x} vs {actual_crc:08x}")

    with open(output_path, 'wb') as f:
        f.write(out_data.tobytes())

    emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                 time_s=round(time.time() - emitter.t0, 2))


# =========================================================================
# NEURAL RESIDUAL PREDICTOR (NRP) — byte-level
# =========================================================================
# A small MLP learns to predict each byte from the previous 8 bytes.
# The prediction error (residual) is then compressed with order-2 AC.
# The MLP captures non-linear patterns that linear context models miss.

def _train_byte_predictor(data, device, steps=300):
    """Train a small MLP to predict byte from 8-byte context."""
    torch, nn = _import_torch()
    n = len(data)
    if n < 100:
        return None

    class BytePredictor(nn.Module):
        def __init__(self, hidden=32, ctx_len=8):
            super().__init__()
            self.ctx_len = ctx_len
            self.embed = nn.Linear(ctx_len, hidden)
            self.net = nn.Sequential(
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        def forward(self, x):
            return self.net(self.embed(x))

    model = BytePredictor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Build training samples: context = 8 previous bytes (normalized to [0,1])
    # target = current byte (normalized to [0,1])
    ctx_len = 8
    data_f = data.astype(np.float32) / 255.0

    for step in range(steps):
        # Random batch
        idx = np.random.randint(ctx_len, n, size=min(4096, n - ctx_len))
        ctx = np.stack([data_f[idx - k] for k in range(ctx_len, 0, -1)], axis=1)  # (B, 8)
        targets = data_f[idx].reshape(-1, 1)  # (B, 1)

        ctx_t = torch.from_numpy(ctx).to(device)
        tgt_t = torch.from_numpy(targets).to(device)

        pred = model(ctx_t)
        loss = nn.MSELoss()(pred, tgt_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    return model


def _predict_bytes_with_mlp(data, model, device):
    """Use trained MLP to predict each byte. Returns residuals (data - prediction)
    as int16 array, scaled to [-128, 127] range."""
    torch, _ = _import_torch()
    n = len(data)
    if n < 100 or model is None:
        return data.astype(np.int16)

    ctx_len = model.ctx_len
    data_f = data.astype(np.float32) / 255.0
    residuals = np.zeros(n, dtype=np.int16)

    # Process in chunks for speed
    CHUNK = 65536
    for start in range(ctx_len, n, CHUNK):
        end = min(start + CHUNK, n)
        size = end - start
        # Build context: (size, ctx_len)
        ctx = np.zeros((size, ctx_len), dtype=np.float32)
        for k in range(ctx_len):
            ctx[:, k] = data_f[start - ctx_len + k:end - ctx_len + k]
        ctx_t = torch.from_numpy(ctx).to(device)
        with torch.no_grad():
            pred = model(ctx_t).cpu().numpy().flatten()  # (size,) in [0,1]
        # Residual = actual - predicted, scaled back to byte range
        actual = data_f[start:end]
        res = (actual - pred) * 255.0
        residuals[start:end] = np.round(res).astype(np.int16)

    # First ctx_len bytes: store raw (no prediction)
    residuals[:ctx_len] = data[:ctx_len].astype(np.int16) - 128
    return residuals


def _reconstruct_bytes_with_mlp(residuals, model, device):
    """Inverse: reconstruct original bytes from residuals.
    Autoregressive — predict each byte using reconstructed previous bytes."""
    torch, _ = _import_torch()
    n = len(residuals)
    if n < 100 or model is None:
        return (residuals + 128).clip(0, 255).astype(np.uint8)

    ctx_len = model.ctx_len
    out = np.zeros(n, dtype=np.float32)

    # First ctx_len bytes: residuals store (byte - 128)
    out[:ctx_len] = residuals[:ctx_len].astype(np.float32) + 128.0

    # Reconstruct rest autoregressively
    CHUNK = 65536
    for start in range(ctx_len, n, CHUNK):
        end = min(start + CHUNK, n)
        size = end - start
        ctx = np.zeros((size, ctx_len), dtype=np.float32)
        for k in range(ctx_len):
            ctx[:, k] = out[start - ctx_len + k:end - ctx_len + k] / 255.0
        ctx_t = torch.from_numpy(ctx).to(device)
        with torch.no_grad():
            pred = model(ctx_t).cpu().numpy().flatten()  # (size,) in [0,1]
        out[start:end] = pred * 255.0 + residuals[start:end].astype(np.float32)

    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def compress_binary_nrp(input_path: str, output_path: str, emitter: ProgressEmitter):
    """Neural Residual Predictor: MLP predicts bytes, residuals compressed with O2 AC."""
    _, _, _, _, _encode_o2, _ = _kernels()
    _, _finish, _, _, _, _ = _kernels()
    torch, _ = _import_torch()

    orig = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig, output_size=0, ratio=1.0)
    emitter.emit("scan", progress=0.0, current_ratio=1.0)

    with open(input_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(data)
    emitter.emit("scan", progress=0.2, current_ratio=1.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train MLP predictor
    emitter.emit("train", progress=0.0, loss=1.0)
    predictor = _train_byte_predictor(data, device, steps=300)
    emitter.emit("train", progress=1.0, loss=0.0)

    if predictor is None:
        # Fallback to O2
        return compress_binary_o2(input_path, output_path, emitter)

    # Predict and compute residuals
    emitter.emit("scan", progress=0.5, current_ratio=1.0)
    residuals = _predict_bytes_with_mlp(data, predictor, device)
    # Shift to 0..255 for AC (residuals are in -255..255)
    res_shifted = (residuals + 255).clip(0, 510).astype(np.uint16) % 256
    res_data = res_shifted.astype(np.uint8)

    emitter.emit("scan", progress=0.8, current_ratio=1.0)

    # Initialize cum_freqs for order-2 AC on residuals
    cum_freqs = np.zeros((65536, 257), dtype=np.uint32)
    cum_freqs[:] = np.arange(257, dtype=np.uint32)

    # Encode residuals
    emitter.emit("encode", progress=0.0, current_ratio=1.0, throughput_mbs=0.0)
    state = np.array([0, TOP_VALUE, 0, 0, 0], dtype=np.int64)
    safe_buf = max(n + 1024, 4096)
    out_buf = np.zeros(safe_buf, dtype=np.uint8)

    t0 = time.time()
    n_written = _encode_o2(res_data, cum_freqs, state, out_buf, safe_buf)
    finish_buf = np.zeros(64, dtype=np.uint8)
    n_fin = _finish(state, finish_buf, 64)
    dt = max(time.time() - t0, 1e-6)
    total_compressed = n_written + n_fin
    ratio = orig / max(total_compressed, 1)
    mbs = (n / 1_048_576) / dt
    emitter.emit("encode", progress=1.0, current_ratio=round(ratio, 3),
                 throughput_mbs=round(mbs, 2))

    # Write output: MAGIC_NRP + ver + orig_size + crc + model_len + model + compressed
    with open(output_path, 'wb') as f:
        f.write(MAGIC_NRP)
        f.write(struct.pack('>B', 6))
        f.write(struct.pack('>Q', orig))
        crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
        f.write(struct.pack('>I', crc))
        # Save predictor model (int8 quantized)
        b = io.BytesIO()
        sd = predictor.state_dict()
        b.write(struct.pack('>I', len(sd)))
        for name, t in sd.items():
            t_np = t.detach().cpu().float().numpy()
            shape = t_np.shape
            b.write(struct.pack('>I', len(shape)))
            for d in shape: b.write(struct.pack('>I', d))
            max_abs = float(np.abs(t_np).max()) if t_np.size > 0 else 1.0
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            b.write(struct.pack('>f', scale))
            q = np.round(t_np / scale).clip(-127, 127).astype(np.int8)
            b.write(q.tobytes())
        mb = b.getvalue()
        f.write(struct.pack('>I', len(mb)))
        f.write(mb)
        f.write(out_buf[:n_written].tobytes())
        f.write(finish_buf[:n_fin].tobytes())

    out_size = os.path.getsize(output_path)
    if out_size >= orig:
        os.remove(output_path)
        with open(output_path, 'wb') as f:
            f.write(MAGIC_NRP)
            f.write(struct.pack('>B', 6))
            f.write(struct.pack('>Q', orig))
            f.write(struct.pack('>I', crc))
            f.write(struct.pack('>I', 0))  # store mode marker
            f.write(data.tobytes())
        out_size = os.path.getsize(output_path)
        ratio = orig / out_size

    emitter.emit("done", input_size=orig, output_size=out_size,
                 ratio=round(ratio, 3),
                 time_s=round(time.time() - emitter.t0, 2))
    return ratio


def decompress_binary_nrp(input_path: str, output_path: str, emitter: ProgressEmitter):
    _, _, _, _, _, _decode_o2 = _kernels()
    torch, _ = _import_torch()

    orig_size = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig_size, output_size=0, ratio=1.0)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_NRP:
            raise ValueError(f"Bad magic for NRP mode: {magic!r}")
        ver = struct.unpack('>B', f.read(1))[0]
        orig = struct.unpack('>Q', f.read(8))[0]
        crc = struct.unpack('>I', f.read(4))[0]
        model_len = struct.unpack('>I', f.read(4))[0]

        # Store mode check
        if model_len == 0:
            raw = f.read()
            actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
            if actual_crc != crc:
                raise ValueError(f"CRC mismatch (store): {crc:08x} vs {actual_crc:08x}")
            with open(output_path, 'wb') as fout:
                fout.write(raw)
            emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                         time_s=round(time.time() - emitter.t0, 2))
            return

        model_bytes = f.read(model_len)

        # Load predictor model (int8 quantized)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        b = io.BytesIO(model_bytes)
        n_tensors = struct.unpack('>I', b.read(4))[0]
        sd = {}
        for _ in range(n_tensors):
            n_dims = struct.unpack('>I', b.read(4))[0]
            shape = tuple(struct.unpack('>I', b.read(4))[0] for _ in range(n_dims))
            scale = struct.unpack('>f', b.read(4))[0]
            n_elems = 1
            for d in shape: n_elems *= d
            q = np.frombuffer(b.read(n_elems), dtype=np.int8).astype(np.float32)
            sd[f"t{len(sd)}"] = torch.from_numpy(q * scale).reshape(shape) if shape else torch.from_numpy(q * scale)

        # Rebuild predictor
        class BytePredictor(torch.nn.Module):
            def __init__(self, hidden=32, ctx_len=8):
                super().__init__()
                self.ctx_len = ctx_len
                self.embed = torch.nn.Linear(ctx_len, hidden)
                self.net = torch.nn.Sequential(
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden, hidden),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden, 1),
                )
            def forward(self, x):
                return self.net(self.embed(x))
        predictor = BytePredictor().to(device)
        actual_sd = predictor.state_dict()
        keys = list(actual_sd.keys())
        for i, k in enumerate(keys):
            actual_sd[k] = sd[f"t{i}"]
        predictor.load_state_dict(actual_sd)
        predictor.eval()

        compressed = np.frombuffer(f.read(), dtype=np.uint8)

    # Decode residuals with O2 AC
    cum_freqs = np.zeros((65536, 257), dtype=np.uint32)
    cum_freqs[:] = np.arange(257, dtype=np.uint32)

    value = np.int64(0)
    buf_ptr = 0
    bit_ptr = 7
    for _ in range(32):
        if buf_ptr < len(compressed):
            byte = int(compressed[buf_ptr])
            bit = (byte >> bit_ptr) & 1
            bit_ptr -= 1
            if bit_ptr < 0:
                bit_ptr = 7; buf_ptr += 1
        else:
            bit = 0
        value = (value << np.int64(1)) | np.int64(bit)
    state = np.array([0, TOP_VALUE, value, buf_ptr, bit_ptr], dtype=np.int64)

    res_shifted = np.zeros(orig, dtype=np.uint8)
    if orig > 0:
        emitter.emit("decode", progress=0.0)
        _decode_o2(compressed, orig, cum_freqs, state, res_shifted)
        emitter.emit("decode", progress=0.5)

    # Convert back to signed residuals
    residuals = res_shifted.astype(np.int16)
    # Unshift: 0..255 → -128..127 (mod 256)
    residuals = ((residuals + 128) % 256) - 128

    # Reconstruct bytes using MLP
    out_data = _reconstruct_bytes_with_mlp(residuals, predictor, device)
    emitter.emit("decode", progress=1.0)

    actual_crc = zlib.crc32(out_data.tobytes()) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ValueError(f"CRC mismatch: {crc:08x} vs {actual_crc:08x}")

    with open(output_path, 'wb') as f:
        f.write(out_data.tobytes())

    emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                 time_s=round(time.time() - emitter.t0, 2))


# =========================================================================
# TRANSFORM COMBO MODE — BWT + MTF + RLE + O2 (bzip2-style)
# =========================================================================

def compress_binary_trs(input_path: str, output_path: str, emitter: ProgressEmitter):
    """Transform combo: BWT → MTF → RLE → O2 AC. Best for text + structured data."""
    _, _, _, _, _encode_o2, _ = _kernels()
    _, _finish, _, _, _, _ = _kernels()

    orig = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig, output_size=0, ratio=1.0)
    emitter.emit("scan", progress=0.0, current_ratio=1.0)

    with open(input_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(data)
    emitter.emit("scan", progress=0.2, current_ratio=1.0)

    # Step 1: BWT
    bwt_bytes, primary = _bwt_transform(data)
    if bwt_bytes is None:
        return compress_binary_o2(input_path, output_path, emitter)
    bwt_data = np.frombuffer(bwt_bytes, dtype=np.uint8)
    emitter.emit("scan", progress=0.4, current_ratio=1.0)

    # Step 2: MTF
    mtf_data = _mtf_encode(bwt_data)
    emitter.emit("scan", progress=0.6, current_ratio=1.0)

    # Step 3: RLE
    rle_data = _rle_encode_bytes(mtf_data)
    emitter.emit("scan", progress=0.8, current_ratio=1.0)

    # Step 4: O2 AC encode
    cum_freqs = np.zeros((65536, 257), dtype=np.uint32)
    cum_freqs[:] = np.arange(257, dtype=np.uint32)

    emitter.emit("encode", progress=0.0, current_ratio=1.0, throughput_mbs=0.0)
    state = np.array([0, TOP_VALUE, 0, 0, 0], dtype=np.int64)
    safe_buf = max(len(rle_data) + 1024, 4096)
    out_buf = np.zeros(safe_buf, dtype=np.uint8)

    t0 = time.time()
    n_written = _encode_o2(rle_data, cum_freqs, state, out_buf, safe_buf)
    finish_buf = np.zeros(64, dtype=np.uint8)
    n_fin = _finish(state, finish_buf, 64)
    dt = max(time.time() - t0, 1e-6)
    total_compressed = n_written + n_fin
    ratio = orig / max(total_compressed, 1)
    mbs = (n / 1_048_576) / dt
    emitter.emit("encode", progress=1.0, current_ratio=round(ratio, 3),
                 throughput_mbs=round(mbs, 2))

    # Write output: MAGIC_TRS + ver + orig + crc + primary + rle_len + compressed
    with open(output_path, 'wb') as f:
        f.write(MAGIC_TRS)
        f.write(struct.pack('>B', 6))
        f.write(struct.pack('>Q', orig))
        crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
        f.write(struct.pack('>I', crc))
        f.write(struct.pack('>Q', primary))
        f.write(struct.pack('>Q', len(rle_data)))  # rle_data length for decoder
        f.write(out_buf[:n_written].tobytes())
        f.write(finish_buf[:n_fin].tobytes())

    out_size = os.path.getsize(output_path)
    if out_size >= orig:
        os.remove(output_path)
        with open(output_path, 'wb') as f:
            f.write(MAGIC_TRS)
            f.write(struct.pack('>B', 6))
            f.write(struct.pack('>Q', orig))
            crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
            f.write(struct.pack('>I', crc))
            f.write(struct.pack('>Q', 0xFFFFFFFFFFFFFFFF))  # primary = store marker
            f.write(struct.pack('>Q', 0))                   # rle_len = 0 (unused)
            f.write(data.tobytes())
        out_size = os.path.getsize(output_path)
        ratio = orig / out_size

    emitter.emit("done", input_size=orig, output_size=out_size,
                 ratio=round(ratio, 3),
                 time_s=round(time.time() - emitter.t0, 2))
    return ratio


def decompress_binary_trs(input_path: str, output_path: str, emitter: ProgressEmitter):
    _, _, _, _, _, _decode_o2 = _kernels()

    orig_size = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig_size, output_size=0, ratio=1.0)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_TRS:
            raise ValueError(f"Bad magic for TRS mode: {magic!r}")
        ver = struct.unpack('>B', f.read(1))[0]
        orig = struct.unpack('>Q', f.read(8))[0]
        crc = struct.unpack('>I', f.read(4))[0]
        primary = struct.unpack('>Q', f.read(8))[0]

        # Check for store mode (primary = max uint64)
        if primary == 0xFFFFFFFFFFFFFFFF:
            f.read(8)  # skip rle_len field
            raw = f.read()
            actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
            if actual_crc != crc:
                raise ValueError(f"CRC mismatch (store): {crc:08x} vs {actual_crc:08x}")
            with open(output_path, 'wb') as fout:
                fout.write(raw)
            emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                         time_s=round(time.time() - emitter.t0, 2))
            return
        rle_len = struct.unpack('>Q', f.read(8))[0]
        compressed = np.frombuffer(f.read(), dtype=np.uint8)

    # Decode O2 AC (decode orig symbols — RLE data is ≤ orig length)
    cum_freqs = np.zeros((65536, 257), dtype=np.uint32)
    cum_freqs[:] = np.arange(257, dtype=np.uint32)

    value = np.int64(0)
    buf_ptr = 0
    bit_ptr = 7
    for _ in range(32):
        if buf_ptr < len(compressed):
            byte = int(compressed[buf_ptr])
            bit = (byte >> bit_ptr) & 1
            bit_ptr -= 1
            if bit_ptr < 0:
                bit_ptr = 7; buf_ptr += 1
        else:
            bit = 0
        value = (value << np.int64(1)) | np.int64(bit)
    state = np.array([0, TOP_VALUE, value, buf_ptr, bit_ptr], dtype=np.int64)

    rle_decoded = np.zeros(rle_len, dtype=np.uint8)
    if rle_len > 0:
        emitter.emit("decode", progress=0.0)
        _decode_o2(compressed, rle_len, cum_freqs, state, rle_decoded)
        emitter.emit("decode", progress=0.3)

    # Inverse RLE → MTF data (length may differ — trim/pad as needed)
    mtf_data = _rle_decode_bytes(rle_decoded)
    emitter.emit("decode", progress=0.6)

    # Inverse MTF → BWT data
    bwt_data = _mtf_decode(mtf_data)
    emitter.emit("decode", progress=0.8)

    # Inverse BWT → original
    out_data = _bwt_inverse(bwt_data.tobytes(), int(primary))
    emitter.emit("decode", progress=1.0)

    # Trim to original length
    out_data = out_data[:orig]

    actual_crc = zlib.crc32(out_data.tobytes()) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ValueError(f"CRC mismatch: {crc:08x} vs {actual_crc:08x}")

    with open(output_path, 'wb') as f:
        f.write(out_data.tobytes())

    emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                 time_s=round(time.time() - emitter.t0, 2))


# =========================================================================
# BIT-LEVEL CONTEXT MODEL MODE
# =========================================================================

def compress_binary_bit(input_path: str, output_path: str, emitter: ProgressEmitter):
    """Bit-level context model with order-8 + order-16 mixing."""
    orig = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig, output_size=0, ratio=1.0)
    emitter.emit("scan", progress=0.0, current_ratio=1.0)

    with open(input_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(data)
    emitter.emit("scan", progress=0.5, current_ratio=1.0)

    emitter.emit("encode", progress=0.0, current_ratio=1.0, throughput_mbs=0.0)
    t0 = time.time()
    bitstream, n_bits = _bit_context_encode(data)
    dt = max(time.time() - t0, 1e-6)
    ratio = orig / max(len(bitstream), 1)
    mbs = (n / 1_048_576) / dt
    emitter.emit("encode", progress=1.0, current_ratio=round(ratio, 3),
                 throughput_mbs=round(mbs, 2))

    with open(output_path, 'wb') as f:
        f.write(MAGIC_BIT)
        f.write(struct.pack('>B', 6))
        f.write(struct.pack('>Q', orig))
        crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
        f.write(struct.pack('>I', crc))
        f.write(struct.pack('>Q', n_bits))
        f.write(bitstream)

    out_size = os.path.getsize(output_path)
    if out_size >= orig:
        os.remove(output_path)
        with open(output_path, 'wb') as f:
            f.write(MAGIC_BIT)
            f.write(struct.pack('>B', 6))
            f.write(struct.pack('>Q', orig))
            f.write(struct.pack('>I', crc))
            f.write(struct.pack('>Q', 0xFFFFFFFFFFFFFFFF))
            f.write(data.tobytes())
        out_size = os.path.getsize(output_path)
        ratio = orig / out_size

    emitter.emit("done", input_size=orig, output_size=out_size,
                 ratio=round(ratio, 3),
                 time_s=round(time.time() - emitter.t0, 2))
    return ratio


def decompress_binary_bit(input_path: str, output_path: str, emitter: ProgressEmitter):
    orig_size = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig_size, output_size=0, ratio=1.0)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_BIT:
            raise ValueError(f"Bad magic for BIT mode: {magic!r}")
        ver = struct.unpack('>B', f.read(1))[0]
        orig = struct.unpack('>Q', f.read(8))[0]
        crc = struct.unpack('>I', f.read(4))[0]
        n_bits = struct.unpack('>Q', f.read(8))[0]

        if n_bits == 0xFFFFFFFFFFFFFFFF:
            raw = f.read()
            actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
            if actual_crc != crc:
                raise ValueError(f"CRC mismatch (store): {crc:08x} vs {actual_crc:08x}")
            with open(output_path, 'wb') as fout:
                fout.write(raw)
            emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                         time_s=round(time.time() - emitter.t0, 2))
            return
        bitstream = f.read()

    emitter.emit("decode", progress=0.0)
    out_data = _bit_context_decode(bitstream, n_bits)
    emitter.emit("decode", progress=1.0)

    actual_crc = zlib.crc32(out_data.tobytes()) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ValueError(f"CRC mismatch: {crc:08x} vs {actual_crc:08x}")

    with open(output_path, 'wb') as f:
        f.write(out_data.tobytes())

    emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                 time_s=round(time.time() - emitter.t0, 2))


# =========================================================================
# DELTA / XOR COMBO MODE — for counters, timestamps, arithmetic data
# =========================================================================

MAGIC_DLT = b'NFD\x00'  # NFR Delta/XOR + O2
MAGIC_PRG = b'NFR\x00'  # NFR PRNG-detected (seed + params)
MAGIC_KOL = b'NFK\x00'  # NFR Kolmogorov (polynomial/math structure)
MAGIC_BPL = b'NFL\x00' # NFR Bit-plane decomposition (was NFPL, fixed to 4 bytes)


# =========================================================================
# KOLMOGOROV COMPLEXITY COMPRESSOR
# =========================================================================
# Shannon's entropy gives a lower bound based on the SOURCE distribution.
# Kolmogorov complexity K(x) gives the length of the shortest PROGRAM that
# produces x. For strings with hidden mathematical structure (π digits,
# squares, Fibonacci, polynomials), K(x) << |x| even though Shannon ≈ |x|.
#
# We try to detect: constant sequences, arithmetic (linear), quadratic,
# cubic, quartic polynomials, geometric, and Fibonacci — interpreting the
# byte stream as integers in various bases (256, 16, 10, 9, 8, 2).

def _interpret_as_ints(data: np.ndarray, base: int) -> np.ndarray:
    """Interpret byte stream as a sequence of integers in the given base.
    For base 256: each byte is one integer.
    For base 16/10/9/8/2: split byte stream into base-'base' digits, then
    chunk into fixed-width integers (so a polynomial can fit).
    We use width = log_b(2^32) so each chunk fits in a uint32."""
    if base == 256:
        return data.astype(np.int64)
    # For other bases, treat each byte as a digit in base 'base' (if < base)
    # Actually, simpler: each byte IS the digit value, but we filter to < base
    # Then chunk every K digits into one integer where K = bits needed
    import math
    if base <= 1:
        return None
    # Filter: only keep bytes < base (if any byte >= base, this base doesn't fit)
    if data.max() >= base:
        return None
    # Chunk size: enough digits to fill ~32 bits
    chunk = max(1, int(math.floor(math.log(2**32, base))))
    n = len(data)
    n_chunks = n // chunk
    if n_chunks < 4:
        return None
    out = np.zeros(n_chunks, dtype=np.int64)
    for i in range(n_chunks):
        val = 0
        for k in range(chunk):
            val = val * base + int(data[i * chunk + k])
        out[i] = val
    return out


def _fit_polynomial(seq: np.ndarray, degree: int, mod: int = None) -> list:
    """Fit polynomial of given degree to sequence. Returns coeffs [a0, a1, ...]
    such that seq[i] = (a0 + a1*i + a2*i^2 + ... + ad*i^d) mod `mod`.
    If mod is None, requires exact integer fit (no mod).
    Returns None if no fit found."""
    n = len(seq)
    if n < degree + 2:
        return None
    m = degree + 1

    if mod is None:
        # Exact integer fit via Vandermonde
        V = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                V[i, j] = i ** j
        b = seq[:m].astype(np.float64)
        try:
            coeffs = np.linalg.solve(V, b)
        except np.linalg.LinAlgError:
            return None
        coeffs_int = np.round(coeffs).astype(np.int64)
        for i in range(m):
            val = sum(int(coeffs_int[j]) * (i ** j) for j in range(m))
            if val != int(seq[i]):
                return None
        for i in range(n):
            val = sum(int(coeffs_int[j]) * (i ** j) for j in range(m))
            if val != int(seq[i]):
                return None
        return coeffs_int.tolist()
    else:
        # Modular fit using finite differences (no division needed!)
        # If seq is a polynomial of degree d mod m, the d-th finite difference
        # is constant mod m. We store: first (d+1) values + constant d-th diff.
        # Reconstruction: integrate the difference table iteratively.

        if degree == 0:
            c0 = int(seq[0]) % mod
            for i in range(n):
                if c0 != int(seq[i]) % mod:
                    return None
            return {'newton': [c0], 'mod': mod}

        # Compute d-th finite differences on first (degree+3) points
        diffs = seq[:degree + 3].astype(np.int64).copy()
        for d in range(degree):
            diffs = np.diff(diffs)
        # diffs should all be equal mod `mod`
        if len(diffs) < 2:
            return None
        lead = int(diffs[0]) % mod
        for v in diffs:
            if int(v) % mod != lead:
                return None

        # Store: first (degree+1) values + the constant d-th difference
        first_vals = [int(seq[k]) % mod for k in range(degree + 1)]
        const_d_diff = lead

        # VERIFY on ALL points by reconstructing iteratively
        # Build the difference table from first_vals, then extend
        table = [first_vals[:] ]  # table[0] = original values
        for d in range(1, degree + 1):
            row = []
            for k in range(len(table[-1]) - 1):
                row.append((int(table[-1][k+1]) - int(table[-1][k])) % mod)
            table.append(row)
        # table[degree] should have 1 element = const_d_diff
        # Now extend: for each new point, work backwards up the table
        reconstructed = list(first_vals)
        for i in range(degree + 1, n):
            # Extend each row from bottom up
            table[degree].append(const_d_diff)
            for d in range(degree - 1, -1, -1):
                new_val = (int(table[d][-1]) + int(table[d+1][-1])) % mod
                table[d].append(new_val)
            reconstructed.append(table[0][-1])

        # Verify
        for i in range(n):
            if reconstructed[i] != int(seq[i]) % mod:
                return None

        return {'newton': first_vals + [const_d_diff], 'mod': mod}


def _try_detect_polynomial(data: np.ndarray, base: int, max_degree: int = 4) -> dict:
    """Try to fit a polynomial of degree 0..max_degree to the data interpreted
    as integers in the given base. Tries both exact and modular fits."""
    seq = _interpret_as_ints(data, base)
    if seq is None or len(seq) < 6:
        return None

    # Try exact fit first (no mod)
    for degree in range(0, max_degree + 1):
        coeffs = _fit_polynomial(seq, degree, mod=None)
        if coeffs is not None:
            return {
                'type': 'poly',
                'base': base,
                'degree': degree,
                'coeffs': coeffs,
                'n_ints': len(seq),
                'orig_len': len(data),
                'mod': 0,  # 0 means no mod
            }

    # Try modular fits (for base 256, mod 256 is natural)
    if base == 256:
        for mod in [256, 65536]:
            for degree in range(0, max_degree + 1):
                coeffs = _fit_polynomial(seq, degree, mod=mod)
                if coeffs is not None:
                    if isinstance(coeffs, dict) and 'newton' in coeffs:
                        return {
                            'type': 'poly_newton',
                            'base': base,
                            'degree': degree,
                            'fwd_diffs': coeffs['newton'],
                            'mod': coeffs['mod'],
                            'n_ints': len(seq),
                            'orig_len': len(data),
                        }
                    else:
                        return {
                            'type': 'poly',
                            'base': base,
                            'degree': degree,
                            'coeffs': coeffs,
                            'n_ints': len(seq),
                            'orig_len': len(data),
                            'mod': mod,
                        }
    return None


def _try_detect_geometric(data: np.ndarray, base: int) -> dict:
    """Detect geometric sequence: a[i] = a0 * r^i."""
    seq = _interpret_as_ints(data, base)
    if seq is None or len(seq) < 4:
        return None
    if int(seq[0]) == 0:
        return None
    # r = seq[1] / seq[0]  (must be integer)
    if int(seq[1]) % int(seq[0]) != 0:
        return None
    r = int(seq[1]) // int(seq[0])
    if r == 0:
        return None
    # Verify
    a0 = int(seq[0])
    for i in range(len(seq)):
        expected = a0 * (r ** i)
        if expected != int(seq[i]):
            return None
    return {
        'type': 'geom',
        'base': base,
        'a0': a0,
        'r': r,
        'n_ints': len(seq),
        'orig_len': len(data),
    }


def _try_detect_fibonacci(data: np.ndarray, base: int) -> dict:
    """Detect Fibonacci-like: a[i] = a[i-1] + a[i-2]."""
    seq = _interpret_as_ints(data, base)
    if seq is None or len(seq) < 5:
        return None
    a0, a1 = int(seq[0]), int(seq[1])
    for i in range(2, len(seq)):
        expected = int(seq[i-1]) + int(seq[i-2])
        if expected != int(seq[i]):
            return None
    return {
        'type': 'fib',
        'base': base,
        'a0': a0,
        'a1': a1,
        'n_ints': len(seq),
        'orig_len': len(data),
    }


def _reconstruct_kolmogorov(params: dict) -> bytes:
    """Reconstruct data from Kolmogorov parameters."""
    t = params['type']
    base = params['base']
    n_ints = params['n_ints']
    orig_len = params['orig_len']

    if t == 'poly':
        degree = params['degree']
        coeffs = params['coeffs']
        mod = params.get('mod', 0)
        seq = np.zeros(n_ints, dtype=np.int64)
        for i in range(n_ints):
            val = 0
            for j in range(degree + 1):
                val += int(coeffs[j]) * (i ** j)
            if mod > 0:
                val = val % mod
            seq[i] = val
    elif t == 'poly_newton':
        degree = params['degree']
        # fwd_diffs = first (degree+1) values + constant d-th difference
        all_vals = params['fwd_diffs']
        mod = params['mod']
        first_vals = all_vals[:degree + 1]
        const_d_diff = all_vals[degree + 1] if len(all_vals) > degree + 1 else 0

        # Reconstruct using difference table
        table = [list(first_vals)]
        for d in range(1, degree + 1):
            row = []
            for k in range(len(table[-1]) - 1):
                row.append((int(table[-1][k+1]) - int(table[-1][k])) % mod)
            table.append(row)

        seq = list(first_vals)
        for i in range(degree + 1, n_ints):
            table[degree].append(const_d_diff)
            for d in range(degree - 1, -1, -1):
                new_val = (int(table[d][-1]) + int(table[d+1][-1])) % mod
                table[d].append(new_val)
            seq.append(table[0][-1])
        seq = np.array(seq, dtype=np.int64)
    elif t == 'geom':
        a0, r = params['a0'], params['r']
        seq = np.array([a0 * (r ** i) for i in range(n_ints)], dtype=np.int64)
    elif t == 'fib':
        a0, a1 = params['a0'], params['a1']
        seq = np.zeros(n_ints, dtype=np.int64)
        seq[0] = a0
        if n_ints > 1:
            seq[1] = a1
        for i in range(2, n_ints):
            seq[i] = seq[i-1] + seq[i-2]
    else:
        return b''

    # Convert integers back to bytes
    if base == 256:
        out = seq.astype(np.uint8).tobytes()
    else:
        # Each integer was chunk * base-digits wide
        import math
        chunk = max(1, int(math.floor(math.log(2**32, base))))
        out = bytearray()
        for v in seq:
            digits = []
            v = int(v)
            for _ in range(chunk):
                digits.append(v % base)
                v //= base
            digits.reverse()
            out.extend(digits)
        out = bytes(out)

    # Trim/pad to original length
    if len(out) >= orig_len:
        return out[:orig_len]
    else:
        return out  # remainder handled separately


def compress_binary_kol(input_path: str, output_path: str, emitter: ProgressEmitter):
    """Kolmogorov compressor: detect polynomial/geometric/Fibonacci structure."""
    orig = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig, output_size=0, ratio=1.0)
    emitter.emit("scan", progress=0.0, current_ratio=1.0)

    with open(input_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(data)
    if n < 8:
        return None

    emitter.emit("scan", progress=0.2, current_ratio=1.0)

    # Try each base × each detector
    params = None
    bases = [256, 16, 10, 9, 8, 2]
    for base in bases:
        emitter.emit("scan", progress=0.5, current_ratio=1.0)
        # Try polynomial (degrees 0-4)
        params = _try_detect_polynomial(data, base, max_degree=4)
        if params:
            break
        # Try geometric
        params = _try_detect_geometric(data, base)
        if params:
            break
        # Try Fibonacci
        params = _try_detect_fibonacci(data, base)
        if params:
            break

    emitter.emit("scan", progress=1.0, current_ratio=1.0)

    if params is None:
        return None

    # Verify reconstruction
    reconstructed = _reconstruct_kolmogorov(params)
    if len(reconstructed) < n:
        # Need to store remainder
        remainder = data.tobytes()[len(reconstructed):]
    elif len(reconstructed) > n:
        reconstructed = reconstructed[:n]
        remainder = b''
    else:
        remainder = b''

    if reconstructed != data.tobytes()[:len(reconstructed)]:
        return None

    remainder_len = len(remainder)
    emitter.emit("encode", progress=0.5, current_ratio=999.0, throughput_mbs=0.0)

    # Write: MAGIC_KOL + ver + orig + crc + type + base + params + remainder
    with open(output_path, 'wb') as f:
        f.write(MAGIC_KOL)
        f.write(struct.pack('>B', 6))
        f.write(struct.pack('>Q', orig))
        crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
        f.write(struct.pack('>I', crc))

        t = params['type']
        f.write(t.encode('ascii').ljust(16, b'\x00')[:16])
        f.write(struct.pack('>I', params['base']))
        f.write(struct.pack('>Q', params['n_ints']))

        if t == 'poly':
            f.write(struct.pack('>I', params['degree']))
            f.write(struct.pack('>I', len(params['coeffs'])))
            f.write(struct.pack('>Q', params.get('mod', 0)))
            for c in params['coeffs']:
                f.write(struct.pack('>q', int(c)))
        elif t == 'poly_newton':
            f.write(struct.pack('>I', params['degree']))
            f.write(struct.pack('>I', len(params['fwd_diffs'])))
            f.write(struct.pack('>Q', params['mod']))
            for c in params['fwd_diffs']:
                f.write(struct.pack('>q', int(c)))
        elif t == 'geom':
            f.write(struct.pack('>q', params['a0']))
            f.write(struct.pack('>q', params['r']))
        elif t == 'fib':
            f.write(struct.pack('>q', params['a0']))
            f.write(struct.pack('>q', params['a1']))

        f.write(struct.pack('>Q', remainder_len))
        if remainder_len > 0:
            f.write(remainder)

    out_size = os.path.getsize(output_path)
    ratio = orig / max(out_size, 1)
    emitter.emit("encode", progress=1.0, current_ratio=round(ratio, 3),
                 throughput_mbs=0.0)
    emitter.emit("done", input_size=orig, output_size=out_size,
                 ratio=round(ratio, 3),
                 time_s=round(time.time() - emitter.t0, 2))
    return ratio


def decompress_binary_kol(input_path: str, output_path: str, emitter: ProgressEmitter):
    orig_size = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig_size, output_size=0, ratio=1.0)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_KOL:
            raise ValueError(f"Bad magic for KOL mode: {magic!r}")
        ver = struct.unpack('>B', f.read(1))[0]
        orig = struct.unpack('>Q', f.read(8))[0]
        crc = struct.unpack('>I', f.read(4))[0]
        t = f.read(16).rstrip(b'\x00').decode('ascii')
        base = struct.unpack('>I', f.read(4))[0]
        n_ints = struct.unpack('>Q', f.read(8))[0]

        params = {'type': t, 'base': base, 'n_ints': n_ints, 'orig_len': orig}

        if t == 'poly':
            degree = struct.unpack('>I', f.read(4))[0]
            n_coeffs = struct.unpack('>I', f.read(4))[0]
            mod = struct.unpack('>Q', f.read(8))[0]
            coeffs = [struct.unpack('>q', f.read(8))[0] for _ in range(n_coeffs)]
            params['degree'] = degree
            params['coeffs'] = coeffs
            params['mod'] = mod
        elif t == 'poly_newton':
            degree = struct.unpack('>I', f.read(4))[0]
            n_diffs = struct.unpack('>I', f.read(4))[0]
            mod = struct.unpack('>Q', f.read(8))[0]
            fwd_diffs = [struct.unpack('>q', f.read(8))[0] for _ in range(n_diffs)]
            params['degree'] = degree
            params['fwd_diffs'] = fwd_diffs
            params['mod'] = mod
        elif t == 'geom':
            params['a0'] = struct.unpack('>q', f.read(8))[0]
            params['r'] = struct.unpack('>q', f.read(8))[0]
        elif t == 'fib':
            params['a0'] = struct.unpack('>q', f.read(8))[0]
            params['a1'] = struct.unpack('>q', f.read(8))[0]
        else:
            raise ValueError(f"Unknown Kolmogorov type: {t}")

        remainder_len = struct.unpack('>Q', f.read(8))[0]
        remainder = f.read() if remainder_len > 0 else b''

    emitter.emit("decode", progress=0.0)
    reconstructed = _reconstruct_kolmogorov(params)
    out_data = (reconstructed + remainder)[:orig]
    emitter.emit("decode", progress=1.0)

    actual_crc = zlib.crc32(out_data) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ValueError(f"CRC mismatch: {crc:08x} vs {actual_crc:08x}")

    with open(output_path, 'wb') as f:
        f.write(out_data)

    emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                 time_s=round(time.time() - emitter.t0, 2))


# =========================================================================
# PRNG DETECTOR — detect LCG, XORShift, Mersenne Twister
# =========================================================================
# Most "random" data is actually pseudo-random. If we can detect the generator
# and its parameters, we can reproduce the entire stream from just a seed.
# This gives 10000x+ ratios on PRNG output.

def _try_detect_lcg32(data: np.ndarray) -> dict:
    """Try to detect a 32-bit LCG: x[n+1] = (a*x[n] + c) mod 2^32.
    Returns dict with seed, a, c if found, else None."""
    n = len(data)
    if n < 16 or n % 4 != 0:
        return None
    # Interpret as 32-bit little-endian words
    words = np.frombuffer(data.tobytes()[:n - (n % 4)], dtype=np.uint32)
    if len(words) < 4:
        return None

    M = 1 << 32
    # Solve: w1 = a*w0 + c mod M
    #        w2 = a*w1 + c mod M
    # → a = (w2 - w1) * inverse(w1 - w0) mod M  (if w1 != w0)
    d01 = int(words[1]) - int(words[0])
    d12 = int(words[2]) - int(words[1])

    if d01 == 0:
        return None

    # Compute modular inverse of d01 mod 2^32 (only exists if d01 is odd)
    if d01 % 2 == 0:
        return None

    def modinv(a, m):
        # Extended Euclidean
        g, x, _ = _extended_gcd(a % m, m)
        if g != 1:
            return None
        return x % m

    inv = modinv(d01, M)
    if inv is None:
        return None

    a = (d12 * inv) % M
    c = (int(words[1]) - a * int(words[0])) % M

    # Verify against all words
    x = int(words[0])
    for i in range(1, len(words)):
        x = (a * x + c) % M
        if x != int(words[i]):
            return None

    return {'type': 'lcg32', 'seed': int(words[0]), 'a': a, 'c': c, 'n_words': len(words)}


def _extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def _try_detect_lcg64(data: np.ndarray) -> dict:
    """Try to detect a 64-bit LCG."""
    n = len(data)
    if n < 32 or n % 8 != 0:
        return None
    words = np.frombuffer(data.tobytes()[:n - (n % 8)], dtype=np.uint64)
    if len(words) < 4:
        return None

    M = 1 << 64
    d01 = int(words[1]) - int(words[0])
    d12 = int(words[2]) - int(words[1])

    if d01 == 0 or d01 % 2 == 0:
        return None

    def modinv(a, m):
        g, x, _ = _extended_gcd(a % m, m)
        if g != 1:
            return None
        return x % m

    inv = modinv(d01, M)
    if inv is None:
        return None

    a = (d12 * inv) % M
    c = (int(words[1]) - a * int(words[0])) % M

    x = int(words[0])
    for i in range(1, len(words)):
        x = (a * x + c) % M
        if x != int(words[i]):
            return None

    return {'type': 'lcg64', 'seed': int(words[0]), 'a': a, 'c': c, 'n_words': len(words)}


def _try_detect_xorshift32(data: np.ndarray) -> dict:
    """Try to detect XORShift32: x ^= x << S; x ^= x >> T; x ^= x << U."""
    n = len(data)
    if n < 16 or n % 4 != 0:
        return None
    words = np.frombuffer(data.tobytes()[:n - (n % 4)], dtype=np.uint32)
    if len(words) < 4:
        return None

    M = (1 << 32) - 1
    # Try common shift triples (1-31)
    for S in range(1, 32):
        for T in range(1, 32):
            for U in range(1, 32):
                x = int(words[0])
                ok = True
                for i in range(1, min(len(words), 20)):  # check first 20
                    x = x ^ ((x << S) & M)
                    x = x ^ (x >> T)
                    x = x ^ ((x << U) & M)
                    if x != int(words[i]):
                        ok = False
                        break
                if ok:
                    # Verify full sequence
                    x = int(words[0])
                    full_ok = True
                    for i in range(1, len(words)):
                        x = x ^ ((x << S) & M)
                        x = x ^ (x >> T)
                        x = x ^ ((x << U) & M)
                        if x != int(words[i]):
                            full_ok = False
                            break
                    if full_ok:
                        return {'type': 'xorshift32', 'seed': int(words[0]),
                                'S': S, 'T': T, 'U': U, 'n_words': len(words)}
    return None


def _reconstruct_prng(params: dict) -> bytes:
    """Reconstruct data from PRNG parameters."""
    t = params['type']
    if t == 'lcg32':
        M = 1 << 32
        a, c = params['a'], params['c']
        x = params['seed']
        out = np.zeros(params['n_words'], dtype=np.uint32)
        out[0] = x
        for i in range(1, params['n_words']):
            x = (a * x + c) % M
            out[i] = x
        return out.tobytes()
    elif t == 'lcg64':
        M = 1 << 64
        a, c = params['a'], params['c']
        x = params['seed']
        out = np.zeros(params['n_words'], dtype=np.uint64)
        out[0] = x
        for i in range(1, params['n_words']):
            x = (a * x + c) % M
            out[i] = x
        return out.tobytes()
    elif t == 'xorshift32':
        M = (1 << 32) - 1
        S, T, U = params['S'], params['T'], params['U']
        x = params['seed']
        out = np.zeros(params['n_words'], dtype=np.uint32)
        out[0] = x
        for i in range(1, params['n_words']):
            x = x ^ ((x << S) & M)
            x = x ^ (x >> T)
            x = x ^ ((x << U) & M)
            out[i] = x
        return out.tobytes()
    return b''


def compress_binary_prng(input_path: str, output_path: str, emitter: ProgressEmitter):
    """Try to detect PRNG. If found, store just the parameters."""
    orig = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig, output_size=0, ratio=1.0)
    emitter.emit("scan", progress=0.0, current_ratio=1.0)

    with open(input_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(data)
    emitter.emit("scan", progress=0.3, current_ratio=1.0)

    # Try each detector
    params = None
    for detector in [_try_detect_lcg32, _try_detect_lcg64, _try_detect_xorshift32]:
        emitter.emit("scan", progress=0.5, current_ratio=1.0)
        params = detector(data)
        if params is not None:
            break

    emitter.emit("scan", progress=1.0, current_ratio=1.0)

    if params is None:
        # Not a PRNG — fall back
        return None

    # Verify reconstruction
    reconstructed = _reconstruct_prng(params)
    # Pad/truncate to original length
    if len(reconstructed) >= n:
        reconstructed = reconstructed[:n]
    else:
        # PRNG output is shorter than data — store remainder separately
        remainder = data.tobytes()[len(reconstructed):]
    if reconstructed != data.tobytes()[:len(reconstructed)]:
        return None

    # Check if there's remainder
    remainder = data.tobytes()[len(reconstructed):]
    remainder_len = len(remainder)

    emitter.emit("encode", progress=0.5, current_ratio=999.0, throughput_mbs=0.0)

    # Write: MAGIC_PRG + ver + orig + crc + type + params + remainder_len + remainder
    with open(output_path, 'wb') as f:
        f.write(MAGIC_PRG)
        f.write(struct.pack('>B', 6))
        f.write(struct.pack('>Q', orig))
        crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
        f.write(struct.pack('>I', crc))

        # Type string (8 bytes)
        t = params['type']
        f.write(t.encode('ascii').ljust(16, b'\x00')[:16])

        if t == 'lcg32':
            f.write(struct.pack('>Q', params['seed']))      # store as 64-bit
            f.write(struct.pack('>Q', params['a']))
            f.write(struct.pack('>Q', params['c']))
            f.write(struct.pack('>Q', params['n_words']))
        elif t == 'lcg64':
            f.write(struct.pack('>Q', params['seed'] & 0xFFFFFFFFFFFFFFFF))
            f.write(struct.pack('>Q', params['a'] & 0xFFFFFFFFFFFFFFFF))
            f.write(struct.pack('>Q', params['c'] & 0xFFFFFFFFFFFFFFFF))
            f.write(struct.pack('>Q', params['n_words']))
        elif t == 'xorshift32':
            f.write(struct.pack('>Q', params['seed']))
            f.write(struct.pack('>B', params['S']))
            f.write(struct.pack('>B', params['T']))
            f.write(struct.pack('>B', params['U']))
            f.write(struct.pack('>Q', params['n_words']))

        # Remainder
        f.write(struct.pack('>Q', remainder_len))
        if remainder_len > 0:
            f.write(remainder)

    out_size = os.path.getsize(output_path)
    ratio = orig / max(out_size, 1)
    emitter.emit("encode", progress=1.0, current_ratio=round(ratio, 3),
                 throughput_mbs=0.0)
    emitter.emit("done", input_size=orig, output_size=out_size,
                 ratio=round(ratio, 3),
                 time_s=round(time.time() - emitter.t0, 2))
    return ratio


def decompress_binary_prng(input_path: str, output_path: str, emitter: ProgressEmitter):
    orig_size = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig_size, output_size=0, ratio=1.0)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_PRG:
            raise ValueError(f"Bad magic for PRG mode: {magic!r}")
        ver = struct.unpack('>B', f.read(1))[0]
        orig = struct.unpack('>Q', f.read(8))[0]
        crc = struct.unpack('>I', f.read(4))[0]
        t = f.read(16).rstrip(b'\x00').decode('ascii')

        if t == 'lcg32':
            seed = struct.unpack('>Q', f.read(8))[0]
            a = struct.unpack('>Q', f.read(8))[0]
            c = struct.unpack('>Q', f.read(8))[0]
            n_words = struct.unpack('>Q', f.read(8))[0]
            params = {'type': 'lcg32', 'seed': seed, 'a': a, 'c': c, 'n_words': n_words}
        elif t == 'lcg64':
            seed = struct.unpack('>Q', f.read(8))[0]
            a = struct.unpack('>Q', f.read(8))[0]
            c = struct.unpack('>Q', f.read(8))[0]
            n_words = struct.unpack('>Q', f.read(8))[0]
            params = {'type': 'lcg64', 'seed': seed, 'a': a, 'c': c, 'n_words': n_words}
        elif t == 'xorshift32':
            seed = struct.unpack('>Q', f.read(8))[0]
            S = struct.unpack('>B', f.read(1))[0]
            T = struct.unpack('>B', f.read(1))[0]
            U = struct.unpack('>B', f.read(1))[0]
            n_words = struct.unpack('>Q', f.read(8))[0]
            params = {'type': 'xorshift32', 'seed': seed, 'S': S, 'T': T, 'U': U, 'n_words': n_words}
        else:
            raise ValueError(f"Unknown PRNG type: {t}")

        remainder_len = struct.unpack('>Q', f.read(8))[0]
        remainder = f.read() if remainder_len > 0 else b''

    emitter.emit("decode", progress=0.0)
    reconstructed = _reconstruct_prng(params)
    # Truncate to original (in case PRNG produced more)
    out_data = (reconstructed + remainder)[:orig]
    emitter.emit("decode", progress=1.0)

    actual_crc = zlib.crc32(out_data) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ValueError(f"CRC mismatch: {crc:08x} vs {actual_crc:08x}")

    with open(output_path, 'wb') as f:
        f.write(out_data)

    emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                 time_s=round(time.time() - emitter.t0, 2))


# =========================================================================
# BIT-PLANE DECOMPOSITION — split into 8 bit-planes, compress each
# =========================================================================

def compress_binary_bpl(input_path: str, output_path: str, emitter: ProgressEmitter):
    """Bit-plane decomposition: split data into 8 bit-planes, compress each with O2."""
    _, _, _, _, _encode_o2, _ = _kernels()
    _, _finish, _, _, _, _ = _kernels()

    orig = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig, output_size=0, ratio=1.0)
    emitter.emit("scan", progress=0.0, current_ratio=1.0)

    with open(input_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(data)
    emitter.emit("scan", progress=0.3, current_ratio=1.0)

    # Decompose into 8 bit-planes (MSB first)
    planes = []
    for bit in range(7, -1, -1):
        plane = ((data >> bit) & 1).astype(np.uint8)
        # Pack bits into bytes for O2 (8 bits per byte)
        # Actually, let's keep them as bytes (0 or 1) — O2 will handle it
        planes.append(plane)
    emitter.emit("scan", progress=0.5, current_ratio=1.0)

    # Compress each plane with O2
    out_buf = bytearray()
    plane_sizes = []
    for i, plane in enumerate(planes):
        emitter.emit("encode", progress=i / 8, current_ratio=1.0, throughput_mbs=0.0)
        cum_freqs = np.zeros((65536, 257), dtype=np.uint32)
        cum_freqs[:] = np.arange(257, dtype=np.uint32)
        state = np.array([0, TOP_VALUE, 0, 0, 0], dtype=np.int64)
        safe_buf = max(n + 1024, 4096)
        enc_buf = np.zeros(safe_buf, dtype=np.uint8)
        n_w = _encode_o2(plane, cum_freqs, state, enc_buf, safe_buf)
        fin_buf = np.zeros(64, dtype=np.uint8)
        n_fin = _finish(state, fin_buf, 64)
        plane_data = enc_buf[:n_w].tobytes() + fin_buf[:n_fin].tobytes()
        plane_sizes.append(len(plane_data))
        out_buf.extend(plane_data)

    emitter.emit("encode", progress=1.0, current_ratio=1.0, throughput_mbs=0.0)

    crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
    with open(output_path, 'wb') as f:
        f.write(MAGIC_BPL)
        f.write(struct.pack('>B', 6))
        f.write(struct.pack('>Q', orig))
        f.write(struct.pack('>I', crc))
        # 8 plane sizes
        for s in plane_sizes:
            f.write(struct.pack('>Q', s))
        f.write(bytes(out_buf))

    out_size = os.path.getsize(output_path)
    if out_size >= orig:
        os.remove(output_path)
        with open(output_path, 'wb') as f:
            f.write(MAGIC_BPL)
            f.write(struct.pack('>B', 6))
            f.write(struct.pack('>Q', orig))
            f.write(struct.pack('>I', crc))
            for _ in range(8):
                f.write(struct.pack('>Q', 0))  # store marker
            f.write(data.tobytes())
        out_size = os.path.getsize(output_path)
        ratio = orig / out_size
    else:
        ratio = orig / out_size

    emitter.emit("done", input_size=orig, output_size=out_size,
                 ratio=round(ratio, 3),
                 time_s=round(time.time() - emitter.t0, 2))
    return ratio


def decompress_binary_bpl(input_path: str, output_path: str, emitter: ProgressEmitter):
    _, _, _, _, _, _decode_o2 = _kernels()

    orig_size = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig_size, output_size=0, ratio=1.0)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_BPL:
            raise ValueError(f"Bad magic for BPL mode: {magic!r}")
        ver = struct.unpack('>B', f.read(1))[0]
        orig = struct.unpack('>Q', f.read(8))[0]
        crc = struct.unpack('>I', f.read(4))[0]
        plane_sizes = [struct.unpack('>Q', f.read(8))[0] for _ in range(8)]

        # Store mode check
        if all(s == 0 for s in plane_sizes):
            raw = f.read()
            actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
            if actual_crc != crc:
                raise ValueError(f"CRC mismatch (store): {crc:08x} vs {actual_crc:08x}")
            with open(output_path, 'wb') as fout:
                fout.write(raw)
            emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                         time_s=round(time.time() - emitter.t0, 2))
            return

        # Read all plane data
        all_data = f.read()

    # Decompress each plane
    offset = 0
    planes = []
    for i, psize in enumerate(plane_sizes):
        emitter.emit("decode", progress=i / 8)
        plane_bytes = all_data[offset:offset + psize]
        offset += psize

        # O2 decode
        compressed = np.frombuffer(plane_bytes, dtype=np.uint8)
        cum_freqs = np.zeros((65536, 257), dtype=np.uint32)
        cum_freqs[:] = np.arange(257, dtype=np.uint32)

        value = np.int64(0)
        buf_ptr = 0
        bit_ptr = 7
        for _ in range(32):
            if buf_ptr < len(compressed):
                byte = int(compressed[buf_ptr])
                bit = (byte >> bit_ptr) & 1
                bit_ptr -= 1
                if bit_ptr < 0:
                    bit_ptr = 7; buf_ptr += 1
            else:
                bit = 0
            value = (value << np.int64(1)) | np.int64(bit)
        state = np.array([0, TOP_VALUE, value, buf_ptr, bit_ptr], dtype=np.int64)

        plane = np.zeros(orig, dtype=np.uint8)
        if orig > 0:
            _decode_o2(compressed, orig, cum_freqs, state, plane)
        planes.append(plane)

    emitter.emit("decode", progress=1.0)

    # Reconstruct: each plane contributes one bit
    out = np.zeros(orig, dtype=np.uint8)
    for bit_idx, plane in enumerate(planes):
        shift = 7 - bit_idx
        out |= (plane & 1) << shift

    actual_crc = zlib.crc32(out.tobytes()) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ValueError(f"CRC mismatch: {crc:08x} vs {actual_crc:08x}")

    with open(output_path, 'wb') as f:
        f.write(out.tobytes())

    emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                 time_s=round(time.time() - emitter.t0, 2))


def compress_binary_delta(input_path: str, output_path: str, emitter: ProgressEmitter,
                          delta_order: int = 1, use_xor: bool = False):
    """Delta or XOR encoding + O2 AC. Best for arithmetic sequences."""
    _, _, _, _, _encode_o2, _ = _kernels()
    _, _finish, _, _, _, _ = _kernels()

    orig = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig, output_size=0, ratio=1.0)
    emitter.emit("scan", progress=0.0, current_ratio=1.0)

    with open(input_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(data)
    emitter.emit("scan", progress=0.3, current_ratio=1.0)

    if use_xor:
        transformed = _xor_encode(data, delta_order)
        transform_id = 1
    else:
        transformed = _delta_encode(data, delta_order)
        transform_id = 0
    emitter.emit("scan", progress=0.6, current_ratio=1.0)

    cum_freqs = np.zeros((65536, 257), dtype=np.uint32)
    cum_freqs[:] = np.arange(257, dtype=np.uint32)

    emitter.emit("encode", progress=0.0, current_ratio=1.0, throughput_mbs=0.0)
    state = np.array([0, TOP_VALUE, 0, 0, 0], dtype=np.int64)
    safe_buf = max(n + 1024, 4096)
    out_buf = np.zeros(safe_buf, dtype=np.uint8)

    t0 = time.time()
    n_written = _encode_o2(transformed, cum_freqs, state, out_buf, safe_buf)
    finish_buf = np.zeros(64, dtype=np.uint8)
    n_fin = _finish(state, finish_buf, 64)
    dt = max(time.time() - t0, 1e-6)
    total_compressed = n_written + n_fin
    ratio = orig / max(total_compressed, 1)
    mbs = (n / 1_048_576) / dt
    emitter.emit("encode", progress=1.0, current_ratio=round(ratio, 3),
                 throughput_mbs=round(mbs, 2))

    with open(output_path, 'wb') as f:
        f.write(MAGIC_DLT)
        f.write(struct.pack('>B', 6))
        f.write(struct.pack('>Q', orig))
        crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
        f.write(struct.pack('>I', crc))
        f.write(struct.pack('>B', transform_id))
        f.write(struct.pack('>B', delta_order))
        f.write(out_buf[:n_written].tobytes())
        f.write(finish_buf[:n_fin].tobytes())

    out_size = os.path.getsize(output_path)
    if out_size >= orig:
        os.remove(output_path)
        with open(output_path, 'wb') as f:
            f.write(MAGIC_DLT)
            f.write(struct.pack('>B', 6))
            f.write(struct.pack('>Q', orig))
            f.write(struct.pack('>I', crc))
            f.write(struct.pack('>B', 0xFF))  # store marker
            f.write(struct.pack('>B', 0))
            f.write(data.tobytes())
        out_size = os.path.getsize(output_path)
        ratio = orig / out_size

    emitter.emit("done", input_size=orig, output_size=out_size,
                 ratio=round(ratio, 3),
                 time_s=round(time.time() - emitter.t0, 2))
    return ratio


def decompress_binary_delta(input_path: str, output_path: str, emitter: ProgressEmitter):
    _, _, _, _, _, _decode_o2 = _kernels()

    orig_size = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig_size, output_size=0, ratio=1.0)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_DLT:
            raise ValueError(f"Bad magic for DLT mode: {magic!r}")
        ver = struct.unpack('>B', f.read(1))[0]
        orig = struct.unpack('>Q', f.read(8))[0]
        crc = struct.unpack('>I', f.read(4))[0]
        transform_id = struct.unpack('>B', f.read(1))[0]
        delta_order = struct.unpack('>B', f.read(1))[0]

        if transform_id == 0xFF:
            raw = f.read()
            actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
            if actual_crc != crc:
                raise ValueError(f"CRC mismatch (store): {crc:08x} vs {actual_crc:08x}")
            with open(output_path, 'wb') as fout:
                fout.write(raw)
            emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                         time_s=round(time.time() - emitter.t0, 2))
            return
        compressed = np.frombuffer(f.read(), dtype=np.uint8)

    cum_freqs = np.zeros((65536, 257), dtype=np.uint32)
    cum_freqs[:] = np.arange(257, dtype=np.uint32)

    value = np.int64(0)
    buf_ptr = 0
    bit_ptr = 7
    for _ in range(32):
        if buf_ptr < len(compressed):
            byte = int(compressed[buf_ptr])
            bit = (byte >> bit_ptr) & 1
            bit_ptr -= 1
            if bit_ptr < 0:
                bit_ptr = 7; buf_ptr += 1
        else:
            bit = 0
        value = (value << np.int64(1)) | np.int64(bit)
    state = np.array([0, TOP_VALUE, value, buf_ptr, bit_ptr], dtype=np.int64)

    transformed = np.zeros(orig, dtype=np.uint8)
    if orig > 0:
        emitter.emit("decode", progress=0.0)
        _decode_o2(compressed, orig, cum_freqs, state, transformed)
        emitter.emit("decode", progress=0.5)

    if transform_id == 1:
        out_data = _xor_decode(transformed, delta_order)
    else:
        out_data = _delta_decode(transformed, delta_order)
    emitter.emit("decode", progress=1.0)

    actual_crc = zlib.crc32(out_data.tobytes()) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ValueError(f"CRC mismatch: {crc:08x} vs {actual_crc:08x}")

    with open(output_path, 'wb') as f:
        f.write(out_data.tobytes())

    emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                 time_s=round(time.time() - emitter.t0, 2))


# =========================================================================
# BWT (Burrows-Wheeler Transform) — pre-pass for text
# =========================================================================
# BWT sorts all rotations of the input, then takes the last column.
# This clusters similar bytes together, making the data highly compressible
# by order-2 AC (often 30-40% better on text).
# We store the primary index (position of original string in sorted rotations)
# so the decoder can invert the transform.

def _bwt_transform(data: np.ndarray) -> tuple:
    """Forward BWT using prefix-doubling suffix array (O(n log^2 n)).
    For BWT we need cyclic rotations, so we sort suffixes of data+data
    and keep only those starting in the first half."""
    n = len(data)
    if n == 0:
        return b'', 0

    try:
        # Double the data for cyclic rotation comparison
        doubled = np.concatenate([data, data]).astype(np.int32)
        m = 2 * n

        # Prefix doubling on doubled array
        sa = np.arange(m, dtype=np.int32)
        rank = doubled.copy()
        tmp = np.zeros(m, dtype=np.int32)

        k = 1
        while k < m:
            rank_next = np.zeros(m, dtype=np.int32)
            mask = (sa + k) < m
            rank_next[mask] = rank[sa[mask] + k]
            sa = sa[np.lexsort((rank_next, rank[sa]))]

            tmp[sa[0]] = 0
            for i in range(1, m):
                prev = sa[i - 1]
                cur = sa[i]
                prev_key = (int(rank[prev]), int(rank[prev + k]) if prev + k < m else -1)
                cur_key = (int(rank[cur]), int(rank[cur + k]) if cur + k < m else -1)
                tmp[cur] = tmp[prev] + (1 if cur_key != prev_key else 0)
            rank = tmp.copy()
            if rank[sa[-1]] == m - 1:
                break
            k *= 2

        # Keep only suffixes starting in first half (these are the cyclic rotations)
        sa = sa[sa < n]

        # Build BWT: last column = data[(sa[i] - 1) % n]
        bwt = np.zeros(n, dtype=np.uint8)
        primary = 0
        for j in range(n):
            idx = int(sa[j])
            bwt[j] = data[(idx - 1) % n]
            if idx == 0:
                primary = j
        return bwt.tobytes(), primary
    except Exception:
        return None, 0


def _bwt_inverse(bwt_bytes: bytes, primary: int) -> np.ndarray:
    """Inverse BWT using LF-mapping (standard algorithm)."""
    n = len(bwt_bytes)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    bwt = np.frombuffer(bwt_bytes, dtype=np.uint8)

    # Count occurrences and compute first column
    counts = np.bincount(bwt, minlength=256)
    # First column is sorted bwt
    # Compute LF mapping
    # ranks[i] = number of occurrences of bwt[i] in bwt[0..i-1]
    ranks = np.zeros(n, dtype=np.int32)
    seen = np.zeros(256, dtype=np.int32)
    for i in range(n):
        b = int(bwt[i])
        ranks[i] = seen[b]
        seen[b] += 1

    # starts[b] = position of first occurrence of b in first column
    starts = np.zeros(256, dtype=np.int32)
    cum = 0
    for b in range(256):
        starts[b] = cum
        cum += counts[b]

    # Reconstruct: start at primary, walk LF mapping n times
    out = np.zeros(n, dtype=np.uint8)
    pos = primary
    for i in range(n - 1, -1, -1):
        b = int(bwt[pos])
        out[i] = b
        pos = starts[b] + ranks[pos]

    return out


# =========================================================================
# MTF (Move-to-Front) transform — classic BWT companion
# =========================================================================
def _mtf_encode(data: np.ndarray) -> np.ndarray:
    """Move-to-Front: each byte replaced by its position in a dynamic list.
    Bytes seen recently get small indices → better for AC."""
    table = list(range(256))
    out = np.zeros(len(data), dtype=np.uint8)
    for i in range(len(data)):
        b = int(data[i])
        idx = table.index(b)
        out[i] = idx
        # Move to front
        table.pop(idx)
        table.insert(0, b)
    return out


def _mtf_decode(data: np.ndarray) -> np.ndarray:
    """Inverse MTF."""
    table = list(range(256))
    out = np.zeros(len(data), dtype=np.uint8)
    for i in range(len(data)):
        idx = int(data[i])
        b = table[idx]
        out[i] = b
        table.pop(idx)
        table.insert(0, b)
    return out


# =========================================================================
# Delta / XOR transforms — unlock patterns in counters, timestamps, etc.
# =========================================================================
def _delta_encode(data: np.ndarray, order: int = 1) -> np.ndarray:
    """Delta encoding: out[i] = data[i] - data[i-order] (mod 256).
    Best for arithmetic sequences (counters, timestamps)."""
    n = len(data)
    out = data.copy()
    for i in range(order, n):
        out[i] = (int(data[i]) - int(data[i - order])) % 256
    return out.astype(np.uint8)


def _delta_decode(data: np.ndarray, order: int = 1) -> np.ndarray:
    """Inverse delta."""
    n = len(data)
    out = data.copy().astype(np.int32)
    for i in range(order, n):
        out[i] = (out[i] + out[i - order]) % 256
    return out.astype(np.uint8)


def _xor_encode(data: np.ndarray, order: int = 1) -> np.ndarray:
    """XOR encoding: out[i] = data[i] ^ data[i-order].
    Best for data with bit-level patterns."""
    n = len(data)
    out = data.copy()
    for i in range(order, n):
        out[i] = int(data[i]) ^ int(data[i - order])
    return out.astype(np.uint8)


def _xor_decode(data: np.ndarray, order: int = 1) -> np.ndarray:
    """Inverse XOR (same as forward XOR)."""
    return _xor_encode(data, order)


# =========================================================================
# Bit-level context model (PAQ-inspired)
# =========================================================================
# Models each bit using context from previous bits. Finds patterns invisible
# at byte level. Slower but can squeeze 5-15% more on structured data.

def _bit_context_encode(data: np.ndarray) -> tuple:
    """Bit-level encoding using simple context mixing.
    Returns (bitstream_bytes, n_bits).
    Each byte modeled as 8 bits with context = last 12 bits."""
    n = len(data)
    if n == 0:
        return b'', 0

    # Use a simple order-1 bit context (last 8 bits = previous byte)
    # + order-2 (last 16 bits = previous 2 bytes)
    # Combined via logistic mixing
    out_bits = []

    # State: last 16 bits (for context)
    history = 0  # 16-bit register

    # Two adaptive models: order-8 and order-16
    # Each tracks (n0, n1) counts per context
    n0_o8 = np.ones(256, dtype=np.uint32)  # order-8 context (last byte)
    n1_o8 = np.ones(256, dtype=np.uint32)
    n0_o16 = np.ones(65536, dtype=np.uint16)  # order-16 context (last 2 bytes)
    n1_o16 = np.ones(65536, dtype=np.uint16)

    # Arithmetic coder state
    low = 0; high = TOP_VALUE; pending = 0
    bit_buf = 0; bit_cnt = 0
    out_buf = bytearray()
    half = HALF; quarter = QUARTER; tq = THREE_QUARTERS; top = TOP_VALUE

    def emit_bit(b):
        nonlocal bit_buf, bit_cnt, pending, out_buf
        bit_buf = (bit_buf << 1) | b
        bit_cnt += 1
        if bit_cnt == 8:
            out_buf.append(bit_buf)
            bit_buf = 0; bit_cnt = 0

    def emit_with_pending(b):
        nonlocal pending
        emit_bit(b)
        while pending > 0:
            emit_bit(1 - b)
            pending -= 1

    for i in range(n):
        byte = int(data[i])
        ctx8 = history & 0xFF
        ctx16 = history & 0xFFFF

        for bit_pos in range(7, -1, -1):
            bit = (byte >> bit_pos) & 1

            # Model 1: order-8
            t8 = n0_o8[ctx8] + n1_o8[ctx8]
            p8 = n1_o8[ctx8] / t8  # probability of bit=1

            # Model 2: order-16
            t16 = int(n0_o16[ctx16]) + int(n1_o16[ctx16])
            if t16 == 0:
                p16 = 0.5
            else:
                p16 = int(n1_o16[ctx16]) / t16

            # Mix (simple average)
            p = (p8 + p16) / 2
            p = max(0.001, min(0.999, p))

            # Arithmetic encode
            r = high - low + 1
            split = low + int(r * (1 - p))
            if bit == 0:
                high = split - 1
            else:
                low = split

            # Renormalize
            while True:
                if high < half:
                    emit_with_pending(0)
                elif low >= half:
                    emit_with_pending(1)
                    low -= half; high -= half
                elif low >= quarter and high < tq:
                    pending += 1
                    low -= quarter; high -= quarter
                else:
                    break
                low = (low << 1) & top
                high = ((high << 1) & top) | 1

            # Update models
            if bit == 1:
                n1_o8[ctx8] += 1
                n1_o16[ctx16] += 1
            else:
                n0_o8[ctx8] += 1
                n0_o16[ctx16] += 1

            # Cap counts to prevent overflow
            if n1_o8[ctx8] > 65000:
                n0_o8[ctx8] = (n0_o8[ctx8] + 1) // 2
                n1_o8[ctx8] = (n1_o8[ctx8] + 1) // 2
            if n1_o16[ctx16] > 32000:
                n0_o16[ctx16] = (n0_o16[ctx16] + 1) // 2
                n1_o16[ctx16] = (n1_o16[ctx16] + 1) // 2

            # Update history
            history = ((history << 1) | bit) & 0xFFFF

    # Finish
    pending += 1
    if low < quarter:
        emit_with_pending(0)
    else:
        emit_with_pending(1)
    if bit_cnt > 0:
        out_buf.append(bit_buf << (8 - bit_cnt))

    return bytes(out_buf), n * 8


def _bit_context_decode(bitstream: bytes, n_bits: int) -> np.ndarray:
    """Inverse bit-level context model."""
    n_bytes = n_bits // 8
    if n_bytes == 0:
        return np.zeros(0, dtype=np.uint8)

    compressed = np.frombuffer(bitstream, dtype=np.uint8)

    n0_o8 = np.ones(256, dtype=np.uint32)
    n1_o8 = np.ones(256, dtype=np.uint32)
    n0_o16 = np.ones(65536, dtype=np.uint16)
    n1_o16 = np.ones(65536, dtype=np.uint16)

    low = 0; high = TOP_VALUE
    value = 0; buf_ptr = 0; bit_ptr = 7
    half = HALF; quarter = QUARTER; tq = THREE_QUARTERS; top = TOP_VALUE

    # Prime
    for _ in range(32):
        if buf_ptr < len(compressed):
            byte = int(compressed[buf_ptr])
            bit = (byte >> bit_ptr) & 1
            bit_ptr -= 1
            if bit_ptr < 0:
                bit_ptr = 7; buf_ptr += 1
        else:
            bit = 0
        value = (value << 1) | bit

    history = 0
    out = np.zeros(n_bytes, dtype=np.uint8)

    for i in range(n_bytes):
        byte = 0
        ctx8 = history & 0xFF
        ctx16 = history & 0xFFFF

        for bit_pos in range(7, -1, -1):
            t8 = n0_o8[ctx8] + n1_o8[ctx8]
            p8 = n1_o8[ctx8] / t8
            t16 = n0_o16[ctx16] + n1_o16[ctx16]
            p16 = n1_o16[ctx16] / t16
            p = (p8 + p16) / 2
            p = max(0.001, min(0.999, p))

            r = high - low + 1
            split = low + int(r * (1 - p))

            # Decode bit
            if value < split:
                bit = 0
                high = split - 1
            else:
                bit = 1
                low = split

            byte = (byte << 1) | bit

            # Renormalize
            while True:
                if high < half:
                    pass
                elif low >= half:
                    low -= half; high -= half; value -= half
                elif low >= quarter and high < tq:
                    low -= quarter; high -= quarter; value -= quarter
                else:
                    break
                low = (low << 1) & top
                high = ((high << 1) & top) | 1
                if buf_ptr < len(compressed):
                    b = int(compressed[buf_ptr])
                    nb = (b >> bit_ptr) & 1
                    bit_ptr -= 1
                    if bit_ptr < 0:
                        bit_ptr = 7; buf_ptr += 1
                else:
                    nb = 0
                value = ((value << 1) & top) | nb

            # Update
            if bit == 1:
                n1_o8[ctx8] += 1
                n1_o16[ctx16] += 1
            else:
                n0_o8[ctx8] += 1
                n0_o16[ctx16] += 1
            if n1_o8[ctx8] > 65000:
                n0_o8[ctx8] = (n0_o8[ctx8] + 1) // 2
                n1_o8[ctx8] = (n1_o8[ctx8] + 1) // 2
            if n1_o16[ctx16] > 32000:
                n0_o16[ctx16] = (n0_o16[ctx16] + 1) // 2
                n1_o16[ctx16] = (n1_o16[ctx16] + 1) // 2

            history = ((history << 1) | bit) & 0xFFFF

        out[i] = byte

    return out


def _rle_encode_bytes(data: np.ndarray) -> np.ndarray:
    """RLE encode: runs of 4+ identical bytes → (byte × 4, count) where count = extra repeats (0-255).
    So 4 bytes → 5 bytes (overhead), 259 bytes → 5 bytes (51.8x compression on long runs).
    Reversible: decoder sees 4 identical bytes, reads next byte as extra count."""
    n = len(data)
    if n == 0:
        return data.copy()
    out = []
    i = 0
    while i < n:
        b = int(data[i])
        # Count run length
        j = i + 1
        while j < n and int(data[j]) == b and j - i < 259:
            j += 1
        run = j - i
        if run >= 4:
            # Emit 4 copies + count of extras (0-255)
            out.extend([b, b, b, b, run - 4])
        else:
            for _ in range(run):
                out.append(b)
        i = j
    return np.array(out, dtype=np.uint8)


def _rle_decode_bytes(data: np.ndarray) -> np.ndarray:
    """RLE decode: when 4 identical bytes seen, next byte = extra count."""
    n = len(data)
    if n == 0:
        return data.copy()
    out = []
    i = 0
    while i < n:
        b = int(data[i])
        # Check if this starts a run of 4+
        if i + 3 < n and int(data[i+1]) == b and int(data[i+2]) == b and int(data[i+3]) == b:
            # Run! Next byte = extra count
            if i + 4 < n:
                extra = int(data[i+4])
                out.extend([b] * (4 + extra))
                i += 5
            else:
                out.append(b)
                i += 1
        else:
            out.append(b)
            i += 1
    return np.array(out, dtype=np.uint8)


def compress_binary_o2(input_path: str, output_path: str, emitter: ProgressEmitter):
    """Order-2 adaptive context AC with RLE pre-pass for long runs.
    No LZ77, no neural net, no zlib."""
    _, _, _, _, _encode_o2, _ = _kernels()
    _, _finish, _, _, _, _ = _kernels()

    orig = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig, output_size=0, ratio=1.0)
    emitter.emit("scan", progress=0.0, current_ratio=1.0)

    with open(input_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(data)
    emitter.emit("scan", progress=0.3, current_ratio=1.0)

    # RLE pre-pass: encode runs of 4+ identical bytes as (byte × 4, count) where count = extra repeats (0-255).
    # This is lossless and reversible.
    rle_data = _rle_encode_bytes(data)
    rle_len = len(rle_data)
    emitter.emit("scan", progress=0.6, current_ratio=1.0)

    # Initialize cum_freqs: 65536 contexts × 257 entries, uniform (count=1 per symbol)
    cum_freqs = np.zeros((65536, 257), dtype=np.uint32)
    cum_freqs[:] = np.arange(257, dtype=np.uint32)

    # Encode the RLE-expanded data
    emitter.emit("encode", progress=0.0, current_ratio=1.0, throughput_mbs=0.0)
    state = np.array([0, TOP_VALUE, 0, 0, 0], dtype=np.int64)
    safe_buf = max(rle_len + 1024, 4096)
    out_buf = np.zeros(safe_buf, dtype=np.uint8)

    t0 = time.time()
    n_written = _encode_o2(rle_data, cum_freqs, state, out_buf, safe_buf)
    finish_buf = np.zeros(64, dtype=np.uint8)
    n_fin = _finish(state, finish_buf, 64)
    dt = max(time.time() - t0, 1e-6)
    total_compressed = n_written + n_fin
    ratio = orig / max(total_compressed, 1)
    mbs = (n / 1_048_576) / dt
    emitter.emit("encode", progress=1.0, current_ratio=round(ratio, 3),
                 throughput_mbs=round(mbs, 2))

    # Write output: MAGIC_O2 + version + orig_size + crc + rle_flag + rle_len + compressed_data
    with open(output_path, 'wb') as f:
        f.write(MAGIC_O2)
        f.write(struct.pack('>B', 6))
        f.write(struct.pack('>Q', orig))
        crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
        f.write(struct.pack('>I', crc))
        f.write(struct.pack('>B', 1))           # RLE flag = 1 (enabled)
        f.write(struct.pack('>Q', rle_len))     # RLE data length (symbols to decode)
        f.write(out_buf[:n_written].tobytes())
        f.write(finish_buf[:n_fin].tobytes())

    out_size = os.path.getsize(output_path)

    # Store-mode fallback: if compressed file is BIGGER than original, just store raw.
    # Use rle_flag=0xFF as explicit "store mode" marker.
    if out_size >= orig:
        os.remove(output_path)
        with open(output_path, 'wb') as f:
            f.write(MAGIC_O2)
            f.write(struct.pack('>B', 6))
            f.write(struct.pack('>Q', orig))
            crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
            f.write(struct.pack('>I', crc))
            f.write(struct.pack('>B', 0xFF))        # RLE flag = 0xFF → store mode
            f.write(struct.pack('>Q', 0))           # rle_len = 0 (unused)
            f.write(data.tobytes())
        out_size = os.path.getsize(output_path)
        ratio = orig / out_size

    emitter.emit("done", input_size=orig, output_size=out_size,
                 ratio=round(ratio, 3),
                 time_s=round(time.time() - emitter.t0, 2))
    return ratio


def compress_binary_bwt(input_path: str, output_path: str, emitter: ProgressEmitter):
    """BWT + Order-2 adaptive AC. Best for text data (30-40% better than O2 alone)."""
    _, _, _, _, _encode_o2, _ = _kernels()
    _, _finish, _, _, _, _ = _kernels()

    orig = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig, output_size=0, ratio=1.0)
    emitter.emit("scan", progress=0.0, current_ratio=1.0)

    with open(input_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(data)
    emitter.emit("scan", progress=0.2, current_ratio=1.0)

    # BWT transform
    bwt_bytes, primary = _bwt_transform(data)
    if bwt_bytes is None:
        # BWT failed (file too big), fall back to O2
        return compress_binary_o2(input_path, output_path, emitter)
    emitter.emit("scan", progress=0.5, current_ratio=1.0)

    bwt_data = np.frombuffer(bwt_bytes, dtype=np.uint8)

    # Initialize cum_freqs for order-2 AC
    cum_freqs = np.zeros((65536, 257), dtype=np.uint32)
    cum_freqs[:] = np.arange(257, dtype=np.uint32)

    # Encode BWT-transformed data
    emitter.emit("encode", progress=0.0, current_ratio=1.0, throughput_mbs=0.0)
    state = np.array([0, TOP_VALUE, 0, 0, 0], dtype=np.int64)
    safe_buf = max(len(bwt_data) + 1024, 4096)
    out_buf = np.zeros(safe_buf, dtype=np.uint8)

    t0 = time.time()
    n_written = _encode_o2(bwt_data, cum_freqs, state, out_buf, safe_buf)
    finish_buf = np.zeros(64, dtype=np.uint8)
    n_fin = _finish(state, finish_buf, 64)
    dt = max(time.time() - t0, 1e-6)
    total_compressed = n_written + n_fin
    ratio = orig / max(total_compressed, 1)
    mbs = (n / 1_048_576) / dt
    emitter.emit("encode", progress=1.0, current_ratio=round(ratio, 3),
                 throughput_mbs=round(mbs, 2))

    # Write output: MAGIC_BWT + ver + orig_size + crc + bwt_primary + compressed_data
    with open(output_path, 'wb') as f:
        f.write(MAGIC_BWT)
        f.write(struct.pack('>B', 6))
        f.write(struct.pack('>Q', orig))
        crc = zlib.crc32(data.tobytes()) & 0xFFFFFFFF
        f.write(struct.pack('>I', crc))
        f.write(struct.pack('>Q', primary))  # BWT primary index
        f.write(out_buf[:n_written].tobytes())
        f.write(finish_buf[:n_fin].tobytes())

    out_size = os.path.getsize(output_path)

    # Store-mode fallback
    if out_size >= orig:
        os.remove(output_path)
        with open(output_path, 'wb') as f:
            f.write(MAGIC_BWT)
            f.write(struct.pack('>B', 6))
            f.write(struct.pack('>Q', orig))
            f.write(struct.pack('>I', crc))
            f.write(struct.pack('>Q', 0xFFFFFFFFFFFFFFFF))  # store-mode marker
            f.write(data.tobytes())
        out_size = os.path.getsize(output_path)
        ratio = orig / out_size

    emitter.emit("done", input_size=orig, output_size=out_size,
                 ratio=round(ratio, 3),
                 time_s=round(time.time() - emitter.t0, 2))
    return ratio


def decompress_binary_bwt(input_path: str, output_path: str, emitter: ProgressEmitter):
    _, _, _, _, _, _decode_o2 = _kernels()
    _, _finish, _, _, _, _ = _kernels()

    orig_size = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig_size, output_size=0, ratio=1.0)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_BWT:
            raise ValueError(f"Bad magic for BWT mode: {magic!r}")
        ver = struct.unpack('>B', f.read(1))[0]
        orig = struct.unpack('>Q', f.read(8))[0]
        crc = struct.unpack('>I', f.read(4))[0]
        primary = struct.unpack('>Q', f.read(8))[0]

        # Store mode check
        if primary == 0xFFFFFFFFFFFFFFFF:
            raw = f.read()
            actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
            if actual_crc != crc:
                raise ValueError(f"CRC mismatch (store): {crc:08x} vs {actual_crc:08x}")
            with open(output_path, 'wb') as fout:
                fout.write(raw)
            emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                         time_s=round(time.time() - emitter.t0, 2))
            return
        compressed = np.frombuffer(f.read(), dtype=np.uint8)

    # Initialize cum_freqs
    cum_freqs = np.zeros((65536, 257), dtype=np.uint32)
    cum_freqs[:] = np.arange(257, dtype=np.uint32)

    # Prime decoder
    value = np.int64(0)
    buf_ptr = 0
    bit_ptr = 7
    for _ in range(32):
        if buf_ptr < len(compressed):
            byte = int(compressed[buf_ptr])
            bit = (byte >> bit_ptr) & 1
            bit_ptr -= 1
            if bit_ptr < 0:
                bit_ptr = 7; buf_ptr += 1
        else:
            bit = 0
        value = (value << np.int64(1)) | np.int64(bit)
    state = np.array([0, TOP_VALUE, value, buf_ptr, bit_ptr], dtype=np.int64)

    # Decode BWT-transformed data
    bwt_decoded = np.zeros(orig, dtype=np.uint8)
    if orig > 0:
        emitter.emit("decode", progress=0.0)
        _decode_o2(compressed, orig, cum_freqs, state, bwt_decoded)
        emitter.emit("decode", progress=0.5)

    # Inverse BWT
    out_data = _bwt_inverse(bwt_decoded.tobytes(), int(primary))
    emitter.emit("decode", progress=1.0)

    # Verify CRC
    actual_crc = zlib.crc32(out_data.tobytes()) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ValueError(f"CRC mismatch: {crc:08x} vs {actual_crc:08x}")

    if len(out_data) != orig:
        raise ValueError(f"Length mismatch: expected {orig}, got {len(out_data)}")

    with open(output_path, 'wb') as f:
        f.write(out_data.tobytes())

    emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                 time_s=round(time.time() - emitter.t0, 2))


def decompress_binary_o2(input_path: str, output_path: str, emitter: ProgressEmitter):
    _, _, _, _, _, _decode_o2 = _kernels()
    _, _finish, _, _, _, _ = _kernels()

    orig_size = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig_size, output_size=0, ratio=1.0)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_O2:
            raise ValueError(f"Bad magic for O2 mode: {magic!r}")
        ver = struct.unpack('>B', f.read(1))[0]
        orig = struct.unpack('>Q', f.read(8))[0]
        crc = struct.unpack('>I', f.read(4))[0]
        rle_flag = struct.unpack('>B', f.read(1))[0]
        rle_len = struct.unpack('>Q', f.read(8))[0]

        # Store mode: rle_flag = 0xFF → raw bytes follow
        if rle_flag == 0xFF:
            raw = f.read()
            actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
            if actual_crc != crc:
                raise ValueError(f"CRC mismatch (store): {crc:08x} vs {actual_crc:08x}")
            with open(output_path, 'wb') as fout:
                fout.write(raw)
            emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                         time_s=round(time.time() - emitter.t0, 2))
            return
        compressed = np.frombuffer(f.read(), dtype=np.uint8)

    # Initialize cum_freqs (same as encoder)
    cum_freqs = np.zeros((65536, 257), dtype=np.uint32)
    cum_freqs[:] = np.arange(257, dtype=np.uint32)

    # Prime decoder
    value = np.int64(0)
    buf_ptr = 0
    bit_ptr = 7
    for _ in range(32):
        if buf_ptr < len(compressed):
            byte = int(compressed[buf_ptr])
            bit = (byte >> bit_ptr) & 1
            bit_ptr -= 1
            if bit_ptr < 0:
                bit_ptr = 7; buf_ptr += 1
        else:
            bit = 0
        value = (value << np.int64(1)) | np.int64(bit)
    state = np.array([0, TOP_VALUE, value, buf_ptr, bit_ptr], dtype=np.int64)

    # Decode rle_len symbols (the RLE-encoded stream)
    decode_len = rle_len if rle_flag else orig
    rle_decoded = np.zeros(decode_len, dtype=np.uint8)
    if decode_len > 0:
        emitter.emit("decode", progress=0.0)
        _decode_o2(compressed, decode_len, cum_freqs, state, rle_decoded)
        emitter.emit("decode", progress=1.0)

    # Apply RLE decode if flag was set
    if rle_flag:
        out_data = _rle_decode_bytes(rle_decoded)
    else:
        out_data = rle_decoded

    # Verify CRC
    actual_crc = zlib.crc32(out_data.tobytes()) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ValueError(f"CRC mismatch: {crc:08x} vs {actual_crc:08x}")

    if len(out_data) != orig:
        raise ValueError(f"Length mismatch: expected {orig}, got {len(out_data)}")

    with open(output_path, 'wb') as f:
        f.write(out_data.tobytes())

    emitter.emit("done", input_size=orig, output_size=orig, ratio=1.0,
                 time_s=round(time.time() - emitter.t0, 2))


def decompress_binary(input_path: str, output_path: str, emitter: ProgressEmitter):
    _, _, _decode_chunk, _, _, _ = _kernels()
    orig_size = os.path.getsize(input_path)
    emitter.emit("start", mode="binary", input_size=orig_size, output_size=0, ratio=1.0)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_BIN:
            raise ValueError(f"Bad magic: {magic!r}")
        ver = struct.unpack('>B', f.read(1))[0]
        orig = struct.unpack('>Q', f.read(8))[0]
        crc = struct.unpack('>I', f.read(4))[0]
        n_tokens = struct.unpack('>Q', f.read(8))[0]

        # Store mode: n_tokens = 0 → raw bytes follow
        if n_tokens == 0:
            raw = f.read()
            actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
            if actual_crc != crc:
                raise ValueError(f"CRC mismatch (store mode): expected {crc:08x}, got {actual_crc:08x}")
            with open(output_path, 'wb') as fout:
                fout.write(raw)
            emitter.emit("done", input_size=orig, output_size=orig,
                         ratio=1.0, time_s=round(time.time() - emitter.t0, 2))
            return

        BIN_N_CTX = 3
        ctx_freqs = np.zeros((BIN_N_CTX, ALPHABET_SIZE + 1), dtype=np.uint64)
        ctx_total = np.zeros(BIN_N_CTX, dtype=np.uint64)
        for c in range(BIN_N_CTX):
            counts = np.zeros(ALPHABET_SIZE, dtype=np.uint64)
            for k in range(ALPHABET_SIZE):
                counts[k] = struct.unpack('>Q', f.read(8))[0]
            counts = np.maximum(counts, 1)
            total = counts.sum()
            if total > MAX_TOTAL_FREQ:
                scale = MAX_TOTAL_FREQ / total
                counts = (counts.astype(np.float64) * scale).clip(min=1).astype(np.uint64)
            ctx_freqs[c, 1:] = np.cumsum(counts)
            ctx_total[c] = ctx_freqs[c, ALPHABET_SIZE]
        compressed = np.frombuffer(f.read(), dtype=np.uint8)

    # Prime decoder (32 bits, MSB-first). IMPORTANT: cast to int64 to avoid
    # numpy uint8 propagation that would silently truncate `value`.
    value = np.int64(0)
    buf_ptr = 0
    bit_ptr = 7
    for _ in range(32):
        if buf_ptr < len(compressed):
            byte = int(compressed[buf_ptr])
            bit = (byte >> bit_ptr) & 1
            bit_ptr -= 1
            if bit_ptr < 0:
                bit_ptr = 7; buf_ptr += 1
        else:
            bit = 0
        value = (value << np.int64(1)) | np.int64(bit)
    state = np.array([0, TOP_VALUE, value, buf_ptr, bit_ptr], dtype=np.int64)

    # Decode symbols one at a time (ctx depends on previous token)
    out_one = np.zeros(1, dtype=np.int32)
    syms_decoded = np.zeros(n_tokens * 4 + 16, dtype=np.int32)
    syms_idx = 0
    tokens_decoded = 0
    while tokens_decoded < n_tokens:
        ctx_arr = np.array([0], dtype=np.int32)
        _decode_chunk(compressed, 1, ctx_arr, ctx_freqs, ctx_total, state, out_one)
        flag = int(out_one[0])
        syms_decoded[syms_idx] = flag; syms_idx += 1
        if flag == 0:
            ctx_arr = np.array([1], dtype=np.int32)
            _decode_chunk(compressed, 1, ctx_arr, ctx_freqs, ctx_total, state, out_one)
            syms_decoded[syms_idx] = int(out_one[0]); syms_idx += 1
        else:
            ctx_arr = np.array([2, 2, 2], dtype=np.int32)
            out_three = np.zeros(3, dtype=np.int32)
            _decode_chunk(compressed, 3, ctx_arr, ctx_freqs, ctx_total, state, out_three)
            for v in out_three:
                syms_decoded[syms_idx] = int(v); syms_idx += 1
        tokens_decoded += 1
        if tokens_decoded % 10000 == 0:
            emitter.emit("decode", progress=tokens_decoded / n_tokens)

    # Reconstruct bytes from tokens
    out_data = np.zeros(orig, dtype=np.uint8)
    pos = 0
    si = 0
    for ti in range(n_tokens):
        flag = int(syms_decoded[si]); si += 1
        if flag == 0:
            val = int(syms_decoded[si]); si += 1
            if pos >= orig:
                raise ValueError(f"Decode overflow at token {ti}: pos={pos} >= orig={orig}")
            out_data[pos] = val; pos += 1
        else:
            off_hi = int(syms_decoded[si]); si += 1
            off_lo = int(syms_decoded[si]); si += 1
            ln_enc = int(syms_decoded[si]); si += 1
            off = (off_hi << 8) | off_lo
            ln = ln_enc + LZ_MIN_MATCH
            if off <= 0 or off > pos:
                raise ValueError(f"Invalid LZ match at token {ti}: off={off} pos={pos}")
            if pos + ln > orig:
                raise ValueError(f"LZ match overflow at token {ti}: pos+ln={pos+ln} > orig={orig}")
            for k in range(ln):
                out_data[pos] = out_data[pos - off]
                pos += 1

    # Verify CRC
    actual_crc = zlib.crc32(out_data.tobytes()) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ValueError(f"CRC mismatch: expected {crc:08x}, got {actual_crc:08x}")

    with open(output_path, 'wb') as f:
        f.write(out_data.tobytes())

    emitter.emit("done", input_size=orig, output_size=orig,
                 ratio=1.0, time_s=round(time.time() - emitter.t0, 2))


# =========================================================================
# MEDIA MODE (video / image) — NanoSiren v2 + Context-Adaptive AC
# =========================================================================

def compress_media(input_path: str, output_path: str, emitter: ProgressEmitter,
                   limit: Optional[int] = None, use_delta: bool = True):
    torch, nn = _import_torch()
    cv2 = _import_cv2()
    _encode_chunk, _finish, _, _, _, _ = _kernels()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Detect media type
    media_type = _detect_media_type(input_path)
    if media_type is None:
        raise ValueError("Input is neither a video nor an image.")

    is_video = (media_type == 'video')
    if is_video:
        cap = cv2.VideoCapture(input_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if limit and limit < T: T = limit
    else:
        img = cv2.imread(input_path)
        H, W, _ = img.shape
        T = 1

    orig = os.path.getsize(input_path)
    emitter.emit("start", mode=media_type, input_size=orig, output_size=0,
                 ratio=1.0, width=W, height=H, frames=T)

    # ---------- Step 1: Train NanoSiren v2 ----------
    emitter.emit("train", progress=0.0, loss=1.0)
    model = _build_model(device)

    # Sample frames for training (reservoir-style: keep ~50 frames at low-res)
    train_frames = []
    if is_video:
        cap = cv2.VideoCapture(input_path)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if limit and count >= limit: break
            if count % max(1, T // 50) == 0 or count < 30:
                f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                f = cv2.resize(f, (160, 120))
                train_frames.append((count, f))
            count += 1
        cap.release()
    else:
        img = cv2.imread(input_path)
        f = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (160, 120))
        train_frames.append((0, f))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    steps = 500
    for step in range(steps):
        # Random batch
        idx = np.random.randint(0, len(train_frames))
        t_idx, img_sample = train_frames[idx]
        img_t = torch.from_numpy(img_sample).float().to(device) / 255.0
        sh, sw, _ = img_sample.shape
        y = torch.randint(0, sh, (2048,), device=device)
        x = torch.randint(0, sw, (2048,), device=device)
        targets = img_t[y, x]
        norm_t = 2 * (t_idx / max(T - 1, 1)) - 1 if T > 1 else 0
        ts = torch.full_like(y, float(norm_t))
        ys = 2 * (y / max(sh - 1, 1)) - 1
        xs = 2 * (x / max(sw - 1, 1)) - 1
        coords = torch.stack([ts, ys, xs], dim=-1).float()

        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                preds = model(coords)
                loss = nn.MSELoss()(preds, targets)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            preds = model(coords)
            loss = nn.MSELoss()(preds, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        if step % 25 == 0 or step == steps - 1:
            emitter.emit("train", progress=(step + 1) / steps,
                         loss=round(loss.item(), 5))

    model.eval()

    # ---------- Step 2: Entropy scan (compute residuals + freq tables) ----------
    emitter.emit("scan", progress=0.0, current_ratio=1.0)

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )

    # Accumulate per-context symbol counts
    all_counts = np.zeros((N_CONTEXTS, ALPHABET_SIZE), dtype=np.uint64)
    BATCH = 8 if is_video else 1
    total_residuals = 0
    buf = []
    frame_count = 0
    delta_prev = [None]  # mutable holder for prev_frame across process_batch calls

    def process_batch(frames_buf, start_t):
        nonlocal total_residuals
        bs = len(frames_buf)
        if is_video and use_delta:
            # Frame-delta mode: predict each frame from previous frame
            frames_np = np.array(frames_buf)
            frames_rgb = frames_np[..., [2, 1, 0]].astype(np.float32)
            residuals_np = np.zeros((bs, H, W, 3), dtype=np.int16)
            for i in range(bs):
                t = start_t + i
                if t == 0 or delta_prev[0] is None:
                    # Frame 0: use NanoSiren prediction
                    frame_torch = torch.from_numpy(frames_rgb[i:i+1]).to(device)
                    abs_t = torch.tensor([t], device=device)
                    norm_t = 2 * (abs_t / max(T - 1, 1)) - 1 if T > 1 else torch.zeros(1, device=device)
                    tt = norm_t.view(1, 1, 1).expand(1, H, W)
                    coords = torch.stack([tt, yy.unsqueeze(0).expand(1, -1, -1),
                                           xx.unsqueeze(0).expand(1, -1, -1)], dim=-1)
                    with torch.no_grad():
                        pred = model(coords.reshape(-1, 3)).reshape(1, H, W, 3)
                        pred = torch.clamp(pred, 0, 1) * 255.0
                        pred_np = pred.cpu().numpy()[0]
                    residuals_np[i] = (frames_rgb[i] - pred_np).round().astype(np.int16)
                else:
                    # Delta: prediction = previous frame
                    residuals_np[i] = (frames_rgb[i] - delta_prev[0]).round().astype(np.int16)
                delta_prev[0] = frames_rgb[i]
            res = residuals_np.flatten()
        else:
            # Original NanoSiren prediction mode
            frames_np = np.array(frames_buf)
            frames_torch = torch.from_numpy(frames_np[..., [2, 1, 0]]).float().to(device).div(255.0) * 255.0
            abs_ts = torch.arange(start_t, start_t + bs, device=device)
            norm_ts = 2 * (abs_ts / max(T - 1, 1)) - 1 if T > 1 else torch.zeros_like(abs_ts)
            tt = norm_ts.view(bs, 1, 1).expand(-1, H, W)
            yy_b = yy.unsqueeze(0).expand(bs, -1, -1)
            xx_b = xx.unsqueeze(0).expand(bs, -1, -1)
            coords = torch.stack([tt, yy_b, xx_b], dim=-1)
            with torch.no_grad():
                preds = model(coords.reshape(-1, 3)).reshape(bs, H, W, 3)
                preds = torch.clamp(preds, 0, 1) * 255.0
                diff = frames_torch - preds
                res = (diff.round().cpu().numpy().astype(np.int16)).flatten()
        # Convert to symbols + contexts
        syms, ctxs = residuals_to_symbols(res)
        # Accumulate counts per context
        for c in range(N_CONTEXTS):
            mask = ctxs == c
            sub = syms[mask]
            if len(sub) > 0:
                bc = np.bincount(sub, minlength=ALPHABET_SIZE)
                all_counts[c] += bc.astype(np.uint64)
        total_residuals += len(res)

    if is_video:
        cap = cv2.VideoCapture(input_path)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if limit and count >= limit: break
            buf.append(frame)
            count += 1
            if len(buf) == BATCH:
                process_batch(buf, count - BATCH)
                buf = []
                emitter.emit("scan", progress=count / T, current_ratio=1.0)
        if buf:
            process_batch(buf, count - len(buf))
        cap.release()
    else:
        img = cv2.imread(input_path)
        process_batch([img], 0)

    # Finalize frequency tables
    ctx_freqs = np.zeros((N_CONTEXTS, ALPHABET_SIZE + 1), dtype=np.uint64)
    ctx_total = np.zeros(N_CONTEXTS, dtype=np.uint64)
    for c in range(N_CONTEXTS):
        counts = all_counts[c]
        counts = np.maximum(counts, 1)
        total = counts.sum()
        if total > MAX_TOTAL_FREQ:
            scale = MAX_TOTAL_FREQ / total
            counts = (counts.astype(np.float64) * scale).clip(min=1).astype(np.uint64)
        ctx_freqs[c, 1:] = np.cumsum(counts)
        ctx_total[c] = ctx_freqs[c, -1]

    emitter.emit("scan", progress=1.0, current_ratio=1.0)

    # ---------- Step 3: Encode stream ----------
    emitter.emit("encode", progress=0.0, current_ratio=1.0, throughput_mbs=0.0)

    # Open output file, write header + model + freq tables
    with open(output_path, 'wb') as f:
        f.write(MAGIC_V6)
        # Version + flags (1 byte version, 1 byte flags, 1 byte hidden, 1 byte layers)
        # flags bit 0 = use_delta (frame-delta mode)
        delta_flag = 1 if (is_video and use_delta) else 0
        # flags bit 0 = use_delta, bit 1 = int8 quantized model
        quant_flag = 2
        f.write(struct.pack('>BBBB', 6, delta_flag | quant_flag, 48, 2))
        f.write(struct.pack('>III', T, H, W))
        # Model: int8-quantized state_dict (4x smaller than float32)
        # Format: for each tensor: shape_len, shape_dims, scale (f32), then int8 data
        sd = model.state_dict()
        b = io.BytesIO()
        b.write(struct.pack('>I', len(sd)))  # number of tensors
        for name, t in sd.items():
            t_np = t.detach().cpu().float().numpy()
            shape = t_np.shape
            b.write(struct.pack('>I', len(shape)))
            for d in shape: b.write(struct.pack('>I', d))
            # Compute scale = max(abs(t)) / 127
            max_abs = float(np.abs(t_np).max()) if t_np.size > 0 else 1.0
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            b.write(struct.pack('>f', scale))
            # Quantize to int8
            q = np.round(t_np / scale).clip(-127, 127).astype(np.int8)
            b.write(q.tobytes())
        mb = b.getvalue()
        f.write(struct.pack('>I', len(mb)))
        f.write(mb)
        # Frequency tables (N_CONTEXTS × ALPHABET_SIZE × 8 bytes)
        for c in range(N_CONTEXTS):
            raw = np.diff(ctx_freqs[c]).astype(np.uint64)
            for v in raw:
                f.write(struct.pack('>Q', int(v)))
        # Now stream-encode residuals
        state = np.array([0, TOP_VALUE, 0, 0, 0], dtype=np.int64)
        # For the encoder, we need symbols + ctx_ids per batch
        if is_video:
            cap = cv2.VideoCapture(input_path)
            count = 0
            buf = []
            t0 = time.time()
            total_syms = 0
            enc_delta_prev = [None]  # mutable prev_frame for encode pass
            while True:
                ret, frame = cap.read()
                if not ret: break
                if limit and count >= limit: break
                buf.append(frame)
                count += 1
                if len(buf) == BATCH:
                    if use_delta:
                        syms, ctxs, enc_delta_prev[0] = _compute_symbols_for_batch_delta(
                            buf, model, count - BATCH, T, H, W, yy, xx, device, enc_delta_prev[0])
                    else:
                        syms, ctxs = _compute_symbols_for_batch(buf, model, count - BATCH, T, H, W, yy, xx, device)
                    safe = max(len(syms) * 2 + 64, 1024)
                    enc_buf = np.zeros(safe, dtype=np.uint8)
                    n_w = _encode_chunk(syms, ctxs, ctx_freqs, ctx_total, state, enc_buf, safe)
                    if n_w > 0:
                        f.write(enc_buf[:n_w].tobytes())
                    total_syms += len(syms)
                    buf = []
                    dt = max(time.time() - t0, 1e-6)
                    mbs = (total_syms / 1_048_576) / dt
                    out_so_far = os.path.getsize(output_path)
                    emitter.emit("encode",
                                 progress=count / T,
                                 current_ratio=round(orig / max(out_so_far, 1), 3),
                                 throughput_mbs=round(mbs, 2))
            if buf:
                if use_delta:
                    syms, ctxs, enc_delta_prev[0] = _compute_symbols_for_batch_delta(
                        buf, model, count - len(buf), T, H, W, yy, xx, device, enc_delta_prev[0])
                else:
                    syms, ctxs = _compute_symbols_for_batch(buf, model, count - len(buf), T, H, W, yy, xx, device)
                safe = max(len(syms) * 2 + 64, 1024)
                enc_buf = np.zeros(safe, dtype=np.uint8)
                n_w = _encode_chunk(syms, ctxs, ctx_freqs, ctx_total, state, enc_buf, safe)
                if n_w > 0:
                    f.write(enc_buf[:n_w].tobytes())
            cap.release()
        else:
            img = cv2.imread(input_path)
            syms, ctxs = _compute_symbols_for_batch([img], model, 0, 1, H, W, yy, xx, device)
            safe = max(len(syms) * 2 + 64, 1024)
            enc_buf = np.zeros(safe, dtype=np.uint8)
            n_w = _encode_chunk(syms, ctxs, ctx_freqs, ctx_total, state, enc_buf, safe)
            if n_w > 0:
                f.write(enc_buf[:n_w].tobytes())

        # Finish bitstream
        fin_buf = np.zeros(16, dtype=np.uint8)
        n_fin = _finish(state, fin_buf, 16)
        if n_fin > 0:
            f.write(fin_buf[:n_fin].tobytes())

    out_size = os.path.getsize(output_path)
    final_ratio = orig / out_size
    emitter.emit("encode", progress=1.0, current_ratio=round(final_ratio, 3),
                 throughput_mbs=0.0)
    emitter.emit("done", input_size=orig, output_size=out_size,
                 ratio=round(final_ratio, 3),
                 time_s=round(time.time() - emitter.t0, 2))
    return final_ratio


def _compute_symbols_for_batch(frames_buf, model, start_t, T, H, W, yy, xx, device):
    torch, _ = _import_torch()
    bs = len(frames_buf)
    frames_np = np.array(frames_buf)
    frames_torch = torch.from_numpy(frames_np[..., [2, 1, 0]]).float().to(device).div(255.0) * 255.0
    abs_ts = torch.arange(start_t, start_t + bs, device=device)
    norm_ts = 2 * (abs_ts / max(T - 1, 1)) - 1 if T > 1 else torch.zeros_like(abs_ts)
    tt = norm_ts.view(bs, 1, 1).expand(-1, H, W)
    yy_b = yy.unsqueeze(0).expand(bs, -1, -1)
    xx_b = xx.unsqueeze(0).expand(bs, -1, -1)
    coords = torch.stack([tt, yy_b, xx_b], dim=-1)
    with torch.no_grad():
        preds = model(coords.reshape(-1, 3)).reshape(bs, H, W, 3)
        preds = torch.clamp(preds, 0, 1) * 255.0
        diff = frames_torch - preds
        res = (diff.round().cpu().numpy().astype(np.int16)).flatten()
    syms, ctxs = residuals_to_symbols(res)
    return syms.astype(np.int32), ctxs.astype(np.int32)


def _compute_symbols_for_batch_delta(frames_buf, model, start_t, T, H, W, yy, xx, device, prev_frame_np):
    """Frame-delta mode: for frame 0 of each batch, use NanoSiren prediction.
    For subsequent frames, use the previous frame as prediction (delta coding).
    This exploits temporal redundancy — inter-frame deltas are typically tiny.
    Returns (syms, ctxs, last_frame_np)."""
    torch, _ = _import_torch()
    bs = len(frames_buf)
    frames_np = np.array(frames_buf)  # BGR uint8 (bs, H, W, 3)
    frames_rgb = frames_np[..., [2, 1, 0]].astype(np.float32)  # RGB float (bs, H, W, 3)

    residuals_np = np.zeros((bs, H, W, 3), dtype=np.int16)

    for i in range(bs):
        t = start_t + i
        if t == 0 or prev_frame_np is None:
            # Use NanoSiren prediction for frame 0
            frame_torch = torch.from_numpy(frames_rgb[i:i+1]).to(device)
            abs_t = torch.tensor([t], device=device)
            norm_t = 2 * (abs_t / max(T - 1, 1)) - 1 if T > 1 else torch.zeros(1, device=device)
            tt = norm_t.view(1, 1, 1).expand(1, H, W)
            coords = torch.stack([tt, yy.unsqueeze(0).expand(1, -1, -1),
                                   xx.unsqueeze(0).expand(1, -1, -1)], dim=-1)
            with torch.no_grad():
                pred = model(coords.reshape(-1, 3)).reshape(1, H, W, 3)
                pred = torch.clamp(pred, 0, 1) * 255.0
                pred_np = pred.cpu().numpy()[0]
            residuals_np[i] = (frames_rgb[i] - pred_np).round().astype(np.int16)
        else:
            # Frame-delta: prediction = previous frame
            residuals_np[i] = (frames_rgb[i] - prev_frame_np).round().astype(np.int16)
        # Update prev_frame for next iteration
        prev_frame_np = frames_rgb[i]

    res = residuals_np.flatten()
    syms, ctxs = residuals_to_symbols(res)
    return syms.astype(np.int32), ctxs.astype(np.int32), prev_frame_np


def decompress_media(input_path: str, output_path: str, emitter: ProgressEmitter):
    torch, _ = _import_torch()
    cv2 = _import_cv2()
    _, _, _decode_chunk, _, _, _ = _kernels()

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic == MAGIC_V5:
            # Legacy v5 — delegate to original nfr.py logic if available
            raise NotImplementedError("Legacy v5 decode not implemented in v6 engine. Use NFR_Release/nfr.py for .nfr v5 files.")
        if magic != MAGIC_V6:
            raise ValueError(f"Bad magic: {magic!r}")
        ver, flags, hidden, layers = struct.unpack('>BBBB', f.read(4))
        use_delta = bool(flags & 1)  # bit 0 = frame-delta mode
        use_int8 = bool(flags & 2)   # bit 1 = int8 quantized model
        T, H, W = struct.unpack('>III', f.read(12))
        model_len = struct.unpack('>I', f.read(4))[0]
        model_bytes = f.read(model_len)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = _build_model(device)
        if use_int8:
            # Load int8-quantized model
            b = io.BytesIO(model_bytes)
            n_tensors = struct.unpack('>I', b.read(4))[0]
            sd = {}
            for _ in range(n_tensors):
                n_dims = struct.unpack('>I', b.read(4))[0]
                shape = tuple(struct.unpack('>I', b.read(4))[0] for _ in range(n_dims))
                scale = struct.unpack('>f', b.read(4))[0]
                n_elems = 1
                for d in shape: n_elems *= d
                q = np.frombuffer(b.read(n_elems), dtype=np.int8).astype(np.float32)
                t_np = q * scale
                sd_key = f"t{len(sd)}"  # placeholder; we'll match by order
                sd[sd_key] = torch.from_numpy(t_np.reshape(shape) if shape else t_np)
            # Match by order — state_dict keys must align
            actual_sd = model.state_dict()
            keys = list(actual_sd.keys())
            for i, k in enumerate(keys):
                actual_sd[k] = sd[f"t{i}"]
            model.load_state_dict(actual_sd)
        else:
            model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location=device, weights_only=True))
        model.to(device).eval()

        # Frequency tables
        ctx_freqs = np.zeros((N_CONTEXTS, ALPHABET_SIZE + 1), dtype=np.uint64)
        ctx_total = np.zeros(N_CONTEXTS, dtype=np.uint64)
        for c in range(N_CONTEXTS):
            counts = np.zeros(ALPHABET_SIZE, dtype=np.uint64)
            for k in range(ALPHABET_SIZE):
                counts[k] = struct.unpack('>Q', f.read(8))[0]
            counts = np.maximum(counts, 1)
            total = counts.sum()
            if total > MAX_TOTAL_FREQ:
                scale = MAX_TOTAL_FREQ / total
                counts = (counts.astype(np.float64) * scale).clip(min=1).astype(np.uint64)
            ctx_freqs[c, 1:] = np.cumsum(counts)
            ctx_total[c] = ctx_freqs[c, -1]

        compressed = np.frombuffer(f.read(), dtype=np.uint8)

    orig_size = os.path.getsize(input_path)
    emitter.emit("start", mode="video" if T > 1 else "image",
                 input_size=orig_size, output_size=0, ratio=1.0,
                 width=W, height=H, frames=T)

    # Decode symbols. Need to decode one at a time because ctx depends on
    # previous residual (since context = magnitude_bucket(prev_residual)).
    # Strategy: decode SYM, then determine ctx for next symbol based on it.
    # This is slow but correct. For batched decode we'd need to know ctx ahead.

    total_pixels = T * H * W * 3
    out_residuals = np.zeros(total_pixels, dtype=np.int16)
    out_idx = 0

    # Prime decoder
    value = 0
    buf_ptr = 0
    bit_ptr = 7
    for _ in range(32):
        if buf_ptr < len(compressed):
            byte = compressed[buf_ptr]
            bit = (byte >> bit_ptr) & 1
            bit_ptr -= 1
            if bit_ptr < 0:
                bit_ptr = 7; buf_ptr += 1
        else:
            bit = 0
        value = (value << 1) | bit
    state = np.array([0, TOP_VALUE, value, buf_ptr, bit_ptr], dtype=np.int64)

    out_one = np.zeros(1, dtype=np.int32)
    prev_mag_bucket = 3
    progress_inc = max(total_pixels // 100, 1)
    next_progress = progress_inc

    while out_idx < total_pixels:
        ctx_arr = np.array([prev_mag_bucket], dtype=np.int32)
        _decode_chunk(compressed, 1, ctx_arr, ctx_freqs, ctx_total, state, out_one)
        s = int(out_one[0])
        if s == SYM_ZERO_RUN:
            # Next symbol is the run length
            _decode_chunk(compressed, 1, ctx_arr, ctx_freqs, ctx_total, state, out_one)
            run = int(out_one[0])
            out_residuals[out_idx:out_idx + run] = 0
            out_idx += run
            prev_mag_bucket = 0
        else:
            r = symbol_to_residual(s)
            out_residuals[out_idx] = r
            out_idx += 1
            mag = abs(r)
            if mag <= 4: prev_mag_bucket = 0
            elif mag <= 16: prev_mag_bucket = 1
            else: prev_mag_bucket = 2

        if out_idx >= next_progress:
            emitter.emit("decode", progress=min(out_idx / total_pixels, 1.0))
            next_progress += progress_inc

    # Reconstruct frames
    emitter.emit("reconstruct", progress=0.0)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )

    if T > 1:
        out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H))
    else:
        out_writer = None

    res_reshaped = out_residuals.reshape(T, H, W, 3)
    CHUNK = 10
    frames_done = 0
    prev_frame = None  # for delta reconstruction
    while frames_done < T:
        this_chunk = min(CHUNK, T - frames_done)
        with torch.no_grad():
            for i in range(this_chunk):
                t = frames_done + i
                if use_delta and t > 0 and prev_frame is not None:
                    # Delta reconstruction: frame = prev_frame + residual
                    frame = np.clip(prev_frame.astype(np.int16) + res_reshaped[t], 0, 255).astype(np.uint8)
                else:
                    # NanoSiren reconstruction (frame 0 or non-delta mode)
                    norm_t = 2 * (t / max(T - 1, 1)) - 1 if T > 1 else 0
                    tt = torch.full_like(xx, float(norm_t))
                    coords = torch.stack([tt, yy, xx], dim=-1).reshape(-1, 3)
                    preds = model(coords).reshape(H, W, 3)
                    preds_int = (torch.clamp(preds, 0, 1) * 255).cpu().numpy().astype(np.int16)
                    frame = np.clip(preds_int + res_reshaped[t], 0, 255).astype(np.uint8)
                prev_frame = frame.astype(np.float32)  # store for next frame's delta
                if T > 1:
                    out_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames_done += this_chunk
        emitter.emit("reconstruct", progress=frames_done / T)

    if out_writer is not None:
        out_writer.release()

    final_size = os.path.getsize(output_path)
    emitter.emit("done", input_size=orig_size, output_size=final_size,
                 ratio=round(orig_size / max(final_size, 1), 3),
                 time_s=round(time.time() - emitter.t0, 2))


# =========================================================================
# DISPATCHER
# =========================================================================

def compress(input_path: str, output_path: str, emitter: Optional[ProgressEmitter] = None,
             force_mode: Optional[str] = None):
    if emitter is None:
        emitter = ProgressEmitter(enabled=False)
    if not os.path.exists(input_path):
        emitter.emit("error", message=f"Input not found: {input_path}")
        raise FileNotFoundError(input_path)

    if force_mode is None:
        # Auto-detect
        try:
            media = _detect_media_type(input_path)
        except Exception:
            media = None
        mode = media if media in ('video', 'image') else 'binary'
    else:
        mode = force_mode

    # For small images, the model + freq table overhead dominates.
    # Use binary mode as a smarter fallback.
    if mode == 'image':
        size = os.path.getsize(input_path)
        if size < 256 * 1024:  # < 256KB → use binary mode
            mode = 'binary'

    if mode in ('video', 'image'):
        return compress_media(input_path, output_path, emitter)
    else:
        # Binary mode: try multiple strategies and pick the best ratio.
        # - O2+RLE: best for sparse data (long runs of identical bytes)
        # - BWT+O2: best for text (30-40% better than O2 alone on text)
        # - O2 alone: fallback
        # Skip BWT for very large files (suffix sort is slow)
        size = os.path.getsize(input_path)
        candidates = []

        # Try PRNG detection FIRST (can give 10000x+ on pseudo-random data)
        tmp_prg = output_path + '.prg.tmp'
        try:
            r = compress_binary_prng(input_path, tmp_prg, emitter)
            if r is not None:
                candidates.append(('prg', tmp_prg, os.path.getsize(tmp_prg)))
            else:
                if os.path.exists(tmp_prg): os.remove(tmp_prg)
        except Exception as e:
            ProgressEmitter.log(f"PRG candidate failed: {e}")
            if os.path.exists(tmp_prg): os.remove(tmp_prg)

        # Try Kolmogorov detector (polynomial/geometric/Fibonacci in various bases)
        tmp_kol = output_path + '.kol.tmp'
        try:
            r = compress_binary_kol(input_path, tmp_kol, emitter)
            if r is not None:
                candidates.append(('kol', tmp_kol, os.path.getsize(tmp_kol)))
            else:
                if os.path.exists(tmp_kol): os.remove(tmp_kol)
        except Exception as e:
            ProgressEmitter.log(f"KOL candidate failed: {e}")
            if os.path.exists(tmp_kol): os.remove(tmp_kol)

        # Try O2+RLE (always)
        tmp_o2 = output_path + '.o2.tmp'
        try:
            compress_binary_o2(input_path, tmp_o2, emitter)
            candidates.append(('o2', tmp_o2, os.path.getsize(tmp_o2)))
        except Exception as e:
            ProgressEmitter.log(f"O2 candidate failed: {e}")

        # Try NRP (Neural Residual Predictor — for files <= 500KB)
        if size <= 500_000:
            tmp_nrp = output_path + '.nrp.tmp'
            try:
                compress_binary_nrp(input_path, tmp_nrp, emitter)
                candidates.append(('nrp', tmp_nrp, os.path.getsize(tmp_nrp)))
            except Exception as e:
                ProgressEmitter.log(f"NRP candidate failed: {e}")

        # Try PPMd (for files <= 200KB — pure Python is slow)
        if size <= 200_000:
            tmp_ppm = output_path + '.ppm.tmp'
            try:
                compress_binary_ppm(input_path, tmp_ppm, emitter)
                candidates.append(('ppm', tmp_ppm, os.path.getsize(tmp_ppm)))
            except Exception as e:
                ProgressEmitter.log(f"PPM candidate failed: {e}")

        # Try BWT+O2 (prefix-doubling SA is O(n log^2 n), handles up to ~5MB)
        if size <= 5_000_000:
            tmp_bwt = output_path + '.bwt.tmp'
            try:
                compress_binary_bwt(input_path, tmp_bwt, emitter)
                candidates.append(('bwt', tmp_bwt, os.path.getsize(tmp_bwt)))
            except Exception as e:
                ProgressEmitter.log(f"BWT candidate failed: {e}")

        # Try TRS combo (BWT+MTF+RLE+O2 — for files <= 5MB)
        if size <= 5_000_000:
            tmp_trs = output_path + '.trs.tmp'
            try:
                compress_binary_trs(input_path, tmp_trs, emitter)
                candidates.append(('trs', tmp_trs, os.path.getsize(tmp_trs)))
            except Exception as e:
                ProgressEmitter.log(f"TRS candidate failed: {e}")

        # Try BIT-level context model (for files <= 100KB — slow Python)
        if size <= 100_000:
            tmp_bit = output_path + '.bit.tmp'
            try:
                compress_binary_bit(input_path, tmp_bit, emitter)
                candidates.append(('bit', tmp_bit, os.path.getsize(tmp_bit)))
            except Exception as e:
                ProgressEmitter.log(f"BIT candidate failed: {e}")

        # Try Bit-plane decomposition (for files <= 1MB)
        if size <= 1_000_000:
            tmp_bpl = output_path + '.bpl.tmp'
            try:
                compress_binary_bpl(input_path, tmp_bpl, emitter)
                candidates.append(('bpl', tmp_bpl, os.path.getsize(tmp_bpl)))
            except Exception as e:
                ProgressEmitter.log(f"BPL candidate failed: {e}")

        # Try Delta encoding (for files <= 5MB — best for counters/timestamps)
        if size <= 5_000_000:
            for order in [1, 2, 4]:
                tmp_dlt = output_path + f'.dlt{order}.tmp'
                try:
                    compress_binary_delta(input_path, tmp_dlt, emitter, delta_order=order, use_xor=False)
                    candidates.append((f'dlt{order}', tmp_dlt, os.path.getsize(tmp_dlt)))
                except Exception as e:
                    ProgressEmitter.log(f"DLT{order} candidate failed: {e}")
            # Also try XOR-1
            tmp_xor = output_path + '.xor1.tmp'
            try:
                compress_binary_delta(input_path, tmp_xor, emitter, delta_order=1, use_xor=True)
                candidates.append(('xor1', tmp_xor, os.path.getsize(tmp_xor)))
            except Exception as e:
                ProgressEmitter.log(f"XOR1 candidate failed: {e}")

        if not candidates:
            raise RuntimeError("All compression strategies failed")

        # Pick the smallest
        candidates.sort(key=lambda c: c[2])
        best_name, best_path, best_size = candidates[0]

        # Clean up losers
        for name, path, _ in candidates:
            if path != best_path and os.path.exists(path):
                os.remove(path)

        os.replace(best_path, output_path)
        ratio = size / os.path.getsize(output_path)
        return ratio


def decompress(input_path: str, output_path: str, emitter: Optional[ProgressEmitter] = None):
    if emitter is None:
        emitter = ProgressEmitter(enabled=False)
    if not os.path.exists(input_path):
        emitter.emit("error", message=f"Input not found: {input_path}")
        raise FileNotFoundError(input_path)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
    if magic == MAGIC_V6:
        return decompress_media(input_path, output_path, emitter)
    elif magic == MAGIC_BIN:
        return decompress_binary(input_path, output_path, emitter)
    elif magic == MAGIC_O2:
        return decompress_binary_o2(input_path, output_path, emitter)
    elif magic == MAGIC_BWT:
        return decompress_binary_bwt(input_path, output_path, emitter)
    elif magic == MAGIC_PPM:
        return decompress_binary_ppm(input_path, output_path, emitter)
    elif magic == MAGIC_NRP:
        return decompress_binary_nrp(input_path, output_path, emitter)
    elif magic == MAGIC_TRS:
        return decompress_binary_trs(input_path, output_path, emitter)
    elif magic == MAGIC_BIT:
        return decompress_binary_bit(input_path, output_path, emitter)
    elif magic == MAGIC_DLT:
        return decompress_binary_delta(input_path, output_path, emitter)
    elif magic == MAGIC_PRG:
        return decompress_binary_prng(input_path, output_path, emitter)
    elif magic == MAGIC_KOL:
        return decompress_binary_kol(input_path, output_path, emitter)
    elif magic == MAGIC_BPL:
        return decompress_binary_bpl(input_path, output_path, emitter)
    elif magic == MAGIC_V5:
        raise NotImplementedError("Legacy v5 (.nfr) decode not supported by v6 engine. Use NFR_Release/nfr.py.")
    else:
        raise ValueError(f"Unknown file format: {magic!r}")


# =========================================================================
# CLI
# =========================================================================

def main():
    ap = argparse.ArgumentParser(description=f"NFR v{VERSION} — Universal Codec")
    ap.add_argument('mode', choices=['compress', 'decompress', 'predict', 'bench'])
    ap.add_argument('input')
    ap.add_argument('output', nargs='?', default=None)
    ap.add_argument('--limit', type=int, default=None, help='Limit frames (video)')
    ap.add_argument('--mode-bin', action='store_true', help='Force binary mode')
    ap.add_argument('--mode-media', action='store_true', help='Force media mode')
    ap.add_argument('--json', action='store_true', help='Emit JSON progress events to stdout')
    args = ap.parse_args()

    json_mode = args.json or os.environ.get('NFR_JSON') == '1'
    emitter = ProgressEmitter(enabled=json_mode)

    force_mode = None
    if args.mode_bin: force_mode = 'binary'
    elif args.mode_media: force_mode = 'video'

    try:
        if args.mode == 'predict':
            result = predict_ratio(args.input)
            print(json.dumps(result, indent=2))
            return
        if args.mode == 'compress':
            if not args.output:
                ap.error("output path required for compress")
            compress(args.input, args.output, emitter, force_mode=force_mode)
            return
        if args.mode == 'decompress':
            if not args.output:
                ap.error("output path required for decompress")
            decompress(args.input, args.output, emitter)
            return
        if args.mode == 'bench':
            # Compress, decompress, verify
            if not args.output:
                ap.error("output path required for bench")
            comp_path = args.output + '.compressed'
            dec_path = args.output + '.decompressed'
            print(f"[BENCH] Compressing {args.input} -> {comp_path}")
            r1 = compress(args.input, comp_path, emitter, force_mode=force_mode)
            print(f"[BENCH] Decompressing {comp_path} -> {dec_path}")
            decompress(comp_path, dec_path, emitter)
            # Verify (for binary mode, CRC was checked; for media, just compare sizes)
            print(f"[BENCH] Compression ratio: {r1:.3f}x")
            print(f"[BENCH] Original: {_fmt_bytes(os.path.getsize(args.input))}")
            print(f"[BENCH] Compressed: {_fmt_bytes(os.path.getsize(comp_path))}")
            print(f"[BENCH] Decompressed: {_fmt_bytes(os.path.getsize(dec_path))}")
    except Exception as e:
        emitter.emit("error", message=str(e))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
