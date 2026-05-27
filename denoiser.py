"""
denoiser.py — Per-call telephone-grade noise canceller (StreamDenoiser).

Operates on INBOUND customer audio (before STT). Does NOT touch the bot's
outbound TTS.

3-stage chain, tuned for 8 kHz phone audio:
  1. Telephone bandpass 300–3400 Hz (Butterworth, scipy SOS):
     kills everything outside the voice band — engine rumble, AC/mains hum,
     line buzz, high-freq hiss, music bleed.
  2. Non-stationary spectral denoise (noisereduce / MMSE Wiener):
     removes broadband + changing noise INSIDE the voice band
     (fans, traffic, a TV in the background). prop_decrease=0.92 is strong
     but conservative enough to avoid chopping the speaker's own voice.
  3. Adaptive noise gate:
     after spectral cleanup, anything still below the running noise floor is
     attenuated with a smooth attack/release envelope (no pumping).

Output is delayed by one ~80 ms chunk. CPU-bound → run via thread pool.
"""
from __future__ import annotations
import numpy as np
import noisereduce as nr
from scipy.signal import butter, sosfilt, sosfilt_zi

# Telephone voice band (PSTN standard).
_BP_LOW_HZ  = 300.0
_BP_HIGH_HZ = 3400.0
_BP_ORDER   = 4

# Noise-floor tracker (exponential): fall fast on quiet frames, rise slowly.
_NF_ATTACK  = 0.20
_NF_RELEASE = 0.02

# Gate behaviour
_GATE_OPEN_DB = 10.0    # open the gate when frame is +10 dB above the floor
_GATE_ATTACK  = 0.40    # how fast the gate opens
_GATE_RELEASE = 0.08    # how fast it closes
_GATE_FLOOR_INIT = 1e-3

# Stationary-mode profiling threshold (RMS of a "silent" frame).
_PROFILE_ENERGY_LIMIT = 0.025


def _design_bandpass(sr: int) -> np.ndarray:
    nyq  = 0.5 * sr
    low  = max(0.01, _BP_LOW_HZ  / nyq)
    high = min(0.99, _BP_HIGH_HZ / nyq)
    return butter(_BP_ORDER, [low, high], btype="bandpass", output="sos")


class StreamDenoiser:
    """Per-call noise canceller (CPU-bound → run via thread pool)."""

    CHUNK_SAMP = 640  # 80 ms @ 8 kHz

    def __init__(
        self,
        sr: int = 8000,
        prop_decrease: float = 0.92,
        profile_sec: float = 2.0,
        stationary: bool = False,
    ) -> None:
        self._sr         = sr
        self._prop       = float(prop_decrease)
        self._stationary = stationary
        self._target     = int(sr * profile_sec)

        # Bandpass filter + persistent state (so frames join seamlessly).
        self._sos   = _design_bandpass(sr)
        self._bp_zi = sosfilt_zi(self._sos)

        # Stationary-mode noise profile
        self._pbuf:    list[np.ndarray] = []
        self._profiled = 0
        self._noise:   np.ndarray | None = None

        # Spectral processing buffers
        self._inbuf  = np.zeros(0, dtype=np.float32)
        self._outbuf = np.zeros(0, dtype=np.float32)

        # Noise-gate state
        self._floor_rms = _GATE_FLOOR_INIT
        self._gate_open = 0.0

    # ── stage 1: bandpass ────────────────────────────────────────────────
    def _bandpass(self, x: np.ndarray) -> np.ndarray:
        y, self._bp_zi = sosfilt(self._sos, x, zi=self._bp_zi)
        return y.astype(np.float32, copy=False)

    # ── stage 2: spectral denoise ────────────────────────────────────────
    def _denoise(self, seg: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            if self._stationary:
                out = nr.reduce_noise(
                    y=seg, y_noise=self._noise, sr=self._sr,
                    stationary=True, prop_decrease=self._prop,
                    n_fft=512, n_jobs=1,
                )
            else:
                out = nr.reduce_noise(
                    y=seg, sr=self._sr,
                    stationary=False, prop_decrease=self._prop,
                    n_fft=512, n_jobs=1, time_constant_s=1.0,
                )
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # ── stage 3: adaptive noise gate ─────────────────────────────────────
    def _gate(self, seg: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(seg * seg))) + 1e-9
        if rms < self._floor_rms:
            self._floor_rms = (1 - _NF_ATTACK) * self._floor_rms + _NF_ATTACK * rms
        else:
            self._floor_rms = (1 - _NF_RELEASE) * self._floor_rms + _NF_RELEASE * rms
        ratio_db = 20.0 * np.log10(rms / max(self._floor_rms, 1e-9))
        target = 1.0 if ratio_db > _GATE_OPEN_DB else 0.0
        coef = _GATE_ATTACK if target > self._gate_open else _GATE_RELEASE
        self._gate_open = (1 - coef) * self._gate_open + coef * target
        return (seg * self._gate_open).astype(np.float32, copy=False)

    def _emit(self, n: int, fallback: np.ndarray) -> bytes:
        out = self._outbuf[:n] if self._outbuf.shape[0] >= n else fallback
        if self._outbuf.shape[0] >= n:
            self._outbuf = self._outbuf[n:]
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return (np.clip(out, -1.0, 1.0) * 32768).astype(np.int16).tobytes()

    # ── main entry ───────────────────────────────────────────────────────
    def feed_sync(self, pcm16: bytes) -> bytes:
        raw   = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        chunk = self._bandpass(raw)

        # Stationary mode: profile silent frames first, then denoise.
        if self._stationary and self._noise is None:
            self._profiled += chunk.shape[0]
            if float(np.sqrt(np.mean(chunk ** 2))) < _PROFILE_ENERGY_LIMIT:
                self._pbuf.append(chunk.copy())
            enough  = sum(a.shape[0] for a in self._pbuf) >= self._target
            timeout = self._profiled >= self._target * 3
            if enough or timeout:
                self._noise = (
                    np.concatenate(self._pbuf) if self._pbuf
                    else np.zeros(self.CHUNK_SAMP, dtype=np.float32)
                )
                self._pbuf.clear()
            return (np.clip(chunk, -1.0, 1.0) * 32768).astype(np.int16).tobytes()

        # Accumulate → denoise → gate → emit.
        self._inbuf = np.concatenate([self._inbuf, chunk])
        if self._inbuf.shape[0] >= self.CHUNK_SAMP:
            seg         = self._inbuf[:self.CHUNK_SAMP]
            self._inbuf = self._inbuf[self.CHUNK_SAMP:]
            try:
                clean = self._denoise(seg)
                gated = self._gate(clean)
            except Exception:
                gated = seg
            self._outbuf = np.concatenate([self._outbuf, gated])

        return self._emit(chunk.shape[0], chunk)
