"""Run the Kriti ONNX export with NumPy + ONNX Runtime only (no NeMo, no PyTorch).

Files expected in the model directory (produced by export_onnx.py):
  encoder.onnx, decoder.onnx, joint.onnx   (or *.int8.onnx)
  model_config.json   preprocessor + decoding settings taken from the checkpoint
  tokens.json         Nepali SentencePiece pieces; index = token id, blank = len(tokens)
  punctuation_head.json  Kriti's acoustic danda head (logistic regression)

This module is deliberately self-contained so it can be copied into other apps unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

TERMINAL_PUNCTUATION = frozenset("।.!?！？\"'”’)]}")


# ── Log-mel features (NeMo FilterbankFeatures, eval mode) ─────────────────────

def _hz_to_mel_slaney(hz: np.ndarray) -> np.ndarray:
    hz = np.asanyarray(hz, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = hz / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(hz >= min_log_hz, min_log_mel + np.log(np.maximum(hz, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    mels = np.asanyarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(mels >= min_log_mel, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """librosa.filters.mel(..., htk=False, norm='slaney'), which NeMo uses."""
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz_slaney(np.linspace(_hz_to_mel_slaney(fmin), _hz_to_mel_slaney(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, len(fftfreqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32)


class FeatureExtractor:
    def __init__(self, pre: dict, stft_pad_mode: str = "constant"):
        self.sr = int(pre.get("sample_rate", 16000))
        self.win_length = int(round(float(pre.get("window_size", 0.025)) * self.sr))
        self.hop = int(round(float(pre.get("window_stride", 0.01)) * self.sr))
        self.n_fft = int(pre.get("n_fft") or 2 ** math.ceil(math.log2(self.win_length)))
        self.n_mels = int(pre.get("features", pre.get("nfilt", 80)))
        self.preemph = pre.get("preemph", 0.97)
        self.mag_power = float(pre.get("mag_power", 2.0))
        self.log = bool(pre.get("log", True))
        guard = pre.get("log_zero_guard_value", 2 ** -24)
        self.log_guard = float(2 ** -24 if guard in (None, "tiny", "eps") else guard)
        self.log_guard_type = pre.get("log_zero_guard_type", "add")
        self.normalize = pre.get("normalize", "per_feature")
        self.pad_value = float(pre.get("pad_value", 0.0))
        self.stft_pad_mode = stft_pad_mode
        window = pre.get("window", "hann")
        if window != "hann":
            raise ValueError(f"unsupported window: {window}")
        win = np.hanning(self.win_length).astype(np.float32)  # torch.hann_window(periodic=False)
        left = (self.n_fft - self.win_length) // 2
        self.window = np.pad(win, (left, self.n_fft - self.win_length - left))
        fmax = pre.get("highfreq") or self.sr / 2
        self.fb = mel_filterbank(self.sr, self.n_fft, self.n_mels, float(pre.get("lowfreq") or 0.0), float(fmax))

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """audio: 1-D float32 at 16 kHz -> (n_mels, frames) float32."""
        x = np.asarray(audio, dtype=np.float32)
        seq_len = len(x) // self.hop + 1
        if self.preemph:
            x = np.concatenate([x[:1], x[1:] - float(self.preemph) * x[:-1]])
        pad = self.n_fft // 2
        x = np.pad(x, (pad, pad), mode="reflect" if self.stft_pad_mode == "reflect" else "constant")
        n_frames = 1 + (len(x) - self.n_fft) // self.hop
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(n_frames)[:, None]
        frames = x[idx] * self.window
        spec = np.abs(np.fft.rfft(frames, n=self.n_fft, axis=1)).astype(np.float32)
        if self.mag_power != 1.0:
            spec = spec ** self.mag_power
        mel = self.fb @ spec.T  # (n_mels, frames)
        if self.log:
            if self.log_guard_type == "clamp":
                mel = np.log(np.maximum(mel, self.log_guard))
            else:
                mel = np.log(mel + self.log_guard)
        seq_len = min(seq_len, mel.shape[1])
        if self.normalize == "per_feature":
            valid = mel[:, :seq_len]
            mean = valid.mean(axis=1, keepdims=True)
            denom = max(seq_len - 1, 1)
            std = np.sqrt(((valid - mean) ** 2).sum(axis=1, keepdims=True) / denom) + 1e-5
            mel = (mel - mean) / std
        mel[:, seq_len:] = self.pad_value
        return mel[:, :seq_len].astype(np.float32)


# ── Model ─────────────────────────────────────────────────────────────────────

class KritiOnnx:
    def __init__(self, model_dir: str | Path, quantized: bool = True, threads: int = 2,
                 stft_pad_mode: Optional[str] = None):
        import onnxruntime as ort

        d = Path(model_dir)
        cfg = json.loads((d / "model_config.json").read_text(encoding="utf-8"))
        self.tokens: list[str] = json.loads((d / "tokens.json").read_text(encoding="utf-8"))
        self.blank = len(self.tokens)
        self.max_symbols = int(((cfg.get("decoding") or {}).get("greedy") or {}).get("max_symbols") or 10)
        pad_mode = stft_pad_mode or cfg.get("stft_pad_mode", "constant")
        self.features = FeatureExtractor(cfg["preprocessor"], stft_pad_mode=pad_mode)

        head_path = d / "punctuation_head.json"
        self.head = json.loads(head_path.read_text(encoding="utf-8")) if head_path.exists() else None

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        suffix = ".int8.onnx" if quantized and (d / "encoder.int8.onnx").exists() else ".onnx"
        self.encoder = ort.InferenceSession(str(d / f"encoder{suffix}"), opts, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(d / f"decoder{suffix}"), opts, providers=["CPUExecutionProvider"])
        self.joint = ort.InferenceSession(str(d / f"joint{suffix}"), opts, providers=["CPUExecutionProvider"])
        state_shape = self.decoder.get_inputs()[1].shape  # (layers, batch, hidden)
        self.state_layers = int(state_shape[0])
        self.state_hidden = int(state_shape[2])

    # Encoder ------------------------------------------------------------------
    def encode(self, feats: np.ndarray) -> np.ndarray:
        """feats (n_mels, T) -> encoded (D, T')."""
        enc, enc_len = self.encoder.run(None, {
            "features": feats[None].astype(np.float32),
            "length": np.array([feats.shape[1]], dtype=np.int64),
        })
        return enc[0, :, : int(enc_len[0])]

    # RNNT greedy --------------------------------------------------------------
    def _predict(self, token: int, h: np.ndarray, c: np.ndarray):
        g, h, c = self.decoder.run(None, {
            "token": np.array([[token]], dtype=np.int64), "h": h, "c": c,
        })
        return g, h, c

    def greedy_ids(self, encoded: np.ndarray) -> list[int]:
        h = np.zeros((self.state_layers, 1, self.state_hidden), dtype=np.float32)
        c = np.zeros_like(h)
        g, h, c = self._predict(self.blank, h, c)  # blank doubles as SOS (blank_as_pad)
        hyp: list[int] = []
        for t in range(encoded.shape[1]):
            f = encoded[:, t][None, None, :].astype(np.float32)
            for _ in range(self.max_symbols):
                logits = self.joint.run(None, {"encoder_frame": f, "prediction": g})[0]
                k = int(np.argmax(logits.reshape(-1)))
                if k == self.blank:
                    break
                hyp.append(k)
                g, h, c = self._predict(k, h, c)
        return hyp

    def detokenize(self, ids: list[int]) -> str:
        text = "".join(self.tokens[i] for i in ids if 0 <= i < len(self.tokens))
        return " ".join(text.replace("▁", " ").split())

    # Danda head ---------------------------------------------------------------
    def danda_probability(self, encoded: np.ndarray) -> Optional[float]:
        if not self.head:
            return None
        mean = encoded.mean(axis=1)
        std = np.sqrt(((encoded - mean[:, None]) ** 2).mean(axis=1))
        vec = np.concatenate([mean, std]).astype(np.float64)
        logit = float(np.dot(vec, np.asarray(self.head["coefficients"], dtype=np.float64)) + float(self.head["intercept"]))
        return 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))

    def restore_danda(self, text: str, encoded: np.ndarray) -> str:
        p = self.danda_probability(encoded)
        if p is None or not text or text[-1] in TERMINAL_PUNCTUATION or p < float(self.head["threshold"]):
            return text
        return f"{text}।"

    # Public -------------------------------------------------------------------
    def transcribe_features(self, feats: np.ndarray, danda: bool = True) -> str:
        encoded = self.encode(feats)
        text = self.detokenize(self.greedy_ids(encoded))
        return self.restore_danda(text, encoded) if danda else text

    def transcribe(self, audio: np.ndarray, danda: bool = True) -> str:
        """audio: 1-D float32, 16 kHz mono."""
        return self.transcribe_features(self.features(audio), danda=danda)
