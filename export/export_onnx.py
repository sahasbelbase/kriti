"""Export Kriti (Nepali RNNT) to ONNX and verify it against the original NeMo model.

Writes to --out:
  encoder.onnx / decoder.onnx / joint.onnx           fp32 graphs
  encoder.int8.onnx / decoder.int8.onnx / joint.int8.onnx   dynamically quantized copies
  model_config.json, tokens.json, punctuation_head.json
  reference/<clip>.audio.npy, reference/<clip>.feats.npy    NeMo inputs/features (16 kHz, no dither)
  reference/transcripts.json                          NeMo transcripts (raw RNNT and with danda)
  parity.json                                         how closely the ONNX runtime reproduces NeMo

Must run in Kriti's runtime environment (pip install -e '.[runtime]'), which pins the AI4Bharat NeMo fork.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_wrappers(model):
    import torch
    from torch import nn

    class EncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, features, length):
            encoded, encoded_length = self.encoder(audio_signal=features, length=length)
            return encoded, encoded_length

    class DecoderStep(nn.Module):
        """One prediction-network step: previous token + LSTM state -> output + new state."""

        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, token, h, c):
            g, state = self.decoder.predict(token, state=[h, c], add_sos=False, batch_size=token.shape[0])
            return g, state[0], state[1]

    class JointStep(nn.Module):
        """Joint network for one encoder frame and one prediction output, Nepali head only."""

        def __init__(self, joint):
            super().__init__()
            self.enc = joint.enc
            self.pred = joint.pred
            layers = list(joint.joint_net)
            head = layers[-1]
            if isinstance(head, nn.ModuleDict):
                head = head["ne"]
            self.body = nn.Sequential(*layers[:-1])
            self.head = head

        def forward(self, encoder_frame, prediction):
            x = self.enc(encoder_frame).unsqueeze(2) + self.pred(prediction).unsqueeze(1)
            return self.head(self.body(x)).reshape(encoder_frame.shape[0], -1)

    return EncoderWrapper(model.encoder).eval(), DecoderStep(model.decoder).eval(), JointStep(model.joint).eval()


def export_graphs(model, out: Path) -> dict:
    import torch

    enc, dec, joint = build_wrappers(model)
    n_mels = int(model.cfg.preprocessor.features)
    layers = int(model.decoder.pred_rnn_layers)
    hidden = int(model.decoder.pred_hidden)
    enc_dim = int(model.joint.enc.in_features)

    with torch.no_grad():
        feats = torch.randn(1, n_mels, 400)
        length = torch.tensor([400], dtype=torch.int64)
        if hasattr(model.encoder, "_prepare_for_export"):
            model.encoder._prepare_for_export()
        torch.onnx.export(
            enc, (feats, length), str(out / "encoder.onnx"),
            input_names=["features", "length"], output_names=["encoded", "encoded_length"],
            dynamic_axes={"features": {0: "batch", 2: "frames"}, "length": {0: "batch"},
                          "encoded": {0: "batch", 2: "steps"}, "encoded_length": {0: "batch"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
        token = torch.tensor([[0]], dtype=torch.int64)
        h = torch.zeros(layers, 1, hidden)
        c = torch.zeros(layers, 1, hidden)
        torch.onnx.export(
            dec, (token, h, c), str(out / "decoder.onnx"),
            input_names=["token", "h", "c"], output_names=["prediction", "h_out", "c_out"],
            dynamic_axes={"token": {0: "batch"}, "h": {1: "batch"}, "c": {1: "batch"},
                          "prediction": {0: "batch"}, "h_out": {1: "batch"}, "c_out": {1: "batch"}},
            opset_version=17, dynamo=False,
        )
        g, _, _ = dec(token, h, c)
        frame = torch.randn(1, 1, enc_dim)
        torch.onnx.export(
            joint, (frame, g), str(out / "joint.onnx"),
            input_names=["encoder_frame", "prediction"], output_names=["logits"],
            dynamic_axes={"encoder_frame": {0: "batch"}, "prediction": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
    return {"pred_rnn_layers": layers, "pred_hidden": hidden, "encoder_dim": enc_dim, "n_mels": n_mels}


def torch_diagnostics(model, kriti, tokens, paths, transcripts, pristine_encoded) -> dict:
    """Run our decoding loop on the original PyTorch modules, and Kriti's own danda head."""
    import numpy as np
    import torch

    from kriti.model import encoder_features, logistic_probability

    _, dec, joint = build_wrappers(model)
    blank = len(tokens)
    max_symbols = int(((model.cfg.get("decoding") or {}).get("greedy") or {}).get("max_symbols") or 10)
    layers, hidden = int(model.decoder.pred_rnn_layers), int(model.decoder.pred_hidden)
    head_feats = encoder_features(model, paths, "cpu", batch_size=1)
    out = {"max_symbols": max_symbols, "clips": {}}
    for path, vec in zip(paths, head_feats):
        stem = Path(path).stem
        enc = torch.from_numpy(pristine_encoded[stem])
        h = torch.zeros(layers, 1, hidden)
        c = torch.zeros(layers, 1, hidden)
        with torch.no_grad():
            g, h, c = dec(torch.tensor([[blank]]), h, c)
            hyp = []
            for t in range(enc.shape[1]):
                f = enc[:, t][None, None, :]
                for _ in range(max_symbols):
                    k = int(joint(f, g).argmax())
                    if k == blank:
                        break
                    hyp.append(k)
                    g, h, c = dec(torch.tensor([[k]]), h, c)
        text = " ".join("".join(tokens[i] for i in hyp).replace("\u2581", " ").split())
        e = pristine_encoded[stem]
        ours = np.concatenate([e.mean(axis=1), np.sqrt(((e - e.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))])
        head = kriti.head
        out["clips"][stem] = {
            "torch_greedy_matches_nemo": text == transcripts[stem]["rnnt"],
            "torch_greedy": text,
            "danda_prob_kriti": logistic_probability(vec, head["coefficients"], head["intercept"]),
            "danda_prob_ours_from_encoder": logistic_probability(ours.tolist(), head["coefficients"], head["intercept"]),
            "head_feature_max_abs_diff": float(np.abs(np.asarray(vec) - ours).max()),
            "threshold": head["threshold"],
        }
    return out


def quantize(out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for name in ("encoder", "decoder", "joint"):
        quantize_dynamic(str(out / f"{name}.onnx"), str(out / f"{name}.int8.onnx"), weight_type=QuantType.QInt8)
        print(f"[export] {name}: {(out / f'{name}.onnx').stat().st_size >> 20} MB -> "
              f"{(out / f'{name}.int8.onnx').stat().st_size >> 20} MB int8")


def nepali_tokens(model) -> list[str]:
    tok = model.tokenizer
    if hasattr(tok, "tokenizers_dict"):
        tok = tok.tokenizers_dict["ne"]
    size = int(tok.vocab_size)
    return [tok.ids_to_tokens([i])[0] for i in range(size)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="kriti.nemo")
    ap.add_argument("--head", required=True, help="punctuation_head.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("clips", nargs="+", help="16 kHz mono wav files for reference outputs")
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from omegaconf import OmegaConf

    from kriti.model import KritiASR

    out = Path(a.out)
    ref = out / "reference"
    ref.mkdir(parents=True, exist_ok=True)

    kriti = KritiASR.from_files(a.model, a.head, device="cpu")  # verifies hashes + parameter counts
    model = kriti.model
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    tokens = nepali_tokens(model)
    head_rows = int(model.joint.joint_net[-1]["ne"].out_features) if hasattr(model.joint.joint_net[-1], "keys") \
        else int(model.joint.joint_net[-1].out_features)
    if head_rows != len(tokens) + 1:
        raise RuntimeError(f"joint has {head_rows} outputs but tokenizer has {len(tokens)} pieces + blank")
    (out / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False))
    shutil.copy(a.head, out / "punctuation_head.json")

    # NeMo reference outputs, taken from the untouched model (export prep can alter modules)
    transcripts = {}
    pristine_encoded = {}
    paths = [str(p) for p in a.clips]

    # Capture exactly what the encoder sees inside NeMo's own transcribe()
    captured = []

    def _hook(module, inputs, kwargs, output):
        sig = kwargs.get("audio_signal", inputs[0] if inputs else None)
        ln = kwargs.get("length", inputs[1] if len(inputs) > 1 else None)
        captured.append((sig.detach().clone(), ln.detach().clone(), output[0].detach().clone(), output[1].detach().clone()))

    handle = model.encoder.register_forward_hook(_hook, with_kwargs=True)
    pre = model.preprocessor.featurizer
    print(f"[diag] preprocessor before transcribe: dither={pre.dither} pad_to={pre.pad_to} "
          f"normalize={getattr(pre, 'normalize', None)} training={model.training}")
    hyps = model.transcribe(paths, batch_size=1, return_hypotheses=True, language_id="ne")
    handle.remove()
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    print(f"[diag] preprocessor after transcribe: dither={pre.dither} pad_to={pre.pad_to}")
    nemo_ids = {Path(pth).stem: [int(x) for x in getattr(h, "y_sequence", [])] for pth, h in zip(paths, hyps)}
    transcribe_encoder_io = {Path(pth).stem: cap for pth, cap in zip(paths, captured)}
    raw = model.transcribe(paths, batch_size=1, logprobs=False, language_id="ne")
    if isinstance(raw, tuple):
        raw = raw[0]
    full = kriti.transcribe(paths, batch_size=1)
    for path, r, f in zip(paths, raw, full):
        stem = Path(path).stem
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1).astype(np.float32)
        np.save(ref / f"{stem}.audio.npy", audio)
        with torch.no_grad():
            feats, flen = model.preprocessor(input_signal=torch.from_numpy(audio)[None],
                                             length=torch.tensor([len(audio)]))
        np.save(ref / f"{stem}.feats.npy", feats[0, :, : int(flen[0])].numpy().astype(np.float32))
        with torch.no_grad():
            enc, elen = model.encoder(audio_signal=feats, length=flen)
        pristine_encoded[stem] = enc[0, :, : int(elen[0])].numpy().astype(np.float32)
        transcripts[stem] = {"rnnt": getattr(r, "text", r), "with_danda": f}
    (ref / "transcripts.json").write_text(json.dumps(transcripts, indent=2, ensure_ascii=False))

    diagnostics = torch_diagnostics(model, kriti, tokens, paths, transcripts, pristine_encoded)
    for stem, (sig, ln, enc_out, enc_len) in transcribe_encoder_io.items():
        ours = np.load(ref / f"{stem}.feats.npy")
        theirs = sig[0, :, : int(ln[0])].numpy()
        n = min(ours.shape[1], theirs.shape[1])
        their_enc = enc_out[0, :, : int(enc_len[0])].numpy()
        m = min(their_enc.shape[1], pristine_encoded[stem].shape[1])
        diagnostics["clips"][stem].update({
            "transcribe_input_shape": list(sig.shape),
            "transcribe_input_len": int(ln[0]),
            "our_feats_len": int(ours.shape[1]),
            "feats_vs_transcribe_input_max_abs_diff": float(np.abs(ours[:, :n] - theirs[:, :n]).max()),
            "encoder_vs_transcribe_encoder_max_abs_diff": float(np.abs(pristine_encoded[stem][:, :m] - their_enc[:, :m]).max()),
            "nemo_token_ids_head": nemo_ids.get(stem, [])[:20],
            "nemo_ids_max": max(nemo_ids.get(stem, [0]) or [0]),
        })
    print(json.dumps({"diagnostics_before_export": diagnostics}, indent=2, ensure_ascii=False))

    dims = export_graphs(model, out)
    quantize(out)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out / "encoder.onnx"), providers=["CPUExecutionProvider"])
    enc_diff = {}
    for stem in transcripts:
        f = np.load(ref / f"{stem}.feats.npy")
        e, el = sess.run(None, {"features": f[None], "length": np.array([f.shape[1]], dtype=np.int64)})
        e = e[0, :, : int(el[0])]
        p = pristine_encoded[stem]
        n = min(e.shape[1], p.shape[1])
        enc_diff[stem] = {"onnx_steps": int(e.shape[1]), "torch_steps": int(p.shape[1]),
                          "max_abs_diff": float(np.abs(e[:, :n] - p[:, :n]).max())}
    diagnostics["onnx_encoder_vs_torch"] = enc_diff
    print(json.dumps({"onnx_encoder_vs_torch": enc_diff}, indent=2))

    cfg = OmegaConf.to_container(model.cfg, resolve=True)
    model_config = {"preprocessor": cfg["preprocessor"], "decoding": cfg.get("decoding"), **dims}
    (out / "model_config.json").write_text(json.dumps(model_config, indent=2, ensure_ascii=False, default=str))

    # Parity: does the NumPy/ONNX runtime reproduce NeMo?
    from kriti_onnx import FeatureExtractor, KritiOnnx

    feat_diff = {}
    for mode in ("constant", "reflect"):
        fx = FeatureExtractor(cfg["preprocessor"], stft_pad_mode=mode)
        diffs = []
        for stem in transcripts:
            ours = fx(np.load(ref / f"{stem}.audio.npy"))
            theirs = np.load(ref / f"{stem}.feats.npy")
            n = min(ours.shape[1], theirs.shape[1])
            diffs.append(float(np.abs(ours[:, :n] - theirs[:, :n]).max()))
        feat_diff[mode] = max(diffs)
    best_mode = min(feat_diff, key=feat_diff.get)
    model_config["stft_pad_mode"] = best_mode
    (out / "model_config.json").write_text(json.dumps(model_config, indent=2, ensure_ascii=False, default=str))

    parity = {"feature_max_abs_diff": feat_diff, "stft_pad_mode": best_mode, "diagnostics": diagnostics, "runs": {}}
    for quantized in (False, True):
        rt = KritiOnnx(out, quantized=quantized)
        label = "int8" if quantized else "fp32"
        exact_nemo_feats = exact_numpy_feats = 0
        samples = {}
        for stem, t in transcripts.items():
            a1 = rt.transcribe_features(np.load(ref / f"{stem}.feats.npy"))
            a2 = rt.transcribe(np.load(ref / f"{stem}.audio.npy"))
            exact_nemo_feats += a1 == t["with_danda"]
            exact_numpy_feats += a2 == t["with_danda"]
            samples[stem] = {"nemo": t["with_danda"], "onnx_nemo_feats": a1, "onnx_numpy_feats": a2}
        parity["runs"][label] = {
            "clips": len(transcripts),
            "exact_match_with_nemo_features": exact_nemo_feats,
            "exact_match_full_numpy_pipeline": exact_numpy_feats,
            "samples": samples,
        }
    (out / "parity.json").write_text(json.dumps(parity, indent=2, ensure_ascii=False))
    print(json.dumps(parity, indent=2, ensure_ascii=False))

    fp32 = parity["runs"]["fp32"]
    if fp32["exact_match_with_nemo_features"] < fp32["clips"]:
        print("::error::fp32 ONNX graphs do not reproduce NeMo transcripts; not publishing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
