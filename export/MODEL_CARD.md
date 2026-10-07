# Kriti — ONNX conversion

An **ONNX conversion of [Kriti](https://github.com/Naamche-Labs/kriti)**, the open Nepali speech
recognition model by **[Naamche Labs](https://naamchelabs.com)**, made so it runs with ONNX Runtime
and NumPy alone (no NeMo or PyTorch). It powers the optional offline Nepali engine in the free,
open-source [Just Talk](https://github.com/sahasbelbase/JustTalk) voice keyboard.

**All credit for the model goes to Naamche Labs and AI4Bharat.** Kriti is derived from the
MIT-licensed [AI4Bharat Nepali IndicConformer](https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large).
See `NOTICE.txt` for the full attribution record.

## License

MIT, as released by the Kriti authors and AI4Bharat (`LICENSE.txt`, `NOTICE.txt`).
Datasets used to train Kriti keep their own licenses and are not included here.

## Changes from the original

Converted by Sahas Belbase for Just Talk from `harrrshall/kriti` (`kriti.nemo`, verified by
Kriti's own SHA-256 and parameter-count checks):

1. Exported the Conformer encoder, the RNNT prediction network (one step) and the Nepali joint head
   to ONNX (`encoder.onnx`, `decoder.onnx`, `joint.onnx`), plus int8 dynamically quantized copies.
2. Extracted the feature settings (`model_config.json`) and the 256 Nepali SentencePiece pieces
   (`tokens.json`); copied Kriti's acoustic danda head (`punctuation_head.json`) unchanged.
3. `kriti_onnx.py` reimplements log-mel features, greedy RNNT decoding and danda restoration in NumPy.
4. `parity.json` records how closely this runtime reproduces the original NeMo model on public
   test clips (FLEURS and NepTel samples, CC BY 4.0). Release builds require exact transcript
   matches for the fp32 graphs.

Accuracy figures belong to the Kriti authors: see their [benchmark](https://github.com/Naamche-Labs/kriti/blob/main/benchmark.md)
(4.1% WER on clean read Nepali; 40.6% on NepTel real-call audio). The int8 files can differ
slightly from the original; see `parity.json`.

## Usage

```python
import soundfile as sf
from kriti_onnx import KritiOnnx

model = KritiOnnx(".")                      # int8 by default; quantized=False for fp32
audio, sr = sf.read("speech_16k_mono.wav", dtype="float32")
print(model.transcribe(audio))
```

## Citation

Please credit Kriti by Naamche Labs (https://github.com/Naamche-Labs/kriti) and the AI4Bharat
IndicConformer it builds on.
