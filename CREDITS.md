# Credits

This bring-up stands on work with clear owners.

- **DeepSeek: DeepSeek-V4.1-Flash** (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
  The model, the technical report, and the reference `inference/` code. The
  reference implementation is the oracle for the quality gate in this
  repository. We run its model code on the CPU behind a plain-torch shim for its
  six TileLang kernels (`dsv41/ref/kernel.py`) and a zero-copy loader
  (`dsv41/ref/load.py`). The gate turns the shim's activation fake-quant off,
  because the device runs bf16 activations. The Engram hashing, the CSA2 attention, the mHC residual
  stream, the DSpark drafter, and the ViT are all ported from that code.

- **Tenstorrent: tt-metal and TT-NN** (https://github.com/tenstorrent/tt-metal)
  The platform. Four things deserve specific credit:
  - **`ttnn.generic_op` with `ProgramDescriptor`**, which lets us put custom
    kernels on the cores from Python with no C++ rebuild and no fork.
  - **Device traces** (`begin_trace_capture` and `execute_trace`), which the
    decode path is built on.
  - **`ttnn.experimental.rotary_embedding_llama`**, one fused op for the 12-op
    rope pair swap, at HiFi4.
  - **The `.tensorbin` format** read by `ttnn.load_tensor` as an mmap, which makes
    the host expert store possible without a copy.

- **Our own Kimi-Linear bring-up**
  (https://github.com/Veso-AI-Open-Source/tenstorrent-wormhole-kimi-linear)
  The traced-mesh engine, the 1x8 `FABRIC_1D_RING` finding, the `all_gather` plus
  sum reduction idiom, and the packaging of this repository.

- **Codys12: MuseGlimmer** (https://github.com/Codys12/MuseGlimmer)
  The packaging model: correctness first, quantified real-hardware numbers with
  prefill and decode defined separately, and honest constraints rather than
  best-case figures.
