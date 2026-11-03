# Findings: traps we hit bringing DeepSeek-V4.1-Flash up on Wormhole

Twenty defects, and the method that caught each. **Five produced no error at
all.** If you run a large MoE on tt-metal, this is the file to read first.

These come from the September bring-up on tt-metal 0.10. The engine has since
moved to tt-metal v0.81, and some fixes below have been replaced. Numbers here
describe the engine of that time, not this release. See `README.md` for current
measurements.

## The five that were silent

The run is green, the log is clean, and the answer is wrong.

**1. Stock `ttnn.rms_norm` is wrong on this model's residual stream.**
Residual rows carry massive-activation channels (max 374 against an rms of 7).
On a 1300-row fixture: median per-row error 3.3%, worst 17%, 28% of rows over
5%. The fp32-accumulate variant does not fit L1 at width 5120. The four-op form
`x * rsqrt(mean(x*x) + eps) * w` measures 0.38%. The September engine used it for
every norm. Today's decode path uses `ttnn.rms_norm` at HiFi4 with fp32 for up to
32 rows, and the four-op form above that. Caught by the sublayer gate against a
float32 torch model.

**2. `ttnn.update_cache` corrupts the whole tile.**
On a `[1,1,128,512]` bf16 ring, writing index 40 changed rows 32..63 and the
written row was wrong. Replaced by an exact one-hot blend
`ring*(1-hot) + hot (x) kv` (three tiny ops; only a 128-float mask changes per
step, so it is trace-friendly). Decode now writes rows with a custom kernel
(`dsv41/kernels/rope_cache.cpp`). Caught by reading the ring back after one write.

**3. `ttnn.slice` leaves tile padding undefined.**
A NaN in a padded row times a zero one-hot is NaN. It poisoned the DSpark
drafter's key ring progressively, and drafts degraded over a run with no
error. Select rows with a zero-padded one-hot matmul instead. Caught by
diffing engine state against a clean run after every pass.

**4. Dropping a `+eps` that "cannot matter" in bf16 cost three decisive flips.**
Sinkhorn normalizes near-one-hot rows; some column sums are ~4e-6 and eps is
25% of the divisor. The sublayer gate did not move to five digits. The
end-to-end gate lost three top-1 tokens. Rule adopted: every numerics change
runs the end-to-end quality check, not only its sublayer gate.

**5. `rotary_embedding_llama` at its default math fidelity.**
Passed every sublayer gate (2-3%) and failed the end-to-end gate (four decisive
flips, decode KL 0.014 -> 0.052). `compute_kernel_config=HiFi4` fixes it.
Rotations touch every key. Same rule as item 4.

## The one that hangs the board

**6. "Allocating device buffers is unsafe due to the existence of an active
trace" is not a warning.**
Anything persistent allocated AFTER a trace capture can land in that trace's
freed intermediates: a later trace's kept outputs, an op's lazily created state
on a graph's first eager run, or a graph with a new SHAPE. The next replay
overwrites it. The next eager op that reads it hangs the device (eight
dispatch threads at 100%, or the main thread spinning; recover with
`tt-smi -r`). Rule: run every graph variant eagerly first, then capture ALL
traces in one go with outputs copied into pre-allocated buffers, then never
allocate again. Caught by a script that captures one sub-graph at a time and
replays it with eager work in between; the variant whose eager gap contained a
first-run graph hung on the next replay. `faulthandler.dump_traceback_later`
names the stuck op even with the GIL held. This defect returned three times (the
first traced decode, the DSpark 32-row draft graph, the K+1 verify graph).

## Layout and memory

**7. TILE layout pads the last TWO dimensions to 32.**
A residual stream stored as `[1, s, 4, d]` was eight times its size (1.34 GB
instead of 168 MB at s=4096) and every op processed the padding. Store small
dimensions before the sequence dimension: `[1, 4, s, d]`. mHC at s=2048 went
from 68 ms to 17 ms. Caught by reading `get_memory_view` per tensor.

**8. `ttnn.rms_norm` over the 20480-wide flattened mHC stream throws.**
"Circular buffers grow to 2.79 MB beyond max L1 1.5 MB". Use
mean-over-copies of mean-over-d, then rsqrt. Exact, and it scales.

**9. The attention-optimized `ttnn.softmax` asserts on some widths.**
"Size of cb: 80 must be divisible by the size of block". Its block is the
largest divisor <= 8 of the width in TILES; 420 tiles (block 7) dies. Hit at
24K context. Pad the width with -1e9 columns to a multiple of 256 (8 tiles).

**10. Dense reach masks per query chunk scale with context.**
A per-pass cache of `[chunk, n_c]` masks reached 1-2 GB at 40K context and was
the first 128K OOM. Build them on device per chunk from two float32 vectors
(`lt` + typecast). float32 because bf16 cannot hold integers above 256.

**11. `fill_cache` and `slice_write` reject bfp4.**
`fill_cache` takes FLOAT32/BFLOAT16/BFLOAT8_B only; `slice_write` goes through
tilize. Whole-tensor `copy_host_to_device_tensor` into fixed slots, and
in-place `ttnn.copy` device-to-device, are exact for bfp4 and bfp8 (18 slot
copies: 0.4 ms). Caught by a slot write probe.

## Numerics that are the model's, not ours

**12. The router must run in float32.**
bf16 picked a different top-6 SET on 7% of real tokens (6th-vs-7th biased
score gaps down to 1e-4); float32 on 1%. Two million weights: free.

**13. MXFP4 -> bfp4 is inherently 11.6% weight noise.**
E2M1 magnitudes in half-units are {0,1,2,3,4,6,8,12}; a 16-block holding both
a 12 and a 1 cannot sit on bfp4's uniform grid. bfp8 is exact. All-bfp4
failed the quality gate; the store is MIXED: encoder layers bfp8, decoder
layers bfp4. Caught by transcoding real weights and comparing matvecs.

**14. The ViT is bf16-fragile, and a 5% gate against a bf16 fixture cannot be
met by anyone.**
The reference's OWN bf16 ViT output is 7.8% off its fp32 self (small-norm
output rows); the aligner 4.3%. Two bf16 implementations differ by ~10%+.
What matched the reference's rounding points: attention scores kept in fp32
(a bf16 score of 20 is off by 0.06 = 6% on that weight), softmax in fp32,
RMSNorm statistics and the weight multiply in fp32 with one rounding, 2D RoPE
in fp32. The gate is now "error vs fp32 <= 2x the reference's own bf16 error".

**15. Speculative decoding cannot be "identical to greedy" across two kernel
paths in bf16.**
The multi-token verify pass matches single-token decode to KL 0.000-0.03 per
row and rollback is exact (states reload-identical), yet greedy argmax flips
at near-ties. Gate the verify path against the reference like any other path.

## Operational

**16. The 449 GB expert store and the 510 GB checkpoint cannot both live in
503 GB of page cache.**
Running the CPU reference evicts the store; the next mesh pass streams from
NVMe and a layer takes 5.5 s instead of 1.0 s. One prefill measurement was
wrong by 10x for exactly this reason. Warm the store before benchmarking.

**17. Reading one device's tensor from the mesh can hang.**
`get_device_tensors(t)[0]` then `to_torch` spun for 27 minutes at 120% CPU
with no progress; killing it wedged the board. Read all eight and pick one.
It costs 10 ms per token and does not hang.

**18. An intermittent hang at the start of full-model runs.**
About one run in six, at mesh open / fabric init or the first collective. Not
root-caused. `tt-smi -r`, rerun.

**19. Sublayer gates do not see rare-row damage.**
Items 4 and 5 both passed their unit gates. A gate on 394 decisive positions
was decided by 2-3 flips run to run of CODE, so the September end-to-end gate used
a 2048-token reference (1709 decisive positions; one flip = 0.06%). The check in this
release is the smaller 512-token one, so read its margin with that in mind.

**20. The gate metric itself saturates.**
Flat top-1 agreement measures text entropy, not engine fidelity. Use top-1
agreement on DECISIVE positions (reference margin >= 0.10) plus mean KL.

## What did not pay, with the reason

**DSpark speculative decoding (built, correct, off).**
A K-token verify pass pages the experts of ALL K drafted tokens whether or not
they are accepted. Decode here is expert-bandwidth-bound, not dispatch-bound,
so speculation multiplies exactly the bottleneck. Measured 0.48x in September.
In November, after kernel work, the best DSpark run reached 4.76 tok/s against
6.2 tok/s for plain decode on the same prompt. It is still slower.

**Custom-kernel prefill (not built).**
Stock ops reach 95 tok/s at 8K passes; the fixed cost per pass is 13 s of
expert streaming (bandwidth-bound at 31 GB/s) plus ~4.4 ms/token of stock-op
glue. The kernel would be a grouped GEMM with fused gather and scatter. It is not
in this release.
