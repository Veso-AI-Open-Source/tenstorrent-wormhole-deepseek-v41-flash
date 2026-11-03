# tenstorrent-wormhole-deepseek-v41-flash

DeepSeek-V4.1-Flash (552B + 196B Engram) on Tenstorrent Wormhole. The forward pass runs on the cards. The host moves bytes and handles a few small steps: Engram hashing, the last part of the router math, expert paging, and token selection.

This release is for technical review by Tenstorrent. It holds the decode path we have checked on real hardware, plus the stock-op prefill that path depends on. It passes its quality check, and it is not yet fast.

## Results

Hardware: four n300 cards (8 Wormhole chips), a 16-core EPYC host, and 499 GB of RAM.
Software: tt-metal `v0.81.0-dev20261020` (`e5b49d7b`).

We measured every number in this section on the cards from a clean copy of this repo. The raw output is in `results/`.

| Run | Median speed | Median time per token |
|---|---|---|
| Same 16-token output, repeated | 5.7 to 6.2 tok/s | 162 to 174 ms |
| The same output, first run | 3.4 to 3.5 tok/s | 288 to 297 ms |
| Same 128-token output, repeated | 4.1 to 4.2 tok/s | 236 to 242 ms |
| The same output, first run | 3.3 tok/s | 300 ms |
| 512 tokens of different text, expert store in page cache | 3.6 tok/s | 277 ms |

**6.2 tok/s is the best case, not the typical speed.** It needs a short context and experts that are already on the cards. Different text is what a real user would send. The mean over those 512 tokens is 3.3 tok/s.

The very first 512-token run was slower: 2.3 tok/s median and 1.5 tok/s mean. The likely reason is that the expert store was not yet in page cache. We did not isolate it, and that run's output is not in `results/`.

Repeat runs settle at either about 162 ms or about 174 ms per token with identical code, and can switch during a run. `results/runs_16.txt` shows the raw lines. We do not know why.

Quality:

- **Short check:** 16 of 16 tokens match the recorded CPU reference output for "The capital of France is". This is one prompt.
- **512-token check:** top-1 agreement is 98.2% on the 394 positions where the reference is decisive (limit 98%). That is 7 flips, and 8 would fail, so the margin is thin. Mean KL is 0.028 (limit 0.05). Every token goes through the same decode path that `bench_real.py` times. The first tokens after each trace capture run eagerly.

The reference is DeepSeek's own `inference/` code, running on the CPU behind our shim, with activation fake-quant off because the device runs bf16 activations. We generated the reference file and the expected tokens in September. We did not regenerate them for this release.

## Limits

- **Speed falls as the conversation grows.** See the 16-token and 128-token rows above. We have not measured past 512 tokens.
- **The decode trace is captured again** every 64 tokens until the context reaches 512 tokens.
- **Fresh text is slow.** Each layer picks 6 of 384 experts. A pick that is not on the cards is copied from the host during the token.
- **Context is capped at 1,024 tokens** in this configuration.
- **Prompts over 126 tokens** take a slower stock-op path.
- **One user, greedy output only.** This release has no server.
- **Experts are stored lossy on layers 20 to 39.** They use bfp4 there and bfp8 on layers 0 to 19. See `FINDINGS.md`, item 13.
- **DSpark is included and off by default.** Its best run, on November 2 with an earlier script, was 4.76 tok/s against about 6.2 tok/s for plain decode. It is slower today.
- **Vision code is included because the model imports it.** We have not checked it on this version.
- **Startup sometimes fails** at mesh open with an ETH heartbeat timeout. A board reset clears it. We have not found the cause.

## Where the time goes

Each sublayer group of a real layer ran in its own trace, replayed 100 times with one sync at the end. From `bench/bench_layer.py`, on layer 3:

| Group | us per layer |
|---|---|
| Attention (includes one `all_gather` + `sum`) | 921 |
| MoE (includes one `all_gather` + `sum`) | 703 |
| Expert slot gather | 427 |
| Feed-forward mix (`fn4` matmul + Sinkhorn) | 333 |
| Attention mix (`fn4` matmul + Sinkhorn) | 332 |
| Router scores | 177 |
| `rms_norm`, attention | 177 |
| `rms_norm`, feed-forward | 177 |
| `post_pre_ms` | 149 |
| `pre_ms` | 75 |
| **Total** | **3,471** |

That is about 139 ms across 40 layers, against 162 to 174 ms measured per token. Host work and buffer copies are the rest. Reaching 20 tok/s means fitting a token in 50 ms, or 1,250 us per layer.

Single stock ops in a replayed trace, from `bench/bench_ops.py`:

| Op | us |
|---|---|
| `all_gather` + `sum`, 5120-wide row | 270 |
| `rms_norm`, 5120-wide row, HiFi4 with fp32 | 174 |
| `matmul`, 5120 to 1280 | 131 |
| `matmul`, 5120 to 512 | 125 |
| `matmul`, 1280 to 4096 | 87 |
| `matmul`, 1024 to 5120 | 87 |
| `matmul`, 288 to 5120 | 40 |
| `typecast` | 22 |
| `add` | 20 |
| `copy`, 5120-wide row | 15 |

**A measuring trap:** do not sync around each replay. During development, an empty pair of device syncs took about 800 us on this box, and it varied enough to produce negative op times. Replay many times, sync once, and divide.

## Questions for Tenstorrent

1. **Collectives.** `all_gather` + `sum` takes 270 us, twice per layer, to combine one row across 8 chips. Most of each tile is padding. What is the supported way to reduce a single row on a 1x8 ring?
2. **`rms_norm`.** It takes 174 us at width 5120. Is there a faster supported form with the same accuracy at HiFi4 with fp32?
3. **Host decisions inside a token.** The host reads router scores once per layer, between two trace replays, so it can pick experts and page them in. What is the supported pattern for a data-dependent step like this?
4. **Mesh open.** On this QuietBox, opening any mesh fails in the topology mapper unless `FABRIC_1D_RING` is set first, even for a 1x1 mesh. Is that expected?
5. **Two speeds.** Identical code settles at either about 162 ms or about 174 ms per token, and can switch during a run. What could cause a steady 7% difference?

## Structure

```
dsv41/live_model.py     engine entry point: load, prefill, step
dsv41/tt/               the engine: model, layers, attention, MoE, mHC, weights, expert store
dsv41/kernels/          custom kernels: Sinkhorn, fused mHC, RoPE and cache writes, expert slot gather
dsv41/ref/              CPU reference shim, packing helpers, gate metrics
bench/                  bench_real, gate_512, bench_layer, bench_ops
results/                raw output of the runs above, plus the 16-token run lines
docker/Dockerfile       the build image
run.sh                  runs a script from this repo on the cards
FINDINGS.md             traps from the September bring-up, still worth reading first
```

## Setup

You need four n300 cards, 16 x 1 GB hugepages, a CPU with AVX-512 (the expert packer requires it), about 500 GB of RAM, and about 1 TB of fast disk. The checkpoint is about 510 GB. The packed expert store grows as experts are used, up to about 450 GB.

For this release we built the image with the recipe below and ran steps 3 and 4 from an empty weights directory. We did not rebuild tt-metal itself, and we did not rerun the `logits` command in step 5.

**1. Build tt-metal and the image.**

```
git clone --branch v0.81.0-dev20261020 --recursive https://github.com/tenstorrent/tt-metal
export TT_METAL_TREE=$PWD/tt-metal
(cd tt-metal && docker buildx bake -f dockerfile/docker-bake.hcl basic-dev)
docker buildx build -f docker/Dockerfile -t dsv41-tt:v081 --load "$TT_METAL_TREE"
docker run --rm -v "$TT_METAL_TREE":/tt-metal -w /tt-metal --entrypoint /bin/bash dsv41-tt:v081 -c \
  './build_metal.sh --release --use-system-sfpi && ./create_venv.sh --python-version 3.10 --env-dir /tt-metal/python_env --bundle-python --force'
```

**2. Download the model.**

```
hf download deepseek-ai/DeepSeek-V4.1-Flash
```

**3. Build the packer and the Engram tables.**

```
(cd dsv41/ref && gcc -O3 -march=native -fopenmp -shared -fPIC -o libfp4.so fp4lib.c -lm)
./run.sh dsv41/ref/make_reference.py engram
```

**4. Run it.**

```
./run.sh bench/bench_real.py
```

The first run compiles kernels, converts weights, and packs experts, so it is slow. Only later runs give the speeds above.

**5. Run the quality check.** This first writes the reference answers on the CPU, which is slow.

```
./run.sh dsv41/ref/make_reference.py logits
./run.sh bench/gate_512.py
```

To time the layer, run `bench/bench_layer.py` and `bench/bench_ops.py` the same way.

## Licence

Apache-2.0. The model weights are DeepSeek's, under their licence. See `CREDITS.md`.
