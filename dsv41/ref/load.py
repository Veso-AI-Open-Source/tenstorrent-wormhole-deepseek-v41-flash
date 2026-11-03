"""Build the OFFICIAL Transformer at full size on CPU with zero-copy mmapped weights.

510 GB of checkpoint does not fit beside anything else in 503 GB of RAM, so nothing is copied: every
parameter's storage is replaced by a view straight into the safetensors shard mmap (page cache does
the rest; Engram touches 48 rows a token). Only tensors whose checkpoint dtype differs from the
module's dtype are materialized (wo_a fp8->bf16 exactly as convert.py does, fp32 promotions).
"""
import glob, json, os, sys, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = sorted(glob.glob(os.path.expanduser(os.environ.get(
    "DSV41_SNAP", "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*"))))[-1]
sys.path[:0] = [HERE, os.path.dirname(HERE), f"{SNAP}/inference"]   # shim `kernel` shadows TileLang's
import torch
import kernel  # noqa: F401  (the shim)
import model as M
from ckpt import Checkpoint

DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I8": torch.int8, "U8": torch.uint8,
      "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "I32": torch.int32, "I64": torch.int64}


def ck_tensor(ck: Checkpoint, name: str) -> torch.Tensor:
    buf, dt, shape = ck.raw(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # read-only mmap: we never write weights
        return torch.from_numpy(buf).view(DT[dt]).view(shape)


def build(max_seq_len=4096, max_batch_size=1, n_layers=None, verbose=True):
    t0 = time.time()
    cfg = json.load(open(f"{SNAP}/inference/config.json"))
    cfg.update(max_seq_len=max_seq_len, max_batch_size=max_batch_size)
    if n_layers is not None:                      # truncated stack for fast bring-up checks
        cfg["n_layers"] = n_layers
        cfg["compress_ratios"] = cfg["compress_ratios"][:n_layers] + cfg["compress_ratios"][-cfg["n_mtp_layers"]:]
        cfg["dspark_target_layer_ids"] = [n_layers - 3, n_layers - 2, n_layers - 1]
    args = M.ModelArgs(**cfg)
    torch.set_default_dtype(torch.bfloat16)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP)     # Engram hashes n-grams over a normalized vocab
    net = M.Transformer(args, tok)                # torch.empty params: untouched virtual memory
    net.tokenizer = tok
    ck = Checkpoint(SNAP)
    missing, cast = [], []
    for name, p in net.named_parameters():
        if name not in ck.weight_map:
            missing.append(name); continue
        t = ck_tensor(ck, name)
        if name.endswith("wo_a.weight"):          # convert.py: block-dequantize fp8 -> bf16
            s = ck_tensor(ck, name.replace("weight", "scale")).float()
            ob, ib = t.size(0) // s.size(0), t.size(1) // s.size(1)
            t = (t.float().unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)) * s[:, None, :, None]).flatten(2, 3).flatten(0, 1).bfloat16()
        elif p.dtype == torch.float4_e2m1fn_x2:
            t = t.view(torch.float4_e2m1fn_x2)
        elif t.dtype != p.dtype:
            cast.append((name, str(t.dtype), str(p.dtype))); t = t.to(p.dtype)
        assert t.shape == p.shape, (name, tuple(t.shape), tuple(p.shape))
        p.data = t
    unused = [k for k in ck.weight_map if k.split(".")[0] not in ("vision", "aligner") or args.vision_enabled]
    have = {n for n, _ in net.named_parameters()}
    unused = [k for k in unused if k not in have and not k.endswith("wo_a.scale")
              and (n_layers is None or not k.startswith("layers.") or int(k.split(".")[1]) < n_layers)]
    if verbose:
        print(f"[load] {sum(1 for _ in net.parameters())} params attached in {time.time()-t0:.0f}s | "
              f"missing-in-ckpt {len(missing)} {missing[:4]} | ckpt-unused {len(unused)} {unused[:4]} | dtype-cast {len(cast)} {cast[:3]}", flush=True)
    net.eval()
    return net, args


class RouteLog:
    """Records (layer, token) -> top-k expert ids by wrapping every backbone Gate.forward."""
    def __init__(self, net):
        self.routes = {}                          # layer -> list of [n_tokens, k] int tensors
        for blk in net.layers:
            gate, lid = blk.ffn.gate, blk.layer_id
            orig = gate.forward
            def fwd(x, image_mask=None, _o=orig, _l=lid):
                w, idx = _o(x, image_mask); self.routes.setdefault(_l, []).append(idx.clone()); return w, idx
            gate.forward = fwd


def full_logits(net):
    """Make net(ids) return logits for EVERY position (the reference returns only the last)."""
    orig = net.head.forward
    net.head.forward = lambda x, full_logits=False: orig(x, full_logits=True)
    return net
