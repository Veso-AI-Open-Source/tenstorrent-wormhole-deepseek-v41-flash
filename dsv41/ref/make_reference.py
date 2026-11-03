#!/usr/bin/env python3
"""Reference outputs from DeepSeek's own inference code, on the CPU.

  engram   weights/engram_hash.npz, the hash tables the engine loads at startup
  logits   dumps/eval/logits_nofakequant.pt, the answers bench/gate_512.py compares with: 512 real tokens, teacher forced,
           activation fake-quant off, because the device runs bf16 activations

Run: ./run.sh dsv41/ref/make_reference.py engram
     ./run.sh dsv41/ref/make_reference.py logits    (slow: the full model on the CPU)
"""
import os
import sys

os.environ["DSV41_FAKEQUANT"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from load import build, full_logits  # noqa: E402

N_TOKENS = 512


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what not in ("engram", "logits"):
        sys.exit(__doc__)
    net, args = build(max_seq_len=1024)
    if what == "engram":
        eh = net.engram_hash
        path = os.path.join(ROOT, "weights", "engram_hash.npz")
        out = os.environ.get("ENGRAM_OUT", path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, token_map=eh.token_map.numpy(), primes=eh.primes.numpy(), offsets=eh.offsets.numpy(),
                 multipliers=eh.multipliers.numpy(), pad_id=np.int64(eh.pad_id), layer_ids=np.array(args.engram_layer_ids))
        print("wrote", out, flush=True)
        return 0
    full_logits(net)
    tok = net.tokenizer
    text = open(os.path.join(ROOT, "eval", "mix.txt")).read()
    ids = ([tok.bos_token_id] + tok.encode(text, add_special_tokens=False))[:N_TOKENS]
    with torch.inference_mode():
        lg = net(torch.tensor([ids]))[1][0].float()
    path = os.path.join(ROOT, "dumps", "eval", "logits_nofakequant.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"ids": ids, "logits": lg}, path)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
