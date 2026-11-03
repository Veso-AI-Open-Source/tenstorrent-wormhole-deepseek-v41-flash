"""One definition of 'the same model', used by every end-to-end gate.

CALIBRATION (2026-09-17, 512 teacher-forced tokens of mixed prose/code/arithmetic): turning the
reference's activation fake-quants OFF, a change that only makes the arithmetic MORE exact, scores
raw top-1 agreement 0.939 (KL 0.018). Every one of its 31 flips sits where the reference itself is
unsure: agreement is 1.000 where ref top-1 prob >= 0.9, 0.994 in [0.5, 0.9), 0.870 below 0.5, and the
two candidates at a flip differ by 0.05 in reference probability on average. 45% of positions on this
text have ref top-1 prob < 0.5, so a flat 'top-1 >= 0.95' measures the text's entropy, not the engine.
The gate therefore counts agreement only on DECISIVE positions (reference top-1 minus top-2
probability >= 0.10) and keeps KL as the distribution-level check. Raw top-1 is still reported.
"""
import torch

MARGIN = 0.10


def compare(A: torch.Tensor, B: torch.Tensor, ids=None) -> dict:
    """A = reference logits [s,V], B = candidate logits [s,V]; ids = the token ids (for perplexity)."""
    la, lb = A.float().log_softmax(-1), B.float().log_softmax(-1)
    pa = la.exp(); top2 = pa.topk(2, -1); conf = top2.values[:, 0]; margin = conf - top2.values[:, 1]
    a1, b1 = la.argmax(-1), lb.argmax(-1); same = a1 == b1; dec = margin >= MARGIN
    kl = (pa * (la - lb)).sum(-1)
    m = {"top1_decisive": float(same[dec].float().mean()) if dec.any() else 1.0, "n_decisive": int(dec.sum()),
         "top1_agree": float(same.float().mean()), "kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "n_tokens": int(len(a1))}
    for lo, hi, tag in ((0.9, 1.01, "p90"), (0.5, 0.9, "p50_90"), (0.0, 0.5, "p00_50")):
        sel = (conf >= lo) & (conf < hi)
        m[f"agree_{tag}"] = float(same[sel].float().mean()) if sel.any() else 1.0
    if (~same).any():
        m["flip_gap_mean"] = float((conf - pa.gather(1, b1[:, None])[:, 0])[~same].mean())
    if ids is not None:
        tgt = torch.as_tensor(ids[1:])
        m["ppl_ref"] = float((-la[:-1].gather(1, tgt[:, None]).mean()).exp()); m["ppl_cand"] = float((-lb[:-1].gather(1, tgt[:, None]).mean()).exp())
    return m
