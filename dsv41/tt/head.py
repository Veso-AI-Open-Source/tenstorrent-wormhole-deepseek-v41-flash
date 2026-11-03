"""Final collapse + norm + vocab-sharded head. Logits come back to the host per chip (16160 each)."""
import torch
import ttnn
from mesh import mm, rms_norm
from mhc import MHC


class Head:
    def __init__(self, dev, top, sink_op=None):
        self.dev, self.norm, self.head, self.sink_op = dev, top["norm"], top["head"], sink_op

    def logits(self, x, pre_mix, last_only=True):
        """x [1,4,s,d], pre_mix [1,4,s,1] -> host logits [s or 1, 129280] float32."""
        s = x.shape[2]
        if (s == 1 or (s <= 8 and not last_only)) and self.sink_op is not None and hasattr(self.sink_op, "pre_ms"):
            h = self.sink_op.pre_ms(x, pre_mix)[0]
        else:
            h = MHC.pre(x, pre_mix)                                                        # [1,1,s,d]
            if last_only and s > 1:
                h = h[:, :, s - 1 : s, :]
        lg = mm(rms_norm(h, self.norm), self.head)                                         # per chip [1,1,s,16160]
        if h.shape[2] <= 8:
            from rope_cache_op import untilize_head_1
            lg = untilize_head_1(self.dev, lg)
        parts = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(ttnn.from_device(lg))]
        return torch.cat(parts, dim=-1)[0, 0]
