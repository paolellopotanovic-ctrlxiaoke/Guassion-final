"""Geometric vector perceptron primitives used by M4."""

import torch as pt


def masked_mean(x, mask, dim=1, eps=1e-6):
    mask = mask.to(dtype=x.dtype, device=x.device)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)


class GVPTupleLayerNorm(pt.nn.Module):
    def __init__(self, scalar_dim, vector_dim, eps=1e-6):
        super().__init__()
        self.scalar_norm = pt.nn.LayerNorm(scalar_dim)
        self.vector_scale = pt.nn.Parameter(pt.ones(vector_dim))
        self.eps = float(eps)

    def forward(self, s, v):
        s = self.scalar_norm(s)
        if v.numel() == 0:
            return s, v
        rms = pt.sqrt(pt.mean(pt.sum(v * v, dim=-1), dim=-1, keepdim=True).clamp_min(self.eps))
        v = v / rms.unsqueeze(-1)
        v = v * self.vector_scale.view(*([1] * (v.dim() - 2)), -1, 1)
        return s, v


class GVP(pt.nn.Module):
    """Geometric Vector Perceptron block with scalar/vector channels."""

    def __init__(self, in_dims, out_dims, activations=(pt.nn.SiLU(), None), vector_gate=True, eps=1e-8):
        super().__init__()
        si, vi = in_dims
        so, vo = out_dims
        self.si, self.vi, self.so, self.vo = int(si), int(vi), int(so), int(vo)
        self.scalar_act, self.vector_act = activations
        self.vector_gate = bool(vector_gate)
        self.eps = float(eps)
        hidden_v = max(self.vi, self.vo, 1)
        self.wh = pt.nn.Linear(self.vi, hidden_v, bias=False) if self.vi > 0 else None
        self.wv = pt.nn.Linear(hidden_v, self.vo, bias=False) if self.vo > 0 else None
        scalar_in = self.si + hidden_v
        self.ws = pt.nn.Linear(scalar_in, self.so)
        self.wg = pt.nn.Linear(self.so, self.vo) if self.vector_gate and self.vo > 0 else None

    def forward(self, s, v):
        if self.vi > 0 and self.vo > 0:
            vh = self.wh(v.transpose(-1, -2)).transpose(-1, -2)
            vn = pt.linalg.norm(vh, dim=-1)
            s_out = self.ws(pt.cat([s, vn], dim=-1))
            v_out = self.wv(vh.transpose(-1, -2)).transpose(-1, -2)
        elif self.vi > 0:
            vh = self.wh(v.transpose(-1, -2)).transpose(-1, -2)
            vn = pt.linalg.norm(vh, dim=-1)
            s_out = self.ws(pt.cat([s, vn], dim=-1))
            v_out = v.new_zeros(*s.shape[:-1], self.vo, 3)
        else:
            extra = s.new_zeros(*s.shape[:-1], 1)
            s_out = self.ws(pt.cat([s, extra], dim=-1))
            v_out = s.new_zeros(*s.shape[:-1], self.vo, 3)
        if self.scalar_act is not None:
            s_out = self.scalar_act(s_out)
        if self.vo > 0:
            if self.vector_gate:
                gate = pt.sigmoid(self.wg(s_out)).unsqueeze(-1)
                v_out = v_out * gate
            elif self.vector_act is not None:
                v_out = self.vector_act(v_out)
        return s_out, v_out


class GVPDropout(pt.nn.Module):
    def __init__(self, scalar_p=0.1, vector_p=0.1):
        super().__init__()
        self.scalar_dropout = pt.nn.Dropout(scalar_p)
        self.vector_p = float(vector_p)

    def forward(self, s, v):
        s = self.scalar_dropout(s)
        if self.training and self.vector_p > 0 and v.numel() > 0:
            keep = (pt.rand(*v.shape[:-1], 1, device=v.device) > self.vector_p).to(dtype=v.dtype)
            v = v * keep / max(1.0 - self.vector_p, 1e-6)
        return s, v


class FullGVPMessageLayer(pt.nn.Module):
    """Full residue-level GVP message passing layer with edge scalar/vector features."""

    def __init__(self, node_dims, edge_dims, dropout=0.1):
        super().__init__()
        ns, nv = node_dims
        es, ev = edge_dims
        msg_in = (ns * 2 + es, nv * 2 + ev)
        self.msg = pt.nn.Sequential()
        self.msg_gvp1 = GVP(msg_in, (ns, nv))
        self.msg_norm1 = GVPTupleLayerNorm(ns, nv)
        self.msg_gvp2 = GVP((ns, nv), (ns, nv))
        self.msg_norm2 = GVPTupleLayerNorm(ns, nv)
        self.dropout = GVPDropout(dropout, dropout)
        self.node_norm = GVPTupleLayerNorm(ns, nv)
        self.ff_gvp1 = GVP((ns, nv), (ns * 2, nv * 2))
        self.ff_gvp2 = GVP((ns * 2, nv * 2), (ns, nv), activations=(None, None))
        self.ff_norm = GVPTupleLayerNorm(ns, nv)

    def forward(self, s, v, edge_s, edge_v, nn_idx):
        neigh_s = s[nn_idx]
        neigh_v = v[nn_idx]
        base_s = s.unsqueeze(1).expand_as(neigh_s)
        base_v = v.unsqueeze(1).expand_as(neigh_v)
        msg_s_in = pt.cat([base_s, neigh_s, edge_s], dim=-1)
        msg_v_in = pt.cat([base_v, neigh_v, edge_v], dim=-2)
        msg_s, msg_v = self.msg_gvp1(msg_s_in, msg_v_in)
        msg_s, msg_v = self.msg_norm1(msg_s, msg_v)
        msg_s, msg_v = self.msg_gvp2(msg_s, msg_v)
        msg_s = msg_s.mean(dim=1)
        msg_v = msg_v.mean(dim=1)
        msg_s, msg_v = self.dropout(msg_s, msg_v)
        s, v = self.node_norm(s + msg_s, v + msg_v)
        ff_s, ff_v = self.ff_gvp1(s, v)
        ff_s, ff_v = self.ff_gvp2(ff_s, ff_v)
        ff_s, ff_v = self.dropout(ff_s, ff_v)
        return self.ff_norm(s + ff_s, v + ff_v)
