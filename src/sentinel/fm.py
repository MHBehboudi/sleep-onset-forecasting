"""Frozen MANAS-2 (Pratibha-2) EEG foundation model, used as an extra packet-level input.

A dependency-free re-implementation of the released encoder (MannasAI, Apache-2.0). The weights are
NOT part of this repository; download them from the model's release and point `load_manas2` to the
folder holding `model.safetensors` and `positions.json`.

Muse use: each 5 s packet (4 x 640 samples at 128 Hz, uV) is FFT-resampled to 200 Hz (1000 samples)
with content below 0.5 Hz removed, z-scored per channel within the packet and clipped at +-15 sd, then
encoded with the Muse electrode coordinates. The 5 one-second patch tokens are averaged -> (4, 512).
The encoder is shared by all ensemble members and never trained.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MUSE = ['TP9', 'AF7', 'AF8', 'TP10']


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8) * self.weight


class GEGLU(nn.Module):
    def forward(self, x):
        value, gate = x.chunk(2, dim=-1)
        gelu = 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (gate + 0.044715 * gate.pow(3))))
        return value * (gate * torch.sigmoid(gate) + gate * gelu)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.in_proj = nn.Linear(dim, 8 * dim); self.gate = GEGLU(); self.out_proj = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.out_proj(self.gate(self.in_proj(x)))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads; self.qkv_proj = nn.Linear(dim, 3 * dim); self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v)]
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.out_proj(x.transpose(1, 2).contiguous().view(b, t, d))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.pre_attn_norm = RMSNorm(dim); self.attn = Attention(dim, heads)
        self.pre_ffn_norm = RMSNorm(dim); self.ffn = FeedForward(dim)

    def forward(self, x):
        x = x + self.attn(self.pre_attn_norm(x))
        return x + self.ffn(self.pre_ffn_norm(x))


class Encoder(nn.Module):
    def __init__(self, dim=512, depth=22, heads=8):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim, heads) for _ in range(depth)]); self.final_norm = RMSNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(200, 512, bias=False)

    def forward(self, eeg):
        return self.linear(eeg.unfold(-1, 200, 180))


class PosEnc(nn.Module):
    def __init__(self):
        super().__init__()
        freqs = torch.linspace(1.0, 10.0, 4)
        self.register_buffer("freq_matrix", torch.cartesian_prod(*([freqs] * 4)).T)
        self.fourier_linear = nn.Linear(512, 512, bias=False)
        self.learned_linear = nn.Sequential(nn.Linear(4, 1024, bias=False), GEGLU(), RMSNorm(512))
        self.final_norm = RMSNorm(512)

    def forward(self, coords):
        phases = coords @ self.freq_matrix.T.contiguous().T
        fourier = self.fourier_linear(torch.cat([phases.sin(), phases.cos()], dim=-1))
        return self.final_norm(fourier + self.learned_linear(coords))


class Manas2(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(); self.pos_enc = PosEnc(); self.encoder = Encoder()

    def forward(self, eeg, positions):               # eeg (B,C,T) at 200 Hz; positions (B,C,3) in cm
        b, c, _ = eeg.shape
        tokens = self.patch_embed(eeg); p = tokens.shape[2]
        spatial = positions.unsqueeze(2).expand(-1, -1, p, -1)
        time = torch.arange(p, device=eeg.device, dtype=torch.float32).view(1, 1, p, 1).expand(b, c, -1, -1)
        coords = torch.cat([spatial, time], -1).reshape(b, c * p, 4).to(self.pos_enc.freq_matrix.dtype)
        enc = self.encoder(tokens.flatten(1, 2) + self.pos_enc(coords))
        return enc.reshape(b, c, p, 512).permute(0, 2, 1, 3).contiguous()   # (B, P, C, 512)


def _freeze(m, device):
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device).eval()


def load_manas2(fm_dir, device='cpu'):
    """Load the released weights (safetensors + electrode positions)."""
    from safetensors.torch import load_file
    fm_dir = Path(fm_dir)
    state = load_file(str(fm_dir / 'model.safetensors'))
    for prefix in ("module.", "wrapped_model.", "model.", "hybrid."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    enc = {k: v for k, v in state.items() if k.startswith(("patch_embed.", "pos_enc.", "encoder."))}
    m = Manas2(); m.load_state_dict(enc, strict=True)
    table = json.loads((fm_dir / 'positions.json').read_text())
    m.register_buffer('muse_pos', torch.tensor([table[n] for n in MUSE], dtype=torch.float32) * 100.0)
    return _freeze(m, device)


def save_compact(m, path):
    """One-file copy for deployment packages: {'state': encoder weights, 'muse_pos': (4,3)}."""
    state = {k: v for k, v in m.state_dict().items() if k != 'muse_pos'}
    torch.save({'state': state, 'muse_pos': m.muse_pos.cpu()}, path)


def load_compact(path, device='cpu'):
    pack = torch.load(path, map_location='cpu', weights_only=True)
    m = Manas2(); m.load_state_dict(pack['state'], strict=True)
    m.register_buffer('muse_pos', pack['muse_pos'])
    return _freeze(m, device)


def prep_packets(x):
    """x (N,4,640) uV at 128 Hz -> (N,4,1000) at 200 Hz, >0.5 Hz, per-channel z-scored, clipped."""
    x = torch.nan_to_num(x.float())
    X = torch.fft.rfft(x, dim=-1)                      # 321 bins, 0.2 Hz spacing
    X[..., :3] = 0                                      # drop DC and < 0.5 Hz
    Y = torch.zeros(*X.shape[:-1], 501, dtype=X.dtype, device=X.device)
    Y[..., :321] = X
    y = torch.fft.irfft(Y, n=1000, dim=-1) * (1000 / 640)
    y = (y - y.mean(-1, keepdim=True)) / (y.std(-1, keepdim=True) + 1e-3)
    return y.clamp(-15, 15)


@torch.no_grad()
def embed_packets(fm, x, chunk=4096):
    """x (N,4,640) -> (N,4,512): mean over the 5 one-second patch tokens (bf16 autocast on GPU)."""
    out = []
    for i in range(0, x.shape[0], chunk):
        xx = prep_packets(x[i:i + chunk])
        pos = fm.muse_pos.unsqueeze(0).expand(xx.shape[0], -1, -1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=xx.is_cuda):
            f = fm(xx, pos)
        out.append(f.float().mean(1))
    return torch.cat(out) if out else x.new_zeros(0, 4, 512)


class GraphedFM:
    """CUDA-graph replay of the frozen encoder for the streamed single-packet shape (1,4,1000).

    The 22-layer encoder runs once per 5 s packet for every recording, so kernel-launch overhead dominated
    streaming time. Capturing it as a CUDA graph cut the per-packet cost enough to fit a 7-member ensemble
    inside the 60-minute evaluation budget. Falls back to eager mode on CPU, other shapes, or capture errors.
    """

    def __init__(self, fm):
        self.fm = fm; self.graph = None; self.failed = False

    def _capture(self, dev):
        self.static_in = torch.zeros(1, 4, 1000, device=dev)
        self.pos = self.fm.muse_pos.unsqueeze(0)
        s = torch.cuda.Stream(dev); s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s), torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for _ in range(3):
                self.fm(self.static_in, self.pos)
        torch.cuda.current_stream(dev).wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            self.static_out = self.fm(self.static_in, self.pos)

    @torch.no_grad()
    def __call__(self, x):                       # x (N,4,640) raw packets
        if x.is_cuda and x.shape[0] == 1 and not self.failed:
            try:
                if self.graph is None:
                    self._capture(x.device)
                self.static_in.copy_(prep_packets(x))
                self.graph.replay()
                return self.static_out.float().mean(1).clone()
            except Exception:
                self.failed = True; self.graph = None
        return embed_packets(self.fm, x)
