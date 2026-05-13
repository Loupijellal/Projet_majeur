"""IGR (Gropp et al. 2020) : surface implicite par perte eikonale + normales."""

import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import trimesh
from skimage import measure


# ───────────────────────── Modèle ─────────────────────────

class IGRNet(nn.Module):
    def __init__(self, d_in=3, d_hidden=512, n_layers=8, skip_layer=4,
                 beta=100.0, radius_init=0.5):
        super().__init__()
        self.d_in = d_in
        self.skip_in = {skip_layer}
        dims = [d_in] + [d_hidden] * n_layers + [1]
        last = len(dims) - 2

        layers = []
        for i in range(len(dims) - 1):
            in_f, out_f = dims[i], dims[i + 1]
            if (i + 1) in self.skip_in:
                out_f -= d_in
            lin = nn.Linear(in_f, out_f)
            self._init_atzmon(lin, i, last, radius_init, d_in, i in self.skip_in)
            layers.append(nn_utils.weight_norm(lin))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.Softplus(beta=beta)

    @staticmethod
    def _init_atzmon(lin, i, last, radius_init, d_in, is_skip_input):
        out_f, in_f = lin.weight.shape
        if i == last:
            nn.init.constant_(lin.weight, math.sqrt(math.pi) / math.sqrt(in_f))
            nn.init.constant_(lin.bias, -radius_init)
        else:
            nn.init.normal_(lin.weight, 0.0, math.sqrt(2.0) / math.sqrt(out_f))
            nn.init.constant_(lin.bias, 0.0)
            if is_skip_input:
                lin.weight.data[:, -d_in:].zero_()

    def forward(self, x):
        h = x
        for i, lin in enumerate(self.layers):
            if i in self.skip_in:
                h = torch.cat([h, x], dim=-1) / math.sqrt(2.0)
            h = lin(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def gradient(self, x, create_graph=True):
        x = x.requires_grad_(True)
        u = self(x)
        (g,) = torch.autograd.grad(
            u, x, torch.ones_like(u),
            create_graph=create_graph, retain_graph=create_graph,
        )
        return g


# ───────────────────────── Loss ─────────────────────────

@dataclass
class LossTerms:
    value: torch.Tensor
    normal: torch.Tensor
    eikonal: torch.Tensor
    total: torch.Tensor


def igr_loss(model, surface_pts, surface_normals, eik_pts,
             tau=1.0, lambda_eik=0.1):
    surface_pts = surface_pts.requires_grad_(True)
    u_s = model(surface_pts)
    value = u_s.abs().mean()

    grad_s = torch.autograd.grad(
        u_s, surface_pts, torch.ones_like(u_s),
        create_graph=True, retain_graph=True,
    )[0]
    normal = (grad_s - surface_normals).norm(dim=-1).mean()

    eik_pts = eik_pts.requires_grad_(True)
    u_e = model(eik_pts)
    grad_e = torch.autograd.grad(
        u_e, eik_pts, torch.ones_like(u_e),
        create_graph=True, retain_graph=True,
    )[0]
    eikonal = ((grad_e.norm(dim=-1) - 1.0) ** 2).mean()

    total = value + tau * normal + lambda_eik * eikonal
    return LossTerms(value, normal, eikonal, total)


def sample_eikonal_points(surface_pts, n_uniform, n_jitter,
                          bbox=(-1.1, 1.1), jitter_sigma=0.1):
    device = surface_pts.device
    uniform = torch.empty(n_uniform, 3, device=device).uniform_(*bbox)
    if n_jitter == 0:
        return uniform
    idx = torch.randint(0, surface_pts.shape[0], (n_jitter,), device=device)
    jitter = surface_pts[idx] + jitter_sigma * torch.randn(n_jitter, 3, device=device)
    return torch.cat([uniform, jitter], dim=0)


# ───────────────────────── Données ─────────────────────────

@dataclass
class Cloud:
    points: np.ndarray
    normals: np.ndarray
    center: np.ndarray
    scale: float

    def denormalize(self, verts):
        return verts * self.scale + self.center


def normalize_cloud(points, normals):
    pts = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    scale = float(np.abs(centered).max())
    pts_n = (centered / scale).astype(np.float32)
    nrm_n = (nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    return Cloud(pts_n, nrm_n, center.astype(np.float32), scale)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ───────────────────────── Entraînement ─────────────────────────

def train(model, cloud, n_iters=3000, batch_size=8192,
          tau=1.0, lambda_eik=0.1, lr=1e-4, device=None, log_every=100):
    device = device or pick_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pts = torch.from_numpy(cloud.points).to(device)
    nrm = torch.from_numpy(cloud.normals).to(device)
    n_pts = pts.shape[0]

    t0 = perf_counter()
    for it in range(1, n_iters + 1):
        idx = torch.randint(0, n_pts, (batch_size,), device=device)
        surface = pts[idx]
        normals = nrm[idx]
        eik = sample_eikonal_points(surface, n_uniform=batch_size,
                                    n_jitter=batch_size // 8)

        loss = igr_loss(model, surface, normals, eik,
                        tau=tau, lambda_eik=lambda_eik)
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        optimizer.step()

        if it % log_every == 0 or it == 1:
            print(f"[iter {it:>5}/{n_iters}] "
                  f"L_value={loss.value.item():.3e}  "
                  f"L_normal={loss.normal.item():.3e}  "
                  f"L_eik={loss.eikonal.item():.3e}  "
                  f"L_total={loss.total.item():.3e}")
    print(f"Temps entraînement : {perf_counter() - t0:.1f}s")
    return model


# ───────────────────────── Extraction mesh ─────────────────────────

@torch.no_grad()
def evaluate_grid(model, resolution, bbox=(-1.1, 1.1), chunk=65536, device=None):
    device = device or next(model.parameters()).device
    xs = torch.linspace(bbox[0], bbox[1], resolution, device=device)
    gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing="ij")
    grid = torch.stack([gx.ravel(), gy.ravel(), gz.ravel()], dim=1)

    out = torch.empty(grid.shape[0], device=device)
    for i in range(0, grid.shape[0], chunk):
        out[i:i + chunk] = model(grid[i:i + chunk]).squeeze(-1)
    return out.reshape(resolution, resolution, resolution).cpu().numpy(), bbox


def extract_mesh(model, resolution=256, bbox=(-1.1, 1.1), device=None):
    field, _ = evaluate_grid(model, resolution, bbox=bbox, device=device)
    spacing = (bbox[1] - bbox[0]) / (resolution - 1)
    verts, faces, _, _ = measure.marching_cubes(field, level=0.0,
                                                spacing=(spacing,) * 3)
    verts += bbox[0]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


# ───────────────────────── Évaluation ─────────────────────────

def chamfer_distance(points, mesh, n_samples=50_000, rng_seed=0):
    from scipy.spatial import cKDTree
    surf, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=rng_seed)
    pc_to_mesh = cKDTree(surf).query(points, k=1)[0].mean()
    mesh_to_pc = cKDTree(points).query(surf, k=1)[0].mean()
    return float(pc_to_mesh), float(mesh_to_pc), float((pc_to_mesh + mesh_to_pc) / 2)


# ───────────────────────── Pipeline ─────────────────────────

if __name__ == "__main__":
    pts = np.load("point_cloud.npy")
    nrm_path = Path("point_cloud_normals.npy")
    if nrm_path.exists():
        normals = np.load(nrm_path)
    else:
        from hoppe import estimate_normals
        print("Estimation des normales (Hoppe)...")
        normals = estimate_normals(pts.astype(np.float64), k=32)
        np.save(nrm_path, normals.astype(np.float32))

    cloud = normalize_cloud(pts, normals)
    print(f"Nuage : {cloud.points.shape[0]} points  scale={cloud.scale:.3f}")

    device = pick_device()
    print(f"Device : {device}")

    model = IGRNet()
    train(model, cloud, n_iters=3000, batch_size=8192, device=device)

    ckpt_path = Path("data/igr_checkpoint.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "center": cloud.center, "scale": cloud.scale}, ckpt_path)
    print(f"Checkpoint sauvegardé : {ckpt_path}")

    print("Extraction marching cubes (256³)...")
    t0 = perf_counter()
    mesh = extract_mesh(model, resolution=256, device=device)
    mesh.vertices = cloud.denormalize(mesh.vertices)
    print(f"  {len(mesh.vertices)} sommets, {len(mesh.faces)} faces "
          f"({perf_counter() - t0:.1f}s)")

    out = Path("data/reconstructed_igr.ply")
    mesh.export(out)
    print(f"Sauvegardé : {out}")

    print("\nDistance de Chamfer (nuage ↔ mesh) :")
    p2m, m2p, sym = chamfer_distance(pts.astype(np.float64), mesh)
    print(f"  IGR    : pc→mesh={p2m:.5f}  mesh→pc={m2p:.5f}  sym={sym:.5f}")
    hoppe_path = Path("data/reconstructed_hoppe.ply")
    if hoppe_path.exists():
        hoppe_mesh = trimesh.load(hoppe_path, process=False)
        p2m, m2p, sym = chamfer_distance(pts.astype(np.float64), hoppe_mesh)
        print(f"  Hoppe  : pc→mesh={p2m:.5f}  mesh→pc={m2p:.5f}  sym={sym:.5f}")
