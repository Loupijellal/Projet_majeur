"""Reconstruction unique avec paramètres personnalisés."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.create_object import build_object
from src.hoppe_reconstruction import reconstruct, save_mesh
from src.sample_pointcloud import sample_points
from src.visualize import _set_equal_axes

N = 30000
K = 32
RES = 160

mesh = build_object()
points = sample_points(mesh, n_points=N, seed=0)

t0 = perf_counter()
recon = reconstruct(points, k_normals=K, k_graph=K, resolution=RES, padding=0.1)
elapsed = perf_counter() - t0
print(f"n={N}, k={K}, res={RES}  ->  "
      f"{len(recon.vertices)} sommets, {len(recon.faces)} faces  ({elapsed:.1f}s)")

stem = f"reconstructed_n{N}_k{K}_res{RES}"
ply_path = Path("data") / f"{stem}.ply"
png_path = Path("data") / f"{stem}.png"
save_mesh(recon, ply_path)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
tri = recon.vertices[recon.faces]
ax.add_collection3d(
    Poly3DCollection(tri, facecolor="#cccccc", edgecolor="black", linewidth=0.05)
)
_set_equal_axes(ax, recon.vertices)
ax.view_init(elev=20, azim=35)
ax.set_title(f"Hoppe : n={N}, k={K}, res={RES}")
fig.tight_layout()
fig.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"Maillage : {ply_path}")
print(f"Figure  : {png_path}")
plt.show()
