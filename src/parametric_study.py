"""Étude paramétrique : influence de k, du nombre de points et de la résolution."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .create_object import build_object
from .hoppe_reconstruction import reconstruct
from .sample_pointcloud import sample_points
from .visualize import _set_equal_axes

K_DEFAULT = 12
N_DEFAULT = 8000
RES_DEFAULT = 96

K_VALUES = [4, 12, 32]
N_VALUES = [2000, 8000, 30000]
RES_VALUES = [48, 96, 160]


def _plot(ax, mesh: trimesh.Trimesh, title: str) -> None:
    triangles = mesh.vertices[mesh.faces]
    poly = Poly3DCollection(
        triangles, facecolor="#cccccc", edgecolor="black", linewidth=0.05, alpha=1.0
    )
    ax.add_collection3d(poly)
    _set_equal_axes(ax, mesh.vertices)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=20, azim=35)


def _run(mesh: trimesh.Trimesh, n: int, k: int, res: int) -> trimesh.Trimesh:
    points = sample_points(mesh, n_points=n, seed=0)
    t0 = perf_counter()
    recon = reconstruct(points, k_normals=k, k_graph=k, resolution=res, padding=0.1)
    print(f"      n={n:>5d}  k={k:>2d}  res={res:>3d}  -> "
          f"{len(recon.faces):>6d} faces  ({perf_counter() - t0:.1f}s)")
    return recon


def run_study(save_path: Path) -> None:
    mesh = build_object()

    print("[1/3] Variation de k (n=8000, res=96)...")
    row_k = [_run(mesh, N_DEFAULT, k, RES_DEFAULT) for k in K_VALUES]

    print("[2/3] Variation de n (k=12, res=96)...")
    row_n = [_run(mesh, n, K_DEFAULT, RES_DEFAULT) for n in N_VALUES]

    print("[3/3] Variation de la résolution (n=8000, k=12)...")
    row_r = [_run(mesh, N_DEFAULT, K_DEFAULT, r) for r in RES_VALUES]

    fig = plt.figure(figsize=(15, 13))
    rows = [
        ("k", K_VALUES, row_k, f"n={N_DEFAULT}, res={RES_DEFAULT}"),
        ("n", N_VALUES, row_n, f"k={K_DEFAULT}, res={RES_DEFAULT}"),
        ("res", RES_VALUES, row_r, f"n={N_DEFAULT}, k={K_DEFAULT}"),
    ]
    for r, (param, values, recons, fixed) in enumerate(rows):
        for c, (val, recon) in enumerate(zip(values, recons)):
            ax = fig.add_subplot(3, 3, r * 3 + c + 1, projection="3d")
            _plot(ax, recon, f"{param}={val}  ({fixed})")

    fig.suptitle(
        "Influence de k (kNN), du nombre de points n et de la résolution",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardée : {save_path}")
    plt.show()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_study(root / "data" / "parametric_study.png")
