"""Application de la reconstruction de Hoppe sur le nuage SDF (point_cloud.npy)."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.hoppe_reconstruction import reconstruct, save_mesh
from src.visualize import _set_equal_axes

CLOUD_PATH = Path("point_cloud.npy")
OUT_PLY = Path("data/reconstructed_sdf.ply")
OUT_PNG = Path("data/reconstructed_sdf.png")

K = 32
RES = 160
PADDING = 0.1


def main() -> None:
    if not CLOUD_PATH.exists():
        raise FileNotFoundError(
            f"{CLOUD_PATH} introuvable. Lancez d'abord : python3.11 src/sdf_point_cloud.py"
        )

    pts = np.load(CLOUD_PATH).astype(np.float64)
    print(f"Nuage chargé : {pts.shape[0]} points")

    t0 = perf_counter()
    mesh = reconstruct(pts, k_normals=K, k_graph=K, resolution=RES, padding=PADDING)
    elapsed = perf_counter() - t0
    print(f"Reconstruction Hoppe (k={K}, res={RES}) : "
          f"{len(mesh.vertices)} sommets, {len(mesh.faces)} faces ({elapsed:.1f}s)")

    save_mesh(mesh, OUT_PLY)
    print(f"Maillage  : {OUT_PLY}")

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    idx = np.random.default_rng(0).choice(len(pts), min(10000, len(pts)), replace=False)
    sub = pts[idx]
    ax1.scatter(
        sub[:, 0], sub[:, 1], sub[:, 2],
        s=0.5, c=sub[:, 1], cmap="viridis", alpha=0.6,
    )
    _set_equal_axes(ax1, pts)
    ax1.view_init(elev=20, azim=35)
    ax1.set_title(f"Nuage SDF ({len(pts)} pts)")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    tri = mesh.vertices[mesh.faces]
    ax2.add_collection3d(
        Poly3DCollection(tri, facecolor="#cccccc", edgecolor="black", linewidth=0.05)
    )
    _set_equal_axes(ax2, mesh.vertices)
    ax2.view_init(elev=20, azim=35)
    ax2.set_title(f"Reconstruction Hoppe (k={K}, res={RES})")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Figure   : {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
