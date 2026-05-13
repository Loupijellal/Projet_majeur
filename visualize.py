"""Comparaison visuelle : nuage de points · Hoppe · IGR."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CLOUD = Path("point_cloud.npy")
HOPPE = Path("data/reconstructed_hoppe.ply")
IGR   = Path("data/reconstructed_igr.ply")
OUT   = Path("data/comparison.png")
ELEV, AZIM = 22, 35


def set_equal_axes(ax, pts):
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    c, r = (mn + mx) / 2, (mx - mn).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def plot_cloud(ax, pts, title):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), min(15_000, len(pts)), replace=False)
    sub = pts[idx]
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2],
               s=0.4, c=sub[:, 1], cmap="viridis", alpha=0.7)
    set_equal_axes(ax, pts)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_title(title)


def plot_mesh(ax, mesh, title, color):
    tri = mesh.vertices[mesh.faces]
    ax.add_collection3d(Poly3DCollection(tri, facecolor=color,
                                         edgecolor="black", linewidth=0.04))
    set_equal_axes(ax, mesh.vertices)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_title(title)


if __name__ == "__main__":
    for p in (CLOUD, HOPPE, IGR):
        if not p.exists():
            raise FileNotFoundError(f"Manquant : {p}")

    pts = np.load(CLOUD)
    hoppe = trimesh.load(HOPPE, process=False)
    igr = trimesh.load(IGR, process=False)

    fig = plt.figure(figsize=(16, 5.5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    plot_cloud(ax1, pts, f"Nuage SDF ({len(pts)} pts)")
    plot_mesh(ax2, hoppe,
              f"Hoppe — {len(hoppe.vertices)} sommets · {len(hoppe.faces)} faces",
              "#d9d9d9")
    plot_mesh(ax3, igr,
              f"IGR — {len(igr.vertices)} sommets · {len(igr.faces)} faces",
              "#cfe2f3")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Figure sauvegardée : {OUT}")
    plt.show()
