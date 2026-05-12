"""Affichage côte à côte : objet original, nuage de points, reconstruction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = (mn + mx) / 2
    radius = (mx - mn).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def _plot_mesh(ax, mesh: trimesh.Trimesh, color: str, title: str) -> None:
    triangles = mesh.vertices[mesh.faces]
    poly = Poly3DCollection(
        triangles, facecolor=color, edgecolor="black", linewidth=0.05, alpha=1.0
    )
    ax.add_collection3d(poly)
    _set_equal_axes(ax, mesh.vertices)
    ax.set_title(title)


def _plot_points(ax, points: np.ndarray, title: str) -> None:
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.6, c="black")
    _set_equal_axes(ax, points)
    ax.set_title(title)


def show(
    object_path: Path,
    cloud_path: Path,
    recon_path: Path,
    save_path: Path | None = None,
) -> None:
    mesh = trimesh.load(object_path, process=False)
    cloud = trimesh.load(cloud_path, process=False)
    recon = trimesh.load(recon_path, process=False)
    points = np.asarray(cloud.vertices)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    _plot_mesh(ax1, mesh, color="#f0e050", title="Objet 3D original")
    _plot_points(ax2, points, title=f"Nuage de points ({len(points)} pts)")
    _plot_mesh(ax3, recon, color="#cccccc", title="Reconstruction (Hoppe)")

    for ax in (ax1, ax2, ax3):
        ax.view_init(elev=20, azim=35)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {save_path}")
    plt.show()


def compare_reconstructions(
    paths: dict[str, Path],
    save_path: Path | None = None,
) -> None:
    n = len(paths)
    fig = plt.figure(figsize=(5 * n, 5))
    for i, (title, path) in enumerate(paths.items(), start=1):
        ax = fig.add_subplot(1, n, i, projection="3d")
        mesh = trimesh.load(path, process=False)
        _plot_mesh(ax, mesh, color="#cccccc", title=title)
        ax.view_init(elev=20, azim=35)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {save_path}")
    plt.show()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    show(
        object_path=data / "object.ply",
        cloud_path=data / "pointcloud.ply",
        recon_path=data / "reconstructed.ply",
        save_path=data / "comparison.png",
    )
    compare_reconstructions(
        {
            "Baseline (8k pts, k=16, res=96)": data / "reconstructed.ply",
            "Haute qualité (30k pts, k=8, res=160)": data / "reconstructed_hq.ply",
            "Normales vraies (30k pts, res=160)": data / "reconstructed_gt.ply",
        },
        save_path=data / "comparison_reconstructions.png",
    )
