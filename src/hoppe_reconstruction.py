"""Reconstruction de surface par la méthode de Hoppe (1992)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from skimage import measure

from .sample_pointcloud import load_points


@dataclass
class TangentPlanes:
    centroids: np.ndarray   # (N, 3) o_i
    normals: np.ndarray     # (N, 3) n_i orientées


def estimate_tangent_planes(points: np.ndarray, k: int) -> TangentPlanes:
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)

    neighborhoods = points[idx]                       # (N, k, 3)
    centroids = neighborhoods.mean(axis=1)            # (N, 3)
    centered = neighborhoods - centroids[:, None, :]  # (N, k, 3)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k

    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]                        # plus petite valeur propre
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return TangentPlanes(centroids=centroids, normals=normals)


def orient_normals_mst(planes: TangentPlanes, k: int) -> np.ndarray:
    centroids = planes.centroids
    normals = planes.normals.copy()
    n = len(centroids)

    tree = cKDTree(centroids)
    _, idx = tree.query(centroids, k=k)

    rows, cols, weights = [], [], []
    for i in range(n):
        for j in idx[i, 1:]:
            w = 1.0 - abs(float(normals[i] @ normals[j]))
            rows.append(i); cols.append(int(j)); weights.append(w + 1e-9)
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))

    mst = minimum_spanning_tree(graph)
    mst = mst + mst.T

    seed = int(np.argmax(centroids[:, 2]))
    if normals[seed, 2] < 0:
        normals[seed] = -normals[seed]

    visited = np.zeros(n, dtype=bool)
    visited[seed] = True
    stack = [seed]
    mst_csr = mst.tocsr()
    while stack:
        i = stack.pop()
        start, end = mst_csr.indptr[i], mst_csr.indptr[i + 1]
        for j in mst_csr.indices[start:end]:
            j = int(j)
            if visited[j]:
                continue
            if normals[i] @ normals[j] < 0:
                normals[j] = -normals[j]
            visited[j] = True
            stack.append(j)

    return normals


def signed_distance_grid(
    planes: TangentPlanes, resolution: int, padding: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroids = planes.centroids
    normals = planes.normals

    mn = centroids.min(axis=0) - padding
    mx = centroids.max(axis=0) + padding
    xs = np.linspace(mn[0], mx[0], resolution)
    ys = np.linspace(mn[1], mx[1], resolution)
    zs = np.linspace(mn[2], mx[2], resolution)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    tree = cKDTree(centroids)
    _, nearest = tree.query(grid, k=1)
    f = np.einsum("ij,ij->i", grid - centroids[nearest], normals[nearest])
    return f.reshape(resolution, resolution, resolution), mn, mx


def reconstruct(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    k_normals: int = 16,
    k_graph: int = 16,
    resolution: int = 96,
    padding: float = 0.1,
) -> trimesh.Trimesh:
    if normals is None:
        planes = estimate_tangent_planes(points, k=k_normals)
        planes.normals = orient_normals_mst(planes, k=k_graph)
    else:
        planes = TangentPlanes(centroids=points.copy(), normals=normals.copy())

    f, mn, mx = signed_distance_grid(planes, resolution=resolution, padding=padding)
    spacing = ((mx - mn) / (resolution - 1)).tolist()

    verts, faces, _, _ = measure.marching_cubes(f, level=0.0, spacing=spacing)
    verts += mn
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def save_mesh(mesh: trimesh.Trimesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    points = load_points(root / "data" / "pointcloud.ply")
    mesh = reconstruct(points)
    save_mesh(mesh, root / "data" / "reconstructed.ply")
    print(f"Maillage reconstruit : {len(mesh.vertices)} sommets, {len(mesh.faces)} faces")
