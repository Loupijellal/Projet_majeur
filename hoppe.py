"""
hoppe.py
Reconstruction de surface par la méthode de Hoppe (1992).
Supporte les nuages de points .npy ET .ply en entrée.
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from skimage import measure

from point_cloud_io import load_cloud, save_cloud


# ─────────────────────────────────────────────
#  PLANS TANGENTS
# ─────────────────────────────────────────────

@dataclass
class TangentPlanes:
    centroids: np.ndarray
    normals:   np.ndarray


def estimate_tangent_planes(points: np.ndarray, k: int) -> TangentPlanes:
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    neighborhoods = points[idx]
    centroids = neighborhoods.mean(axis=1)
    centered  = neighborhoods - centroids[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    _, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return TangentPlanes(centroids=centroids, normals=normals)


# ─────────────────────────────────────────────
#  ORIENTATION DES NORMALES
# ─────────────────────────────────────────────

def orient_normals_mst(planes: TangentPlanes, k: int) -> np.ndarray:
    centroids = planes.centroids
    normals   = planes.normals.copy()
    n = len(centroids)

    tree = cKDTree(centroids)
    _, idx = tree.query(centroids, k=k)

    rows, cols, weights = [], [], []
    for i in range(n):
        for j in idx[i, 1:]:
            w = 1.0 - abs(float(normals[i] @ normals[j]))
            rows.append(i); cols.append(int(j)); weights.append(w + 1e-9)

    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    mst   = minimum_spanning_tree(graph)
    mst   = (mst + mst.T).tocsr()

    seed = int(np.argmax(centroids[:, 2]))
    if normals[seed, 2] < 0:
        normals[seed] = -normals[seed]

    visited = np.zeros(n, dtype=bool)
    visited[seed] = True
    stack = [seed]
    while stack:
        i = stack.pop()
        for j in mst.indices[mst.indptr[i]:mst.indptr[i + 1]]:
            j = int(j)
            if visited[j]:
                continue
            if normals[i] @ normals[j] < 0:
                normals[j] = -normals[j]
            visited[j] = True
            stack.append(j)

    return normals


def estimate_normals(points: np.ndarray, k: int = 32) -> np.ndarray:
    planes = estimate_tangent_planes(points, k=k)
    return orient_normals_mst(planes, k=k)


# ─────────────────────────────────────────────
#  GRILLE SDF + MARCHING CUBES
# ─────────────────────────────────────────────

def signed_distance_grid(planes: TangentPlanes,
                         resolution: int,
                         padding: float):
    centroids, normals = planes.centroids, planes.normals
    mn = centroids.min(axis=0) - padding
    mx = centroids.max(axis=0) + padding

    xs = np.linspace(mn[0], mx[0], resolution)
    ys = np.linspace(mn[1], mx[1], resolution)
    zs = np.linspace(mn[2], mx[2], resolution)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    tree = cKDTree(centroids)
    _, nearest = tree.query(grid, k=1)
    f = np.einsum("ij,ij->i",
                  grid - centroids[nearest],
                  normals[nearest])
    return f.reshape(resolution, resolution, resolution), mn, mx


def reconstruct(points:  np.ndarray,
                normals: np.ndarray | None = None,
                k: int = 32,
                resolution: int = 160,
                padding: float = 0.1) -> trimesh.Trimesh:
    if normals is None:
        planes = estimate_tangent_planes(points, k=k)
        planes.normals = orient_normals_mst(planes, k=k)
    else:
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        planes  = TangentPlanes(centroids=points.copy(), normals=normals)

    f, mn, mx = signed_distance_grid(planes, resolution=resolution, padding=padding)
    spacing   = ((mx - mn) / (resolution - 1)).tolist()
    verts, faces, _, _ = measure.marching_cubes(f, level=0.0, spacing=spacing)
    verts += mn
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hoppe 1992 surface reconstruction")
    parser.add_argument("input", nargs="?", default="point_cloud.npy",
                        help="Nuage de points : .npy ou .ply  (défaut: point_cloud_.npy)")
    parser.add_argument("--k",          type=int,   default=32,  help="Voisinage k")
    parser.add_argument("--resolution", type=int,   default=160, help="Résolution grille")
    parser.add_argument("--padding",    type=float, default=0.1, help="Marge bbox")
    parser.add_argument("--out",        default="data/reconstructed_hoppe.ply",
                        help="Fichier de sortie (.ply)")
    args = parser.parse_args()

    # ── Chargement (npy ou ply) ──────────────────────────────────────────
    pts, nrm = load_cloud(args.input)
    pts = pts.astype(np.float64)
    if nrm is not None:
        nrm = nrm.astype(np.float64)
        print("Normales fournies — étape d'orientation ignorée.")
    print(f"Nuage chargé : {pts.shape[0]:,} points")

    # ── Reconstruction ───────────────────────────────────────────────────
    t0   = perf_counter()
    mesh = reconstruct(pts, normals=nrm,
                       k=args.k,
                       resolution=args.resolution,
                       padding=args.padding)
    dt   = perf_counter() - t0
    print(f"Reconstruction : {len(mesh.vertices):,} sommets, "
          f"{len(mesh.faces):,} faces  ({dt:.1f}s)")

    # ── Sauvegarde ───────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"Sauvegardé : {out}")