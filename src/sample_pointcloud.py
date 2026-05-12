"""Échantillonnage d'un nuage de points à la surface d'un maillage."""

from pathlib import Path

import numpy as np
import trimesh


def sample_points(mesh: trimesh.Trimesh, n_points: int = 8000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=int(rng.integers(1 << 31)))
    return np.asarray(points, dtype=np.float64)


def sample_points_with_normals(
    mesh: trimesh.Trimesh, n_points: int = 8000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    points, face_idx = trimesh.sample.sample_surface(
        mesh, n_points, seed=int(rng.integers(1 << 31))
    )
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return np.asarray(points, dtype=np.float64), normals


def save_xyz(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, points, fmt="%.6f")


def save_ply(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = trimesh.PointCloud(points)
    cloud.export(path)


def load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".xyz":
        return np.loadtxt(path, dtype=np.float64)
    if suffix == ".ply":
        cloud = trimesh.load(path, process=False)
        return np.asarray(cloud.vertices, dtype=np.float64)
    raise ValueError(f"Format non supporté : {suffix}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    mesh = trimesh.load(root / "data" / "object.ply", process=False)
    points = sample_points(mesh, n_points=8000)
    save_ply(points, root / "data" / "pointcloud.ply")
    save_xyz(points, root / "data" / "pointcloud.xyz")
    print(f"Nuage de points sauvegardé : {len(points)} points")
