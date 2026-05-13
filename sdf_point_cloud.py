"""Génère un nuage de points par rejection sampling sur une SDF analytique."""

import numpy as np


CUBE_HALF  = 0.5
CYLINDER_R = 0.35
CYLINDER_H = 2.0
HOLE_R     = 0.30
HOLE_H     = 0.6


def sdf_cylinder(p, radius, half_height):
    d_xz = np.linalg.norm(p[:, [0, 2]], axis=1) - radius
    d_y  = np.abs(p[:, 1]) - half_height
    d    = np.column_stack([d_xz, d_y])
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(np.max(d, axis=1), 0.0)


def sdf_box(p, half_extents):
    q = np.abs(p) - np.asarray(half_extents)
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)


def sdf_scene(p):
    d_cyl  = sdf_cylinder(p, CYLINDER_R, CYLINDER_H)
    d_cube = sdf_box(p, [CUBE_HALF] * 3)

    d_hole_x = sdf_cylinder(p[:, [1, 0, 2]], HOLE_R, HOLE_H)
    d_hole_y = sdf_cylinder(p, HOLE_R, HOLE_H)
    d_hole_z = sdf_cylinder(p[:, [0, 2, 1]], HOLE_R, HOLE_H)

    d_cube = np.maximum(d_cube, -d_hole_x)
    d_cube = np.maximum(d_cube, -d_hole_y)
    d_cube = np.maximum(d_cube, -d_hole_z)
    d_cyl  = np.maximum(d_cyl,  -d_hole_x)
    d_cyl  = np.maximum(d_cyl,  -d_hole_z)
    return np.minimum(d_cyl, d_cube)


def generate_point_cloud(n_samples=500_000, surface_threshold=0.02,
                         noise_std=0.005, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    bbox = np.array([[-0.6, -2.1, -0.6], [0.6, 2.1, 0.6]])
    pts = rng.uniform(bbox[0], bbox[1], (n_samples, 3)).astype(np.float32)

    sdf_vals = np.empty(n_samples, dtype=np.float32)
    for i in range(0, n_samples, 50_000):
        sdf_vals[i:i + 50_000] = sdf_scene(pts[i:i + 50_000])

    surface = pts[np.abs(sdf_vals) < surface_threshold]
    surface += rng.normal(0, noise_std, surface.shape).astype(np.float32)
    return surface


if __name__ == "__main__":
    pts = generate_point_cloud()
    print(f"Nuage généré : {pts.shape[0]} points")
    np.save("point_cloud.npy", pts)
    print("Sauvegardé : point_cloud.npy")
