"""Génération d'un objet 3D : cube avec cavités sphériques traversé par un cylindre."""

from pathlib import Path

import numpy as np
import trimesh


def build_object(
    cube_size: float = 1.0,
    sphere_radius: float = 0.55,
    cylinder_radius: float = 0.25,
    cylinder_height: float = 3.0,
    sections: int = 64,
) -> trimesh.Trimesh:
    cube = trimesh.creation.box(extents=(cube_size, cube_size, cube_size))
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=sphere_radius)
    cube_carved = trimesh.boolean.difference([cube, sphere], engine="manifold")

    cylinder = trimesh.creation.cylinder(
        radius=cylinder_radius, height=cylinder_height, sections=sections
    )

    mesh = trimesh.boolean.union([cube_carved, cylinder], engine="manifold")
    mesh.process(validate=True)
    return mesh


def save_object(mesh: trimesh.Trimesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "object.ply"
    mesh = build_object()
    save_object(mesh, out)
    print(f"Objet 3D sauvegardé : {out} ({len(mesh.vertices)} sommets, {len(mesh.faces)} faces)")
