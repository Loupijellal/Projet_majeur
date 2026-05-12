"""Pipeline complet : objet 3D -> nuage de points -> reconstructions de Hoppe."""

from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
from src.create_object import build_object, save_object
from src.hoppe_reconstruction import reconstruct, save_mesh
from src.sample_pointcloud import (
    sample_points,
    sample_points_with_normals,
    save_ply,
    save_xyz,
)
from src.visualize import show


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    object_path = data_dir / "object.ply"
    cloud_ply = data_dir / "pointcloud.ply"
    cloud_xyz = data_dir / "pointcloud.xyz"

    print("[1/4] Construction de l'objet 3D...")
    mesh = build_object()
    save_object(mesh, object_path)
    print(f"      {len(mesh.vertices)} sommets, {len(mesh.faces)} faces")

    print("[2/4] Échantillonnage du nuage de points (baseline)...")
    points = sample_points(mesh, n_points=8000)
    save_ply(points, cloud_ply)
    save_xyz(points, cloud_xyz)
    print(f"      {len(points)} points")

    print("[3/4] Reconstruction baseline (8000 pts, k=16, res=96)...")
    recon = reconstruct(points, k_normals=16, k_graph=16, resolution=96, padding=0.1)
    save_mesh(recon, data_dir / "reconstructed.ply")
    print(f"      {len(recon.vertices)} sommets, {len(recon.faces)} faces")

    print("[4/4] Reconstructions diagnostiques...")
    points_hq, normals_hq = sample_points_with_normals(mesh, n_points=30000)

    print("      (a) haute qualité (30000 pts, k=8, res=160)...")
    recon_hq = reconstruct(
        points_hq, k_normals=8, k_graph=12, resolution=160, padding=0.1
    )
    save_mesh(recon_hq, data_dir / "reconstructed_hq.ply")
    print(f"          {len(recon_hq.vertices)} sommets, {len(recon_hq.faces)} faces")

    print("      (b) normales vraies (30000 pts, res=160)...")
    recon_gt = reconstruct(
        points_hq, normals=normals_hq, resolution=160, padding=0.1
    )
    save_mesh(recon_gt, data_dir / "reconstructed_gt.ply")
    print(f"          {len(recon_gt.vertices)} sommets, {len(recon_gt.faces)} faces")

    show(
    object_path=data_dir / "object.ply",
    cloud_path=data_dir / "pointcloud.ply",
    recon_path=data_dir / "reconstructed.ply",
    save_path=data_dir / "comparison.png",
    )


if __name__ == "__main__":
    main()
