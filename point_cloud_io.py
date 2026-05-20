"""
point_cloud_io.py
Chargement / sauvegarde d'un nuage de points depuis/vers .npy ou .ply.

Formats supportés en lecture :
  .npy  — tableau NumPy (N,3) ou (N,6) [xyz + normals]
  .ply  — nuage de points PLY via trimesh
            · attributs attendus : x, y, z  (obligatoires)
            · attributs optionnels : nx, ny, nz  (normales)

Usage rapide :
    from point_cloud_io import load_cloud, save_cloud

    pts, nrm = load_cloud("scan.ply")    # nrm peut être None
    save_cloud("out.ply", pts, nrm)
"""

from pathlib import Path
import numpy as np


# ─────────────────────────────────────────────
#  LECTURE
# ─────────────────────────────────────────────

def load_cloud(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Charge un nuage de points.

    Retourne
    --------
    pts     : (N, 3) float32  — coordonnées XYZ
    normals : (N, 3) float32  ou None si absentes
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        return _load_npy(path)
    elif suffix == ".ply":
        return _load_ply(path)
    else:
        raise ValueError(f"Format non supporté : '{suffix}'  (attendu : .npy ou .ply)")


def _load_npy(path: Path):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError(
            f"{path.name} : forme inattendue {arr.shape} "
            f"(attendu (N,3) ou (N,6))"
        )
    pts = arr[:, :3].astype(np.float32)
    nrm = arr[:, 3:6].astype(np.float32) if arr.shape[1] == 6 else None
    return pts, nrm


def _load_ply(path: Path):
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh est requis pour lire les fichiers .ply.\n"
            "Installez-le : pip install trimesh"
        )

    loaded = trimesh.load(str(path), process=False)

    # trimesh peut retourner un Trimesh (maillage) ou un PointCloud
    if isinstance(loaded, trimesh.PointCloud):
        pts = np.asarray(loaded.vertices, dtype=np.float32)
        # Normales stockées dans les métadonnées ou les attributs de vertex
        nrm = _extract_ply_normals_pointcloud(loaded)

    elif isinstance(loaded, trimesh.Trimesh):
        # Fichier PLY contenant un maillage — on extrait juste les sommets
        pts = np.asarray(loaded.vertices, dtype=np.float32)
        nrm = _extract_ply_normals_mesh(loaded)
        if nrm is None and len(loaded.vertex_normals) == len(pts):
            nrm = np.asarray(loaded.vertex_normals, dtype=np.float32)

    else:
        raise ValueError(
            f"{path.name} : type trimesh non reconnu ({type(loaded).__name__})"
        )

    print(f"PLY chargé : {len(pts):,} points"
          + ("  + normales" if nrm is not None else "  (pas de normales)"))
    return pts, nrm


def _extract_ply_normals_pointcloud(cloud):
    """Cherche nx/ny/nz dans les métadonnées d'un trimesh.PointCloud."""
    md = getattr(cloud, "metadata", {})
    for key in ("normals", "vertex_normals"):
        if key in md:
            n = np.asarray(md[key], dtype=np.float32)
            if n.shape[1] == 3:
                return n
    return None


def _extract_ply_normals_mesh(mesh):
    """Cherche nx/ny/nz dans les métadonnées d'un trimesh.Trimesh."""
    md = getattr(mesh, "metadata", {})
    for key in ("vertex_normals", "normals"):
        if key in md:
            n = np.asarray(md[key], dtype=np.float32)
            if n.ndim == 2 and n.shape[1] == 3:
                return n
    return None


# ─────────────────────────────────────────────
#  ÉCRITURE
# ─────────────────────────────────────────────

def save_cloud(path: str | Path,
               pts: np.ndarray,
               normals: np.ndarray | None = None) -> None:
    """
    Sauvegarde un nuage de points au format .npy ou .ply.

    pts     : (N, 3)
    normals : (N, 3) ou None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = pts if normals is None else np.concatenate([pts, normals], axis=1)
        np.save(path, arr.astype(np.float32))

    elif suffix == ".ply":
        _save_ply(path, pts, normals)

    else:
        raise ValueError(f"Format non supporté : '{suffix}'")

    print(f"Nuage sauvegardé : {path}  ({len(pts):,} points)")


def _save_ply(path: Path, pts: np.ndarray, normals: np.ndarray | None):
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh requis pour écrire les .ply.")

    cloud = trimesh.PointCloud(vertices=pts.astype(np.float32))
    if normals is not None:
        # trimesh.PointCloud n'a pas de champ normals natif →
        # on exporte manuellement avec plyfile si disponible,
        # sinon on concatène dans les métadonnées.
        try:
            _save_ply_with_normals(path, pts, normals)
            return
        except ImportError:
            pass  # plyfile absent → export sans normales
    cloud.export(str(path))


def _save_ply_with_normals(path: Path, pts: np.ndarray, normals: np.ndarray):
    """Export PLY avec normales via plyfile (pip install plyfile)."""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        raise ImportError(
            "plyfile est requis pour exporter les normales dans un .ply.\n"
            "Installez-le : pip install plyfile"
        )
    pts = pts.astype(np.float32)
    nrm = normals.astype(np.float32)
    vertex = np.empty(len(pts), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ])
    for i, name in enumerate(["x", "y", "z"]):
        vertex[name] = pts[:, i]
    for i, name in enumerate(["nx", "ny", "nz"]):
        vertex[name] = nrm[:, i]
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))


# ─────────────────────────────────────────────
#  UTILITAIRE : détection automatique du format
# ─────────────────────────────────────────────

def auto_load(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Alias de load_cloud() avec affichage du résumé.
    Utilisable comme point d'entrée universel.
    """
    path = Path(path)
    pts, nrm = load_cloud(path)
    print(f"[auto_load] {path.name} → {pts.shape}  "
          f"normales={'oui' if nrm is not None else 'non'}")
    return pts, nrm