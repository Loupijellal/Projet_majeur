"""
Génération d'un nuage de points désorganisés via SDF (Signed Distance Functions)
Figure : cylindre vertical traversant un cube avec des trous circulaires sur chaque face.

Basé sur les formules de https://iquilezles.org/articles/distfunctions/

Structure conçue pour migration facile vers PyTorch (remplacer np par torch).
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  PRIMITIVES SDF  (traduit GLSL → NumPy)
# ─────────────────────────────────────────────

def sdf_cylinder_infinite(p, radius):
    """
    Cylindre infini d'axe Y (vertical).
    p : (N, 3)  →  retourne (N,)
    """
    xz = p[:, [0, 2]]                   # composantes X et Z
    return np.linalg.norm(xz, axis=1) - radius


def sdf_cylinder_capped(p, radius, half_height):
    """
    Cylindre fini d'axe Y, centré à l'origine.
    Formule iquilezles : sdCappedCylinder
    """
    d_xz = np.linalg.norm(p[:, [0, 2]], axis=1) - radius   # dist radiale
    d_y  = np.abs(p[:, 1]) - half_height                    # dist axiale
    d    = np.column_stack([d_xz, d_y])
    return (np.linalg.norm(np.maximum(d, 0.0), axis=1)
            + np.minimum(np.max(d, axis=1), 0.0))


def sdf_box(p, half_extents):
    """
    Boite axe-alignée centrée à l'origine.
    half_extents : (3,) — demi-dimensions en x, y, z
    Formule iquilezles : sdBox
    """
    b = np.asarray(half_extents)
    q = np.abs(p) - b
    return (np.linalg.norm(np.maximum(q, 0.0), axis=1)
            + np.minimum(np.max(q, axis=1), 0.0))


def sdf_sphere(p, center, radius):
    """Sphère centrée en 'center'."""
    return np.linalg.norm(p - np.asarray(center), axis=1) - radius


def sdf_torus(p, R, r):
    """
    Tore centré à l'origine, orienté dans le plan XZ.
    p : array (N, 3) points
    R : rayon majeur (distance du centre de la section à l'axe Y)
    r : rayon mineur (rayon du tube)
    Retourne : array (N,) distances signées
    """
    # Distance au cercle central dans le plan XZ
    q = np.array([np.linalg.norm(p[:, [0, 2]], axis=1) - R, p[:, 1]]).T
    return np.linalg.norm(q, axis=1) - r

def sdf_twisted_torus(p, R, r, k=10.0):
    """
    Tore avec déformation de torsion autour de l'axe Y.
    p : points (N, 3)
    R : rayon majeur
    r : rayon mineur
    k : facteur d'intensité de la torsion
    """
    # Calcul des rotations 2D dans le plan XZ fonction de la hauteur y
    c = np.cos(k * p[:, 1])
    s = np.sin(k * p[:, 1])

    # Rotation de (x, z) pour chaque point
    x_rot = c * p[:, 0] - s * p[:, 2]
    z_rot = s * p[:, 0] + c * p[:, 2]

    # Points transformés (y inchangé)
    q = np.column_stack([x_rot, p[:, 1], z_rot])

    # Évaluation SDF du tore sur les points tordus
    return sdf_torus(q, R, r)

# ─────────────────────────────────────────────
#  OPÉRATIONS BOOLÉENNES
# ─────────────────────────────────────────────

def op_union(d1, d2):
    return np.minimum(d1, d2)

def op_subtract(d_base, d_tool):
    """Soustraction : retire d_tool de d_base."""
    return np.maximum(d_base, -d_tool)

def op_intersect(d1, d2):
    return np.maximum(d1, d2)


# ─────────────────────────────────────────────
#  SDF GLOBALE DE LA FIGURE
# ─────────────────────────────────────────────

# Paramètres géométriques
CUBE_HALF    = 0.5          # demi-côté du cube
CYLINDER_R   = 0.35         # rayon du cylindre principal
CYLINDER_H   = 2.0          # demi-hauteur du cylindre principal
HOLE_R       = 0.30         # rayon des trous circulaires sur les faces du cube
HOLE_H       = 0.6          # profondeur (demi-longueur) des cylindres de découpe


def sdf_scene(p):
    """
    SDF de la scène complète :
      - Cylindre vertical (infini dans la figure = long)
      - Cube dont on soustrait 3 paires de trous cylindriques (une par paire d'axes)
    p : (N, 3) → (N,)
    """

    R, r = 0.5, 0.2
    # --- Cylindre principal (axe Y) ---
    d_cyl = sdf_cylinder_capped(p, CYLINDER_R, CYLINDER_H)

    # --- Cube de base ---
    d_cube = sdf_box(p, [CUBE_HALF, CUBE_HALF, CUBE_HALF])

    # --- Trous sur les faces ±X (cylindres d'axe X) ---
    p_rot_x = p[:, [1, 0, 2]]   # permute pour mettre X en "axe Y" du cylindre
    d_hole_x = sdf_cylinder_capped(p_rot_x, HOLE_R, HOLE_H)

    # --- Trous sur les faces ±Y (cylindres d'axe Y) ---
    d_hole_y = sdf_cylinder_capped(p, HOLE_R, HOLE_H)

    # --- Trous sur les faces ±Z (cylindres d'axe Z) ---
    p_rot_z = p[:, [0, 2, 1]]   # permute pour mettre Z en "axe Y" du cylindre
    d_hole_z = sdf_cylinder_capped(p_rot_z, HOLE_R, HOLE_H)

    # Cube avec trous (soustraction booléenne)
    d_holed_cube = op_subtract(d_cube, d_hole_x)
    d_holed_cube = op_subtract(d_holed_cube, d_hole_y)
    d_holed_cube = op_subtract(d_holed_cube, d_hole_z)
    d_cyl = op_subtract(d_cyl, d_hole_x)
    d_cyl = op_subtract(d_cyl, d_hole_z)

    d_twisted_tornus = sdf_twisted_torus(p, R, r, k=10.0)

    # Union finale : cylindre + cube troué
    return d_twisted_tornus


# ─────────────────────────────────────────────
#  GÉNÉRATION DU NUAGE DE POINTS
# ─────────────────────────────────────────────

def generate_point_cloud(n_samples=200_000, surface_threshold=0.02,
                         rng_seed=42):
    """
    Méthode rejection sampling :
      1. Tire aléatoirement N points dans une boîte englobante.
      2. Évalue la SDF sur chaque point.
      3. Garde uniquement les points proches de la surface (|SDF| < threshold).
      4. Ajoute un bruit gaussien pour simuler une acquisition réelle.

    Retourne : (M, 3) float32
    """
    rng = np.random.default_rng(rng_seed)

    # Boîte englobante légèrement plus grande que la scène
    bbox = np.array([[-0.6, -2.1, -0.6],
                     [ 0.6,  2.1,  0.6]])

    pts = rng.uniform(bbox[0], bbox[1], (n_samples, 3)).astype(np.float32)

    # Évaluation SDF par batch pour économiser la mémoire
    batch_size = 50_000
    sdf_vals = np.empty(n_samples, dtype=np.float32)
    for i in range(0, n_samples, batch_size):
        sdf_vals[i:i+batch_size] = sdf_scene(pts[i:i+batch_size])

    # Sélection des points proches de la surface
    mask = np.abs(sdf_vals) < surface_threshold
    surface_pts = pts[mask]

    print(f"Points candidats : {n_samples:,}")
    print(f"Points sur surface (|SDF| < {surface_threshold}) : {len(surface_pts):,}")


    return surface_pts

# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────

def visualize(pts, max_pts=10_000):
    """Affiche le nuage de points en 3D (sous-échantillonné pour la vitesse)."""
    idx = np.random.choice(len(pts), min(max_pts, len(pts)), replace=False)
    sub = pts[idx]

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2],
               s=0.5, c=sub[:, 1], cmap='viridis', alpha=0.6)
    ax.set_title(f"Nuage de points SDF ({len(pts):,} pts affiché : {len(sub):,})")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect([1, 3, 1])
    plt.tight_layout()
    plt.savefig("./point_cloud_preview.png", dpi=150)
    print("Aperçu sauvegardé : point_cloud_preview.png")
    plt.show()





# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Génération du nuage de points ===")
    point_cloud = generate_point_cloud(
        n_samples=500_000,
        surface_threshold=0.02,
    )
    print(f"Nuage final : {point_cloud.shape}")

    # Sauvegarde
    np.save("./point_cloud.npy", point_cloud)
    print("Sauvegardé : point_cloud.npy")

    visualize(point_cloud)




# ─────────────────────────────────────────────
#  NOTE MIGRATION PYTORCH
# ─────────────────────────────────────────────
"""
Pour migrer vers PyTorch (réseau de neurones sur SDF) :

    import torch

    def sdf_scene_torch(p: torch.Tensor) -> torch.Tensor:
        # Même logique, remplacer np par torch :
        #   np.abs       → torch.abs
        #   np.linalg.norm(x, axis=1) → torch.norm(x, dim=1)
        #   np.maximum   → torch.maximum  (ou torch.clamp)
        #   np.minimum   → torch.minimum
        #   np.max(x, axis=1) → x.max(dim=1).values
        ...

    # Chargement du nuage
    pts = torch.from_numpy(np.load("point_cloud.npy"))  # (N, 3)

    # Réseau DeepSDF-like : entrée (x,y,z) → sortie scalaire SDF
    class SDFNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(3, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, 1),
            )
        def forward(self, x):
            return self.net(x)

    # Loss : prédire sdf_scene_torch(pts) ≈ 0 sur le nuage de surface,
    # + eikonal loss ||∇f|| = 1 pour les autres points.
"""
