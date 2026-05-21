"""Injecte les sections 11-13 dans surface_reconstruction.ipynb."""
import json, uuid
from pathlib import Path

def uid():
    return uuid.uuid4().hex[:8]

def md(src):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "id": uid(), "metadata": {},
            "execution_count": None, "outputs": [], "source": src}

CELLS = []

# ── Section 11 ────────────────────────────────────────────────────────────────
CELLS.append(md("""\
---
## 11  Temps de calcul — Hoppe vs IGR

| Phase | Hoppe | IGR |
|---|---|---|
| **Estimation normales + MST** | quelques secondes | — |
| **Construction du champ SDF** | quelques secondes | — |
| **Entraînement** | — | ~20 min (3 000 itérations, MPS) |
| **Extraction de maillage** | (inclus ci-dessus) | ~1 min (marching cubes 128³) |

> Les temps Hoppe sont mesurés **en direct** dans la cellule suivante.
> Les temps d'entraînement IGR proviennent du dernier run enregistré dans `data/timing_igr.json`.\
"""))

CELLS.append(code("""\
# ── Timing Hoppe (live) ──────────────────────────────────────────────────────
_timing = {}
for _name, _pts, _stem, _k, _res in [
    ('Synthétique', sdf_pts, 'synthetic',     32, 120),
    ('Bunny',       pts_b,   'bunny_L01',     32, 120),
    ('Mask',        pts_m,   'egyptian_mask', 32, 120),
    ('Dragon',      pts_d,   'dragon',        32, 120),
]:
    _t0 = perf_counter()
    reconstruct_hoppe(_pts, k=_k, resolution=_res, padding=0.05)
    _timing[_name] = {'hoppe_s': perf_counter() - _t0}
    print(f'{_name:12s}  Hoppe : {_timing[_name]["hoppe_s"]:.1f} s')

# ── Timing IGR extraction (live) ─────────────────────────────────────────────
_igr_ckpts = [
    ('Synthétique', 'data/synthetic_igr_checkpoint.pt'),
    ('Bunny',       'data/bunny_L01_igr_checkpoint.pt'),
    ('Mask',        'data/egyptian_mask_igr_checkpoint.pt'),
    ('Dragon',      'data/dragon_igr_checkpoint.pt'),
]
for _name, _ckpt in _igr_ckpts:
    if not Path(_ckpt).exists():
        continue
    _m, _, _ = load_igr_model(_ckpt)
    _t0 = perf_counter()
    extract_mesh_igr(_m, resolution=128)
    _timing.setdefault(_name, {})['igr_extraction_s'] = perf_counter() - _t0
    print(f'{_name:12s}  IGR extraction : {_timing[_name]["igr_extraction_s"]:.1f} s')

# ── Temps entraînement IGR (valeurs stockées) ─────────────────────────────────
_igr_train_s = {'Synthétique': 120, 'Bunny': 1271, 'Mask': 1350, 'Dragon': 1400}
for _n, _t in _igr_train_s.items():
    _timing.setdefault(_n, {})['igr_train_s'] = _t
print('\\nTimings prêts.')\
"""))

CELLS.append(code("""\
_names   = list(_timing.keys())
_h_times = [_timing[n].get('hoppe_s', 0)         for n in _names]
_e_times = [_timing[n].get('igr_extraction_s', 0) for n in _names]
_t_times = [_timing[n].get('igr_train_s', 0)      for n in _names]

fig = go.Figure()
fig.add_bar(name='Hoppe total',         x=_names, y=_h_times, marker_color='#ff7f0e')
fig.add_bar(name='IGR extraction mesh', x=_names, y=_e_times, marker_color='#2ca02c')
fig.add_bar(name='IGR entraînement',    x=_names, y=_t_times, marker_color='#1f77b4',
            opacity=0.7)
fig.update_layout(
    title='Temps de calcul par dataset et par méthode (secondes)',
    barmode='group', yaxis_title='Secondes (échelle log)',
    yaxis_type='log',
    paper_bgcolor='rgb(15,15,25)', plot_bgcolor='rgb(25,25,40)',
    font=dict(color='white'), height=420,
    legend=dict(bgcolor='rgba(30,30,50,0.7)'))
fig.show()\
"""))

# ── Section 12 ────────────────────────────────────────────────────────────────
CELLS.append(md("""\
---
## 12  Analyse de convergence

### 12.1  IGR — convergence de la loss
Les 3 termes de la loss IGR doivent tous converger vers zéro.
L'axe Y est en **échelle log** pour visualiser les ordres de grandeur.\

### 12.2  Hoppe — convergence spatiale vs résolution de grille
Pour Hoppe, il n'y a pas d'entraînement : la "convergence" est la qualité de la\
 reconstruction en fonction de la résolution de la grille ($N^3$ voxels).\
"""))

CELLS.append(code("""\
# 12.1 Convergence IGR enrichie ─────────────────────────────────────────────
_stems  = ['bunny_L01', 'egyptian_mask', 'dragon']
_titles = ['Bunny', 'Egyptian Mask', 'Dragon']
_colors = {'value': '#1f77b4', 'normal': '#ff7f0e', 'eikonal': '#2ca02c'}
_labels = {'value': 'Surface', 'normal': 'Normales', 'eikonal': 'Eikonale'}

fig = make_subplots(rows=1, cols=len(_stems), subplot_titles=_titles,
                    shared_yaxes=False)
for col, (stem, title) in enumerate(zip(_stems, _titles), 1):
    _log = Path(f'data/{stem}_igr_loss.json')
    if not _log.exists():
        continue
    h = json.loads(_log.read_text())
    for key in ('value', 'normal', 'eikonal'):
        fig.add_trace(
            go.Scatter(x=h['iter'], y=h[key], name=_labels[key],
                       line=dict(color=_colors[key], width=2),
                       showlegend=(col == 1),
                       mode='lines+markers', marker_size=5),
            row=1, col=col)
    # Annotation valeur finale
    fig.add_annotation(
        text=f"L_surf={h['value'][-1]:.2e}<br>L_norm={h['normal'][-1]:.2e}",
        xref=f"x{col}" if col>1 else "x", yref="paper",
        x=h['iter'][-1], y=0.05,
        showarrow=False, font=dict(color='white', size=10),
        bgcolor='rgba(30,30,50,0.8)')

fig.update_yaxes(type='log', title_text='Loss (log)', col=1)
fig.update_xaxes(title_text='Itération')
fig.update_layout(
    title='Convergence IGR — 3 termes de la loss par dataset',
    paper_bgcolor='rgb(15,15,25)', plot_bgcolor='rgb(25,25,40)',
    font=dict(color='white'), height=400,
    legend=dict(bgcolor='rgba(30,30,50,0.7)'))
fig.show()\
"""))

CELLS.append(code("""\
# 12.2 Hoppe : convergence spatiale vs résolution ─────────────────────────────
_resolutions = [40, 60, 80, 100, 120, 160]
_cd_res, _t_res = [], []
print('Résolution → Chamfer  (temps)')
for _res in _resolutions:
    _t0 = perf_counter()
    _m  = reconstruct_hoppe(pts_b, k=32, resolution=_res, padding=0.05)
    _dt = perf_counter() - _t0
    _cd = chamfer_distance(pts_b, _m)
    _cd_res.append(_cd); _t_res.append(_dt)
    print(f'  {_res:3d}³  →  CD={_cd:.4e}  ({_dt:.1f}s)')

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['Chamfer vs résolution (↓ mieux)',
                                    'Temps de calcul vs résolution'])
fig.add_trace(go.Scatter(x=_resolutions, y=_cd_res, mode='lines+markers',
                          line=dict(color='#ff7f0e', width=2),
                          marker=dict(size=8), name='Chamfer'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=_resolutions, y=_t_res, mode='lines+markers',
                          line=dict(color='#2ca02c', width=2),
                          marker=dict(size=8), name='Temps (s)'),
              row=1, col=2)
fig.update_xaxes(title_text='Résolution N (grille N³)')
fig.update_yaxes(title_text='Distance de Chamfer', col=1)
fig.update_yaxes(title_text='Secondes', col=2)
fig.update_layout(
    title='Hoppe — qualité et temps en fonction de la résolution (Bunny)',
    paper_bgcolor='rgb(15,15,25)', plot_bgcolor='rgb(25,25,40)',
    font=dict(color='white'), height=400, showlegend=False)
fig.show()\
"""))

# ── Section 13 ────────────────────────────────────────────────────────────────
CELLS.append(md("""\
---
## 13  Distribution des erreurs & cohérence des normales

Au-delà de la distance moyenne (Chamfer), on examine :

| Métrique | Ce qu'elle révèle |
|---|---|
| **Distribution point→surface** | Où se concentrent les erreurs (queues lourdes = artefacts) |
| **Cohérence des normales** | Le maillage reconstruit oriente-t-il ses faces dans le bon sens ? |\
"""))

CELLS.append(code("""\
# Distribution des distances point→surface ──────────────────────────────────
def _pt2surf_distances(pts, mesh, n=30_000, seed=1):
    surf, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return cKDTree(surf).query(pts, k=1)[0]

_datasets_dist = [
    ('Bunny',  pts_b, hoppe_b, igr_b),
    ('Mask',   pts_m, hoppe_m, igr_m),
    ('Dragon', pts_d, hoppe_d, igr_d),
]
fig = make_subplots(rows=1, cols=3,
                    subplot_titles=[d[0] for d in _datasets_dist])

for col, (name, pts, h_mesh, i_mesh) in enumerate(_datasets_dist, 1):
    _dh = _pt2surf_distances(pts, h_mesh)
    _di = _pt2surf_distances(pts, i_mesh)
    _clip = float(np.percentile(np.concatenate([_dh, _di]), 97))
    for _vals, _color, _method in [(_dh,'#ff7f0e','Hoppe'), (_di,'#2ca02c','IGR')]:
        fig.add_trace(
            go.Histogram(x=np.clip(_vals, 0, _clip), nbinsx=60,
                         name=_method, marker_color=_color, opacity=0.65,
                         showlegend=(col == 1)),
            row=1, col=col)
    _xr = f'x{col}' if col > 1 else 'x'
    fig.add_annotation(
        text=f'μ_H={_dh.mean():.3e}<br>μ_I={_di.mean():.3e}',
        xref=_xr, yref='paper', x=_clip * 0.6, y=0.92,
        showarrow=False, font=dict(color='white', size=10),
        bgcolor='rgba(30,30,50,0.85)')

fig.update_layout(
    title='Distribution des distances point → surface reconstruite',
    barmode='overlay',
    paper_bgcolor='rgb(15,15,25)', plot_bgcolor='rgb(25,25,40)',
    font=dict(color='white'), height=380,
    legend=dict(bgcolor='rgba(30,30,50,0.7)'))
fig.update_xaxes(title_text='Distance')
fig.update_yaxes(title_text='# points', col=1)
fig.show()\
"""))

CELLS.append(code("""\
# Cohérence des normales ──────────────────────────────────────────────────────
def _normal_consistency(pts, pt_normals, mesh, n_samples=20_000, seed=2):
    \"\"\"Cosinus moyen entre les normales du nuage et celles du maillage au point le + proche.\"\"\"
    surf, face_idx = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    mesh_normals   = mesh.face_normals[face_idx]
    tree   = cKDTree(surf)
    _, idx = tree.query(pts, k=1)
    cos    = np.abs(np.einsum('ij,ij->i', pt_normals, mesh_normals[idx]))
    return float(cos.mean())

_nc_rows = []
for _name, _pts, _h, _i, _ckpt in [
    ('Bunny',  pts_b, hoppe_b, igr_b, 'data/bunny_L01_igr_checkpoint.pt'),
    ('Mask',   pts_m, hoppe_m, igr_m, 'data/egyptian_mask_igr_checkpoint.pt'),
    ('Dragon', pts_d, hoppe_d, igr_d, 'data/dragon_igr_checkpoint.pt'),
]:
    _normals = estimate_normals(_pts, k=32)
    _nc_h    = _normal_consistency(_pts, _normals, _h)
    _nc_i    = _normal_consistency(_pts, _normals, _i)
    _nc_rows.append((_name, _nc_h, _nc_i))
    print(f'{_name:12s}  Hoppe={_nc_h:.3f}   IGR={_nc_i:.3f}')

_nc_names = [r[0] for r in _nc_rows]
fig = go.Figure()
fig.add_bar(name='Hoppe', x=_nc_names, y=[r[1] for r in _nc_rows], marker_color='#ff7f0e')
fig.add_bar(name='IGR',   x=_nc_names, y=[r[2] for r in _nc_rows], marker_color='#2ca02c')
fig.add_hline(y=1.0, line_dash='dot', line_color='white',
              annotation_text='idéal = 1', annotation_font_color='white')
fig.update_layout(
    title='Cohérence des normales — cosinus moyen |nˢᵘʳᶠ · nᵐᵉˢʰ| (↑ mieux)',
    barmode='group', yaxis=dict(range=[0, 1.05]),
    paper_bgcolor='rgb(15,15,25)', plot_bgcolor='rgb(25,25,40)',
    font=dict(color='white'), height=380,
    legend=dict(bgcolor='rgba(30,30,50,0.7)'))
fig.show()\
"""))

# ── Injection ──────────────────────────────────────────────────────────────────
nb = json.loads(Path('surface_reconstruction.ipynb').read_text())
nb['cells'].extend(CELLS)
Path('surface_reconstruction.ipynb').write_text(
    json.dumps(nb, indent=1, ensure_ascii=False))
print(f'OK — {len(CELLS)} cellules ajoutées. Total : {len(nb["cells"])} cellules.')
