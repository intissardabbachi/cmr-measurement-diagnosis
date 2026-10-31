# =============================================================================
# STEP 3 — Strong baseline of the same family: ACDC-challenge-style pipeline
#   segmentation (our fold-wise U-Nets, weights from step 1, no retraining)
#   -> rich hand-crafted features (21) -> random forest
# Same protocol as step 2: classifier fitted on REFERENCE features of the training
# patients of each fold (hyper-parameters by inner 5-fold CV), applied to the
# PREDICTED features of the held-out patients. Paired comparison with our model.
# Also saves the predicted masks (needed later for uncertainty / XAI steps).
# CPU is enough (about 15-25 min, mostly the U-Net inference).
# No body-size indexing (height missing in M&Ms-1 for GE/Philips).
# =============================================================================
import os, glob, zipfile, time
import numpy as np, pandas as pd, nibabel as nib, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
STEP1 = os.environ.get('STEP1_DIR', f'{DRIVE}/step1_unet')
OUT = f'{DRIVE}/step3_challenge_baseline'
ROOT = os.environ.get('ACDC_ROOT', '/content/acdc/training')
os.makedirs(f'{OUT}/pred_masks', exist_ok=True)
if not os.path.isdir(ROOT):
    zipfile.ZipFile('/content/drive/MyDrive/ACDC_calibration_article/training-20261021T163705Z-1-001.zip').extractall('/content/acdc')
SEED, SIZE, WIDTH, B = 42, int(os.environ.get('SIZE', 224)), int(os.environ.get('WIDTH', 32)), 2000
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(os.cpu_count() or 1)

# ------------------------------------------------ same U-Net and preprocessing as step 1
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(s, c=32, n=4, p=0.1):
        super().__init__()
        s.e1, s.e2, s.e3, s.e4 = block(1, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        s.b = block(8 * c, 16 * c)
        s.drop = nn.Dropout2d(p)
        s.u4, s.d4 = nn.ConvTranspose2d(16 * c, 8 * c, 2, 2), block(16 * c, 8 * c)
        s.u3, s.d3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), block(8 * c, 4 * c)
        s.u2, s.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), block(4 * c, 2 * c)
        s.u1, s.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), block(2 * c, c)
        s.out = nn.Conv2d(c, n, 1)

    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(F.max_pool2d(e1, 2)); e3 = s.e3(F.max_pool2d(e2, 2)); e4 = s.e4(F.max_pool2d(e3, 2))
        b = s.drop(s.b(F.max_pool2d(e4, 2)))
        d4 = s.d4(torch.cat([s.u4(b), e4], 1)); d3 = s.d3(torch.cat([s.u3(d4), e3], 1))
        d2 = s.d2(torch.cat([s.u2(d3), e2], 1)); d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
        return s.out(d1)

def read_cfg(p):
    return {k.strip(): v.strip() for k, v in (l.split(':', 1) for l in open(p) if ':' in l)}

def nii(base):
    for e in ('.nii.gz', '.nii'):
        if os.path.exists(base + e):
            return base + e
    raise FileNotFoundError(base)

def norm(v):
    lo, hi = np.percentile(v, [1, 99])
    v = np.clip(v, lo, hi)
    return ((v - v.mean()) / (v.std() + 1e-8)).astype(np.float32)

@torch.no_grad()
def predict_mask(net, v):
    H, W, Z = v.shape
    x = torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1)))[:, None].to(dev)
    x = F.interpolate(x, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
    p = F.interpolate(net(x).float().softmax(1), size=(H, W), mode='bilinear', align_corners=False)
    return p.argmax(1).permute(1, 2, 0).cpu().numpy().astype(np.uint8)       # (H, W, Z) labels

# ------------------------------------------------ rich features (challenge style)
def thickness_per_slice(myo, zooms):
    out = []
    for z in range(myo.shape[2]):
        m = myo[:, :, z]
        if m.sum() >= 10:
            out.append(2 * distance_transform_edt(m, sampling=zooms[:2]).max() - np.mean(zooms[:2]))
    return np.array(out) if out else np.array([0.0])

def rich_features(lab_ed, lab_es, z_ed, z_es):
    ved, ves = np.prod(z_ed) / 1000, np.prod(z_es) / 1000
    v = {f'{s}_{ph}': (lab == c).sum() * vv for ph, lab, vv in (('ED', lab_ed, ved), ('ES', lab_es, ves))
         for s, c in (('RV', 1), ('MYO', 2), ('LV', 3))}
    t_ed, t_es = thickness_per_slice(lab_ed == 2, z_ed), thickness_per_slice(lab_es == 2, z_es)
    eps = 1e-6
    return dict(**v,
                LV_EF=100 * (v['LV_ED'] - v['LV_ES']) / max(v['LV_ED'], eps),
                RV_EF=100 * (v['RV_ED'] - v['RV_ES']) / max(v['RV_ED'], eps),
                LV_mass=1.05 * v['MYO_ED'],
                RV_LV_ED=v['RV_ED'] / max(v['LV_ED'], eps), RV_LV_ES=v['RV_ES'] / max(v['LV_ES'], eps),
                MYO_LV_ED=v['MYO_ED'] / max(v['LV_ED'], eps), MYO_LV_ES=v['MYO_ES'] / max(v['LV_ES'], eps),
                WT_ED_max=t_ed.max(), WT_ED_mean=t_ed.mean(), WT_ED_std=t_ed.std(), WT_ED_min=t_ed.min(),
                WT_ES_max=t_es.max(), WT_ES_mean=t_es.mean(), WT_ES_std=t_es.std(), WT_ES_min=t_es.min())

# ------------------------------------------------ features: reference (all) + predicted (held-out only)
folds = pd.read_csv(f'{DRIVE}/folds.csv')
order = folds.patient.values
classes = np.array(sorted(folds.label.unique()))
y = np.searchsorted(classes, folds.label.values)
f = folds.fold.values
ref, pred = {}, {}
t0 = time.time()
for k in range(5):
    net = UNet(c=WIDTH).to(dev)
    net.load_state_dict(torch.load(f'{STEP1}/unet_fold{k}.pt', map_location=dev))
    net.eval()
    for p in order[f == k]:
        d = f'{ROOT}/{p}'
        info = read_cfg(f'{d}/Info.cfg')
        labs, gts, zs = [], [], []
        for ph in ('ED', 'ES'):
            img = nib.load(nii(f'{d}/{p}_frame{int(info[ph]):02d}'))
            gts.append(np.rint(nib.load(nii(f'{d}/{p}_frame{int(info[ph]):02d}_gt')).get_fdata()).astype(np.uint8))
            zs.append(tuple(float(z) for z in img.header.get_zooms()[:3]))
            labs.append(predict_mask(net, norm(img.get_fdata())))
        np.savez_compressed(f'{OUT}/pred_masks/{p}.npz', ED=labs[0], ES=labs[1], zooms_ED=zs[0], zooms_ES=zs[1])
        pred[p] = rich_features(labs[0], labs[1], zs[0], zs[1])
        ref[p] = rich_features(gts[0], gts[1], zs[0], zs[1])
    print(f'fold {k}: masks predicted ({time.time() - t0:.0f}s)')

R = pd.DataFrame.from_dict(ref, orient='index').loc[order]
Q = pd.DataFrame.from_dict(pred, orient='index').loc[order]
FEATS = list(R.columns)
pd.concat([R.assign(source='gt'), Q.assign(source='pred')]).rename_axis('patient').to_csv(f'{OUT}/rich_features.csv')
print(len(FEATS), 'features:', FEATS)

# ------------------------------------------------ random forest, same protocol as step 2
P = {'challenge_RF_pred': np.zeros((len(y), len(classes))), 'challenge_RF_GT_upper': np.zeros((len(y), len(classes)))}
for k in range(5):
    tr, te = f != k, f == k
    grid = GridSearchCV(RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=SEED, n_jobs=-1),
                        {'max_features': ['sqrt', 0.5], 'min_samples_leaf': [1, 3]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy')
    grid.fit(R.values[tr], y[tr])
    P['challenge_RF_pred'][te] = grid.predict_proba(Q.values[te])
    P['challenge_RF_GT_upper'][te] = grid.predict_proba(R.values[te])
    print(f'fold {k}: {grid.best_params_}')

for name, path in (('concept_pred_soft (ours)', f'{STEP1}/oof_probs_concept_pred_soft.npy'),
                   ('cinema_frozen_probe', f'{DRIVE}/cinema_oof_probs.npy')):
    if os.path.exists(path):
        P[name] = np.load(path)

def ece(Pm, yy, bins=15):
    conf, acc = Pm.max(1), (Pm.argmax(1) == yy).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    return sum(((conf > lo) & (conf <= hi)).mean() * abs(acc[(conf > lo) & (conf <= hi)].mean() - conf[(conf > lo) & (conf <= hi)].mean())
               for lo, hi in zip(edges[:-1], edges[1:]) if ((conf > lo) & (conf <= hi)).any())

def metrics(Pm, i):
    return (balanced_accuracy_score(y[i], Pm[i].argmax(1)),
            roc_auc_score(y[i], Pm[i], multi_class='ovr', average='macro', labels=range(len(classes))), ece(Pm[i], y[i]))

rng = np.random.default_rng(SEED)
boots = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(B)) if len(np.unique(y[i])) == len(classes)]
rows = []
for name, Pm in P.items():
    pt, bs = metrics(Pm, np.arange(len(y))), np.array([metrics(Pm, i) for i in boots])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    rows.append(dict(model=name, bal_acc=pt[0], bal_lo=lo[0], bal_hi=hi[0], AUC=pt[1], AUC_lo=lo[1], AUC_hi=hi[1],
                     ECE=pt[2], macro_F1=f1_score(y, Pm.argmax(1), average='macro')))
res = pd.DataFrame(rows)
print('\n', res.round(3).to_string(index=False))

ours = 'concept_pred_soft (ours)'
if ours in P:
    print(f'\nPaired difference: {ours} minus challenge_RF_pred')
    d = np.array([np.subtract(metrics(P[ours], i)[:2], metrics(P['challenge_RF_pred'], i)[:2]) for i in boots])
    for j, m in enumerate(['bal_acc', 'AUC']):
        print(f'  {m}: {d[:, j].mean():+.3f}  95% CI [{np.percentile(d[:, j], 2.5):+.3f}, {np.percentile(d[:, j], 97.5):+.3f}]')

print('\nConfusion matrix, challenge_RF_pred (rows = truth):', [str(c) for c in classes])
print(confusion_matrix(y, P['challenge_RF_pred'].argmax(1)))
res.to_csv(f'{OUT}/step3_results.csv', index=False)
for name in ('challenge_RF_pred', 'challenge_RF_GT_upper'):
    np.save(f'{OUT}/oof_probs_{name}.npy', P[name])
print('\nSaved to', OUT)
