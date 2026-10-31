# =============================================================================
# STEP 8 — Per-patient measurement uncertainty -> diagnosis (sample-based, no Gaussian)
#
# Step 7 used ONE error covariance for all patients -> heavy-tailed errors inflated the
# doubt of everyone. Here each patient gets its OWN measurement uncertainty, taken from
# its own segmentation:
#   Part A (GPU, ~10-20 min): for each patient, the U-Net of the fold where the patient
#     was held out produces S stochastic segmentations (MC dropout + flips + mild gamma).
#     The 21 concepts are computed on every sample -> concept_samples.npz (n, S, 21).
#   Part B (CPU, ~1 min): concept samples are spread around the deterministic prediction,
#     scaled by ONE factor kappa (fitted on training patients only, moment matching of
#     |pred - gt| to the sample SD), pushed through the linear concept head, softmax-averaged.
#
# Pre-specified variants (written before seeing results):
#   plain   : step 2b model, no uncertainty
#   pp_raw  : per-patient samples, kappa = 1
#   pp_cal  : per-patient samples, kappa fitted on training patients
# Outputs: accuracy/calibration, conformal sets (LAC, alpha = 0.1, CV+ style),
#          error detection (AUROC of uncertainty for flagging misdiagnoses),
#          per-patient explanation of the doubt.
# =============================================================================
import os, glob, zipfile, time
import numpy as np, pandas as pd, nibabel as nib, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
STEP1 = os.environ.get('STEP1_DIR', f'{DRIVE}/step1_unet')
ROOT = os.environ.get('ACDC_ROOT', '/content/acdc/training')
OUT = f'{DRIVE}/step8_patient_uncertainty'
os.makedirs(OUT, exist_ok=True)
SEED, SIZE, WIDTH = 42, int(os.environ.get('SIZE', 224)), int(os.environ.get('WIDTH', 32))
S, ALPHA = int(os.environ.get('S', 20)), 0.10
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)

folds = pd.read_csv(f'{DRIVE}/folds.csv')
order = folds.patient.values
classes = np.array(sorted(folds.label.unique()))
y = np.searchsorted(classes, folds.label.values)
f = folds.fold.values
n, K = len(y), len(classes)
rf = pd.read_csv(f'{DRIVE}/step3_challenge_baseline/rich_features.csv')
FEATS = [c for c in rf.columns if c not in ('patient', 'source')]
C_gt = rf[rf.source == 'gt'].set_index('patient').loc[order, FEATS].values
C_pr = rf[rf.source == 'pred'].set_index('patient').loc[order, FEATS].values

# ------------------------------------------------ Part A: stochastic segmentations
SAMPLES = f'{OUT}/concept_samples.npz'
if not os.path.exists(SAMPLES):
    if not os.path.isdir(ROOT):
        zipfile.ZipFile('/content/drive/MyDrive/ACDC_calibration_article/training-20261021T163705Z-1-001.zip').extractall('/content/acdc')

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
    def sample_mask(net, v, s):
        """s = 0: deterministic; s > 0: MC dropout + random flips + mild gamma."""
        H, W, Z = v.shape
        x = torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1)))[:, None].to(dev)
        x = F.interpolate(x, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
        fl = []
        if s > 0:
            g = float(np.exp(rng.normal(0, 0.1)))
            x = torch.sign(x) * x.abs() ** g
            fl = [d for d in (2, 3) if rng.random() < 0.5]
            if fl: x = torch.flip(x, fl)
        p = net(x).float().softmax(1)
        if fl: p = torch.flip(p, fl)
        p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False)
        return p.argmax(1).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    def thickness_per_slice(myo, zooms):
        out = []
        for z in range(myo.shape[2]):
            m = myo[:, :, z]
            if m.sum() >= 10:
                out.append(2 * distance_transform_edt(m, sampling=zooms[:2]).max() - np.mean(zooms[:2]))
        return np.array(out) if out else np.array([0.0])

    def rich_features(lab_ed, lab_es, z_ed, z_es):     # identical to step 3
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

    CS = np.zeros((n, S, len(FEATS)))
    t0 = time.time()
    pos = {p: i for i, p in enumerate(order)}
    for k in range(5):
        net = UNet(c=WIDTH).to(dev)
        net.load_state_dict(torch.load(f'{STEP1}/unet_fold{k}.pt', map_location=dev))
        net.eval(); net.drop.train()                      # MC dropout only (BatchNorm stays in eval)
        for p in order[f == k]:
            d = f'{ROOT}/{p}'; info = read_cfg(f'{d}/Info.cfg')
            vols, zs = [], []
            for ph in ('ED', 'ES'):
                img = nib.load(nii(f'{d}/{p}_frame{int(info[ph]):02d}'))
                vols.append(norm(img.get_fdata())); zs.append(tuple(float(z) for z in img.header.get_zooms()[:3]))
            for s in range(S):
                ft = rich_features(sample_mask(net, vols[0], s + 1), sample_mask(net, vols[1], s + 1), zs[0], zs[1])
                CS[pos[p], s] = [ft[c] for c in FEATS]
        print(f'fold {k}: sampled ({time.time() - t0:.0f}s)')
    np.savez_compressed(SAMPLES, CS=CS, feats=np.array(FEATS), patients=order)
CS = np.load(SAMPLES)['CS']
print(f'concept samples: {CS.shape}')

# ------------------------------------------------ Part B: propagate to the diagnosis
dev_s = CS - CS.mean(1, keepdims=True)                        # per-patient spread around its own mean
sd = CS.std(1) + 1e-9                                         # (n, D) per-patient SD

def fit_head(idx):
    g = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                     {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy')
    g.fit(C_gt[idx], y[idx])
    return g.best_estimator_

def fit_kappa(idx):
    """one global inflation factor: median |error| / median sample SD, in SD units (robust)."""
    r = np.abs(C_pr[idx] - C_gt[idx]) / sd[idx]
    return float(np.median(r) / 0.6745)

def predict(model, idx, mode, kappa):
    if mode == 'plain':
        return model.predict_proba(C_pr[idx])
    kap = 1.0 if mode == 'pp_raw' else kappa
    X = C_pr[idx][:, None, :] + kap * dev_s[idx]               # (m, S, D)
    m, s_, D = X.shape
    return model.predict_proba(X.reshape(-1, D)).reshape(m, s_, K).mean(1)

def qhat(s):
    m = len(s); return np.quantile(s, min(1, np.ceil((m + 1) * (1 - ALPHA)) / m), method='higher')

MODES = ['plain', 'pp_raw', 'pp_cal']
P = {m: np.zeros((n, K)) for m in MODES}
SET = {m: np.zeros((n, K), bool) for m in MODES}
kap_log = []
for k in range(5):
    tr, te = np.where(f != k)[0], np.where(f == k)[0]
    cal = {m: np.zeros(len(tr)) for m in MODES}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y[tr]):
        mdl, kap = fit_head(tr[a]), fit_kappa(tr[a])
        for m in MODES:
            Pc = predict(mdl, tr[b], m, kap)
            cal[m][b] = 1 - Pc[np.arange(len(b)), y[tr[b]]]
    mdl, kap = fit_head(tr), fit_kappa(tr)
    kap_log.append(kap)
    for m in MODES:
        Pt = predict(mdl, te, m, kap)
        P[m][te] = Pt
        SET[m][te] = (1 - Pt) <= qhat(cal[m])
    print(f'fold {k}: kappa = {kap:.2f}')

def ece(Pm, bins=15):
    conf, acc = Pm.max(1), (Pm.argmax(1) == y).astype(float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

rows = []
for m in MODES:
    wrong = P[m].argmax(1) != y
    cov = SET[m][np.arange(n), y]
    rows.append(dict(model=m, bal_acc=balanced_accuracy_score(y, P[m].argmax(1)),
                     AUC=roc_auc_score(y, P[m], multi_class='ovr'), ECE=ece(P[m]),
                     NLL=-np.mean(np.log(np.clip(P[m][np.arange(n), y], 1e-12, 1))),
                     err_detect_AUROC=roc_auc_score(wrong, 1 - P[m].max(1)) if 0 < wrong.sum() < n else np.nan,
                     coverage=cov.mean(), mean_size=SET[m].sum(1).mean(), singleton=(SET[m].sum(1) == 1).mean(),
                     **{f'cov_{c}': cov[y == i].mean() for i, c in enumerate(classes)}))
res = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print(f'\nkappa per fold: {np.round(kap_log, 2)} | target coverage {1 - ALPHA:.2f}')
print(res.round(3).to_string(index=False))

# is the per-patient spread informative? (does the sample SD track the real error?)
from scipy.stats import spearmanr
print('\nSpearman: per-patient sample SD vs real |pred - gt| (per measurement)')
for j, c in enumerate(FEATS):
    if c in ('LV_EF', 'RV_EF', 'LV_ED', 'RV_ED', 'MYO_ED', 'RV_LV_ED', 'MYO_LV_ED', 'WT_ED_max'):
        r, pv = spearmanr(sd[:, j], np.abs(C_pr[:, j] - C_gt[:, j]))
        print(f'  {c:10s}: rho = {r:+.2f}  p = {pv:.2g}')

# per-patient explanation: which measurement's own uncertainty flips the decision most often
expl = []
for k in range(5):
    tr, te = np.where(f != k)[0], np.where(f == k)[0]
    mdl, kap = fit_head(tr), fit_kappa(tr)
    for t in te:
        base = mdl.predict_proba(C_pr[[t]])[0].argmax()
        flips = []
        for j in range(len(FEATS)):                              # perturb ONE measurement by its own samples
            X = np.repeat(C_pr[[t]], S, 0); X[:, j] += kap * dev_s[t, :, j]
            flips.append(np.mean(mdl.predict(X) != base))
        flips = np.array(flips); top = np.argsort(-flips)[:2]
        expl.append(dict(patient=order[t], label=classes[y[t]], pred=classes[base],
                         p_flip_any=np.mean(P['pp_cal'][t].argmax() != base),
                         doubt1=FEATS[top[0]], doubt1_flip=flips[top[0]], doubt2=FEATS[top[1]], doubt2_flip=flips[top[1]]))
ex = pd.DataFrame(expl)
w = ex[ex.label != ex.pred]
print(f'\nMisdiagnosed ({len(w)}) with their main source of doubt:')
print(w[['patient', 'label', 'pred', 'doubt1', 'doubt1_flip', 'doubt2', 'doubt2_flip']].round(2).to_string(index=False))
print('\nPatients with a measurement that flips the decision in >= 20% of its samples:',
      f'wrong {np.mean(w.doubt1_flip >= 0.2) * 100:.0f}% vs correct {np.mean(ex[ex.label == ex.pred].doubt1_flip >= 0.2) * 100:.0f}%')

res.to_csv(f'{OUT}/step8_results.csv', index=False)
ex.to_csv(f'{OUT}/doubt_per_patient.csv', index=False)
for m in MODES:
    np.save(f'{OUT}/oof_probs_{m}.npy', P[m]); np.save(f'{OUT}/sets_LAC_{m}.npy', SET[m])
print('\nSaved to', OUT)
