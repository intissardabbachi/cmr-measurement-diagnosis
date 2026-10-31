# =============================================================================
# STEP 9 — External test on M&Ms-1 (multi-vendor). LOCKED protocol, nothing tuned here.
#
# Locked before running (from ACDC only):
#   * segmentation : the 5 ACDC fold U-Nets (step 1), patient i uses U-Net (i mod 5)
#   * concepts     : the same 21 measurements (step 3 code)
#   * head         : logistic regression on ACDC reference concepts (all 100, C by 5-fold CV)
#   * uncertainty  : S = 20 samples (MC dropout + flips + gamma), kappa fitted on ACDC
#   * flip flag    : a single measurement flips the decision in >= 20% of its samples
#   * conformal    : LAC threshold from ACDC out-of-fold probabilities (alpha = 0.10)
#   * primary population: Pathology in {DCM, HCM, NOR}; decision = argmax over these 3
# M&Ms label convention is LV=1, MYO=2, RV=3 -> remapped to ACDC (RV=1, MYO=2, LV=3).
# Results reported overall and per vendor.
# GPU (Colab T4): about 30-40 min. Resumes from a cache if the session drops.
# =============================================================================
import os, glob, zipfile, io, time, re
import numpy as np, pandas as pd, nibabel as nib, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr, fisher_exact, beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
STEP1 = f'{DRIVE}/step1_unet'
S8 = f'{DRIVE}/step8_patient_uncertainty'
OUT = f'{DRIVE}/step9_mnms_external'
LOCAL = '/content/mnms'
os.makedirs(OUT, exist_ok=True); os.makedirs(LOCAL, exist_ok=True)
SEED, SIZE, WIDTH, S, ALPHA, FLAG, B = 42, 224, 32, 20, 0.10, 0.20, 2000
PRIMARY = ['DCM', 'HCM', 'NOR']
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
print('device:', dev)

# ------------------------------------------------ ACDC side (locked model)
folds = pd.read_csv(f'{DRIVE}/folds.csv')
order = folds.patient.values
classes = np.array(sorted(folds.label.unique()))
y_acdc = np.searchsorted(classes, folds.label.values)
K = len(classes)
rf = pd.read_csv(f'{DRIVE}/step3_challenge_baseline/rich_features.csv')
FEATS = [c for c in rf.columns if c not in ('patient', 'source')]
A_gt = rf[rf.source == 'gt'].set_index('patient').loc[order, FEATS].values
A_pr = rf[rf.source == 'pred'].set_index('patient').loc[order, FEATS].values
A_cs = np.load(f'{S8}/concept_samples.npz')['CS']
head = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                    {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy').fit(A_gt, y_acdc).best_estimator_
KAPPA = float(np.median(np.abs(A_pr - A_gt) / (A_cs.std(1) + 1e-9)) / 0.6745)
P_oof = np.load(f'{S8}/oof_probs_plain.npy')
cal = 1 - P_oof[np.arange(len(y_acdc)), y_acdc]
QHAT = np.quantile(cal, min(1, np.ceil((len(cal) + 1) * (1 - ALPHA)) / len(cal)), method='higher')
pidx = np.searchsorted(classes, PRIMARY)
print(f'locked: kappa = {KAPPA:.2f}, conformal qhat = {QHAT:.3f}, primary classes {PRIMARY} -> idx {pidx}')

# ------------------------------------------------ M&Ms files
ZIP = (glob.glob('/content/drive/MyDrive/**/M&M_dataset.zip', recursive=True) +
       glob.glob('/content/drive/MyDrive/**/M&M*.zip', recursive=True))[0]
z = zipfile.ZipFile(ZIP)
names = z.namelist()
meta_name = [n for n in names if 'MnMs_Dataset_information' in n and n.lower().endswith('.csv')][0]
meta = pd.read_csv(io.BytesIO(z.read(meta_name)))
col = lambda *keys: next(c for c in meta.columns if any(k.lower() == c.lower().strip() for k in keys))
C_CODE, C_PATH, C_ED, C_ES = col('External code'), col('Pathology'), col('ED'), col('ES')
C_VEND = next((c for c in meta.columns if c.lower().strip() in ('vendorname', 'vendor name')), col('Vendor'))
meta[C_CODE] = meta[C_CODE].astype(str).str.strip()
img_of = {re.sub(r'_sa\.nii(\.gz)?$', '', os.path.basename(n)): n for n in names if re.search(r'_sa\.nii(\.gz)?$', n)}
gt_of = {re.sub(r'_sa_gt\.nii(\.gz)?$', '', os.path.basename(n)): n for n in names if re.search(r'_sa_gt\.nii(\.gz)?$', n)}
M = meta[meta[C_CODE].isin(gt_of) & meta[C_CODE].isin(img_of) & meta[C_PATH].isin(PRIMARY)].reset_index(drop=True)
print(f'M&Ms primary population with ground truth: {len(M)}')
print(pd.crosstab(M[C_VEND], M[C_PATH]).to_string())

# ------------------------------------------------ same U-Net / features as ACDC steps
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

def norm(v):
    lo, hi = np.percentile(v, [1, 99])
    v = np.clip(v, lo, hi)
    return ((v - v.mean()) / (v.std() + 1e-8)).astype(np.float32)

@torch.no_grad()
def seg(net, v, stochastic):
    H, W, Z = v.shape
    x = torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1)))[:, None].to(dev)
    x = F.interpolate(x, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
    fl = []
    if stochastic:
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
    for k in range(myo.shape[2]):
        m = myo[:, :, k]
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

REMAP = np.array([0, 3, 2, 1], np.uint8)          # M&Ms (LV=1, MYO=2, RV=3) -> ACDC (RV=1, MYO=2, LV=3)
def dice(a, b, c):
    a, b = a == c, b == c
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)

# ------------------------------------------------ Part A: segment + measure (cached)
CACHE = f'{OUT}/mnms_concepts.npz'
done = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}
res = done.get('res', np.array([], dtype=object)).tolist() if done else []
seen = {r['code'] for r in res}
nets = []
for k in range(5):
    net = UNet(c=WIDTH).to(dev)
    net.load_state_dict(torch.load(f'{STEP1}/unet_fold{k}.pt', map_location=dev))
    net.eval(); net.drop.train()                    # MC dropout only; deterministic pass below turns it off
    nets.append(net)
t0 = time.time()
for i, r in M.iterrows():
    code = r[C_CODE]
    if code in seen: continue
    fi, fg = (z.extract(img_of[code], LOCAL), z.extract(gt_of[code], LOCAL))
    im, gt = nib.load(fi), nib.load(fg)
    zooms = tuple(float(v) for v in im.header.get_zooms()[:3])
    I, G = im.get_fdata(), np.rint(gt.get_fdata()).astype(np.uint8)
    ed, es = int(r[C_ED]), int(r[C_ES])
    if G[..., ed].max() == 0 or G[..., es].max() == 0:            # fall back to the labelled frames
        lab_t = [t for t in range(G.shape[3]) if G[..., t].max() > 0]
        ed, es = lab_t[0], lab_t[-1]
    vols = [norm(I[..., ed]), norm(I[..., es])]
    gts = [REMAP[G[..., ed]], REMAP[G[..., es]]]
    net = nets[i % 5]
    net.drop.eval(); det = [seg(net, v, False) for v in vols]; net.drop.train()
    samp = []
    for s in range(S):
        ft = rich_features(seg(net, vols[0], True), seg(net, vols[1], True), zooms, zooms)
        samp.append([ft[c] for c in FEATS])
    fg_ = rich_features(gts[0], gts[1], zooms, zooms); fp_ = rich_features(det[0], det[1], zooms, zooms)
    res.append(dict(code=code, vendor=r[C_VEND], label=r[C_PATH],
                    gt=np.array([fg_[c] for c in FEATS]), pred=np.array([fp_[c] for c in FEATS]), samples=np.array(samp),
                    dice={f'{nm}_{ph}': dice(det[j], gts[j], c) for j, ph in enumerate(('ED', 'ES'))
                          for nm, c in (('RV', 1), ('MYO', 2), ('LV', 3))}))
    os.remove(fi); os.remove(fg)
    if len(res) % 20 == 0 or i == len(M) - 1:
        np.savez(CACHE, res=np.array(res, dtype=object))
        print(f'{len(res)}/{len(M)} patients ({time.time() - t0:.0f}s)')
np.savez(CACHE, res=np.array(res, dtype=object))

# ------------------------------------------------ Part B: locked evaluation
R = pd.DataFrame(res)
n = len(R)
Cg, Cp, CS = np.stack(list(R['gt'])), np.stack(list(R['pred'])), np.stack(list(R['samples']))
yv = np.searchsorted(PRIMARY, R.label.values)                     # 0..2 within primary classes
vend = R.vendor.values
dv = CS - CS.mean(1, keepdims=True)

def restrict(P5):
    Q = P5[:, pidx]; return Q / Q.sum(1, keepdims=True)

P5 = head.predict_proba(Cp)
P3 = restrict(P5)
P3_gt = restrict(head.predict_proba(Cg))                          # upper bound: reference measurements
pred3 = P3.argmax(1)
wrong = pred3 != yv

flips = np.zeros((n, len(FEATS)))
for t in range(n):
    for j in range(len(FEATS)):
        X = np.repeat(Cp[[t]], S, 0); X[:, j] += KAPPA * dv[t, :, j]
        flips[t, j] = np.mean(restrict(head.predict_proba(X)).argmax(1) != pred3[t])
fmax = flips.max(1); doubt = np.array(FEATS)[flips.argmax(1)]
flag = fmax >= FLAG
SET = (1 - P5) <= QHAT                                            # conformal sets (5 classes)
cover = SET[np.arange(n), pidx[yv]]

D = pd.DataFrame(list(R['dice']))
print('\nSegmentation Dice (deterministic) per vendor:')
print(D.assign(vendor=vend).groupby('vendor').mean().round(3).to_string())

def summary(m):
    yy, P, w, fl = yv[m], P3[m], wrong[m], flag[m]
    out = dict(n=m.sum(), bal_acc=balanced_accuracy_score(yy, P.argmax(1)),
               bal_acc_refmeas=balanced_accuracy_score(yy, P3_gt[m].argmax(1)),
               AUC=roc_auc_score(yy, P, multi_class='ovr', labels=range(3)) if len(np.unique(yy)) == 3 else np.nan,
               errors=w.sum(), flag_sens=fl[w].mean() if w.any() else np.nan, false_flag=fl[~w].mean(),
               AUROC_flip=roc_auc_score(w, fmax[m]) if 0 < w.sum() < m.sum() else np.nan,
               AUROC_conf=roc_auc_score(w, 1 - P.max(1)) if 0 < w.sum() < m.sum() else np.nan,
               conformal_cov=cover[m].mean(), set_size=SET[m].sum(1).mean())
    return out
rows = [dict(vendor='ALL', **summary(np.ones(n, bool)))] + [dict(vendor=v, **summary(vend == v)) for v in sorted(set(vend))]
T = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print('\nExternal results (primary: DCM / HCM / NOR):')
print(T.round(3).to_string(index=False))
print('\nConfusion (rows = truth):', PRIMARY); print(confusion_matrix(yv, pred3, labels=range(3)))

a, b_, c, d_ = (flag & wrong).sum(), (~flag & wrong).sum(), (flag & ~wrong).sum(), (~flag & ~wrong).sum()
odds, pv = fisher_exact([[a, b_], [c, d_]], alternative='greater')
def cp(k, m):
    return (beta.ppf(0.025, k, m - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, m - k) if k < m else 1.0)
print(f'\nFlip flag: sensitivity {a}/{a + b_} CI {np.round(cp(a, a + b_), 2)} | false-flag {c}/{c + d_} CI {np.round(cp(c, c + d_), 2)}'
      f' | Fisher p = {pv:.2g}, OR = {odds:.1f}')

bs = []
for _ in range(B):
    i = rng.integers(0, n, n)
    if 0 < wrong[i].sum() < n:
        bs.append(roc_auc_score(wrong[i], fmax[i]) - roc_auc_score(wrong[i], 1 - P3[i].max(1)))
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f'Paired AUROC difference (flip flag minus confidence): {T.AUROC_flip[0] - T.AUROC_conf[0]:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]')

print('\nSpearman: per-patient sample SD vs real |pred - gt| (does uncertainty still track error?)')
sd = CS.std(1)
for c_ in ('LV_EF', 'RV_EF', 'LV_ED', 'RV_ED', 'MYO_ED', 'RV_LV_ED', 'MYO_LV_ED'):
    j = FEATS.index(c_); r_, p_ = spearmanr(sd[:, j], np.abs(Cp[:, j] - Cg[:, j]))
    print(f'  {c_:10s}: rho = {r_:+.2f}  p = {p_:.2g}')

print('\nMain source of doubt among errors:'); print(pd.Series(doubt[wrong]).value_counts().head(6).to_string())

T.to_csv(f'{OUT}/step9_results.csv', index=False)
pd.DataFrame(dict(code=R.code, vendor=vend, label=R.label, pred=np.array(PRIMARY)[pred3], flip_max=fmax, doubt=doubt,
                  flag=flag, in_set=cover, **{f'p_{c_}': P3[:, j] for j, c_ in enumerate(PRIMARY)})).to_csv(f'{OUT}/per_patient.csv', index=False)
D.assign(code=R.code, vendor=vend).to_csv(f'{OUT}/dice.csv', index=False)
print('\nSaved to', OUT)
