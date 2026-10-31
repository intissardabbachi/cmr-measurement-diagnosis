# =============================================================================
# STEP 4 — Joint, differentiable concept-bottleneck model  (contribution C1)
#
#   image -> U-Net soft masks -> concepts computed in closed form (mL, %, g, mm)
#         -> linear head -> diagnosis,  trained END TO END.
#
# Per outer fold k (leakage-free, same patients as steps 1-3):
#   - U-Net initialised from step-1 weights of fold k (trained on the same patients)
#   - head initialised from a logistic regression on the reference concepts of the
#     training patients (C by inner 5-fold CV), i.e. exactly the step-2b model
#   - loss = diagnosis CE + LAMBDA_C * concept loss + LAMBDA_S * segmentation loss
#       concept loss: Huber on standardised concepts vs reference (training patients)
#       the 13 volume-based concepts are differentiable (sum of soft probabilities x
#       voxel volume); the 8 wall-thickness concepts come from the hard mask (no grad)
#   - epoch chosen on the inner-validation patients (diagnosis NLL), then the
#     held-out fold is predicted once.
# Ablation for H3: LAMBDA_C=0 (no concept supervision) -> RUN_TAG=_noconcept
#
# Env: LAMBDA_C (1.0), LAMBDA_S (1.0), EPOCHS (30), RUN_TAG ('')
# GPU needed (about 5-10 min per fold on a T4).
# =============================================================================
import os, glob, time, copy, zipfile
import numpy as np, pandas as pd, nibabel as nib, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.utils.checkpoint import checkpoint
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
if KAGGLE:
    DRIVE = '/kaggle/working'
    ROOT = os.path.dirname(os.path.dirname(sorted(glob.glob('/kaggle/input/**/patient001/Info.cfg', recursive=True))[0]))
    FOLDS = sorted(glob.glob('/kaggle/input/**/folds.csv', recursive=True))[0]
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except (ImportError, NotImplementedError):
        pass
    DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
    ROOT = '/content/acdc/training'
    FOLDS = f'{DRIVE}/folds.csv'
    if not os.path.isdir(ROOT):
        zipfile.ZipFile('/content/drive/MyDrive/ACDC_calibration_article/training-20261021T163705Z-1-001.zip').extractall('/content/acdc')
STEP1 = os.environ.get('STEP1_DIR', f'{DRIVE}/step1_unet')
OUT = f"{DRIVE}/step4_joint{os.environ.get('RUN_TAG', '')}"
os.makedirs(OUT, exist_ok=True)
LAMBDA_C, LAMBDA_S = float(os.environ.get('LAMBDA_C', 1.0)), float(os.environ.get('LAMBDA_S', 1.0))
EPOCHS, BP, SEED, SIZE, WIDTH, BOOT = int(os.environ.get('EPOCHS', 30)), 4, 42, 224, 32, 2000
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device {dev} | LAMBDA_C {LAMBDA_C} | LAMBDA_S {LAMBDA_S} | EPOCHS {EPOCHS} | out {OUT}')

# ------------------------------------------------ U-Net (same as step 1)
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

# ------------------------------------------------ data
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

def rs(a, mode):                                          # (Z, H, W) -> (Z, S, S)
    t = torch.from_numpy(np.ascontiguousarray(a))[:, None].float()
    if mode == 'nearest':
        return F.interpolate(t, size=(SIZE, SIZE), mode='nearest')[:, 0]
    return F.interpolate(t, size=(SIZE, SIZE), mode='bilinear', align_corners=False)[:, 0]

folds = pd.read_csv(FOLDS)
order = folds.patient.values
classes = np.array(sorted(folds.label.unique()))
y = np.searchsorted(classes, folds.label.values)
f = folds.fold.values
D = []
t0 = time.time()
for p in order:
    d, info = f'{ROOT}/{p}', read_cfg(f'{ROOT}/{p}/Info.cfg')
    item = {}
    for ph in ('ED', 'ES'):
        img = nib.load(nii(f'{d}/{p}_frame{int(info[ph]):02d}'))
        gt = np.rint(nib.load(nii(f'{d}/{p}_frame{int(info[ph]):02d}_gt')).get_fdata()).astype(np.int64)
        v = norm(img.get_fdata())
        item[ph] = dict(x=rs(v.transpose(2, 0, 1), 'bilinear')[:, None], ys=rs(gt.transpose(2, 0, 1), 'nearest').long(),
                        gt=gt, hw=v.shape[:2], vox=float(np.prod(img.header.get_zooms()[:3])) / 1000,
                        zoom=tuple(float(z) for z in img.header.get_zooms()[:3]))
    D.append(item)
print(f'loaded {len(D)} patients in {time.time() - t0:.0f}s')

# ------------------------------------------------ concepts (same 21 as steps 2b and 3)
def wt_stats(myo, zooms):
    t = [2 * distance_transform_edt(myo[:, :, z], sampling=zooms[:2]).max() - np.mean(zooms[:2])
         for z in range(myo.shape[2]) if myo[:, :, z].sum() >= 10]
    t = np.array(t) if t else np.array([0.0])
    return [t.max(), t.mean(), t.std(), t.min()]

def concepts(prob, item):
    """prob[ph]: (Z, 4, H, W) torch probabilities (or one-hot). Returns (21,) torch tensor."""
    v = {f'{s}_{ph}': prob[ph][:, c].sum() * item[ph]['vox'] for ph in ('ED', 'ES') for s, c in (('RV', 1), ('MYO', 2), ('LV', 3))}
    e = 1e-6
    diff = [v['RV_ED'], v['MYO_ED'], v['LV_ED'], v['RV_ES'], v['MYO_ES'], v['LV_ES'],
            100 * (v['LV_ED'] - v['LV_ES']) / (v['LV_ED'] + e), 100 * (v['RV_ED'] - v['RV_ES']) / (v['RV_ED'] + e),
            1.05 * v['MYO_ED'], v['RV_ED'] / (v['LV_ED'] + e), v['RV_ES'] / (v['LV_ES'] + e),
            v['MYO_ED'] / (v['LV_ED'] + e), v['MYO_ES'] / (v['LV_ES'] + e)]
    wt = []
    for ph in ('ED', 'ES'):
        hard = (prob[ph].detach().argmax(1) == 2).permute(1, 2, 0).cpu().numpy()
        wt += wt_stats(hard, item[ph]['zoom'])
    return torch.stack(diff + [torch.tensor(w, dtype=torch.float32, device=diff[0].device) for w in wt])

NAMES = ['RV_ED', 'MYO_ED', 'LV_ED', 'RV_ES', 'MYO_ES', 'LV_ES', 'LV_EF', 'RV_EF', 'LV_mass', 'RV_LV_ED', 'RV_LV_ES',
         'MYO_LV_ED', 'MYO_LV_ES', 'WT_ED_max', 'WT_ED_mean', 'WT_ED_std', 'WT_ED_min',
         'WT_ES_max', 'WT_ES_mean', 'WT_ES_std', 'WT_ES_min']
N_DIFF = 13

def onehot_prob(item, ph):
    return F.one_hot(torch.from_numpy(item[ph]['gt'].transpose(2, 0, 1)), 4).permute(0, 3, 1, 2).float()

C_ref = torch.stack([concepts({ph: onehot_prob(it, ph) for ph in ('ED', 'ES')}, it) for it in D]).numpy()

def forward_probs(net, item, ph, aug_k=None):
    x = item[ph]['x'].to(dev)
    if aug_k is not None:
        x = torch.rot90(x, aug_k[0], (-2, -1)).flip(-1) if aug_k[1] else torch.rot90(x, aug_k[0], (-2, -1))
    if torch.is_grad_enabled():                               # gradient checkpointing, 4 slices at a time:
        logits = torch.cat([checkpoint(net, x[j:j + 4], use_reentrant=False)  # activations are recomputed in
                            for j in range(0, x.shape[0], 4)]).float()        # backward instead of stored
    else:
        logits = net(x).float()
    if aug_k is not None:                                     # undo the augmentation on the outputs
        logits = logits.flip(-1) if aug_k[1] else logits
        logits = torch.rot90(logits, -aug_k[0], (-2, -1))
    p = F.interpolate(logits.softmax(1), size=item[ph]['hw'], mode='bilinear', align_corners=False)
    return logits, p                                          # logits at S x S, probs at native size

def seg_loss(logits, ys):
    p = logits.softmax(1)
    oh = F.one_hot(ys, 4).permute(0, 3, 1, 2).float()
    dice = 1 - ((2 * (p * oh).sum((0, 2, 3)) + 1) / (p.sum((0, 2, 3)) + oh.sum((0, 2, 3)) + 1))[1:].mean()
    return F.cross_entropy(logits, ys) + dice

# ------------------------------------------------ training per fold
P_joint = np.zeros((len(y), len(classes)))
C_joint = np.zeros_like(C_ref)
curves, picks, dice_rows = [], [], []
for k in range(5):
    t0 = time.time()
    tr_all = np.where(f != k)[0]
    a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(tr_all, y[tr_all]))
    tr, va, te = tr_all[a], tr_all[b], np.where(f == k)[0]          # same inner split as steps 1 and cinema

    grid = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                        {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy')
    grid.fit(C_ref[tr_all], y[tr_all])
    sc, lr = grid.best_estimator_.named_steps['standardscaler'], grid.best_estimator_.named_steps['logisticregression']
    mu, sd = torch.tensor(sc.mean_, dtype=torch.float32, device=dev), torch.tensor(sc.scale_, dtype=torch.float32, device=dev)
    head = nn.Linear(21, len(classes)).to(dev)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(lr.coef_, dtype=torch.float32)); head.bias.copy_(torch.tensor(lr.intercept_, dtype=torch.float32))
    cw = torch.tensor(len(tr) / (len(classes) * np.bincount(y[tr], minlength=len(classes))), dtype=torch.float32, device=dev)

    np.random.seed(SEED + k); torch.manual_seed(SEED + k)
    net = UNet(c=WIDTH).to(dev)
    net.load_state_dict(torch.load(f'{STEP1}/unet_fold{k}.pt', map_location=dev))
    opt = torch.optim.AdamW([{'params': net.parameters(), 'lr': 1e-4}, {'params': head.parameters(), 'lr': 1e-3}], weight_decay=1e-4)

    def run(idx, train):
        net.train(train); head.train(train)
        if train:                                               # BatchNorm statistics stay frozen (batches = one patient)
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        logits_all, conc_all = [], []
        with torch.set_grad_enabled(train):
            for i in idx:
                augk = (int(np.random.randint(4)), bool(np.random.randint(2))) if train else None
                lp = {ph: forward_probs(net, D[i], ph, augk) for ph in ('ED', 'ES')}
                c = concepts({ph: lp[ph][1].float() for ph in ('ED', 'ES')}, D[i])
                if train:
                    yield i, lp, c
                else:
                    logits_all.append(head((c - mu) / sd)); conc_all.append(c)
        if not train:
            yield torch.stack(logits_all), torch.stack(conc_all)

    def evaluate(idx):
        lo, cc = next(run(idx, False))
        return lo, cc

    best, best_state, best_ep = np.inf, None, 0
    lo0, _ = evaluate(va)                                       # epoch 0 = the sequential model
    best = F.cross_entropy(lo0, torch.tensor(y[va], device=dev)).item()
    best_state = (copy.deepcopy(net.state_dict()), copy.deepcopy(head.state_dict()))
    curves.append(dict(fold=k, epoch=0, val_nll=best, val_bal_acc=balanced_accuracy_score(y[va], lo0.argmax(1).cpu().numpy())))
    for ep in range(1, EPOCHS + 1):
        rng = np.random.default_rng(SEED * 100 + k * 1000 + ep)
        perm = rng.permutation(tr)
        tot = {'dx': 0.0, 'concept': 0.0, 'seg': 0.0}
        for s in range(0, len(perm) - BP + 1, BP):
            opt.zero_grad(set_to_none=True)
            for i, lp, c in run(perm[s:s + BP], True):            # one patient at a time, gradients accumulated
                z = (c - mu) / sd
                l_dx = cw[y[i]] * F.cross_entropy(head(z)[None], torch.tensor([y[i]], device=dev))   # class-balanced
                zr = (torch.tensor(C_ref[i], device=dev) - mu) / sd
                l_c = F.smooth_l1_loss(z[:N_DIFF], zr[:N_DIFF])
                l_s = sum(seg_loss(lp[ph][0], D[i][ph]['ys'].to(dev)) for ph in ('ED', 'ES')) / 2
                ((l_dx + LAMBDA_C * l_c + LAMBDA_S * l_s) / BP).backward()
                for key, val in (('dx', l_dx), ('concept', l_c), ('seg', l_s)):
                    tot[key] += val.item() / BP
                del lp, c, z
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(head.parameters()), 5.0)
            opt.step()
        lo, _ = evaluate(va)
        nll = F.cross_entropy(lo, torch.tensor(y[va], device=dev)).item()
        n_b = max(1, len(perm) // BP)
        curves.append(dict(fold=k, epoch=ep, train_dx=tot['dx'] / n_b, train_concept=tot['concept'] / n_b,
                           train_seg=tot['seg'] / n_b, val_nll=nll,
                           val_bal_acc=balanced_accuracy_score(y[va], lo.argmax(1).cpu().numpy())))
        if nll < best:
            best, best_ep = nll, ep
            best_state = (copy.deepcopy(net.state_dict()), copy.deepcopy(head.state_dict()))

    net.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    torch.save({'unet': best_state[0], 'head': best_state[1], 'mu': mu.cpu(), 'sd': sd.cpu()}, f'{OUT}/joint_fold{k}.pt')
    lo, cc = evaluate(te)
    P_joint[te] = lo.softmax(1).detach().cpu().numpy()
    C_joint[te] = cc.detach().cpu().numpy()
    picks.append(dict(fold=k, best_epoch=best_ep, inner_val_nll=best, C=grid.best_params_['logisticregression__C']))
    print(f'fold {k}: best epoch {best_ep}/{EPOCHS} (0 = sequential), inner-val NLL {best:.3f}, {time.time() - t0:.0f}s')
    pd.DataFrame(curves).to_csv(f'{OUT}/learning_curves.csv', index=False)
    with torch.no_grad():                                       # 3D Dice of the joint U-Net on held-out patients
        for i in te:
            row = dict(patient=order[i], fold=k)
            for ph in ('ED', 'ES'):
                pred = forward_probs(net, D[i], ph)[1].argmax(1).cpu().numpy().transpose(1, 2, 0)
                for n, c in (('RV', 1), ('MYO', 2), ('LV', 3)):
                    a_, b_ = pred == c, D[i][ph]['gt'] == c
                    row[f'{ph}_{n}'] = 2 * (a_ & b_).sum() / max(a_.sum() + b_.sum(), 1)
            dice_rows.append(row)
    del net, head, opt
    torch.cuda.empty_cache()

# ------------------------------------------------ results
np.save(f'{OUT}/oof_probs_joint.npy', P_joint)
pd.DataFrame(C_joint, columns=NAMES).assign(patient=order, source='joint').to_csv(f'{OUT}/concepts_joint.csv', index=False)
pd.DataFrame(picks).to_csv(f'{OUT}/model_selection.csv', index=False)
dice = pd.DataFrame(dice_rows)
dice.to_csv(f'{OUT}/dice_test.csv', index=False)
print('\nHeld-out 3D Dice, joint U-Net:\n', dice.drop(columns=['patient', 'fold']).mean().round(3).to_string())

P = {'joint (ours)': P_joint}
for name, path in (('sequential concept21 (step 2b)', f'{DRIVE}/step2b_concepts21/oof_probs_concept21_LR_pred.npy'),
                   ('challenge_RF_pred (step 3)', f'{DRIVE}/step3_challenge_baseline/oof_probs_challenge_RF_pred.npy')):
    if os.path.exists(path):
        P[name] = np.load(path)

def ece(Pm, yy, bins=15):
    conf, acc = Pm.max(1), (Pm.argmax(1) == yy).astype(float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = (conf > lo_) & (conf <= hi_)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

def metrics(Pm, i):
    return (balanced_accuracy_score(y[i], Pm[i].argmax(1)),
            roc_auc_score(y[i], Pm[i], multi_class='ovr', average='macro', labels=range(len(classes))), ece(Pm[i], y[i]))

rng = np.random.default_rng(SEED)
boots = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(BOOT)) if len(np.unique(y[i])) == len(classes)]
rows = []
for name, Pm in P.items():
    pt, bs = metrics(Pm, np.arange(len(y))), np.array([metrics(Pm, i) for i in boots])
    lo_, hi_ = np.percentile(bs, [2.5, 97.5], axis=0)
    rows.append(dict(model=name, bal_acc=pt[0], bal_lo=lo_[0], bal_hi=hi_[0], AUC=pt[1], AUC_lo=lo_[1], AUC_hi=hi_[1],
                     ECE=pt[2], macro_F1=f1_score(y, Pm.argmax(1), average='macro')))
res = pd.DataFrame(rows)
print('\n', res.round(3).to_string(index=False))
for ref in list(P)[1:]:
    d = np.array([np.subtract(metrics(P['joint (ours)'], i)[:2], metrics(P[ref], i)[:2]) for i in boots])
    print(f'\nPaired difference: joint minus {ref}')
    for j, m in enumerate(['bal_acc', 'AUC']):
        print(f'  {m}: {d[:, j].mean():+.3f}  95% CI [{np.percentile(d[:, j], 2.5):+.3f}, {np.percentile(d[:, j], 97.5):+.3f}]')

print('\nConcept agreement on held-out patients (joint minus reference): bias, 95% LoA, SD ratio')
ba = pd.DataFrame([dict(concept=n, bias=(C_joint[:, j] - C_ref[:, j]).mean(),
                        loa=1.96 * (C_joint[:, j] - C_ref[:, j]).std(ddof=1),
                        sd_ratio=C_joint[:, j].std() / C_ref[:, j].std()) for j, n in enumerate(NAMES)])
print(ba.round(2).to_string(index=False))
ba.to_csv(f'{OUT}/concept_agreement.csv', index=False)
res.to_csv(f'{OUT}/step4_results.csv', index=False)
print('\nConfusion matrix, joint (rows = truth):', [str(c) for c in classes])
print(confusion_matrix(y, P_joint.argmax(1)))
print('\nSaved to', OUT)
