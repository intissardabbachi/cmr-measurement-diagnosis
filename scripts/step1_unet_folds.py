# =============================================================================
# STEP 1 — 2D U-Net per outer fold (leakage-free) + learning curves + biomarkers
# Article: physically grounded concept bottleneck for CMR diagnosis (ACDC)
#
# Run in Colab or Kaggle (GPU) as ONE cell. Time: about 5-10 min per fold on a T4.
# Settings can be changed with environment variables before %run:
#   SIZE (224), WIDTH (32), EPOCHS (40), RUN_TAG ('' -> step1_unet, '_pilot' -> step1_unet_pilot)
#   CPU pilot: SIZE=128, WIDTH=16, EPOCHS=15, RUN_TAG=_pilot  (code check only, not final results)
#
# Leakage rules (fixed):
#   - Outer folds = folds.csv (patient-level, seed 42).
#   - For outer fold k: U-Net trains on 90% of the training patients;
#     the other 10% (inner validation, stratified) choose the best epoch.
#   - Held-out fold-k patients are seen only once, at the end, for prediction.
#
# Outputs (Drive: CMR_concept_bottleneck/step1_unet/):
#   unet_fold{k}.pt          best weights per fold (selected on inner validation)
#   learning_curves.csv      fold, epoch, train loss, val loss, val Dice per structure
#   split_check.csv          train / inner-val / test patients per fold
#   biomarkers.csv           per patient: source = gt | pred_soft | pred_hard
#   dice_test.csv            per held-out patient: 3D Dice RV, MYO, LV at ED and ES
# =============================================================================
import os, glob, zipfile, time, copy
import numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):   # Kaggle ships google.colab but cannot mount Drive
    pass

KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ   # set only inside Kaggle notebooks (Colab also has /kaggle/input)
if KAGGLE:                                   # Kaggle: data and folds.csv come from an uploaded dataset
    DRIVE = '/kaggle/working'
    ROOT = os.path.dirname(os.path.dirname(sorted(glob.glob('/kaggle/input/**/patient001/Info.cfg', recursive=True))[0]))
    FOLDS = sorted(glob.glob('/kaggle/input/**/folds.csv', recursive=True))[0]
else:
    DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
    ROOT = os.environ.get('ACDC_ROOT', '/content/acdc/training')
    FOLDS = f'{DRIVE}/folds.csv'
# ACDC zip stays in its upload folder (not duplicated, to save Drive space)
ZIP = '/content/drive/MyDrive/ACDC_calibration_article/training-20261021T163705Z-1-001.zip'
OUT = f"{DRIVE}/step1_unet{os.environ.get('RUN_TAG', '')}"
os.makedirs(OUT, exist_ok=True)

SEED, BS, LR = 42, 16, 1e-3
SIZE, WIDTH, EPOCHS = (int(os.environ.get(k, d)) for k, d in (('SIZE', 224), ('WIDTH', 32), ('EPOCHS', 40)))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(os.cpu_count() or 1)
print(f'device: {dev} | SIZE {SIZE} | WIDTH {WIDTH} | EPOCHS {EPOCHS} | data {ROOT} | out {OUT}')

if not os.path.isdir(ROOT):
    print('unzipping ACDC ...')
    with zipfile.ZipFile(ZIP) as z:
        z.extractall('/content/acdc')
pids_all = sorted(os.path.basename(p) for p in glob.glob(f'{ROOT}/patient*'))
print('patients found:', len(pids_all))

# ---------------------------------------------------------------- data
def read_cfg(p):
    d = {}
    for line in open(p):
        if ':' in line:
            k, v = line.split(':', 1)
            d[k.strip()] = v.strip()
    return d

def norm(v):
    lo, hi = np.percentile(v, [1, 99])
    v = np.clip(v, lo, hi)
    return ((v - v.mean()) / (v.std() + 1e-8)).astype(np.float32)

def nii(base):                                 # .nii.gz (official) or .nii (some Kaggle copies)
    for e in ('.nii.gz', '.nii'):
        if os.path.exists(base + e):
            return base + e
    raise FileNotFoundError(base + '.nii[.gz]')

def load_patient(pid):
    d = f'{ROOT}/{pid}'
    info = read_cfg(f'{d}/Info.cfg')
    out = {'info': info}
    for ph in ('ED', 'ES'):
        fr = int(info[ph])
        img = nib.load(nii(f'{d}/{pid}_frame{fr:02d}'))
        gt = nib.load(nii(f'{d}/{pid}_frame{fr:02d}_gt'))
        out[ph] = (norm(img.get_fdata()),
                   np.rint(gt.get_fdata()).astype(np.int64),   # 0 bg, 1 RV, 2 MYO, 3 LV
                   tuple(float(z) for z in img.header.get_zooms()[:3]))
    return out

t0 = time.time()
P = {p: load_patient(p) for p in pids_all}
print(f'loaded in {time.time() - t0:.0f}s')

def rs(a, mode):
    t = torch.from_numpy(np.ascontiguousarray(a))[None, None].float()
    if mode == 'nearest':
        return F.interpolate(t, size=(SIZE, SIZE), mode='nearest')[0, 0]
    return F.interpolate(t, size=(SIZE, SIZE), mode='bilinear', align_corners=False)[0, 0]

def slices(pids):
    X, Y = [], []
    for p in pids:
        for ph in ('ED', 'ES'):
            v, g, _ = P[p][ph]
            for z in range(v.shape[2]):
                X.append(rs(v[:, :, z], 'bilinear'))
                Y.append(rs(g[:, :, z], 'nearest').long())
    return torch.stack(X)[:, None], torch.stack(Y)

# ---------------------------------------------------------------- model
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(s, c=32, n=4, p=0.1):
        super().__init__()
        s.e1, s.e2, s.e3, s.e4 = block(1, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        s.b = block(8 * c, 16 * c)
        s.drop = nn.Dropout2d(p)          # kept for MC dropout in a later step
        s.u4, s.d4 = nn.ConvTranspose2d(16 * c, 8 * c, 2, 2), block(16 * c, 8 * c)
        s.u3, s.d3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), block(8 * c, 4 * c)
        s.u2, s.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), block(4 * c, 2 * c)
        s.u1, s.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), block(2 * c, c)
        s.out = nn.Conv2d(c, n, 1)

    def forward(s, x):
        e1 = s.e1(x)
        e2 = s.e2(F.max_pool2d(e1, 2))
        e3 = s.e3(F.max_pool2d(e2, 2))
        e4 = s.e4(F.max_pool2d(e3, 2))
        b = s.drop(s.b(F.max_pool2d(e4, 2)))
        d4 = s.d4(torch.cat([s.u4(b), e4], 1))
        d3 = s.d3(torch.cat([s.u3(d4), e3], 1))
        d2 = s.d2(torch.cat([s.u2(d3), e2], 1))
        d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
        return s.out(d1)

def dice_loss(logits, y):
    p = logits.softmax(1)
    oh = F.one_hot(y, 4).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (den + 1))[1:].mean()

def aug(x, y):
    if torch.rand(1) < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if torch.rand(1) < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    k = int(torch.randint(0, 4, (1,)))
    x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
    n = x.size(0)
    x = x * (1 + 0.1 * torch.randn(n, 1, 1, 1, device=x.device)) + 0.1 * torch.randn(n, 1, 1, 1, device=x.device)
    return x, y

@torch.no_grad()
def evaluate(net, X, Y):
    net.eval()
    L, inter, den = 0.0, np.zeros(3), np.zeros(3)
    for b in range(0, len(X), 64):
        x, y = X[b:b + 64].to(dev), Y[b:b + 64].to(dev)
        lo = net(x).float()
        L += F.cross_entropy(lo, y, reduction='none').mean((1, 2)).sum().item()
        pr = lo.argmax(1)
        for c in (1, 2, 3):
            inter[c - 1] += ((pr == c) & (y == c)).sum().item()
            den[c - 1] += (pr == c).sum().item() + (y == c).sum().item()
    return L / len(X), 2 * inter / np.maximum(den, 1)

@torch.no_grad()
def predict_probs(net, v):
    """v: (H, W, Z) normalised volume -> probs (4, H, W, Z) at native resolution."""
    net.eval()
    H, W, Z = v.shape
    x = torch.stack([rs(v[:, :, z], 'bilinear') for z in range(Z)])[:, None].to(dev)
    p = net(x).float().softmax(1)                                   # (Z, 4, S, S)
    p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False)
    return p.permute(1, 2, 3, 0).cpu().numpy()                      # (4, H, W, Z)

# ---------------------------------------------------------------- biomarkers
def wall_thickness_mm(myo, zooms):
    best = 0.0
    for z in range(myo.shape[2]):
        m = myo[:, :, z]
        if m.sum() < 10:
            continue
        # 2 x (distance to background) minus one pixel: half-pixel offset on each side
        best = max(best, 2 * distance_transform_edt(m, sampling=zooms[:2]).max() - np.mean(zooms[:2]))
    return best

def biomarkers(prob_ed, prob_es, zooms_ed, zooms_es, info):
    """prob_*: (4, H, W, Z) soft probabilities or one-hot. Classes 1 RV, 2 MYO, 3 LV."""
    ved, ves = np.prod(zooms_ed) / 1000.0, np.prod(zooms_es) / 1000.0   # mL per voxel
    lv_edv, lv_esv = prob_ed[3].sum() * ved, prob_es[3].sum() * ves
    rv_edv, rv_esv = prob_ed[1].sum() * ved, prob_es[1].sum() * ves
    myo_ed = prob_ed[2].sum() * ved
    h, w = float(info.get('Height', 'nan')), float(info.get('Weight', 'nan'))
    bsa = np.sqrt(h * w / 3600.0)                                       # Mosteller, m2
    return dict(LV_EDV=lv_edv, LV_ESV=lv_esv, LV_EF=100 * (lv_edv - lv_esv) / max(lv_edv, 1e-6),
                RV_EDV=rv_edv, RV_ESV=rv_esv, RV_EF=100 * (rv_edv - rv_esv) / max(rv_edv, 1e-6),
                LV_mass=1.05 * myo_ed,
                MaxWT=wall_thickness_mm(prob_ed[2] > 0.5, zooms_ed),
                LV_EDVi=lv_edv / bsa, RV_EDVi=rv_edv / bsa, LV_massi=1.05 * myo_ed / bsa,
                LV_RV_EDV_ratio=lv_edv / max(rv_edv, 1e-6), BSA=bsa)

def onehot(lbl):
    return np.stack([(lbl == c).astype(np.float32) for c in range(4)])

def dice3d(pred, gt, c):
    a, b = pred == c, gt == c
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan

# ---------------------------------------------------------------- run
folds = pd.read_csv(FOLDS)
lab = dict(zip(folds.patient, folds.label))
fold_of = dict(zip(folds.patient, folds.fold))
assert set(folds.patient) == set(pids_all), 'folds.csv and data folder do not match'

bio_rows, dice_rows, curve_rows, split_rows = [], [], [], []

for p in pids_all:                                   # reference biomarkers from GT masks
    ed, es = P[p]['ED'], P[p]['ES']
    bio_rows.append(dict(patient=p, fold=fold_of[p], label=lab[p], source='gt',
                         **biomarkers(onehot(ed[1]), onehot(es[1]), ed[2], es[2], P[p]['info'])))

for k in range(5):
    t0 = time.time()
    tr_all = folds.patient[folds.fold != k].values
    te = folds.patient[folds.fold == k].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    i_tr, i_va = next(sss.split(tr_all, [lab[p] for p in tr_all]))
    tr, va = tr_all[i_tr], tr_all[i_va]
    assert not (set(tr) & set(te)) and not (set(va) & set(te)) and not (set(tr) & set(va))
    for role, ps in (('train', tr), ('inner_val', va), ('test', te)):
        split_rows += [dict(fold=k, role=role, patient=p) for p in ps]

    Xtr, Ytr = slices(tr)
    Xva, Yva = slices(va)
    torch.manual_seed(SEED + k)
    net = UNet(c=WIDTH).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    use_amp = dev == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best, best_ep = -1.0, 0
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr))
        tl, seen = 0.0, 0
        for b in range(0, len(perm) - BS + 1, BS):
            idx = perm[b:b + BS]
            x, y = aug(Xtr[idx].to(dev), Ytr[idx].to(dev))
            with torch.autocast(device_type=dev, enabled=use_amp):
                lo = net(x)
            loss = F.cross_entropy(lo.float(), y) + dice_loss(lo.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(idx)
            seen += len(idx)
        sched.step()
        vl, vd = evaluate(net, Xva, Yva)
        curve_rows.append(dict(fold=k, epoch=ep + 1, train_loss=tl / max(seen, 1), val_ce=vl,
                               val_dice_RV=vd[0], val_dice_MYO=vd[1], val_dice_LV=vd[2], val_dice_mean=vd.mean()))
        if vd.mean() > best:
            best, best_ep = vd.mean(), ep + 1
            best_state = copy.deepcopy(net.state_dict())   # kept in memory, written once

    net.load_state_dict(best_state)
    torch.save(best_state, f'{OUT}/unet_fold{k}.pt')
    for p in te:                                      # held-out patients: predicted once
        ed, es = P[p]['ED'], P[p]['ES']
        pe, ps_ = predict_probs(net, ed[0]), predict_probs(net, es[0])
        he, hs = pe.argmax(0), ps_.argmax(0)
        bio_rows.append(dict(patient=p, fold=k, label=lab[p], source='pred_soft',
                             **biomarkers(pe, ps_, ed[2], es[2], P[p]['info'])))
        bio_rows.append(dict(patient=p, fold=k, label=lab[p], source='pred_hard',
                             **biomarkers(onehot(he), onehot(hs), ed[2], es[2], P[p]['info'])))
        dice_rows.append(dict(patient=p, fold=k, label=lab[p],
                              **{f'{ph}_{n}': dice3d(h, g[1], c)
                                 for ph, h, g in (('ED', he, ed), ('ES', hs, es))
                                 for n, c in (('RV', 1), ('MYO', 2), ('LV', 3))}))
    print(f'fold {k}: best epoch {best_ep}/{EPOCHS}, inner-val Dice {best:.3f}, {time.time() - t0:.0f}s')
    pd.DataFrame(curve_rows).to_csv(f'{OUT}/learning_curves.csv', index=False)   # saved each fold

pd.DataFrame(split_rows).to_csv(f'{OUT}/split_check.csv', index=False)
pd.DataFrame(bio_rows).to_csv(f'{OUT}/biomarkers.csv', index=False)
dice = pd.DataFrame(dice_rows)
dice.to_csv(f'{OUT}/dice_test.csv', index=False)

print('\nHeld-out 3D Dice (mean over 100 patients):')
print(dice.drop(columns=['patient', 'fold', 'label']).mean().round(3).to_string())
print('\nSaved to', OUT)
