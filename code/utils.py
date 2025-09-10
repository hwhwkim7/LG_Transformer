import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, balanced_accuracy_score
)

from dataloader import WindowDataset

# 데이터로더 생성
def make_loader(windows_path, labels_path, sid_path, batch_size, shuffle, num_workers=4, pin_memory=True):
    ds = WindowDataset(windows_path, labels_path, sid_path)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    return loader, ds

# Positive class weight 계산 (클래스 불균형 대응용)
def compute_pos_weight(train_labels_path, device):
    y = np.load(train_labels_path)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    w = float(neg) / max(float(pos), 1.0) if neg > 0 else 1.0
    return torch.tensor([w], dtype=torch.float32, device=device)

# threshold 계산
def find_best_threshold(scores, targets, metric="f1"):
    best_t, best_v = 0.5, -1
    for t in np.linspace(0.05, 0.95, 181):  # 0.05~0.95(간격 0.005)
        preds = (scores >= t).astype(int)
        if metric == "f1":
            v = f1_score(targets, preds, zero_division=0)
        elif metric == "mcc":
            v = matthews_corrcoef(targets, preds)
        elif metric == "ba":
            v = balanced_accuracy_score(targets, preds)
        else:
            raise ValueError("metric must be f1/mcc/ba")
        if v > best_v:
            best_v, best_t = v, t
    return best_t, best_v

# best threshold 선정
def pick_final_threshold(scores, targets):
    # 후보 threshold: metric별 최적
    t_f1, _  = find_best_threshold(scores, targets, metric="f1")
    t_mcc, _ = find_best_threshold(scores, targets, metric="mcc")
    t_ba, _  = find_best_threshold(scores, targets, metric="ba")
    cands = sorted(set([t_f1, t_mcc, t_ba]))
    best_t, best_combo = cands[0], -1
    from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score
    for t in cands:
        preds = (scores >= t).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        mcc = matthews_corrcoef(targets, preds)
        ba = balanced_accuracy_score(targets, preds)
        # 종합 점수 (평균)
        combo = (f1 + mcc + ba) / 3.0
        if combo > best_combo:
            best_combo, best_t = combo, t

    return best_t, best_combo, {"t_f1": t_f1, "t_mcc": t_mcc, "t_ba": t_ba}

# 평가
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    logits_all, targets_all, seq_ids_all = [], [], []

    for x, y, sid in loader:
        x = x.to(device)  # (B,L,C)
        y = y.float().to(device)

        logits = model(x)
        loss = criterion(logits.squeeze(), y)
        total_loss += loss.item()

        logits_all.append(logits.detach().cpu())
        targets_all.append(y.detach().cpu())
        seq_ids_all.append(torch.as_tensor(sid))

    # numpy 변환
    logits_all = torch.cat(logits_all).numpy()
    targets_all = torch.cat(targets_all).numpy()

    # 예측 (동적인 threshold 기준)
    # 점수는 시그모이드 확률로 맞춰서 임계값 탐색/ROC에 사용
    scores = 1.0 / (1.0 + np.exp(-logits_all))

    # --- 윈도우 레벨: 최종 임계값 선택 (F1/MCC/BA 후보 → 평균점수 최대) ---
    best_t, combo, parts = pick_final_threshold(scores, targets_all)
    preds = (scores >= best_t).astype(int)

    # confusion matrix
    ## 정상을 정상으로 / 정상을 누수로
    ## 누수를 정상으로 / 누수를 누수로
    cm = confusion_matrix(targets_all, preds)
    print("Confusion Matrix:")
    print(cm)

    # AUROC
    try:
        auroc = roc_auc_score(targets_all, logits_all)
    except ValueError:
        auroc = float('nan')  # 한쪽 클래스만 있으면 계산 불가

    # Precision, Recall, F1, Accuracy
    precision = precision_score(targets_all, preds, zero_division=0)
    recall    = recall_score(targets_all, preds, zero_division=0)
    f1        = f1_score(targets_all, preds, zero_division=0)
    acc       = accuracy_score(targets_all, preds)

    avg_loss = total_loss / len(loader)

    # Sequence-level 평가 (시퀀스 내 윈도우들을 다수결로 집계)
    seq_metrics = None
    if len(seq_ids_all) > 0:
        seq_ids_all = torch.cat(seq_ids_all).numpy()  # (N,)
        # 시퀀스별 윈도우 예측을 다수결로 집계
        # 시퀀스 GT는 윈도우 라벨의 max(원래 동일해야 하지만 안전하게)
        from collections import defaultdict
        bins_pred = defaultdict(list)
        bins_prob = defaultdict(list)
        bins_tgt = defaultdict(list)

        for p, pr, t, s in zip(preds, logits_all, targets_all, seq_ids_all):
            bins_pred[s].append(p)
            bins_prob[s].append(pr)
            bins_tgt[s].append(t)

        seq_preds, seq_targets, seq_scores = [], [], []
        for s in bins_pred.keys():
            votes = np.array(bins_pred[s])
            probs = np.array(bins_prob[s])
            tgt = int(np.max(bins_tgt[s]))  # robust GT
            # --- 다수결 ---
            seq_pred = int(votes.sum() >= (len(votes) / 2.0))
            # 시퀀스 스코어는 평균 확률(보고용)
            seq_score = float(np.mean(probs))

            seq_preds.append(seq_pred)
            seq_targets.append(tgt)
            seq_scores.append(seq_score)

        seq_preds = np.array(seq_preds)
        seq_targets = np.array(seq_targets)
        seq_scores = np.array(seq_scores)

        cm_seq = confusion_matrix(seq_targets, seq_preds)
        acc_s = accuracy_score(seq_targets, seq_preds)
        prec_s = precision_score(seq_targets, seq_preds, zero_division=0)
        rec_s = recall_score(seq_targets, seq_preds, zero_division=0)
        f1_s = f1_score(seq_targets, seq_preds, zero_division=0)
        try:
            auroc_s = roc_auc_score(seq_targets, seq_scores)
        except ValueError:
            auroc_s = float('nan')

        print("[SEQUENCE | MAJORITY] Confusion Matrix:")
        print(cm_seq)

        seq_metrics = (cm_seq, auroc_s, acc_s, prec_s, rec_s, f1_s)

    return avg_loss, cm, auroc, acc, precision, recall, f1, seq_metrics

# 1 epoch 학습
def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    for x, y, _ in loader:
        x = x.to(device)                  # (B,L,C)
        y = y.float().to(device)          # BCE용 float (0/1)
        logits = model(x)                 # (B,)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

# 체크포인트 저장
def save_ckpt(path, model, optimizer, epoch, best_val):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val
    }, path)
