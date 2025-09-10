import os
import argparse
import numpy as np
import csv

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import WindowDataset
from model import TimeTransformer
from utils import *

def train(args):
    # GPU 선택 (cuda:{args.gpu})가 가능하면 GPU, 아니면 CPU
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    # 체크포인트/메트릭 저장 경로 준비
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_model.pth")
    metrics_path = os.path.join(args.save_dir, "metrics.csv")  # 여기에 계속 append

    # CSV 헤더 생성(한 번만)
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "phase", "epoch", "lr",
                "loss", "auroc", "acc", "precision", "recall", "f1",
                "tn", "fp", "fn", "tp", "best_val_so_far"
            ])

    # 데이터 로더 생성
    train_loader, train_ds = make_loader(
        os.path.join(args.data_dir, "train_windows.npy"),
        os.path.join(args.data_dir, "train_labels.npy"),
        os.path.join(args.data_dir, "train_seq_ids.npy"),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader, _ = make_loader(
        os.path.join(args.data_dir, "val_windows.npy"),
        os.path.join(args.data_dir, "val_labels.npy"),
        os.path.join(args.data_dir, "val_seq_ids.npy"),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader, _ = make_loader(
        os.path.join(args.data_dir, "test_windows.npy"),
        os.path.join(args.data_dir, "test_labels.npy"),
        os.path.join(args.data_dir, "test_seq_ids.npy"),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    c_in = train_ds.X.shape[-1]  # 채널 수

    # 모델 구성
    model = TimeTransformer(
        c_in=c_in,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dropout=args.dropout,
        use_cls=args.use_cls
    ).to(device)

    # loss / optimizer / scheduler
    # 클래스 불균형 보정을 위한 pos_weight 계산 (BCEWithLogitsLoss에 전달)
    pos_weight = compute_pos_weight(os.path.join(args.data_dir, "train_labels.npy"), device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = -1.0
    wait = 0

    print('--- Train ---')

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # ---- Validate (윈도우 기준 + 시퀀스 다수결) ----
        val_loss, cm, val_auroc, val_acc, val_prec, val_rec, val_f1, val_seq = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        # 출력
        print(f"[{epoch:03d}] train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_auroc={val_auroc:.4f} | "
              f"val_acc={val_acc:.4f} | val_prec={val_prec:.4f} | val_rec={val_rec:.4f} | val_f1={val_f1:.4f}")

        # CSV 기록 (VAL window)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        with open(metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            # train row (loss만)
            w.writerow(["train", epoch, lr_now, f"{train_loss:.6f}", "", "", "", "", "", "", "", "", f"{best_val:.6f}"])
            # val window row
            w.writerow(["val_window", epoch, lr_now, f"{val_loss:.6f}", f"{val_auroc:.6f}", f"{val_acc:.6f}",
                        f"{val_prec:.6f}", f"{val_rec:.6f}", f"{val_f1:.6f}",
                        tn, fp, fn, tp, f"{best_val:.6f}"])
            # val sequence row (있으면)
            if val_seq is not None:
                cm_s, auroc_s, acc_s, prec_s, rec_s, f1_s = val_seq
                tn_s, fp_s, fn_s, tp_s = int(cm_s[0, 0]), int(cm_s[0, 1]), int(cm_s[1, 0]), int(cm_s[1, 1])
                w.writerow(["val_sequence", epoch, lr_now, "", f"{auroc_s:.6f}", f"{acc_s:.6f}",
                            f"{prec_s:.6f}", f"{rec_s:.6f}", f"{f1_s:.6f}",
                            tn_s, fp_s, fn_s, tp_s, f"{best_val:.6f}"])

        # --- early stopping on val AUROC ---
        score = val_auroc if not np.isnan(val_auroc) else -1.0
        if score > best_val:
            best_val = score
            wait = 0
            save_ckpt(ckpt_path, model, optimizer, epoch, best_val)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best val AUROC={best_val:.4f}")
                break

    print('--- Test ---')

     # --- best ckpt 로드 후 test 평가 ---
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        print(f"Loaded best checkpoint from epoch {state['epoch']}, best_val={state['best_val']:.4f}")

        # Test (윈도우 + 시퀀스 다수결)
    test_loss, cm_t, test_auroc, test_acc, test_prec, test_rec, test_f1, test_seq = evaluate(
        model, test_loader, criterion, device
    )

    # CSV에 TEST도 기록
    tn, fp, fn, tp = int(cm_t[0, 0]), int(cm_t[0, 1]), int(cm_t[1, 0]), int(cm_t[1, 1])
    with open(metrics_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_window", "final", "", f"{test_loss:.6f}", f"{test_auroc:.6f}", f"{test_acc:.6f}",
                    f"{test_prec:.6f}", f"{test_rec:.6f}", f"{test_f1:.6f}",
                    tn, fp, fn, tp, f"{best_val:.6f}"])
        if test_seq is not None:
            cm_s, auroc_s, acc_s, prec_s, rec_s, f1_s = test_seq
            tn_s, fp_s, fn_s, tp_s = int(cm_s[0, 0]), int(cm_s[0, 1]), int(cm_s[1, 0]), int(cm_s[1, 1])
            w.writerow(["test_sequence", "final", "", "", f"{auroc_s:.6f}", f"{acc_s:.6f}",
                        f"{prec_s:.6f}", f"{rec_s:.6f}", f"{f1_s:.6f}",
                        tn_s, fp_s, fn_s, tp_s, f"{best_val:.6f}"])

