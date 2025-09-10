import os
import glob
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# 특정 키워드가 포함된 CSV 파일 목록 가져오기
def get_csv_files(input_dir, include_keywords):
    all_csvs = glob.glob(os.path.join(input_dir, "*.csv"))
    selected = []
    for path in all_csvs:
        fname = os.path.basename(path)
        if any(kw in fname for kw in include_keywords):
            selected.append(path)
    return selected

# 시계열 데이터를 윈도우 단위로 자르기
def make_windows(seq, window_size, stride):
    T, C = seq.shape
    starts = np.arange(0, T - window_size + 1, stride)
    windows = [seq[s:s+window_size] for s in starts]
    return np.array(windows)  # (윈도우개수, window_size, C)

# 윈도우 샘플링 (최대 max_k개까지)
def sample_windows(windows, max_k, seed=42):
    np.random.seed(seed)
    M = len(windows)
    if M <= max_k:
        return windows
    idx = np.random.choice(M, size=max_k, replace=False)
    return windows[idx]

# CSV 읽기 (여러 인코딩 시도)
def read_csv_encod(path):
    READ_ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]
    for enc in READ_ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except UnicodeDecodeError:
            continue
    else:
        return None

# CSV에서 시퀀스 데이터 불러오기 (숫자형 컬럼만 사용)
def load_sequence_from_csv(path, drop_cols=None):
    df = read_csv_encod(path)
    if df is None: return None
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    num_df = df.select_dtypes(include="number")

    if num_df.empty:
        raise ValueError(f"No numeric columns found in {path}")

    return num_df.values.astype(np.float32)  # (T, C)

# Manifest 파일 생성 (메타데이터: path, row/col 개수, label)
def make_manifest(files, output_path):
    rows = []
    normal = 0
    leak = 0
    for path in files:
        try:
            df = read_csv_encod(path)
            num_df = df.select_dtypes(include="number")
            n_row, n_col = num_df.shape
            if '정상' in path: label = 0
            elif '누수' in path: label = 1
        except Exception as e:
            print(f"⚠️ {path} 읽기 실패: {e}")
            n_row, n_col = None, None
            label = None
        if label == 1:
            leak += 1
        elif label == 0:
            normal += 1

        rows.append({
            "path": path,
            "n_col": n_col,
            "n_row": n_row,
            "label": label
        })

    manifest = pd.DataFrame(rows)
    summary_row = pd.DataFrame([{
        "path": "TOTAL",
        "n_col": "",
        "n_row": "",
        "label": f"정상:{normal}, 누수:{leak}"
    }])
    manifest = pd.concat([manifest, summary_row], ignore_index=True)
    manifest.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Manifest 저장 완료: {output_path}")
    return manifest

# Train/Val/Test 분할 후 manifest 갱신
def split_manifest(manifest_path, train, test, valid, seed=42):
    df = pd.read_csv(manifest_path)
    total_row = df[df["path"] == "TOTAL"].copy()
    df_files = df[df["path"] != "TOTAL"].copy()

    updated = []
    for label in [0, 1]:
        group = df_files[df_files["label"] == str(label)]
        if group.empty:
            continue

        train_df, temp_df = train_test_split(
            group, test_size=(1 - train), random_state=seed, shuffle=True
        )
        val_ratio = valid / (valid + test)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_ratio), random_state=seed, shuffle=True)

        train_df.loc[:, "split"] = "train"
        val_df.loc[:, "split"] = "val"
        test_df.loc[:, "split"] = "test"

        updated.extend([train_df, val_df, test_df])

    out_df = pd.concat(updated, ignore_index=True)
    split_counts = out_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    train_norm, train_leak = split_counts.loc["train", '0'], split_counts.loc["train", '1']
    val_norm, val_leak = split_counts.loc["val", '0'], split_counts.loc["val", '1']
    test_norm, test_leak = split_counts.loc["test", '0'], split_counts.loc["test", '1']

    summary_row = pd.DataFrame([{
        "path": "TOTAL",
        "n_col": "",
        "n_row": "",
        "label": (
            f"정상:{(out_df['label'] == '0').sum()}, "
            f"누수:{(out_df['label'] == '1').sum()}, "
            f"train(정상:{train_norm}, 누수:{train_leak}), "
            f"val(정상:{val_norm}, 누수:{val_leak}), "
            f"test(정상:{test_norm}, 누수:{test_leak})"
        )
    }])

    final_df = pd.concat([out_df, summary_row], ignore_index=True)
    final_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print("✅ manifest 업데이트 완료 (split 요약 포함)")
    return final_df

# 시퀀스 단위 z-score 정규화
def zscore_per_sequence(arr, eps=1e-8):
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True)
    std  = np.where(std < eps, 1.0, std)   # 0 나눗셈 방지
    zarr = (arr - mean) / std
    return zarr.astype(np.float32, copy=False), mean.squeeze(0), std.squeeze(0)

# manifest 기반으로 윈도우 데이터셋 만들기 + 저장
def build_windows_from_manifest(manifest_path, window_size, stride, max_k, out_dir):
    df = pd.read_csv(manifest_path)
    df_files = df[df["path"] != "TOTAL"].copy()
    os.makedirs(out_dir, exist_ok=True)

    stats_rows = []  # CSV로 저장할 행들
    for split in ["train", "val", "test"]:
        split_files = df_files[df_files["split"] == split]
        all_wins, all_labels, all_ids = [], [], []
        s_id = 0
        for _, row in split_files.iterrows():
            seq = load_sequence_from_csv(row["path"], drop_cols=None)
            if seq is None or len(seq) < window_size:
                continue

            # ---- 디바이스별 z-score ----
            seq_z, mean_c, std_c = zscore_per_sequence(seq)
            # 통계 저장 (평균/표준편차 벡터를 문자열로 저장)
            stats_rows.append({
                "path": row["path"],
                "label": row["label"],
                "split": split,
                "mean": ",".join(map(str, mean_c)),
                "std": ",".join(map(str, std_c))
            })
            # -----------------------------------
            wins = make_windows(seq_z, window_size, stride)
            wins = sample_windows(wins, max_k)

            if len(wins) == 0:
                continue

            all_wins.append(wins)
            all_labels.extend([row["label"]] * len(wins))
            all_ids.extend([s_id] * len(wins))
            s_id += 1

        if not all_wins:
            print(f"⚠️ {split} split에 윈도우 없음")
            continue

        X = np.concatenate(all_wins, axis=0).astype(np.float32)  # (N, L, C)
        y = np.array(all_labels, dtype=np.int64)  # (N,)
        seq_ids = np.array(all_ids, dtype=np.int64)  # (N,)

        np.save(os.path.join(out_dir, f"{split}_windows.npy"), X)
        np.save(os.path.join(out_dir, f"{split}_labels.npy"), y)
        np.save(os.path.join(out_dir, f"{split}_seq_ids.npy"), seq_ids)  # ➜ 저장
        # print(f"✅ {split}: X={X.shape}, y={y.shape}, seq_ids={seq_ids.shape}")

        # ---- 통계 CSV 저장 ----
    stats_df = pd.DataFrame(stats_rows)
    stats_csv_path = os.path.join(out_dir, "per_sequence_zscore_stats.csv")
    stats_df.to_csv(stats_csv_path, index=False, encoding="utf-8-sig")
    print(f"📊 Z-score 통계 저장 완료: {stats_csv_path}")

def processor(args):
    keywords = ["normal", "leak", '정상', '누수']  # 파일명에 포함되어야 하는 문자열들
    files = get_csv_files(args.input_folder_path, keywords)
    manifest_path = '../dataset/manifest.csv'
    make_manifest(files, manifest_path)
    split_manifest(manifest_path, train=0.6, test=0.2, valid=0.2, seed=42)
    build_windows_from_manifest(manifest_path, args.window_size, args.stride, args.max_window_num, args.output_folder_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--max_window_num', type=int, default=24)
    parser.add_argument('--input_folder_path', type=str, default='../dataset/final_raw')
    parser.add_argument('--output_folder_path', type=str, default='../dataset/final_npy')
    args = parser.parse_args()

    processor(args)