import argparse
import csv
import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import load_wav, speech_before, frame_energy_db, f0_contour


def extract_features(x, sr, pause_start):
    """Uses only audio strictly before pause_start."""
    seg = speech_before(x, sr, pause_start, window_s=1.8)
    n_feat = 16
    if len(seg) < sr // 20:
        return np.zeros(n_feat, dtype=np.float32)

    e = frame_energy_db(seg, sr)
    if len(e) < 2:
        return np.zeros(n_feat, dtype=np.float32)

    energy_final = float(np.mean(e[-4:]))
    energy_mean = float(np.mean(e))
    energy_std = float(np.std(e))
    energy_min = float(np.min(e))
    energy_max = float(np.max(e))

    t = np.arange(len(e), dtype=np.float64)
    energy_slope = float(np.polyfit(t, e, 1)[0]) if len(e) >= 4 else 0.0

    n_last = max(2, int(0.25 / 0.01))
    energy_drop = (
        float(np.mean(e[:-n_last]) - np.mean(e[-n_last:]))
        if len(e) > n_last + 2 else 0.0
    )
    energy_rel_final = energy_final - energy_mean

    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]

    if len(voiced) >= 3:
        f0_final = float(np.mean(voiced[-3:]))
        f0_mean = float(np.mean(voiced))
        f0_std = float(np.std(voiced))
        last_v = voiced[-min(8, len(voiced)):]
        f0_slope = float(np.polyfit(np.arange(len(last_v)), last_v, 1)[0])
        voicing_ratio = float(len(voiced) / max(1, len(f0)))
    else:
        f0_final = f0_mean = f0_std = f0_slope = voicing_ratio = 0.0

    run = 0
    for value in f0[::-1]:
        if value > 0:
            run += 1
        else:
            break
    last_voiced_len = run * 0.01

    speech_dur = len(seg) / float(sr)
    speaking_rate = voicing_ratio / max(speech_dur, 0.05)
    turn_progress = float(min(1.0, pause_start / max(pause_start + 0.8, 1.5)))

    return np.array([
        energy_final, energy_mean, energy_std, energy_min, energy_max,
        energy_slope, energy_drop, energy_rel_final,
        f0_final, f0_mean, f0_std, f0_slope,
        voicing_ratio, last_voiced_len, speaking_rate, turn_progress,
    ], dtype=np.float32)


def load_folder(data_dir, language_name):
    """Read one language folder and return features and metadata."""
    labels_path = os.path.join(data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))
    cache = {}
    X, y, groups, keys = [], [], [], []

    for r in rows:
        audio_path = os.path.join(data_dir, r["audio_file"])
        if audio_path not in cache:
            cache[audio_path] = load_wav(audio_path)

        x, sr = cache[audio_path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        y.append(1 if r["label"] == "eot" else 0)

        # Prefix prevents en001 and hi001 from accidentally becoming same group
        groups.append(f"{language_name}_{r['turn_id']}")
        keys.append((r["turn_id"], r["pause_index"]))

    return X, y, groups, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dirs",
        nargs="+",
        required=True,
        help="One or more folders, e.g. ../eot_data/english ../eot_data/hindi",
    )
    ap.add_argument("--out", default="combined_train_predictions.csv")
    ap.add_argument("--model_out", default="model.pkl")
    args = ap.parse_args()

    X_all, y_all, groups_all, keys_all = [], [], [], []

    for data_dir in args.data_dirs:
        language_name = os.path.basename(os.path.normpath(data_dir))
        X, y, groups, keys = load_folder(data_dir, language_name)
        X_all.extend(X)
        y_all.extend(y)
        groups_all.extend(groups)
        keys_all.extend(keys)
        print(f"loaded {len(y)} pauses from {language_name}")

    X = np.asarray(X_all, dtype=np.float32)
    y = np.asarray(y_all, dtype=np.int32)

    tr, te = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        .split(X, y, groups_all)
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            learning_rate=0.06,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])

    pipe.fit(X[tr], y[tr])
    print(
        f"held-out-turn accuracy: {pipe.score(X[te], y[te]):.3f} "
        f"(chance ~ {max(np.mean(y), 1 - np.mean(y)):.3f})"
    )

    pipe.fit(X, y)
    with open(args.model_out, "wb") as f:
        pickle.dump({"model": pipe, "n_features": X.shape[1]}, f)

    print(f"saved combined model -> {args.model_out}")

    # This CSV is only a training sanity output; final CSVs come from predict.py.
    p = pipe.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pause_index), probability in zip(keys_all, p):
            w.writerow([tid, pause_index, f"{probability:.4f}"])

    print(f"wrote {len(keys_all)} training predictions -> {args.out}")


if __name__ == "__main__":
    main()