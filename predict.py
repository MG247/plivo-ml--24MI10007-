"""Final predict.py – loads saved model, never trains.

Usage:
    python predict.py --data_dir ../eot_data/english --out predictions_en.csv
    python predict.py --data_dir ../eot_data/hindi   --out predictions_hi.csv
"""
import argparse
import csv
import os
import pickle

import numpy as np

from features import load_wav, speech_before, frame_energy_db, f0_contour


def extract_features(x, sr, pause_start):
    """MUST be identical to the one used in train.py."""
    seg = speech_before(x, sr, pause_start, window_s=1.8)
    n_feat = 16
    if len(seg) < sr // 20:
        return np.zeros(n_feat, dtype=np.float32)

    e = frame_energy_db(seg, sr)
    if len(e) < 2:
        return np.zeros(n_feat, dtype=np.float32)

    energy_final = float(np.mean(e[-4:]))
    energy_mean  = float(np.mean(e))
    energy_std   = float(np.std(e))
    energy_min   = float(np.min(e))
    energy_max   = float(np.max(e))

    t = np.arange(len(e), dtype=np.float64)
    energy_slope = float(np.polyfit(t, e, 1)[0]) if len(e) >= 4 else 0.0

    n_last = max(2, int(0.25 / 0.01))
    if len(e) > n_last + 2:
        energy_drop = float(np.mean(e[:-n_last]) - np.mean(e[-n_last:]))
    else:
        energy_drop = 0.0

    energy_rel_final = energy_final - energy_mean

    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]

    if len(voiced) >= 3:
        f0_final = float(np.mean(voiced[-3:]))
        f0_mean  = float(np.mean(voiced))
        f0_std   = float(np.std(voiced))
        last_v   = voiced[-min(8, len(voiced)):]
        f0_slope = float(np.polyfit(np.arange(len(last_v)), last_v, 1)[0])
        voicing_ratio = float(len(voiced) / max(1, len(f0)))
    else:
        f0_final = f0_mean = f0_std = f0_slope = voicing_ratio = 0.0

    run = 0
    for v in f0[::-1]:
        if v > 0:
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
        voicing_ratio, last_voiced_len,
        speaking_rate, turn_progress,
    ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="model.pkl")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    pipe = bundle["model"]

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    out_rows = []

    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        feat = extract_features(x, sr, float(r["pause_start"])).reshape(1, -1)
        p_eot = float(pipe.predict_proba(feat)[0, 1])
        out_rows.append({
            "turn_id": r["turn_id"],
            "pause_index": r["pause_index"],
            "p_eot": f"{p_eot:.4f}",
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()