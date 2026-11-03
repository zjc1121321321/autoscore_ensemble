"""Create independent toy data for checking the pipeline, not clinical results."""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    output = Path(__file__).resolve().parents[1] / "data" / "demo"
    targets = [output / "train_data.csv", output / "test_data.csv"]
    if any(p.exists() for p in targets):
        raise SystemExit("Demo data already exists. Reuse it or move it before generating again.")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1200, 11))
    probability = 1 / (1 + np.exp(-(-0.8 + x[:, 0] - 0.7 * x[:, 1] + 0.5 * x[:, 2])))
    df = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(1, 12)])
    df.insert(0, "label", rng.binomial(1, probability))
    output.mkdir(parents=True, exist_ok=True)
    df.iloc[:960].to_csv(targets[0], index=False)
    df.iloc[960:].to_csv(targets[1], index=False)
    print("Created independent demo data in data/demo (960 train / 240 test rows).")


if __name__ == "__main__":
    main()
