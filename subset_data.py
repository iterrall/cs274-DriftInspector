from pathlib import Path
import argparse
import numpy as np


def make_adult_subset(
    src_dir="data",
    out_dir="data_subset_tiny",
    train_rows=500,
    test_rows=500,
    seed=42,
    overwrite=False,
):
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)

    train_src = src_dir / "adult.data"
    test_src = src_dir / "adult.test"

    if not train_src.exists():
        raise FileNotFoundError(f"Missing file: {train_src}")
    if not test_src.exists():
        raise FileNotFoundError(f"Missing file: {test_src}")

    if out_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {out_dir}\n"
            f"Use --overwrite to replace it."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # adult.data
    train_lines = train_src.read_text(encoding="utf-8", errors="replace").splitlines()
    train_lines = [x for x in train_lines if x.strip()]
    train_rows = min(train_rows, len(train_lines))
    train_idx = np.sort(rng.choice(len(train_lines), size=train_rows, replace=False))
    sampled_train = [train_lines[i] for i in train_idx]

    # adult.test
    test_lines = test_src.read_text(encoding="utf-8", errors="replace").splitlines()
    test_lines = [x for x in test_lines if x.strip()]
    header = test_lines[0] if test_lines and test_lines[0].startswith("|") else None
    body = test_lines[1:] if header else test_lines
    test_rows = min(test_rows, len(body))
    test_idx = np.sort(rng.choice(len(body), size=test_rows, replace=False))
    sampled_test = [body[i] for i in test_idx]

    train_out = out_dir / "adult.data"
    test_out = out_dir / "adult.test"

    train_out.write_text("\n".join(sampled_train) + "\n", encoding="utf-8")
    if header:
        test_out.write_text(
            header + "\n" + "\n".join(sampled_test) + "\n",
            encoding="utf-8",
        )
    else:
        test_out.write_text("\n".join(sampled_test) + "\n", encoding="utf-8")

    print(f"Created subset in {out_dir}")
    print(f"adult.data rows: {len(sampled_train)}")
    print(f"adult.test rows: {len(sampled_test)}")

    return train_out, test_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a sampled Adult dataset subset.")
    parser.add_argument("--src-dir", default="data")
    parser.add_argument("--out-dir", default="data_subset_tiny")
    parser.add_argument("--train-rows", type=int, default=500)
    parser.add_argument("--test-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    make_adult_subset(
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        train_rows=args.train_rows,
        test_rows=args.test_rows,
        seed=args.seed,
        overwrite=args.overwrite,
    )


    """
    python subset_data.py --out-dir data_subset_tiny --train-rows 500 --test-rows 500 --overwrite
    python subset_data.py --out-dir data_subset_small --train-rows 2000 --test-rows 2000 --overwrite
    python subset_data.py --out-dir data_subset_medium --train-rows 5000 --test-rows 5000 --overwrite
    """