import os
import pandas as pd

# ======================
# PATH
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# ======================
# LOAD CSV
# ======================
def load_csv(file_path, label):

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    print(f"\nLoaded {len(df)} rows from {file_path}")
    print("Columns:", df.columns.tolist())

    # 👉 FIX: dùng title
    if "title" not in df.columns:
        print(f"❌ No 'title' column in {file_path}")
        return pd.DataFrame()

    df = df[["title"]].copy()
    df.rename(columns={"title": "text"}, inplace=True)
    df["label"] = label

    return df


# ======================
# MAIN
# ======================
def main():

    print("RAW DIR:", RAW_DIR)
    print("Files:", os.listdir(RAW_DIR))

    dfs = []

    dfs.append(load_csv(os.path.join(RAW_DIR, "politifact_real.csv"), 0))
    dfs.append(load_csv(os.path.join(RAW_DIR, "politifact_fake.csv"), 1))
    dfs.append(load_csv(os.path.join(RAW_DIR, "gossipcop_real.csv"), 0))
    dfs.append(load_csv(os.path.join(RAW_DIR, "gossipcop_fake.csv"), 1))

    dfs = [df for df in dfs if not df.empty]

    if len(dfs) == 0:
        print("❌ No data loaded")
        return

    df = pd.concat(dfs, ignore_index=True)

    print("\nTotal BEFORE clean:", len(df))

    # CLEAN
    df = df.dropna()
    df = df.drop_duplicates(subset=["text"])

    # ⚠️ vì title ngắn → giảm threshold
    df = df[df["text"].str.split().str.len() > 5]

    print("Total AFTER clean:", len(df))

    # shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # SAVE
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n✅ Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()