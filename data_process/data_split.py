from pathlib import Path
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

project_path = Path(__file__).resolve().parents[1]
data_path = project_path / "data" / "Penn_Action"
split_path = project_path / "splits"

records = []

for label_path in sorted((data_path / "labels").glob("*.mat")):
    label = loadmat(label_path)

    video_id = label_path.stem
    action = str(label["action"].item()).strip()
    train_value = int(label["train"].item())

    if train_value == 1:
        official_split = "train"
    elif train_value == -1:
        official_split = "test"
    else:
        raise ValueError(f"Unexpected split value in {label_path}")

    records.append({
        "video_id": video_id,
        "action": action,
        "official_split": official_split,
    })

df = pd.DataFrame(records)

print(df.head())
print(df["official_split"].value_counts())
print(df.groupby(["action", "official_split"]).size().unstack(fill_value=0))

official_train = df[df["official_split"] == "train"].copy()
test_df = df[df["official_split"] == "test"].copy()

train_df, val_df = train_test_split(
    official_train,
    test_size=0.15,
    random_state=233,
    shuffle=True,
    stratify=official_train["action"],
)

train_df = train_df.sort_values("video_id").reset_index(drop=True)
val_df = val_df.sort_values("video_id").reset_index(drop=True)
test_df = test_df.sort_values("video_id").reset_index(drop=True)

all_classes = set(df["action"])

assert set(train_df["action"]) == all_classes
assert set(val_df["action"]) == all_classes

missing_test_classes = all_classes - set(test_df["action"])
if missing_test_classes:
    print("Classes missing from official test set:", sorted(missing_test_classes))

split_path.mkdir(parents=True, exist_ok=True)

train_df["video_id"].to_csv(
    split_path / "train.txt",
    index=False,
    header=False
)

val_df["video_id"].to_csv(
    split_path / "val.txt",
    index=False,
    header=False
)

test_df["video_id"].to_csv(
    split_path / "test.txt",
    index=False,
    header=False
)