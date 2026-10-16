from pathlib import Path

import pandas as pd
from scipy.io import loadmat


project_path = Path(__file__).resolve().parents[1]
label_path = project_path / "data" / "Penn_Action" / "labels"
split_path = project_path / "splits"

actions = set()

for path in sorted(label_path.glob("*.mat")):
    label = loadmat(path)
    actions.add(str(label["action"].item()).strip())

if not actions:
    raise ValueError("No action labels found")

class_names = sorted(actions)

classes_df = pd.DataFrame({
    "label_id": range(len(class_names)),
    "action": class_names,
})

split_path.mkdir(parents=True, exist_ok=True)
classes_df.to_csv(split_path / "classes.csv", index=False)

print(classes_df)