import numpy as np

try:
    label_map = np.load("label_map.npy", allow_pickle=True).item()
    print("Label Map Contents:")
    for label, name in label_map.items():
        print(f"Label: {label} -> Name: {name}")
except Exception as e:
    print(f"Error loading label map: {e}")
