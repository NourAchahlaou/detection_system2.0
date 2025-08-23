import pandas as pd
import matplotlib.pyplot as plt

base_dir = "c:/Users/hp/Desktop/airbus2.0/detection_system2.0/shared_data/models"  # Change this to your dataset path
df = pd.read_csv(r"c:/Users/hp/Desktop/airbus2.0/detection_system2.0/shared_data/models/combined_results.csv")

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Now plot
plt.figure(figsize=(12, 6))
plt.plot(df["epoch_number"], df["metrics/precision(B)"], label="Precision", marker='o')
plt.plot(df["epoch_number"], df["metrics/recall(B)"], label="Recall", marker='s')
plt.plot(df["epoch_number"], df["metrics/mAP50(B)"], label="mAP50", marker='^')
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Training Metrics over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
