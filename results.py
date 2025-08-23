import os
import pandas as pd

# Path to the directory containing epoch folders
base_dir = "c:/Users/hp/Desktop/airbus2.0/detection_system2.0/shared_data/models"  # Change this to your dataset path
# List to store all dataframes
all_dfs = []

# Loop over all folders in the base directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path) and folder_name.startswith("G053_epoch_"):
        # Extract epoch number from folder name
        epoch_num = int(folder_name.split("_")[-1])
        csv_file = os.path.join(folder_path, "results.csv")
        if os.path.exists(csv_file):
            # Read the CSV
            df = pd.read_csv(csv_file)
            # Add or overwrite epoch column
            df.insert(0, "epoch_number", epoch_num)
            all_dfs.append(df)

# Combine all dataframes
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    # Optional: sort by epoch_number for clarity
    combined_df.sort_values(by="epoch_number", inplace=True)
    # Save combined CSV
    combined_df.to_csv(os.path.join(base_dir, "combined_results.csv"), index=False)
    print("Combined results saved to combined_results.csv")
else:
    print("No results.csv files found in the epoch folders.")