import os
import pandas as pd

def summarize_farm_harvest_dates(output_root, summary_filename="farm_harvest_summary.csv"):
    print("📊 Summarizing farm harvest dates...")
    summary_data = {}

    for crop_name in os.listdir(output_root):
        crop_dir = os.path.join(output_root, crop_name)
        farms_dir = os.path.join(crop_dir, "Farms")
        if not os.path.isdir(farms_dir):
            continue

        for farm_id in os.listdir(farms_dir):
            farm_folder = os.path.join(farms_dir, farm_id)
            if not os.path.isdir(farm_folder):
                continue
            
            for file in os.listdir(farm_folder):
                if file.endswith("_summary.csv"):
                    df = pd.read_csv(os.path.join(farm_folder, file))
                    pick = df[df["Harvest"].isin(["End", "Start"])]
                    harvest_date = pick.iloc[0]["Date"] if not pick.empty else ""
                    summary_data[farm_id] = harvest_date
            out = pd.DataFrame(
                [{"FarmID": fid, "HarvestDate": summary_data[fid]} for fid in sorted(summary_data)]
            )
            out.to_csv(os.path.join(output_root, summary_filename), index=False)

    # Build output rows
    farm_ids = list(summary_data.keys())
    harvest_dates = [summary_data[fid] for fid in farm_ids]

    # Create DataFrame with one column per farm
    df_out = pd.DataFrame([farm_ids, harvest_dates])
    summary_path = os.path.join(output_root, summary_filename)
    df_out.to_csv(summary_path, index=False, header=False)
    
    print(f"✅ Harvest summary saved to: {summary_path}")