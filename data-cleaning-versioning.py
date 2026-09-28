import pandas as pd
import boto3
import mlflow
from datetime import date


# ==============================
# 1. Load raw CSV
# ==============================

path = r"D:\AI-ML\DataSets\Mlops_house_predication_raw_data.csv"

df = pd.read_csv(path)

print("============== Before Cleaning ==============")
print(df.isnull().sum())
print(f"Shape Before: {df.shape}")
print()


# ==============================
# 2. Clean data
# ==============================

df_clean = df.dropna()

print("============== After Cleaning ===============")
print(df_clean.isnull().sum())
print(f"Shape After: {df_clean.shape}")


# ==============================
# 3. Save cleaned CSV locally
# ==============================

clean_path = r"D:\AI-ML\DataSets\Mlops_house_predication_clean_v2.csv"

df_clean.to_csv(clean_path, index=False)

print("\nClean CSV saved successfully.")


# ==============================
# 4. Upload processed data to S3
# ==============================

s3 = boto3.client("s3")

BUCKET = "mlopsbuckethouseprice"


def upload_processed_data(local_path):

    key = f"processed/{date.today()}/Mlops_house_predication_clean_v2.csv"

    s3.upload_file(
        local_path,
        BUCKET,
        key
    )

    print(f"\nUploaded to:")
    print(f"s3://{BUCKET}/{key}")

    return key


upload_processed_data(clean_path)


# ==============================
# 5. Features and Target
# ==============================

X = df_clean[[
    "sqft",
    "bedrooms",
    "bathrooms"
]]

y = df_clean["price"]


# ==============================
# 6. MLflow Experiment
# ==============================

mlflow.set_experiment("house-price-prediction")

print("\nExperiment configured successfully.")
print("Features:", X.columns.tolist())
print("Target:", y.name)