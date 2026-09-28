import pandas as pd
import boto3
from io import StringIO

s3 = boto3.client("s3")

BUCKET = "mlopsbuckethouseprice"
KEY = "Mlops_house_predication_raw_data.csv"


def fetch_data():

    response = s3.get_object(
        Bucket=BUCKET,
        Key=KEY
    )

    df = pd.read_csv(
        StringIO(
            response["Body"].read().decode("utf-8")
        )
    )

    return df


df = fetch_data()

print(f"Fetched shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# features/ Tragets
X=df[['sqft','bedrooms','bathrooms','age_years','garage','location_score']]
y=df['price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
mlflow.set_experiment("mlops-house-prediction")