import pathlib
import sys
from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# Utility function to handle hyphenated field names
def convert_hyphen_to_underscore(field_name: str) -> str:
    """Replace underscores with hyphens in field names."""
    return field_name.replace("_", "-")


# Define the input schema using Pydantic
class SalaryInput(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="United-States")

    class Config:
        alias_generator = convert_hyphen_to_underscore
        populate_by_name = True

BASE_DIR = pathlib.Path(__file__).parent


# Initialize the FastAPI app
app = FastAPI(
    title="Salary Prediction API",
    description="An API to predict if a person's salary exceeds $50K based on their attributes.",
    version="1.0.0"
)

@app.on_event("startup")
def load_artifacts():
    global model, encoder, label_binarizer
    print("[STARTUP] loading model artifacts…", file=sys.stderr)
    model_path   = BASE_DIR / "model" / "model.pkl"
    encoder_path = BASE_DIR / "model" / "encoder.pkl"
    lb_path      = BASE_DIR / "model" / "lb.pkl"

    # these opens should succeed in CI if the files are in place
    with open(model_path, "rb")   as f: model = pickle.load(f)
    with open(encoder_path, "rb") as f: encoder = pickle.load(f)
    with open(lb_path, "rb")      as f: label_binarizer = pickle.load(f)
    print("[STARTUP] done loading model, encoder, lb", file=sys.stderr)



# Root endpoint
@app.get("/")
async def root():
    """Root endpoint to verify the API is running."""
    return {"message": "Welcome to the Salary Prediction API!"}


# Prediction endpoint
@app.post("/predict")
async def predict_salary(input_data: SalaryInput):
    """Predict if a person's salary exceeds $50K based on input data."""
    try:
        # Convert input data to a DataFrame
        input_dict = input_data.dict(by_alias=True)
        input_df = pd.DataFrame([input_dict])

        # Preprocess the input data
        X, _, _, _ = process_data(
            X=input_df,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=label_binarizer,
        )

        # Perform inference
        prediction = inference(model, X)
        salary_category = label_binarizer.inverse_transform(prediction)[0]

        return {"salary_prediction": salary_category.strip()}

    except Exception as error:
        print("Error:", error)
        raise HTTPException(status_code=500, detail=f"Prediction error: {error}")
