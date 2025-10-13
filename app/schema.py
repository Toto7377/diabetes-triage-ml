from pydantic import BaseModel, Field

FEATURES = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]

class PredictRequest(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

class PredictResponse(BaseModel):
    prediction: float = Field(..., description="Progression index; higher=worse.")

class HealthResponse(BaseModel):
    status: str
    model_version: str
