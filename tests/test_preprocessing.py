import pandas as pd
from backend.src.ml.preprocessing import preprocess

def test_preprocessing():
    df = pd.DataFrame({"temperature":[10,20]})
    result = preprocess(df)
    assert result is not None