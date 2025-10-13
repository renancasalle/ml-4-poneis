# %%

import pandas as pd
import mlflow.sklearn
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.sklearn.load_model("models:/churn_model/6")

features = model.feature_names_in_
features

# %%

# load some data

df = pd.read_csv('../data/abt_churn.csv')
sample = df[df['dtRef'] == df['dtRef'].max()].sample(3)
sample

# %%

prediction = model_df['model'].predict_proba(sample[model_df['features']])[:,1]
prediction
# %%

sample['proba'] = prediction
sample
# %%
