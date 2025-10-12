# %%

import pandas as pd

# import the model

model_df = pd.read_pickle('churn_model.pkl')
model_df
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
