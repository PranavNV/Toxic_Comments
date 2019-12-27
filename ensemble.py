import numpy as np, pandas as pd
lstm = pd.read_csv('submission.csv')
lr = pd.read_csv('submission2.csv')
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
final = lstm.copy()
final[label_cols] = (lstm[label_cols]*0.2 + lr[label_cols]*0.8) 
final.to_csv('submission_final.csv', index=False)