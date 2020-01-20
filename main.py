import numpy as np
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
gender_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

tcoi = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
numcols = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
doi = pd.DataFrame(train_data[tcoi])
survive = train_data.Survived
tdoi = pd.DataFrame(test_data[tcoi])

numeric_transformer = SimpleImputer(strategy='mean')
categoric_transformer = Pipeline(steps=[
    ('simpimp', SimpleImputer(strategy='constant', fill_value='Z')),
    ('labenc', LabelEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numcols),
    ('cat', categoric_transformer, ['Embarked', 'Sex'])
])

model = XGBRegressor(n_estimators=1000)
model.fit(doi, survive, early_stopping_rounds=5, eval_set=[(doi, survive)], verbose=False)
main_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

main_pipeline.fit(doi, survive)
preds = main_pipeline.predict(tdoi)

scores = -1 * cross_val_score(main_pipeline, doi, survive, cv=5, scoring='neg_mean_absolute_error')
