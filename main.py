import pandas as pd 
import numpy as np 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

df = pd.read_csv('../data/train.csv')


df = df.drop(['Name','Ticket','Cabin'], axis=1)
df.head(2)
from scipy.stats import mode
df['Embarked'] = df['Embarked'].fillna(mode(df['Embarked'])[0][0])
df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(int)
df = pd.concat([df,pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
df = df.drop(['Sex','Embarked'], axis=1)
age_mean = df['Age'].mean()
age_median = df['Age'].median()
train_data = df.values


imputer = Imputer(missing_values='NaN')

classifier = RandomForestClassifier(n_estimators=100)

pipeline = Pipeline([
    ('imp', imputer),
    ('clf', classifier),
])

parameter_grid = {
    'imp__strategy': ['mean', 'median'],
    'clf__max_features': [0.5, 1],
    'clf__max_depth': [5, None],
}

grid_search = GridSearchCV(pipeline, parameter_grid, cv=5, verbose=3)
grid_search.fit(train_data[0:,2:], train_data[0:,1])

print sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score)
print grid_search.best_score_
print grid_search.best_params_

df['Age'] = df['Age'].fillna(age_mean)
train_data = df.values

model = RandomForestClassifier(n_estimators = 100, max_features=0.5, max_depth=5)
model = model.fit(train_data[0:,2:],train_data[0:,1])

df_test = pd.read_csv('../data/test.csv')

df_test = df_test.drop(['Name','Ticket','Cabin'], axis=1)
df_test['Age'] = df_test['Age'].fillna(age_mean)

fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
df_test['Fare'] = df_test[['Fare','Pclass']].apply(lambda x: fare_means[x['Pclass']] if pd.isnull(x['Fare']) 
                                                 else x['Fare'], axis=1)

df_test['Gender'] = df_test['Sex'].map({'female':0, 'male':1}).astype(int)

df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')], axis=1)
df_test = df_test.drop(['Sex','Embarked'], axis=1)

test_data = df_test.values
output = model.predict(test_data[:,1:])
results = np.c_[test_data[:,0].astype(int), output.astype(int)]
df_result = pd.DataFrame(results, columns = ['PassengerId', 'Survived'])

print df_result