'''In this notebook we train and evaluate Random Forest algorithm. 
Final model has a score 0.95753 on the leaderboard.'''
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
x_train = pd.read_csv('../../Data/Data/Clean/x_train_clean.csv')
x_test = pd.read_csv('../../Data/Data/Clean/x_test_clean.csv')
y_train = pd.read_csv('../../Data/Data/Clean/Y_train_clean.csv').values.ravel()
#******************************************************************************************
#%% Train random forest classifier with all defult params
forest_clasifier_base = RandomForestClassifier()
forest_clasifier_base.fit(x_train, y_train)

# forest_clasifier.predict([x_test.iloc[0, :]])

# %% Evaluate (cross val with three folds)
rf_defult = cross_val_score(forest_clasifier_base, x_train, y_train, cv=3, scoring="accuracy")
print(" Random forest classifier with defult params scores:")
print(rf_defult)

#%%*******************************************************************************************
# Does scaling the input improve results?
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
rf_scaled = cross_val_score(forest_clasifier_base, x_train_scaled, y_train, cv=3,
                             scoring="accuracy")
print(" Random forest classifier with defult params + scaled x_train scores:")
print(rf_scaled)
# As expected scaling does not improve scores for RF classifier.
# %%

#***********************************************************************
# Fine tune the hyperparams with grid search (runs for a long time)
forest_clasifier = RandomForestClassifier(max_features='sqrt', oob_score = True, random_state = 7)
param_grid = {'max_depth': [10, 40, 70],
 'n_estimators': [200, 400, 1000, 1200]}
model_grid_search = GridSearchCV(forest_clasifier, param_grid=param_grid,
                                 n_jobs=2, cv=2, verbose=10)
model_grid_search.fit(x_train, y_train)
joblib.dump(model_grid_search.best_estimator_, 'grid_search_random_forest.pkl', compress = 1)
#%%
print(f"The best set of parameters is: "
      f"{model_grid_search.best_params_}")
# %%
cv_results = pd.DataFrame(model_grid_search.cv_results_).sort_values(
    "mean_test_score", ascending=False)
cv_results.head()
# %% Some validation and learning curves

# Max depth validation curve
cv = ShuffleSplit(n_splits=3, test_size=0.2)
max_depth = [1, 5, 10, 15, 20, 25]
train_scores, test_scores = validation_curve(
    forest_clasifier, x_train, y_train, param_name="max_depth", param_range=max_depth,
    cv=cv, scoring="accuracy", n_jobs=2)
#%%
plt.plot(max_depth, train_scores.mean(axis=1), label="Training error")
plt.plot(max_depth, test_scores.mean(axis=1), label="Testing error")
plt.legend()

plt.xlabel("Maximum depth of decision tree")
plt.ylabel("Accuracy")
plt.title("Validation curve for decision tree")
plt.savefig('../../Figures/Validation_random_forest_max_depth.png')
# %% Check effect of n_estimators by using out of bag samples
forest_clasifier1 = RandomForestClassifier(max_depth=12,
                                           max_features='sqrt',
                                            oob_score = True, random_state = 7)
oob_error = []
for n_est in [i for i in range(10, 1000, 40)]:
    forest_clasifier1.set_params(n_estimators=n_est)
    forest_clasifier1.fit(x_train, y_train)
    oob_error.append(1 - forest_clasifier1.oob_score_)

plt.plot(range(10, 1000, 40), oob_error)
plt.title('oob_error vs n_estimators (max_depth =12, max_features = sqrt)')
plt.ylabel('oob error')
plt.xlabel('n estimators')
plt.savefig('../../Figures/Obb_error_vs_n_estimators_random_forest.png')
# %% Train the final model
forest_clasifier_final = RandomForestClassifier(max_depth=12,
                                           max_features='sqrt',
                                        n_estimators=400, random_state = 7)
forest_clasifier_final.fit(x_train, y_train)

predictions = forest_clasifier_final.predict(x_test)
# Create a file to submit
submission = pd.DataFrame()
submission['ImageId'] = [i for i in range(1, len(predictions)+1)]
submission['Label'] = predictions
submission.to_csv('../../Predictions/Random_forest.csv', index=False)