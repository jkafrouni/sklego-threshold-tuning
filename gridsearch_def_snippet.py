from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

model = LogisticRegression()
pipe = make_pipeline(preprocessor, model)

param_grid = {'logisticregression__C': np.logspace(-3, 3, 20)}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='average_precision',
                    n_jobs=-1, verbose=1, return_train_score=True)
grid.fit(X_train, y_train)

print(grid.best_score_)
best_model = grid.best_estimator_