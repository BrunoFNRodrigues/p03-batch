from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators=100, random_state=195)
model.fit(X_train, y_train)