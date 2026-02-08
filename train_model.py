import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Importing the 6 Algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# 1. Load Data
df = pd.read_csv('DATASET CAR PRICE.csv')
df.columns = df.columns.str.strip()

# 2. Features and Target
X = df.drop(['Price', 'Car ID'], axis=1)
y = df['Price']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Hybrid Pipeline Construction
# Note: StandardScaler is added because Linear models (Ridge/Lasso) perform better with scaled data
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# Defining the 6 Algorithms
m1 = LinearRegression()
m2 = Ridge(alpha=1.0)
m3 = Lasso(alpha=1.0)
m4 = DecisionTreeRegressor(random_state=42)
m5 = RandomForestRegressor(n_estimators=100, random_state=42)
m6 = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Create the Hybrid Voting Regressor (Combines all 6)
hybrid_model = VotingRegressor(estimators=[
    ('lr', m1), ('ridge', m2), ('lasso', m3),
    ('dt', m4), ('rf', m5), ('gb', m6)
])

# Create the Final Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', hybrid_model)
])

# 4. Train the Hybrid Model
print("Training Mega-Hybrid Ensemble (6 Algorithms)...")
model_pipeline.fit(X_train, y_train)

# 5. Evaluate
preds = model_pipeline.predict(X_test)
acc = r2_score(y_test, preds)
mse = mean_squared_error(y_test, preds)

# 6. Save Files
model_data = {
    "model": model_pipeline,
    "accuracy": round(acc * 100, 2),
    "mse": round(mse, 2),
    "algo": "Mega-Hybrid (6-Model Ensemble)",
    "feature_names": X.columns.tolist()
}

with open('car_price_models.pkl', 'wb') as f:
    pickle.dump(model_data, f)

ui_info = {
    'brands': sorted(df['Brand'].unique().tolist()),
    'fuel_types': sorted(df['Fuel Type'].unique().tolist()),
    'transmissions': sorted(df['Transmission'].unique().tolist())
}
with open('ui_data.pkl', 'wb') as f:
    pickle.dump(ui_info, f)

print(f"✅ SUCCESS! Hybrid Accuracy: {model_data['accuracy']}%")
