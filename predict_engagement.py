
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
df = pd.read_csv("social_media_data.csv")

# 2. Create the target variable
df["total_engagement"] = (
    df["likes"] + df["comments"] + df["shares"]
)

# 3. Select input features and target
X = df[["post_type", "posting_hour", "reach"]]
y = df["total_engagement"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 5. Convert post_type into numerical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "post_type",
            OneHotEncoder(handle_unknown="ignore"),
            ["post_type"]
        )
    ],
    remainder="passthrough"
)

# 6. Create the Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    )
])

# 7. Train the model
model.fit(X_train, y_train)

# 8. Evaluate the model on unseen test rows
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== MACHINE LEARNING RESULTS =====")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Mean Absolute Error (MAE):", round(mae, 2))
print("R-squared (R2):", round(r2, 3))

# 9. Compare actual and predicted engagement
results = pd.DataFrame({
    "Actual Engagement": y_test.values,
    "Predicted Engagement": predictions.round(1)
})

print("\nActual vs Predicted:")
print(results.to_string(index=False))

# 10. Predict engagement for a new example post
new_post = pd.DataFrame({
    "post_type": ["Reel"],
    "posting_hour": [18],
    "reach": [2000]
})

predicted_engagement = model.predict(new_post)[0]

print("\n===== NEW POST PREDICTION =====")
print("Post type: Reel")
print("Posting hour: 18:00")
print("Reach: 2000")
print(
    "Predicted total engagement:",
    round(max(0, predicted_engagement))
)

print("\nPrediction completed successfully!")
