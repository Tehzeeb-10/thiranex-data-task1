import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PREDICTIVE MODELING USING MACHINE LEARNING
# ============================================================

print("=" * 60)
print("PREDICTIVE MODELING USING MACHINE LEARNING")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "D:/thiranex1/cleaned_dataset.xlsx"

df = pd.read_excel(file_path)

print("\nOriginal dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 2. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("\nDataset after removing duplicates:")
print(df.shape)


# ============================================================
# 3. CHECK DATA
# ============================================================

print("\n" + "=" * 60)
print("DATA CHECK")
print("=" * 60)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. CLEAN NUMERICAL COLUMNS
# ============================================================

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

df["Purchase Amount"] = (
    df["Purchase Amount"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+(?:\.\d+)?)")[0]
)

df["Purchase Amount"] = pd.to_numeric(
    df["Purchase Amount"],
    errors="coerce"
)


# ============================================================
# 5. REMOVE ROWS WITHOUT TARGET
# ============================================================

df = df.dropna(subset=["Purchase Amount"])

print("\nDataset after cleaning:")
print(df.shape)


# ============================================================
# 6. CREATE AGE GROUP
# ============================================================

def age_group(age):
    if pd.isna(age):
        return "Unknown"
    elif age < 25:
        return "Young"
    elif age < 40:
        return "Adult"
    elif age < 60:
        return "Middle_Age"
    else:
        return "Senior"


df["Age_Group"] = df["Age"].apply(age_group)


# ============================================================
# 7. CREATE CATEGORY + SEASON INTERACTION
# ============================================================

df["Category_Season"] = (
    df["Category"].astype(str)
    + "_"
    + df["Season"].astype(str)
)


# ============================================================
# 8. CREATE SUBSCRIPTION FLAG
# ============================================================

df["Subscription_Flag"] = (
    df["Subscription Status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)


# ============================================================
# 9. FILL MISSING VALUES
# ============================================================

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Subscription_Flag"] = df["Subscription_Flag"].fillna(0)


# ============================================================
# 10. TARGET CORRELATION
# ============================================================

print("\n" + "=" * 60)
print("TARGET CORRELATION")
print("=" * 60)

print(
    df[
        ["Age", "Subscription_Flag", "Purchase Amount"]
    ].corr()["Purchase Amount"].sort_values(
        ascending=False
    )
)


# ============================================================
# 11. AVERAGE PURCHASE AMOUNT
# ============================================================

print("\n" + "=" * 60)
print("AVERAGE PURCHASE AMOUNT BY CATEGORY")
print("=" * 60)

for column in [
    "Gender",
    "Items Purchased",
    "Category",
    "Shipping Type",
    "Profession",
    "Subscription Status",
    "Season"
]:

    print("\n" + column)

    print(
        df.groupby(column)["Purchase Amount"]
        .mean()
        .sort_values(ascending=False)
    )


# ============================================================
# 12. REMOVE CUSTOMER ID
# ============================================================

if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

print("\nCustomerID removed.")


# ============================================================
# 13. DEFINE TARGET AND FEATURES
# ============================================================

target = "Purchase Amount"

X = df.drop(columns=[target])

y = df[target]


# ============================================================
# 14. DEFINE NUMERICAL FEATURES
# ============================================================

numerical_features = [
    "Age",
    "Subscription_Flag"
]


# ============================================================
# 15. DEFINE CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "Gender",
    "Items Purchased",
    "Category",
    "Shipping Type",
    "Profession",
    "Subscription Status",
    "Season",
    "Country",
    "Age_Group",
    "Category_Season"
]


print("\n" + "=" * 60)
print("DATA PREPARATION")
print("=" * 60)

print("\nDataset after cleaning:")
print(df.shape)

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nTarget variable:")
print(target)

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 16. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Split temporary 50/50
# Result = 15% validation and 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print("\nTraining records:")
print(len(X_train))

print("\nValidation records:")
print(len(X_val))

print("\nTesting records:")
print(len(X_test))


# ============================================================
# 17. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 18. FUNCTION FOR MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_data, y_data):

    predictions = model.predict(X_data)

    mae = mean_absolute_error(
        y_data,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_data,
            predictions
        )
    )

    r2 = r2_score(
        y_data,
        predictions
    )

    return mae, rmse, r2


# ============================================================
# 19. MODEL 1 - LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("MODEL 1 - LINEAR REGRESSION")
print("=" * 60)


linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)


linear_model.fit(
    X_train,
    y_train
)


linear_mae, linear_rmse, linear_r2 = evaluate_model(
    linear_model,
    X_val,
    y_val
)


print(f"Validation MAE : {linear_mae:.2f}")
print(f"Validation RMSE: {linear_rmse:.2f}")
print(f"Validation R2  : {linear_r2:.4f}")


# ============================================================
# 20. MODEL 2 - DECISION TREE
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2 - DECISION TREE REGRESSION")
print("=" * 60)


tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            DecisionTreeRegressor(
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            )
        )
    ]
)


tree_model.fit(
    X_train,
    y_train
)


tree_mae, tree_rmse, tree_r2 = evaluate_model(
    tree_model,
    X_val,
    y_val
)


print(f"Validation MAE : {tree_mae:.2f}")
print(f"Validation RMSE: {tree_rmse:.2f}")
print(f"Validation R2  : {tree_r2:.4f}")


# ============================================================
# 21. MODEL 3 - RANDOM FOREST TUNING
# ============================================================

print("\n" + "=" * 60)
print("MODEL 3 - RANDOM FOREST TUNING")
print("=" * 60)


rf_settings = [
    {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 4
    },
    {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_split": 10,
        "min_samples_leaf": 4
    },
    {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 4
    },
    {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 4
    }
]


rf_results = []

best_rf_model = None
best_rf_r2 = -999


for settings in rf_settings:

    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=settings["n_estimators"],
                    max_depth=settings["max_depth"],
                    min_samples_split=settings["min_samples_split"],
                    min_samples_leaf=settings["min_samples_leaf"],
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )


    rf_model.fit(
        X_train,
        y_train
    )


    rf_mae, rf_rmse, rf_r2 = evaluate_model(
        rf_model,
        X_val,
        y_val
    )


    rf_results.append(
        {
            "Model": "Random Forest",
            "Settings": str(settings),
            "MAE": rf_mae,
            "RMSE": rf_rmse,
            "R2": rf_r2
        }
    )


    print("\nSettings:")
    print(settings)

    print(f"MAE: {rf_mae:.2f}")
    print(f"RMSE: {rf_rmse:.2f}")
    print(f"R2: {rf_r2:.4f}")


    if rf_r2 > best_rf_r2:

        best_rf_r2 = rf_r2
        best_rf_model = rf_model


# ============================================================
# 22. MODEL 4 - GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 60)
print("MODEL 4 - GRADIENT BOOSTING REGRESSION")
print("=" * 60)


gradient_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=2,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            )
        )
    ]
)


gradient_model.fit(
    X_train,
    y_train
)


gradient_mae, gradient_rmse, gradient_r2 = evaluate_model(
    gradient_model,
    X_val,
    y_val
)


print(f"Validation MAE : {gradient_mae:.2f}")
print(f"Validation RMSE: {gradient_rmse:.2f}")
print(f"Validation R2  : {gradient_r2:.4f}")


# ============================================================
# 23. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    [
        {
            "Model": "Linear Regression",
            "MAE": linear_mae,
            "RMSE": linear_rmse,
            "R2": linear_r2
        },
        {
            "Model": "Decision Tree",
            "MAE": tree_mae,
            "RMSE": tree_rmse,
            "R2": tree_r2
        },
        {
            "Model": "Random Forest",
            "MAE": rf_results[
                np.argmax(
                    [x["R2"] for x in rf_results]
                )
            ]["MAE"],
            "RMSE": rf_results[
                np.argmax(
                    [x["R2"] for x in rf_results]
                )
            ]["RMSE"],
            "R2": best_rf_r2
        },
        {
            "Model": "Gradient Boosting",
            "MAE": gradient_mae,
            "RMSE": gradient_rmse,
            "R2": gradient_r2
        }
    ]
)


print("\n" + "=" * 60)
print("MODEL COMPARISON - VALIDATION")
print("=" * 60)

print(comparison.to_string(index=False))


# ============================================================
# 24. SELECT BEST MODEL
# ============================================================

best_index = comparison["R2"].idxmax()

best_model_name = comparison.loc[
    best_index,
    "Model"
]


print("\n" + "=" * 60)
print("SELECTED MODEL")
print("=" * 60)

print(
    "Model selected using validation R2:",
    best_model_name
)


# ============================================================
# 25. SELECT MODEL OBJECT
# ============================================================

if best_model_name == "Linear Regression":

    selected_model = linear_model

elif best_model_name == "Decision Tree":

    selected_model = tree_model

elif best_model_name == "Random Forest":

    selected_model = best_rf_model

else:

    selected_model = gradient_model


# ============================================================
# 26. FINAL TEST RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)


test_predictions = selected_model.predict(
    X_test
)


test_mae = mean_absolute_error(
    y_test,
    test_predictions
)


test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)


test_r2 = r2_score(
    y_test,
    test_predictions
)


print("\nSelected Model:")
print(best_model_name)

print("\nTest MAE:")
print(f"{test_mae:.2f}")

print("\nTest RMSE:")
print(f"{test_rmse:.2f}")

print("\nTest R2:")
print(f"{test_r2:.4f}")


# ============================================================
# 27. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)


sample_predictions = pd.DataFrame(
    {
        "Actual": y_test.values[:10],
        "Predicted": np.round(
            test_predictions[:10],
            2
        )
    }
)


print(
    sample_predictions.to_string(
        index=False
    )
)


# ============================================================
# 28. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.6
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Purchase Amount")
plt.ylabel("Predicted Purchase Amount")

plt.title(
    "Actual vs Predicted Purchase Amount"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 29. MODEL COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    comparison["Model"],
    comparison["R2"]
)

plt.xlabel("Model")
plt.ylabel("Validation R2")

plt.title(
    "Model Comparison Using R2"
)

plt.xticks(
    rotation=20
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()


# ============================================================
# 30. SAVE MODEL COMPARISON
# ============================================================

results_file = "D:/thiranex1/predictive_modeling_results.xlsx"

comparison.to_excel(
    results_file,
    index=False
)


# ============================================================
# 31. SAVE TEST PREDICTIONS
# ============================================================

prediction_file = "D:/thiranex1/test_predictions.xlsx"

prediction_output = pd.DataFrame(
    {
        "Actual Purchase Amount": y_test.values,
        "Predicted Purchase Amount": test_predictions
    }
)


prediction_output.to_excel(
    prediction_file,
    index=False
)


# ============================================================
# 32. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREDICTIVE MODELING COMPLETED")
print("=" * 60)

print("\nCustomerID was removed.")

print(
    "Items Purchased was treated as categorical "
    "because it contains product names."
)

print("Missing numerical values were handled.")

print("Age Group was treated as categorical.")

print("Category + Season interaction was created.")

print("Subscription status was converted to a numerical flag.")

print(
    "70% training, 15% validation and 15% testing were used."
)

print("Random Forest parameters were tuned.")

print("Gradient Boosting was included.")

print("MAE, RMSE and R2 were calculated.")

print("Actual vs Predicted graph was displayed.")

print("Model comparison graph was displayed.")

print("\nResults saved as:")
print(results_file)
print(prediction_file)

print("\nFinal selected model:")
print(best_model_name)

print("\nFinal Test R2:")
print(f"{test_r2:.4f}")

print("\n" + "=" * 60)
