import polars as pl
from src.preprocessing import load_and_merge, handle_missing, prepare_features
from src.pipeline import build_pipeline

# --- Downloading ---
print("Downloading data...")
train = load_and_merge(
    'data/train_transactions.csv',
    'data/train_users.csv'
)
test = load_and_merge(
    'data/test_transactions.csv',
    'data/test_users.csv'
)

# --- Clearing data ---
print("Handling missing data...")
train = handle_missing(train)
test = handle_missing(test)


#--- Features ---
#Misha will add his features here,  before method prepare_features()
X_train, y_train = prepare_features(train, target_col="is_fraud")
X_test, _ = prepare_features(test)

# --- Model ---
# Аліна замінить модель тут:
# from lightgbm import LGBMClassifier
# pipeline = build_pipeline(model=LGBMClassifier(...))
pipline = build_pipeline()

print("Training model...")
pipline.fit(X_train, y_train)

#---Results---
print("Generating predictions...")
predictions = pipline.predict(X_test)

submission = pl.DataFrame({
    'id_user': test['id_user'],
    'is_fraud': predictions
})
submission.write_csv('submission.csv')
print("Ready! submission.csv is saved.")