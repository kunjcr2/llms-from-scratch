####################################################################################################################################
# I dont think if running this is feasable !
# The core issue: 
    # Scikit-surprise requires numpy < 2.0.0, since written in C++ and is old.
    # numpy < 2.0.0 is not compatible with Windows
####################################################################################################################################

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Load ratings data
ratings = pd.read_csv("models/SVD/ratings_small.csv")
print(ratings.head())

# Surprise expects (user, item, rating)
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

# Initialize SVD model
svd = SVD(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

# Cross-validation (replaces deprecated evaluate)
cv_results = cross_validate(
    svd,
    data,
    measures=["RMSE", "MAE"],
    cv=5,
    verbose=True
)

# Train on full dataset (for actual predictions)
trainset = data.build_full_trainset()
svd.fit(trainset)

# Inspect ratings of a specific user
user_id = 1
print(ratings[ratings["userId"] == user_id].head())

# Predict rating: user 1 → movie 302
prediction = svd.predict(uid=1, iid=302)
print(prediction)