# Import necessary modules and functions
from ML_Pipeline.processing import preprocess_data, preprocess_resample_data
from ML_Pipeline.modeling import *

# Read and preprocess the raw data
processed_df = preprocess_data("../data/raw/final_telco.csv")


# Resample and preprocess the data
x, y, df_reason = preprocess_resample_data(processed_df)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split_data(x, y)

# Train a Random Forest model
rf_model = train_random_forest(x_train, y_train, x_test, y_test)

# Preprocess and train a Neural Network model
y_mlb = preprocess_target_data(y, df_reason)
x_mlb = x.values
x_train_mlb, x_test_mlb, y_train_mlb, y_test_mlb = train_test_split_data(x_mlb, y_mlb)
nn_model = train_neural_network(x_train_mlb, y_train_mlb, x_test_mlb, y_test_mlb)


# Save the trained models
save_model(rf_model, "../models/random_forest.model")
nn_model.save("../models/dnn_mlb.model")
print("Models saved")
