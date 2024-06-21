# Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
import pandas as pd

def train_test_split_data(x, y):
    """
    Split data into training and testing sets.

    Parameters:
    - x (pandas.DataFrame): Features.
    - y (pandas.Series or numpy.array): Target variable.

    Returns:
    - x_train, x_test, y_train, y_test (tuple): Split datasets for training and testing.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2, test_size=0.2)
    return x_train, x_test, y_train, y_test

def train_model(method, x_train, y_train, x_test, y_test):
    """
    Train the model, make predictions, and evaluate its performance.

    Parameters:
    - method (object): Machine learning or deep learning model.
    - x_train, x_test, y_train, y_test (pandas.DataFrame or numpy.array): Training and testing data.

    Returns:
    - None
    """
    # Train the model
    print("Training Model......")
    method.fit(x_train, y_train)
    print("Model Trained")

    # Make predictions on test data
    predictions = method.predict(x_test)

    # Evaluate model performance and print results
    print("Model accuracy: ", '{:.2%}'.format(accuracy_score(y_test, predictions)))
    print(classification_report(y_test, predictions))

def train_random_forest(x_train, y_train, x_test, y_test):
    """
    Train a Random Forest model.

    Parameters:
    - x_train, x_test, y_train, y_test (pandas.DataFrame or numpy.array): Training and testing data.

    Returns:
    - rf (object): Trained Random Forest model.
    """
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1)
    train_model(rf, x_train, y_train, x_test, y_test)
    return rf

def train_neural_network(x_train, y_train, x_test, y_test):
    """
    Train a Neural Network model.

    Parameters:
    - x_train, x_test, y_train, y_test (pandas.DataFrame or numpy.array): Training and testing data.

    Returns:
    - model (object): Trained Neural Network model.
    """
    n_inputs, n_outputs = x_train.shape[1], y_train.shape[1]
    model = Sequential()
    model.add(Dense(50, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(25, activation='tanh'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=30)
    acc = model.evaluate(x_test, y_test, verbose=0)[1] * 100.0
    print('Categorical Accuracy: >%.3f' % acc)
    return model

def save_model(model, filepath):
    """
    Save the trained model to disk.

    Parameters:
    - model (object): Trained model object.
    - filepath (str): File path to save the model.

    Returns:
    - None
    """
    pickle.dump(model, open(filepath, 'wb'))
    print("Model saved successfully.")

def preprocess_target_data(y, df_reason):
    """
    Preprocess the target variable data.

    Parameters:
    - y (pandas.Series or numpy.array): Target variable.
    - df_reason (pandas.Series or numpy.array): Series containing churn reasons.

    Returns:
    - y_mlb (numpy.array): Preprocessed target variable in MultiLabelBinarizer format.
    """
    y_df = pd.DataFrame(y)
    y_df['Churn Reason'] = df_reason
    mlb = MultiLabelBinarizer()
    y_mlb = mlb.fit_transform(y_df.values)
    return y_mlb
