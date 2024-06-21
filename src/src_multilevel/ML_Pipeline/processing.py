# Import Libraries
import pandas as pd
import h3
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def encode_data(dataframe_series):
    """
    Encode categorical data using LabelEncoder.

    Parameters:
    - dataframe_series (pandas.Series): Series containing categorical data.

    Returns:
    - Encoded categorical data.
    """
    if dataframe_series.dtype == 'object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

def clean_churn_category(category, reason):
    """
    Clean churn category based on churn reason.

    Parameters:
    - category (str): Current churn category.
    - reason (str): Churn reason.

    Returns:
    - Updated churn category.
    """
    # Handling various churn reasons and updating the category accordingly
    if reason in ['Lack of affordable download/upload speed', 'Limited range of services',
                  'Network reliability'] or 'dissatisfaction' in reason:
        category = "Dissatisfaction"
    if "Price" in reason:
        category = "Price"
    if "Competitor" in reason:
        category = "Competitor"
    if "support" in reason or reason in ['Lack of self-service on Website']:
        category = "Support"
    if category in ["bcvjhdjcb", "Other", "Unknown", "Attitude"] or reason == 'Unknown':
        category = "Other"
    if reason in ['Attitude of service provider']:
        category = "Support"
    if reason in ['Extra data charges', 'Long distance charges']:
        category = "Price"
    return category

def clean_churn_reason(df):
    """
    Clean churn reasons in the dataframe.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing churn data.

    Returns:
    - DataFrame with cleaned churn reasons.
    """
    # Remove irrelevant churn reasons and update churn category
    df = df[df['Churn Reason'] != 'Moved']
    df = df[df['Churn Reason'] != 'Deceased']
    df['Churn Category'] = df[['Churn Category', 'Churn Reason']]\
        .apply(lambda x: clean_churn_category(x['Churn Category'], x['Churn Reason']), axis=1)
    df = df[df['Churn Reason'] != '43tgeh']
    df.drop(df[(df['Churn Category'] == 'Competitor') & (df['Churn Reason'] == 'Unknown')].index, inplace=True)
    df['Churn Reason'] = df[['Churn Reason', 'Churn Category']].apply(
        lambda x: 'Unknown' if x['Churn Category'] == 'Other' else x['Churn Reason'], axis=1)
    return df

def generate_hex_counts(df):
    """
    Generate hexagonal counts based on geographical coordinates.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing geographical coordinates.

    Returns:
    - DataFrame with hexagonal counts.
    """
    hex_level = 5
    df['hex_id'] = df.apply(lambda x: h3.geo_to_h3(x['latitude'], x['longitude'], hex_level), axis=1)
    hex_counts = df.groupby('hex_id')['Customer ID'].nunique().reset_index(name='total_clients')
    hex_counts['center'] = hex_counts['hex_id'].apply(lambda x: h3.h3_to_geo(x))
    return df, hex_counts

def calculate_tenure(df):
    """
    Calculate tenure in months.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing tenure information.

    Returns:
    - DataFrame with calculated tenure.
    """
    df['Tenure Months'] = df['Month'] - df['Month of Joining']
    return df

def preprocess_data(csv_file_path):
    """
    Preprocess raw data from a CSV file.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing raw data.

    Returns:
    - Preprocessed DataFrame.
    """
    df = pd.read_csv(csv_file_path)
    df = clean_churn_reason(df)
    df, hex_counts = generate_hex_counts(df)
    df = calculate_tenure(df)
    data = df.copy()
    data = data.drop(["Location ID", "Service ID", "area_codes", "Status ID"], axis=1)
    data = data.drop(['Customer ID', 'zip_code', 'state', 'county', 'latitude', 'longitude',
                      'night_pck_user', 'fb_user', 'Customer Status'], axis=1)
    data['Internet Type'].fillna("Other", inplace=True)
    data['total_rech_data'].fillna(data['total_rech_data'].mean(), inplace=True)
    data = data.apply(lambda x: encode_data(x))
    return data

def resample_data(x, y):
    """
    Resample the data using SMOTE and RandomUnderSampler.

    Parameters:
    - x (pandas.DataFrame): Features.
    - y (pandas.Series): Target variable.

    Returns:
    - Resampled features and target variable.
    """
    over = SMOTE(sampling_strategy=0.25)
    under = RandomUnderSampler(sampling_strategy=0.7)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    x_resampled, y_resampled = pipeline.fit_resample(x, y)
    return x_resampled, y_resampled

def preprocess_resample_data(data):
    """
    Preprocess and resample the data.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing raw data.

    Returns:
    - Preprocessed and resampled features, target variable, and churn reasons.
    """
    x = data.drop("Churn Value", axis=1)
    y = data['Churn Value']
    x_resampled, y_resampled = resample_data(x, y)
    df_reason = x_resampled['Churn Reason']
    y_resampled = x_resampled['Churn Category']
    x_resampled = x_resampled.drop(['Churn Category', 'Churn Reason'], axis=1)
    return x_resampled, y_resampled, df_reason
