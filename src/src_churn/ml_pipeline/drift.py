import warnings
warnings.filterwarnings('ignore')

import traceback

from deepchecks.tabular import Dataset
from deepchecks.tabular import Suite
from deepchecks.tabular.checks import WholeDatasetDrift, DataDuplicates, NewLabelTrainTest, TrainTestFeatureDrift, TrainTestLabelDrift
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureLabelCorrelationChange, ConflictingLabels, OutlierSampleDetection 
from deepchecks.tabular.checks import WeakSegmentsPerformance, RocReport, ConfusionMatrixReport, TrainTestPredictionDrift, CalibrationScore, BoostingOverfit

from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_auc_score
import pandas as pd
import xgboost as xgb
from ml_pipeline.processing import categorical_cols, cts_cols


pred_cat_cols=[
       'Gender_Female', 'Gender_Male', 'Gender_Not Specified', 'Gender_Other',
       'Married_No', 'Married_Not Specified', 'Married_Yes', 'Dependents_No',
       'Dependents_Not Specified', 'Dependents_Yes', 'offer_A', 'offer_B',
       'offer_C', 'offer_D', 'offer_E', 'offer_F', 'offer_G', 'offer_H',
       'offer_I', 'offer_J', 'offer_No Offer', 'Referred a Friend_No',
       'Referred a Friend_Yes', 'Phone Service_No', 'Phone Service_Yes',
       'Multiple Lines_No', 'Multiple Lines_None', 'Multiple Lines_Yes',
       'Internet Service_No', 'Internet Service_Yes', 'Internet Type_Cable',
       'Internet Type_DSL', 'Internet Type_Fiber Optic', 'Internet Type_None',
       'Internet Type_Not Applicable', 'Online Security_No',
       'Online Security_Yes', 'Online Backup_No', 'Online Backup_Yes',
       'Device Protection Plan_No', 'Device Protection Plan_Yes',
       'Premium Tech Support_No', 'Premium Tech Support_Yes',
       'Streaming TV_No', 'Streaming TV_Yes', 'Streaming Movies_No',
       'Streaming Movies_Yes', 'Streaming Music_No', 'Streaming Music_Yes',
       'Unlimited Data_No', 'Unlimited Data_None', 'Unlimited Data_Yes',
       'Payment Method_Bank Withdrawal', 'Payment Method_Credit Card',
       'Payment Method_Wallet Balance']

pred_cts_cols=['Age', 'Number of Dependents', 'roam_ic', 'roam_og', 'loc_og_t2t',
       'loc_og_t2m', 'loc_og_t2f', 'loc_og_t2c', 'std_og_t2t', 'std_og_t2m',
       'std_og_t2f', 'std_og_t2c', 'isd_og', 'spl_og', 'og_others',
       'loc_ic_t2t', 'loc_ic_t2m', 'loc_ic_t2f', 'std_ic_t2t', 'std_ic_t2m',
       'std_ic_t2f', 'std_ic_t2o', 'spl_ic', 'isd_ic', 'ic_others',
       'total_rech_amt', 'total_rech_data', 'vol_4g', 'vol_5g', 'arpu_5g',
       'arpu_4g', 'arpu', 'aug_vbc_5g', 'Number of Referrals',
       'Streaming Data Consumption', 'Satisfaction Score', 'total_recharge']


def preprocess_steps(data,encoded_features, encoder, scaler):
    '''
    For further processing data while testing
    '''
    df=data.copy()
    drop_cols=['Customer ID', 'Quarter', 'Quarter of Joining', 'Month',
       'Month of Joining', 'zip_code','Location ID', 'Service ID',
       'state', 'county', 'timezone', 'area_codes', 'country', 'latitude',
       'longitude','Status ID']
    df=df.drop(columns=drop_cols)
    
    processed_data=df.copy()
    processed_data[encoded_features] = encoder.transform(processed_data[categorical_cols])
    processed_data=processed_data.drop(categorical_cols,axis=1)
    processed_data[cts_cols]  = scaler.transform(processed_data[cts_cols]) 

    return processed_data



def check_data_drift(ref_df:pd.DataFrame, cur_df:pd.DataFrame, predictors:list, job_id:str):
    """
    Check for data drifts between two datasets and decide whether to retrain the model. 
    A report will be saved in the results directory.
    :param ref_df: Reference dataset
    :param cur_df: Current dataset
    :param predictors: Predictors to check for drifts
    :param target: Target variable to check for drifts
    :param job_id: Job ID
    :return: boolean
    """
    ref_features = [col for col in predictors if col in ref_df.columns]
    cur_features = [col for col in predictors if col in cur_df.columns]
    ref_cat_features = [col for col in pred_cat_cols if col in ref_df.columns]
    cur_cat_features = [col for col in pred_cat_cols if col in cur_df.columns]
    ref_dataset = Dataset(ref_df,  features=ref_features, cat_features=ref_cat_features)
    cur_dataset = Dataset(cur_df, features=cur_features, cat_features=cur_cat_features)
    
    suite = Suite("data drift",
        WholeDatasetDrift().add_condition_overall_drift_value_less_than(0.2), #0.2 
        TrainTestFeatureDrift().add_condition_drift_score_less_than(0.2), #0.1   
        )
    r = suite.run(train_dataset=ref_dataset, test_dataset=cur_dataset)
    retrain = (len(r.get_not_ran_checks())>0) or (len(r.get_not_passed_checks())>0)
    
    try:
        r.save_as_html(f"../reports/{job_id}_data_drift_report.html")
        print("[INFO] Data drift report saved as {}".format(f"{job_id}_data_drift_report.html"))
    except Exception as e:
        print(f"[WARNING][DRIFTS.check_DATA_DRIFT] {traceback.format_exc()}")
    return {"report": r, "retrain": retrain}





def inference_pipeline(inference_data,reference_data,job_id,predictors_cols, encoded_features, encoder, scaler):
    #write data cleaning steps if necessary

    #data preprocessing
    clean_inf_data=preprocess_steps(inference_data, encoded_features, encoder, scaler)

    #data drift
    data_drift=check_data_drift(ref_df=reference_data, cur_df=clean_inf_data, predictors=predictors_cols,  job_id=job_id)
    print(f"Data Drift Retrain: {data_drift['retrain']}")

    return data_drift
    



def check_data_drift_with_label(ref_df:pd.DataFrame, cur_df:pd.DataFrame, target:str, predictors:list, job_id:str):
    """
    Check for data drifts between two datasets and decide whether to retrain the model. 
    A report will be saved in the results directory.
    :param ref_df: Reference dataset
    :param cur_df: Current dataset
    :param predictors: Predictors to check for drifts
    :param target: Target variable to check for drifts
    :param job_id: Job ID
    :return: boolean
    """
    ref_features = [col for col in predictors if col in ref_df.columns]
    cur_features = [col for col in predictors if col in cur_df.columns]
    ref_cat_features = [col for col in pred_cat_cols if col in ref_df.columns]
    cur_cat_features = [col for col in pred_cat_cols if col in cur_df.columns]
    ref_dataset = Dataset(ref_df, label=target, features=ref_features, cat_features=ref_cat_features)
    cur_dataset = Dataset(cur_df, label=target,features=cur_features, cat_features=cur_cat_features)
    
    suite = Suite("data drift",
        NewLabelTrainTest(),
        WholeDatasetDrift().add_condition_overall_drift_value_less_than(0.2), 
        FeatureLabelCorrelationChange().add_condition_feature_pps_difference_less_than(0.2), 
        TrainTestFeatureDrift().add_condition_drift_score_less_than(0.2), 
        TrainTestLabelDrift(balance_classes=True).add_condition_drift_score_less_than(0.4) 
    )
    r = suite.run(train_dataset=ref_dataset, test_dataset=cur_dataset)
    retrain = (len(r.get_not_ran_checks())>0) or (len(r.get_not_passed_checks())>0)
    
    try:
        r.save_as_html(f"../reports/{job_id}_data_drift_report.html")
        print("[INFO] Data drift report saved as {}".format(f"{job_id}_data_drift_report.html"))
    except Exception as e:
        print(f"[WARNING][DRIFTS.check_DATA_DRIFT] {traceback.format_exc()}")
    return {"report": r, "retrain": retrain}

def check_model_drift(model,pred_data,label):
    label_pred=model.predict(pred_data)
    test_f1_score = f1_score(label,label_pred)
    test_recall = recall_score(label, label_pred)

    print("\n Test Results")
    print(f'F1 Score: {test_f1_score}')
    print(f'Recall Score: {test_recall}')
    print(f'Confusion Matrix: \n{confusion_matrix(label, label_pred)}')
    print(f'Area Under Curve: {roc_auc_score(label, label_pred)}')

    #condition for model retraining according to business
    model_retrain= (test_recall<0.80) or (test_f1_score<0.5)
    print(f"\n Model Drift Retrain: {model_retrain}")
    return model_retrain,label_pred

def inference_pipeline_with_label(inference_data,reference_data,job_id,trained_model,target_col_name,target_value,predictors_cols,encoded_features, encoder, scaler):
    #write data cleaning steps if necessary

    #data preprocessing
    clean_inf_data=preprocess_steps(inference_data, encoded_features, encoder, scaler)
    clean_inf_data[target_col_name]=target_value

    #data drift
    data_drift=check_data_drift_with_label(ref_df=reference_data, cur_df=clean_inf_data, predictors=predictors_cols, target=target_col_name, job_id=job_id)
    print(f"Data Drift Retrain: {data_drift['retrain']}")

    #model drift
    model_retrain,predictions=check_model_drift(model=trained_model,pred_data=xgb.DMatrix(clean_inf_data.drop(columns=target_col_name)),label=target_value)
    

    return  data_drift,model_retrain,predictions


