import sys
import os
import joblib
from pathlib import Path
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

import prediction_model

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset



filepath = os.path.join(config.DATAPATH,'loan_data_defaults.csv')
loan_data_defaults = pd.read_csv(filepath)

pd.options.display.max_rows = None
print(loan_data_defaults.isnull().sum())



PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT,'dataset_LGD_EAD')
print(DATAPATH)


#Spliting data for LGD 
#LGD model stage 1 datasets: recovery rate 0 or greater than 0.
lgd_inputs_stage_1_train, lgd_inputs_stage_1_test, lgd_targets_stage_1_train, lgd_targets_stage_1_test = train_test_split(loan_data_defaults.drop([ 'recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), loan_data_defaults['recovery_rate_0_1'], test_size = 0.2, random_state = 42)


lgd_inputs_stage_1_train.to_csv(os.path.join(DATAPATH,'lgd_inputs_stage_1_train.csv'))
lgd_inputs_stage_1_test.to_csv(os.path.join(DATAPATH,'lgd_inputs_stage_1_test.csv'))
lgd_targets_stage_1_train.to_csv(os.path.join(DATAPATH,'lgd_targets_stage_1_train.csv'))
lgd_targets_stage_1_test.to_csv(os.path.join(DATAPATH,'lgd_targets_stage_1_test.csv'))

#Stage 2 – Linear Regression
# Here we take only rows where the original recovery rate variable is greater than one,
# i.e. where the indicator variable we created is equal to 1.
lgd_stage_2_data = loan_data_defaults[loan_data_defaults['recovery_rate_0_1'] == 1]

# LGD model stage 2 datasets: how much more than 0 is the recovery rate
lgd_inputs_stage_2_train, lgd_inputs_stage_2_test, lgd_targets_stage_2_train, lgd_targets_stage_2_test = train_test_split(lgd_stage_2_data.drop([ 'recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), lgd_stage_2_data['recovery_rate'], test_size = 0.2, random_state = 42)

lgd_inputs_stage_2_train.to_csv(os.path.join(DATAPATH,'lgd_inputs_stage_2_train.csv'))
lgd_inputs_stage_2_test.to_csv(os.path.join(DATAPATH,'lgd_inputs_stage_2_test.csv'))
lgd_targets_stage_2_train.to_csv(os.path.join(DATAPATH,'lgd_targets_stage_2_train.csv'))
lgd_targets_stage_2_test.to_csv(os.path.join(DATAPATH,'lgd_targets_stage_2_test.csv'))


#Spliting data for EAD 
ead_inputs_train, ead_inputs_test, ead_targets_train, ead_targets_test = train_test_split(loan_data_defaults.drop([ 'recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), loan_data_defaults['CCF'], test_size = 0.2, random_state = 42)

ead_inputs_train.to_csv(os.path.join(DATAPATH,'ead_inputs_train.csv'))
ead_inputs_test.to_csv(os.path.join(DATAPATH,'ead_inputs_test.csv'))
ead_targets_train.to_csv(os.path.join(DATAPATH,'ead_targets_train.csv'))
ead_targets_test.to_csv(os.path.join(DATAPATH,'ead_targets_test.csv'))
