import pandas as pd
import os
import json
import numpy as np
import random
from sklearn.preprocessing import RobustScaler

with open('config/SETTINGS.json') as f:
    config = json.load(f)
print(config)

#--------------- read inputs ---------------
targets = pd.read_excel(config['data_paths']['train_targets'],index_col=0)
cntm = pd.read_csv(config['data_paths']['train_cntm'],index_col=0)
cntm_test = pd.read_csv(config['data_paths']['test_cntm'],index_col=0)
meta_qt = pd.read_excel(config['data_paths']['train_meta_qt'],index_col=0)
meta_qt_test = pd.read_excel(config['data_paths']['test_meta_qt'],index_col=0)
meta_ct = pd.read_excel(config['data_paths']['train_meta_ct'],index_col=0)
meta_ct_test = pd.read_excel(config['data_paths']['test_meta_ct'],index_col=0)
print('loaded')

#--------------- normalize train and test cntm using the same average and std---------------
average = cntm.mean()
std = cntm.std()
cntm, cntm_test = (cntm-average)/std, (cntm_test-average)/std

#--------------- scale meta data ---------------
meta = pd.concat([meta_qt.drop('MRI_Track_Age_at_Scan',axis=1),(meta_ct[['Barratt_Barratt_P1_Occ','Barratt_Barratt_P2_Occ']].sum(1)/5).rename('Occ').to_frame()],axis=1)
meta_test = pd.concat([meta_qt_test.drop('MRI_Track_Age_at_Scan',axis=1),(meta_ct_test[['Barratt_Barratt_P1_Occ','Barratt_Barratt_P2_Occ']].sum(1)/5).rename('Occ').to_frame()],axis=1)

scaler = RobustScaler()
meta = pd.DataFrame(scaler.fit_transform(meta),index=meta.index,columns=meta.columns).fillna(0)
meta_test = pd.DataFrame(scaler.transform(meta_test),index=meta_test.index,columns=meta_test.columns).fillna(0)

#--------------- sample and save clean train/val data ---------------
os.makedirs(config['data_paths']['clean_data_dir'], exist_ok=True)
for i in range(config['num_seeds']):
    random.seed(i)
    pids_val = []
    for n,g in targets.apply(tuple,axis=1).to_frame().groupby(0):
        pids_val += random.sample(g.index.to_list(),60)
    pids_train = list(set(cntm.index).difference(set(pids_val)))

    pids_train_sampled = []
    for n,g in targets.loc[pids_train].apply(tuple,axis=1).to_frame().groupby(0):
        pids_train_sampled += random.choices(g.index.to_list(),k=200)
    np.save(os.path.join(config['data_paths']['clean_data_dir'],'train_val'+str(i)),{
        'cntm_train':cntm.loc[pids_train_sampled],
        'cntm_val':cntm.loc[pids_val],
        'meta_train':meta.loc[pids_train_sampled],
        'meta_val':meta.loc[pids_val],
        'targets_train':targets.loc[pids_train_sampled],
        'targets_val':targets.loc[pids_val],
    })
    print(f'saved train val data fold {i}')

#--------------- save clean train data ---------------
np.save(os.path.join(config['data_paths']['clean_data_dir'],'test'),{
    'cntm':cntm_test,
    'meta':meta_test,
    })
print('saved test data')

