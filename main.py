from sklearn.metrics import f1_score
import random
import sklearn
from sklearn import decomposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import scipy
import copy
import statsmodels.api as sm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_col", help='ADHD_Outcome or Sex_F', required=True)
parser.add_argument("--seed", help='0, 1, 2', required=True)
parser.add_argument("--latent_dim", help='256', required=True)
parser.add_argument("--dropout", help='0.5', required=True)

# Parse arguments
args = parser.parse_args()
target_col = args.target_col
seed = args.seed
latent_dim = args.latent_dim
dropout = args.dropout

class Autoencoder(nn.Module):
    def __init__(self, input_dim, meta_dim, hidden_dim, output_dim, dropout):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            #nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim*8),
            nn.ReLU(),
            
            #nn.BatchNorm1d(hidden_dim*8),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*8, hidden_dim*4),
            nn.ReLU(),
            
            #nn.BatchNorm1d(hidden_dim*4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.ReLU(),
            
            #nn.BatchNorm1d(hidden_dim*2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            #nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            
            #nn.BatchNorm1d(hidden_dim*2),
            #nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim*4),
            nn.ReLU(),
            
            #nn.BatchNorm1d(hidden_dim*4),
            #nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, hidden_dim*8),
            nn.ReLU(),
            
            #nn.BatchNorm1d(hidden_dim*8),
            #nn.Dropout(dropout),
            nn.Linear(hidden_dim*8, input_dim),
        )
        # Decoder
        self.mlp = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            #nn.Dropout(dropout),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            
            #nn.BatchNorm1d(hidden_dim),
            #nn.Dropout(dropout),
            nn.Linear(hidden_dim+meta_dim, hidden_dim+meta_dim),
            nn.Linear(hidden_dim+meta_dim, output_dim),
        )
        
    def forward(self, x, meta):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,self.mlp(torch.cat([encoded,meta],dim=1)),decoded
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

#--------------- read inputs ---------------
targets = pd.read_excel('/mnt/computational-bio-data/user-scratch/byao/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx',index_col=0)
cntm = pd.read_csv('/mnt/computational-bio-data/user-scratch/byao/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv',index_col=0)
cntm_test = pd.read_csv('/mnt/computational-bio-data/user-scratch/byao/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv',index_col=0)
meta_qt = pd.read_excel('/mnt/computational-bio-data/user-scratch/byao/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx',index_col=0)
meta_qt_test = pd.read_excel('/mnt/computational-bio-data/user-scratch/byao/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx',index_col=0)
meta_ct = pd.read_excel('/mnt/computational-bio-data/user-scratch/byao/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx',index_col=0)
meta_ct_test = pd.read_excel('/mnt/computational-bio-data/user-scratch/byao/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx',index_col=0)
print('loaded')
#--------------- scale QUANTITATIVE ---------------
meta = pd.concat([meta_qt.drop('MRI_Track_Age_at_Scan',axis=1),(meta_ct[['Barratt_Barratt_P1_Occ','Barratt_Barratt_P2_Occ']].sum(1)/5).rename('Occ').to_frame()],axis=1)
meta_test = pd.concat([meta_qt_test.drop('MRI_Track_Age_at_Scan',axis=1),(meta_ct_test[['Barratt_Barratt_P1_Occ','Barratt_Barratt_P2_Occ']].sum(1)/5).rename('Occ').to_frame()],axis=1)

scaler = sklearn.preprocessing.RobustScaler()
meta = pd.DataFrame(scaler.fit_transform(meta),index=meta.index,columns=meta.columns).fillna(0)
meta_test = pd.DataFrame(scaler.transform(meta_test),index=meta_test.index,columns=meta_test.columns).fillna(0)

cntm_centered, cntm_centered_test = cntm,cntm_test

set_seed(int(seed))
batch_size = 64
pids_all = cntm.index
pids_val = []
for n,g in targets.apply(tuple,axis=1).to_frame().groupby(0):
    e = g.index.to_list()
    pids_val += random.sample(e,60)
pids_train = list(set(pids_all).difference(set(pids_val)))

pids_train_sampled = []
for n,g in targets.loc[pids_train].apply(tuple,axis=1).to_frame().groupby(0):
    e = g.index.to_list()
    pids_train_sampled += random.choices(e,k=200)


average = cntm_centered.loc[pids_train_sampled].mean()
cntm_centered, cntm_centered_test = cntm_centered-average, cntm_centered_test-average
std = cntm_centered.loc[pids_train_sampled].std()
cntm_centered, cntm_centered_test = cntm_centered/std, cntm_centered_test/std

#--------------- train and predict ---------------

cntm_centered_concat = pd.concat([cntm_centered,cntm_centered_test])

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(cntm_centered_concat.loc[pids_train_sampled].to_numpy()),
    torch.tensor(targets.loc[pids_train_sampled,target_col].to_numpy()),
    torch.tensor(meta.loc[pids_train_sampled].to_numpy()),
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(cntm_centered_concat.loc[pids_val].to_numpy()),
    torch.tensor(targets.loc[pids_val,target_col].to_numpy()),
    torch.tensor(meta.loc[pids_val].to_numpy()),
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

submit_dataset = torch.utils.data.TensorDataset(
    torch.tensor(cntm_centered_concat.loc[cntm_centered_test.index].to_numpy()),
    torch.tensor(meta_test.loc[cntm_centered_test.index].to_numpy())
)
submit_loader = DataLoader(submit_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

autoencoder = Autoencoder(cntm_centered_concat.shape[1],meta.shape[1],latent_dim,1, dropout=dropout).to(device)
autoencoder.apply(weights_init_uniform_rule)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
loss_mse = torch.nn.MSELoss(reduction="mean")
loss_bce = torch.nn.BCEWithLogitsLoss()

def train(autoencoder,epoch,trainloader):
    autoencoder.train()
    train_loss = 0
    for batch in trainloader:
        x = batch[0].float().to(device)
        meta = batch[2].float().to(device)
        ecd,cls,recon = autoencoder(x,meta)
        y = batch[1].float().to(device)
        l1_norm = sum(p.abs().sum() for p in autoencoder.mlp.parameters())
        loss = loss_mse(recon, x)+loss_bce(cls.squeeze(),y)*50+l1_norm*0.01
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
        train_loss += loss.item()
        optimizer.step()
    train_loss = train_loss / len(trainloader.dataset)
    train_losses.append(train_loss)
    #print('====> Epoch: {} Average Train loss: {:.4f}'.format(epoch, train_loss))
    return train_loss

def validate(autoencoder,epoch,valloader):
    autoencoder.eval()
    encoded,xs,recs,target,metas,clses = [],[],[],[],[],[]
    with torch.no_grad():
        for batch in valloader:
            x = batch[0].float().to(device)
            meta = batch[2].float().to(device)
            ecd,cls,recon = autoencoder(x,meta)
            target.append(batch[1].cpu().numpy())
            xs.append(x.cpu().numpy())
            recs.append(recon.cpu().numpy())
            encoded.append(ecd.cpu().numpy())
            metas.append(batch[2].numpy())
            clses.append(torch.sigmoid(cls).cpu().numpy())
    encoded,xs,recs,target,metas,clses = np.vstack(encoded),np.vstack(xs),np.vstack(recs),np.hstack(target),np.vstack(metas),np.vstack(clses)
    return encoded,xs,recs,target,metas,clses

best_model_wts = None
best_score,best_std = -float('inf'),-float('inf')
train_losses,val_losses,rs = [],[],[]
epochs = 40
for epoch in range(1, epochs + 1):
    train_loss = train(autoencoder,epoch,train_loader)
    encoded_val,xs_val,recs_val,target_val,metas_val,clses_val = validate(autoencoder,epoch,val_loader)
    pred = (clses_val>0.5).astype(int)
    f1 = f1_score(target_val,pred)
    #print(epoch,round(f1,6),round(pred.std(),6))
    if (f1 > best_score) and (pred.std()>0):
        best_score = f1
        best_std = pred.std()
        best_model_wts = copy.deepcopy(autoencoder.state_dict())
print(best_score,best_std)

def submit(test_loader):
    autoencoder.eval()
    preds = []
    for batch in test_loader:
        x = batch[0].float().to(device)
        meta = batch[1].float().to(device)
        ecd,cls,recon = autoencoder(x,meta)
        pred = cls.detach().cpu().numpy()
        pred = (pred>0.5).astype(int)
        preds += pred.tolist()
    return preds
    
if best_model_wts is not None:
    autoencoder.load_state_dict(best_model_wts)
    preds_submit = submit(submit_loader)
    preds_submit = pd.DataFrame(preds_submit,index=cntm_centered_test.index,columns=[target_col+'|'+str(round(best_score,3))])
    preds_submit.to_csv(f'result/{target_col}autoenc|{seed}.csv')

lm = sklearn.ensemble.HistGradientBoostingClassifier(class_weight={0:1,1:1})
lm.fit(pd.concat([cntm_centered,meta],axis=1).loc[pids_train_sampled], targets.loc[pids_train_sampled,target_col])
pred = lm.predict(pd.concat([cntm_centered,meta],axis=1).loc[pids_val])
f1 = f1_score(targets.loc[pids_val,target_col],pred)
print(round(f1,3), round(pred.std(),3))
preds_submit = lm.predict(pd.concat([cntm_centered_test,meta_test],axis=1))
preds_submit = pd.DataFrame(preds_submit,index=cntm_centered_test.index,columns=[target_col+'|'+str(round(f1,3))])
preds_submit.to_csv(f'result/{target_col}hgb1|{seed}.csv')

lm = sklearn.ensemble.HistGradientBoostingClassifier(class_weight={0:1,1:0.1})
lm.fit(pd.concat([cntm_centered,meta],axis=1).loc[pids_train_sampled], targets.loc[pids_train_sampled,target_col])
pred = lm.predict(pd.concat([cntm_centered,meta],axis=1).loc[pids_val])
f1 = f1_score(targets.loc[pids_val,target_col],pred)
print(round(f1,3), round(pred.std(),3))
preds_submit = lm.predict(pd.concat([cntm_centered_test,meta_test],axis=1))
preds_submit = pd.DataFrame(preds_submit,index=cntm_centered_test.index,columns=[target_col+'|'+str(round(f1,3))])
preds_submit.to_csv(f'result/{target_col}hgb01|{seed}.csv')

lm = sklearn.ensemble.HistGradientBoostingClassifier(class_weight={0:1,1:10})
lm.fit(pd.concat([cntm_centered,meta],axis=1).loc[pids_train_sampled], targets.loc[pids_train_sampled,target_col])
pred = lm.predict(pd.concat([cntm_centered,meta],axis=1).loc[pids_val])
f1 = f1_score(targets.loc[pids_val,target_col],pred)
print(round(f1,3), round(pred.std(),3))
preds_submit = lm.predict(pd.concat([cntm_centered_test,meta_test],axis=1))
preds_submit = pd.DataFrame(preds_submit,index=cntm_centered_test.index,columns=[target_col+'|'+str(round(f1,3))])
preds_submit.to_csv(f'result/{target_col}hgb10|{seed}.csv')
