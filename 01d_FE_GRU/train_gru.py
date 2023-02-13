import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.system("pip install torch")
os.system("pip install torchmetrics==0.10.0")

import uuid
import pickle

out_filename = str(uuid.uuid4())

import argparse

parser = argparse.ArgumentParser(
    description='Hyperparameters for model training'
)
parser.add_argument(
    '--exp-name', 
    type=str,
    help='experiment name'
)
parser.add_argument(
    '--neg-factor', 
    type=str,
    help='experiment name'
)
parser.add_argument(
    '--emb-width', 
    type=str,
    help='experiment name'
)
parser.add_argument(
    '--nip-train-bs', 
    type=str,
    help='experiment name'
)
parser.add_argument(
    '--igfold', 
    type=str,
    help='experiment name'
)
parser.add_argument(
    '--ty', 
    type=str,
    help='experiment name'
)
parser.add_argument(
    '--use-pretrained', 
    type=str,
    help='experiment name'
)
parser.add_argument(
    '--lr', 
    type=str,
    help='experiment name'
)
parser.add_argument(
    '--temp', 
    type=str,
    help='experiment name'
)
parser.add_argument(
    '--gru-layers', 
    type=str,
    help='experiment name'
)
parser.add_argument(
    '--gru-dropout', 
    type=str,
    help='experiment name'
)
parser.add_argument(
    '--ls', 
    type=str,
    help='experiment name'
)

args = parser.parse_args()

config = {}

if args.use_pretrained == 'Yes':
    config['use_pretrained'] = True
else:
    config['use_pretrained'] = False
config['neg_factor'] = int(args.neg_factor)
import numpy as np
config['emb_width'] = int(args.emb_width)
emb_max = 1856000
if config['use_pretrained']:
    config['emb_width'] = 50
config['nip_train_bs'] = int(args.nip_train_bs)
nip_valid_bs = 128

config['igfold'] = int(args.igfold)
config['ty'] = args.ty
config['lr'] = float(args.lr)
config['temp'] = float(args.temp)
config['gru_layers'] = int(args.gru_layers)
config['gru_dropout'] = float(args.gru_dropout)
config['ls'] = float(args.ls)
config['exp_name'] = args.exp_name

import pickle

w2v = pickle.load(open('./data_folds/fold_' + str(config['igfold']) + '/word2vec.emb', 'rb'))

import cudf
import glob
import gc
import numpy as np

from tqdm import tqdm

from merlin.io import Dataset
from merlin.loader.torch import Loader

from to_sparse import to_sparse_tensor

from tqdm import tqdm

import torch

class GRU(torch.nn.Module):
    def __init__(self, w2v, use_pretrained):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=config['emb_width'], 
            hidden_size=config['emb_width'], 
            batch_first=True,
            bidirectional=False, 
            num_layers=config['gru_layers'], 
            dropout=config['gru_dropout']
        )
        emb = torch.nn.Embedding(
            emb_max+1, 
            config['emb_width']
        ) 
        weight = emb.weight.detach().numpy()
        weight[0] = weight[0]*0
        if use_pretrained:
            for el in w2v.keys():
                weight[el+2] = w2v[el]
        emb.weight = torch.nn.Parameter(torch.Tensor(weight))
        self.emb_item = emb
        
        emb = torch.nn.Embedding(
            10, 
            8
        ) 
        weight = emb.weight.detach().numpy()
        weight[0] = weight[0]*0
        emb.weight = torch.nn.Parameter(torch.Tensor(weight))
        self.emb_type = emb
        
        emb = torch.nn.Embedding(
            50, 
            8
        ) 
        weight = emb.weight.detach().numpy()
        weight[0] = weight[0]*0
        emb.weight = torch.nn.Parameter(torch.Tensor(weight))
        self.emb_rank = emb
        
        self.inp_mlp = torch.nn.Sequential(
                torch.nn.Linear(config['emb_width']+16, config['emb_width'])
        )
        
        self.neg_samples = True

    def forward(self, x, prediction=False):
        x_aid = to_sparse_tensor(x[0]['aid'], 20, x[0]['aid'][0].device)
        x_type = to_sparse_tensor(x[0]['type'], 20, x[0]['type'][0].device)
        x_rank = to_sparse_tensor(x[0]['rank'], 20, x[0]['rank'][0].device)
        x_sess_len = x[0]['session_len'].long()
        
        x_emb_item = self.emb_item(x_aid)
        x_emb_type = self.emb_type(x_type)
        x_emb_rank = self.emb_rank(x_rank)
        
        x_out = torch.concat([x_emb_item, x_emb_type, x_emb_rank], axis=2)
        x_out = self.inp_mlp(x_out)

        x_rnn = self.rnn(x_out)
        x_last = x_rnn[0][torch.arange(x_rnn[0].size(0)), x_sess_len-1, :]
        
        if prediction:
            x_last = x_last @ model.emb_item.weight.t()
            return x_last
        else:
            if self.neg_samples:
                return self.calc_neg_samples(x_last, x[0]['target'].squeeze(), x_aid)
            else:
                x_last = x_last @ model.emb_item.weight.t()
                return x_last, x[0]['target'].squeeze()
    
    def calc_neg_samples(self, x, labels_all, x_aid):
        if config['neg_factor']>0:
            uni_neg = torch.randint(low=1, high=emb_max, size=(int(config['neg_factor']*config['nip_train_bs']),)).cuda()
            unique_labels = torch.unique(torch.cat([labels_all,uni_neg]))
        else:
            unique_labels = torch.unique(labels_all)
            
        emb_idx = self.emb_item.weight[unique_labels]
        x = x @ emb_idx.t()
        
        if True:
            bs = x.shape[0]
            mask = []
            for i in range(bs):
                mask.append(
                    torch.isin(unique_labels, x_aid[i])
                )
            mask = torch.stack(mask)
            xx = torch.reshape(
                torch.Tensor.repeat(unique_labels, bs),
                (bs, -1)
            )
            # Keeping Current Label
            mask2 = (xx==torch.unsqueeze(labels_all, axis=1))
            mask[mask2] = False
            mask = ~mask
            x = x*mask

        
        pos = torch.unsqueeze(labels_all, axis=1)==unique_labels
        neg = ~pos
        x_pos = torch.unsqueeze(x[pos], axis=1)
        x_neg = torch.reshape(x[neg], (x_pos.shape[0], -1))
        x = torch.concat([x_pos, x_neg], axis=1)
        y = torch.zeros_like(labels_all)
        
        return(x/config['temp'], y)

def nip_train_epoch():
    running_loss = 0.0
    model.train()
    for i, batch in enumerate(tqdm(nip_train_dl)):
        optimizer.zero_grad()
        x,y = model(batch)
        loss = torch.nn.CrossEntropyLoss(label_smoothing=config['ls'])(x, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(running_loss / i)
        if i % 30000 == 29999:
            break
            
def nip_pred(dl):
    sessions = []
    preds = []
    scores = []
    model.eval()
    for i, batch in enumerate(tqdm(dl)):
        with torch.no_grad():
            x = model(batch, prediction=True)
        pred = torch.topk(x, 50)
        preds.append(pred[1].detach().cpu().numpy())
        scores.append(torch.nn.Softmax(dim=1)(pred[0]).detach().cpu().numpy())
        sessions.append(batch[0]['session'].detach().cpu().numpy())
    
    sessions = np.hstack(sessions)
    preds = np.vstack(preds)
    scores = np.vstack(scores)
    
    df = cudf.DataFrame({
        'session': sessions,
        'scores': scores.tolist(),
    })
    for i in range(50):
        df['rec_' + str(i)] = df.scores.list.get(i, default=-1)
    df.drop(['scores'], inplace=True, axis=1)
    df = cudf.melt(df, id_vars=['session'], value_vars=['rec_' + str(i) for i in range(50)])
    df['rank'] = (1+df['variable'].cat.codes)
    df.drop(['variable'], axis=1, inplace=True)
    df_score = df.rename(columns={'value': 'gru_scores'}).copy()
    del df
    gc.collect()

    df = cudf.DataFrame({
        'session': sessions,
        'preds': (preds-2).tolist(),
    })
    for i in range(50):
        df['rec_' + str(i)] = df.preds.list.get(i, default=-1)
    df.drop(['preds'], inplace=True, axis=1)
    df = cudf.melt(df, id_vars=['session'], value_vars=['rec_' + str(i) for i in range(50)])
    df['rank'] = (1+df['variable'].cat.codes)
    df.drop(['variable'], axis=1, inplace=True)
    
    df = df.merge(
        df_score,
        how='left',
        on=['session', 'rank']
    ).sort_values(['session', 'rank'])
    
    df = df.rename(columns={'value': 'cand'})
    df.drop(['rank'], axis=1, inplace=True)
    return(df)
            
def nip_eval():
    sessions = []
    preds = []
    model.eval()
    for i, batch in enumerate(tqdm(nip_valid_dl)):
        with torch.no_grad():
            x = model(batch, prediction=True)
        pred = torch.topk(x, 20)
        preds.append(pred[1].detach().cpu().numpy())
        sessions.append(batch[0]['session'].detach().cpu().numpy())
    
    sessions = np.hstack(sessions)
    preds = np.vstack(preds)
    
    df = cudf.DataFrame({
        'session': sessions,
        'preds': (preds-2).tolist()
    })
    df = df.rename(columns={'preds': 'labels'})
    
    import pickle
    sessions = pickle.load(open('./data/sessions_eval.pickle', 'rb'))
    if config['igfold'] == 0:
        sess_eval = sessions[0]+sessions[1]
    elif config['igfold'] == 1:
        sess_eval = sessions[2]+sessions[3]
    elif config['igfold'] == 2:
        sess_eval = sessions[4]+sessions[5]
    elif config['igfold'] == 3:
        sess_eval = sessions[6]+sessions[7]
    elif config['igfold'] == 4:
        sess_eval = sessions[8]+sessions[9]

    print(len(sess_eval))
    
    df = df[df['session'].isin(sess_eval)]
    
    score = 0
    for t in [config['ty']]:
        sub = df.to_pandas()
        test_labels = cudf.read_parquet('./data/xgb_train_y.parquet')
        test_labels = test_labels[test_labels['session'].isin(sess_eval)]
        test_labels = test_labels[['session', 'aid', 'type']].groupby(['session', 'type']).agg(list).reset_index()
        test_labels = test_labels.loc[test_labels['type']==t].to_pandas()
        test_labels = test_labels.merge(sub, how='left', on=['session'])
        test_labels['hits'] = test_labels.apply(lambda df: len(set(df.aid).intersection(set(df.labels))), axis=1)
        test_labels['gt_count'] = test_labels.aid.str.len().clip(0,20)
        recall = test_labels['hits'].sum() / test_labels['gt_count'].sum()
    return(recall)

nip_train_files = glob.glob('./data_folds/fold_' + str(config['igfold']) + '/data_gru/train/' + config['ty'] + '/*.parquet')
nip_valid_files = glob.glob('./data_folds/fold_' + str(config['igfold']) + '/data_gru/valid/*.parquet')
nip_test_files = glob.glob('./data_folds/fold_' + str(config['igfold']) + '/data_gru/test/*.parquet')

train_ds = Dataset(nip_train_files)
train_ds.schema = train_ds.schema.remove_col('target_type')
nip_train_dl = Loader(train_ds, batch_size=config['nip_train_bs'])

valid_ds = Dataset(nip_valid_files)
valid_ds.schema = valid_ds.schema.remove_col('target_type')
nip_valid_dl = Loader(valid_ds, batch_size=nip_valid_bs)

test_ds = Dataset(nip_test_files)
test_ds.schema = test_ds.schema.remove_col('target_type')
nip_test_dl = Loader(test_ds, batch_size=nip_valid_bs)

model = GRU(
    use_pretrained=config['use_pretrained'],
    w2v=w2v
).cuda()

if config['use_pretrained']:
    model.emb_item.weight.requires_grad = False
    
item_w = model.emb_item.weight.detach().cpu().numpy()
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

recalls = []
os.system('mkdir -p ./gru_results/' + config['exp_name'] + '/' + out_filename + '/')
pickle.dump([config, recalls], 
            open('./gru_results/' + config['exp_name'] + '/' + out_filename + '/results.pickle', 'wb'))
for e in range(1):
    nip_train_epoch()
    if config['use_pretrained']:
        assert np.isclose(item_w, model.emb_item.weight.detach().cpu().numpy()).mean() == 1.0
    recall = nip_eval()
    print(recall)
    recalls.append(recall)
    os.system('mkdir -p ./gru_results/' + config['exp_name'] + '/' + out_filename + '/')
    pickle.dump([config, recalls], 
                open('./gru_results/' + config['exp_name'] + '/' + out_filename + '/results.pickle', 'wb'))

df = nip_pred(nip_valid_dl)
df.to_parquet(
    './data_folds/fold_' + str(config['igfold']) + '/data_gru/gru_pred_xgb_' + str(config['use_pretrained']) + '_' + str(config['ty']) + '.parquet')
del df
gc.collect()
df = nip_pred(nip_test_dl)
df.to_parquet(
    './data_folds/fold_' + str(config['igfold']) + '/data_gru/gru_pred_sub_' + str(config['use_pretrained']) + '_' + str(config['ty']) + '.parquet')