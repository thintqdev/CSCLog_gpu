#!/usr/bin/env python
"""
CSCLog Training Script
Converted from main.ipynb for automated training
Usage: python train_csclog.py [--config train_config.yaml]
"""

import sys
import argparse
import numpy as np
import random
import pandas as pd
import time
import os
import collections
from tqdm import tqdm
import json
import math

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader
import dateutil.parser

from utils import pytorchtools
from collections import Counter
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from torch_geometric.nn import GCNConv

import warnings
warnings.filterwarnings("ignore")


def seed_torch(seed=42):
    """Set random seeds for reproducibility"""
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


# Import model definitions from main.ipynb
class MLPLayer(nn.Module):
    def __init__(self, dmodel, hid_size, drop, training):
        super(MLPLayer, self).__init__()
        self.dmodel = dmodel
        self.hid_size = hid_size
        self.drop = drop
        
        self.fc0 = nn.Linear(dmodel, hid_size)
        self.fc1 = nn.Linear(hid_size, hid_size)
        
    def forward(self, x):
        x = f.relu(self.fc0(x))
        x = f.dropout(x, p=self.drop, training=self.training)
        x = f.relu(self.fc1(x))
        return x


class CNNLayer(nn.Module):
    def __init__(self, dmodel, hid_size, ksize, drop, training):
        super(CNNLayer, self).__init__()
        self.dmodel = dmodel
        self.hid_size = hid_size
        self.ksize = ksize
        self.drop = drop
        
        self.cnn0 = nn.Conv1d(dmodel, hid_size, kernel_size=ksize, stride=1, padding=0)
        self.cnn1 = nn.Conv1d(hid_size, hid_size, kernel_size=ksize, stride=1, padding=0)
        
    def forward(self, x):
        x = f.relu(self.cnn0(x))
        x = f.dropout(x, p=self.drop, training=self.training)
        x = f.relu(self.cnn1(x))
        return x


class FTEncoder(nn.Module):
    def __init__(self, sen_size, hidden_size, alpha=0.5, pattern=0):
        super(FTEncoder, self).__init__()
        self.pattern = pattern
        assert self.pattern in [0, 1, 2], "pattern just in cat_first 0 or nn_first 1 or add_first 2"
        if pattern == 1:
            self.alpha = alpha
            assert alpha < 1 and alpha > 0, "alpha is rate just in (0,1)"
            sen_fc_size = int(hidden_size * alpha)
            time_fc_size = hidden_size - sen_fc_size
            print('x:{}, sen_x:{}, time_x:{}'.format(hidden_size, sen_fc_size, time_fc_size))
            self.sen_fc = nn.Linear(sen_size, sen_fc_size)
            self.time_fc = nn.Linear(1, time_fc_size)
        elif pattern == 0:
            self.cat_fc = nn.Linear(sen_size + 1, hidden_size)
        elif pattern == 2:
            self.sen_fc = nn.Linear(sen_size, hidden_size)
            self.time_fc = nn.Linear(1, hidden_size)
    
    def forward(self, x):
        if self.pattern == 0:
            cat_x = torch.cat((x[0], x[1].unsqueeze(-1)), -1)
            cat_x = self.cat_fc(cat_x)
        elif self.pattern == 1:
            sen_x = self.sen_fc(x[0])
            time_x = self.time_fc(x[1].unsqueeze(-1))
            cat_x = torch.cat((sen_x, time_x), -1)
        elif self.pattern == 2:
            cat_x = self.sen_fc(x[0]) + self.time_fc(x[1].unsqueeze(-1))
        return cat_x


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]


class IREncoder(nn.Module):
    def __init__(self, dmodel, mlp_hid_size, gcn_hid_size, drop, com_num, training):
        super(IREncoder, self).__init__()
        self.dmodel = dmodel
        self.mlp_hid_size = mlp_hid_size
        self.gcn_hid_size = gcn_hid_size
        self.drop = drop
        self.com_num = com_num
        self.training = training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.edge_mlp = MLPLayer(2*dmodel, mlp_hid_size, drop, training)
        self.mlp_out = nn.Linear(mlp_hid_size, 1)
        
        self.cnn_out = nn.Conv1d(mlp_hid_size, 1, kernel_size=1, stride=1, padding=0)
        
        self.GCN0 = GCNConv(dmodel, gcn_hid_size)
        self.GCN1 = GCNConv(gcn_hid_size, dmodel)
        
    def rel_pyg(self, node):
        rec2edge_index = []
        send2edge_index = []
        for i in range(len(node)):
            for j in range(i + 1, len(node)):
                rec2edge_index.append(node[i])
                send2edge_index.append(node[j])
        rec2edge_index = torch.as_tensor(rec2edge_index, dtype=torch.long).to(self.device)
        send2edge_index = torch.as_tensor(send2edge_index, dtype=torch.long).to(self.device)
        return torch.stack([rec2edge_index, send2edge_index])
    
    def gumbel_softmax(self, x, axis=1):
        trans_input = x.transpose(axis, 0).contiguous()
        soft_max_1d = f.softmax(trans_input)
        return soft_max_1d.transpose(axis, 0)
    
    def node2edge(self, x, index):
        edge_index = self.rel_pyg(index)
        edge_x = torch.cat([x[edge_index[0]], x[edge_index[1]]], -1)
        return edge_x, edge_index
    
    def forward(self, x, index):
        padding_x = torch.zeros([self.com_num, self.dmodel], requires_grad=True).to(self.device)
        padding_x[index] = x
        edge_x, edge_index = self.node2edge(padding_x, index)
        
        edge_x = self.edge_mlp(edge_x)
        edge_x = self.mlp_out(edge_x)
        
        edge_weight = self.gumbel_softmax(edge_x)
        
        out = f.relu(self.GCN0(padding_x, edge_index, edge_weight))
        out = f.dropout(out, self.drop, training=self.training)
        out = self.GCN1(out, edge_index, edge_weight)
        
        return out[index]


class Model(nn.Module):
    def __init__(self, input_size, com_num, hidden_size, alpha, pattern,
                 num_layers, num_keys, drop=0.1, training=True):
        super(Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ft_hid_size, lstm_hid_size, mlp_hid_size, gcn_hid_size, out_hid_size = hidden_size
        self.lstm_hid_size = lstm_hid_size
        self.com_num = com_num
        
        self.ftencoder = FTEncoder(input_size, ft_hid_size, alpha, pattern)
        self.lstm0 = LSTMEncoder(ft_hid_size, lstm_hid_size, num_layers)
        self.lstm2 = LSTMEncoder(ft_hid_size, lstm_hid_size, num_layers)
        self.irencoder = IREncoder(lstm_hid_size, mlp_hid_size, gcn_hid_size, drop, com_num, training)
        
        self.att_fc = nn.Linear(lstm_hid_size, lstm_hid_size)
        self.fc1 = nn.Linear(2*lstm_hid_size, out_hid_size)
        self.fc2 = nn.Linear(out_hid_size, num_keys)
        
        self.u_att = Variable(torch.zeros(1, lstm_hid_size), requires_grad=True).to(self.device)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.u_att, gain=nn.init.calculate_gain('relu'))
        print('Variable inited.')
        
    def resolve(self, per_x, per_index):
        res = collections.OrderedDict()
        for idx in range(per_x.shape[0]):
            if per_index[idx].item() not in res.keys():
                res[per_index[idx].item()] = []
            res[per_index[idx].item()].append(per_x[idx].to(self.device))
        return res
    
    def attention_net(self, x):
        sequence_len = x.shape[1]
        re_x = x.reshape(-1, self.lstm_hid_size)
        re_x = torch.mm(re_x, self.u_att.T).reshape(-1, sequence_len)
        re_x = f.softmax(re_x, dim=1).unsqueeze(-1)
        
        x = torch.sum(x * re_x, 1)
        x = f.relu(self.att_fc(x))
        return x
    
    def forward(self, x, index, q_x, t_x):
        x = self.ftencoder((x, t_x))
        batch_as_x = self.lstm0(x)
        
        batch_ac_x = []
        for idx in range(x.shape[0]):
            res = self.resolve(x[idx], index[idx])
            ac_x = []
            for item in res.items():
                list_x = torch.stack(item[1]).unsqueeze(0)
                out_x = self.lstm2(list_x)
                ac_x.append(out_x.squeeze(0))
            ac_x = torch.stack(ac_x)
            
            if ac_x.shape[0] != 1:
                ac_x = self.irencoder(ac_x, list(res.keys()))
            
            ac_x = self.attention_net(ac_x.unsqueeze(0))
            batch_ac_x.append(ac_x)
        
        batch_ac_x = torch.stack(batch_ac_x).squeeze(1)
        multi_out = torch.cat((batch_as_x, batch_ac_x), -1)
        out = f.relu(self.fc1(multi_out))
        out = self.fc2(out)
        return out


def getDateTimeFromISO8601String(s):
    d = dateutil.parser.parse(s, yearfirst=True)
    return d


def generate_train(name, train_path, logTemp_path, encoder_path, com_path, window_size):
    """Generate training dataset with memory-efficient processing"""
    print(f"Loading training data from {train_path}...")
    train_datas = pd.read_csv(train_path, engine='c', na_filter=False, memory_map=True)
    logTemp = pd.read_csv(logTemp_path, index_col='EventId', engine='c', na_filter=False, memory_map=True)
    mapping = {index: i for i, index in enumerate(logTemp.index.unique())}
    emb = json.load(open(encoder_path, 'r'))
    cop = json.load(open(com_path, 'r'))
    num_keys = len(logTemp.index.unique())
    
    # Process in chunks to avoid memory issues
    chunk_size = 100000  # Process 100K sequences at a time
    all_tensors = []
    total_samples = 0
    attr_num = None
    
    print(f"Processing {len(train_datas)} sequences in chunks of {chunk_size}...")
    
    for chunk_start in range(0, len(train_datas), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(train_datas))
        chunk_data = train_datas.iloc[chunk_start:chunk_end]
        
        inputs, outputs = [], []
        for idx, row in tqdm(chunk_data.iterrows(), total=len(chunk_data), 
                            desc=f"Processing chunk {chunk_start//chunk_size + 1}/{(len(train_datas)-1)//chunk_size + 1}"):
            seqs = eval(row['EventSequence'])
            len_seq = len(seqs)
            # Need at least window_size + 1 events to create input/output pairs
            if len_seq > window_size:
                inputs.extend([seqs[i:i + window_size] for i in range(len_seq - window_size)])
                outputs.extend([mapping[seqs[i + window_size][0]] for i in range(len_seq - window_size)])
            # If sequence length equals window_size, use the whole sequence as input
            # and predict the last event (self-prediction for training)
            elif len_seq == window_size:
                inputs.append(seqs)
                outputs.append(mapping[seqs[-1][0]])
        
        if not inputs:
            continue
            
        inputs_encoded, coms_encoded, quans_encoded, time_encoded = [], [], [], []
        for idx, events in enumerate(tqdm(inputs, desc=f"Encoding chunk {chunk_start//chunk_size + 1}")):
            quan_pattern = [0] * num_keys
            log_counter = Counter([mapping[event] for event, _, _ in events])
            for key in log_counter:
                quan_pattern[key] = log_counter[key]
            quans_encoded.append(quan_pattern)
            
            inp, com, tm = [], [], []
            start_time = getDateTimeFromISO8601String(events[0][2])
            for event, component, time in events:
                cur_time = getDateTimeFromISO8601String(time)
                inp.append(emb[event])
                com.append(cop[str(component)] if str(component) in cop else cop.get(component, -1))
                tm.append((cur_time - start_time).seconds)
                
            inputs_encoded.append(inp)
            coms_encoded.append(com)
            time_encoded.append(tm)
        
        # Store attr_num from first chunk
        if attr_num is None:
            attr_num = len(inputs_encoded[0][0])
        
        # Convert to tensors and store
        chunk_tensors = (
            torch.as_tensor(inputs_encoded, dtype=torch.float),
            torch.as_tensor(coms_encoded),
            torch.as_tensor(quans_encoded, dtype=torch.float),
            torch.as_tensor(time_encoded, dtype=torch.float),
            torch.as_tensor(outputs)
        )
        all_tensors.append(chunk_tensors)
        total_samples += len(inputs_encoded)
        
        print(f"Chunk {chunk_start//chunk_size + 1} processed: {len(inputs_encoded)} samples (Total: {total_samples})")
        
        # Clear memory
        del inputs, outputs, inputs_encoded, coms_encoded, quans_encoded, time_encoded
        import gc
        gc.collect()
    
    # Concatenate all chunks
    print("Concatenating all chunks...")
    final_tensors = tuple(torch.cat([chunk[i] for chunk in all_tensors], dim=0) for i in range(5))
    dataset = TensorDataset(*final_tensors)
    
    print(f'Number of {name}_seqs: {len(dataset)}, components: {len(cop)}')
    
    return dataset, attr_num, len(logTemp.index.unique()), len(cop)


def generate_pre(name, log_path, logTemp_path, encoder_path, com_path, window_size):
    """Generate validation/test dataset"""
    print(f"Loading {name} data from {log_path}...")
    pre_data = pd.read_csv(log_path, engine='c', na_filter=False, memory_map=True)
    logTemp = pd.read_csv(logTemp_path, index_col='EventId', engine='c', na_filter=False, memory_map=True)
    mapping = {index: i for i, index in enumerate(logTemp.index.unique())}
    emb = json.load(open(encoder_path, 'r'))
    emb_len = len(list(emb.items())[0][1])
    cop = json.load(open(com_path, 'r'))
    num_keys = len(logTemp.index.unique())
    
    inputs = []
    for idx, row in tqdm(pre_data.iterrows(), total=len(pre_data), desc=f"Processing {name}"):
        seqs = eval(row['EventSequence'])
        len_seq = len(seqs)
        inp, comp, quanp, timep, lab = [], [], [], [], []
        for i in range(len(seqs) - window_size):
            seq, com, tm = [], [], []
            
            quan_pattern = [0] * num_keys
            log_counter = Counter([mapping[event] for event, _, _ in seqs[i:i + window_size]])
            for key in log_counter:
                quan_pattern[key] = log_counter[key]
            quanp.append(quan_pattern)
            
            start_time = getDateTimeFromISO8601String(seqs[i:i + window_size][0][2])
            for event in seqs[i:i + window_size]:
                cur_time = getDateTimeFromISO8601String(event[2])
                seq.append([-1]*emb_len) if event[0] == -1 else seq.append(emb[event[0]])
                com.append(-1) if event[0] == -1 else com.append(cop[str(event[1])] if str(event[1]) in cop else cop.get(event[1], -1))
                tm.append(-1) if event[0] == -1 else tm.append((cur_time - start_time).seconds)
                
            inp.append(seq)
            comp.append(com)
            timep.append(tm)
            lab.append(mapping[seqs[i + window_size][0]] if seqs[i + window_size] != -1 else -1)
        if inp:
            inputs.append((inp, comp, quanp, timep, lab))
    
    print(f'Number of {name}_seqs(session): {len(inputs)}')
    return inputs, len(inputs), len(cop)


def evaluation(output, label, valid_loss, pattern='macro'):
    accuracy = accuracy_score(label, output)
    precision, recall, F1, _ = precision_recall_fscore_support(label, output, average=pattern)
    return accuracy, precision, recall, F1, np.average(valid_loss)


def eval_handle_topK(nordl, anodl, model, window_size, num_candidates, anomaly_rate=1, device='cuda'):
    nor_hit = dict()
    ano_hit = dict()
    total_loss = []
    with torch.no_grad():
        model.eval()
        for seq, com, quan, timp, label in nordl:
            assert len(seq) == len(label), 'seqs len not equal labels len'
            seq = torch.as_tensor(seq, dtype=torch.float).to(device)
            quan = torch.as_tensor(quan, dtype=torch.float).to(device)
            com = torch.as_tensor(com).to(device)
            timp = torch.as_tensor(timp, dtype=torch.float).to(device)
            label = torch.as_tensor(label).to(device)
            
            output = model(seq, com, quan, timp).to(device)
            
            for num_can in num_candidates:
                if num_can not in nor_hit.keys():
                    nor_hit[num_can] = []
                indice = torch.argsort(output, 1, descending=True)[:, 0:num_can].contiguous()
                fcnt = (torch.isin(label, indice) == False).sum().item()
                nor_hit[num_can].append(1) if fcnt >= anomaly_rate else nor_hit[num_can].append(0)

        for seq, com, quan, timp, label in anodl:
            assert len(seq) == len(label), 'seqs len not equal labels len'
            seq = torch.as_tensor(seq, dtype=torch.float).to(device)
            quan = torch.as_tensor(quan, dtype=torch.float).to(device)
            com = torch.as_tensor(com).to(device)
            timp = torch.as_tensor(timp, dtype=torch.float).to(device)
            label = torch.as_tensor(label).to(device)
            
            output = model(seq, com, quan, timp).to(device)
            
            for num_can in num_candidates:
                if num_can not in ano_hit.keys():
                    ano_hit[num_can] = []
                indice = torch.argsort(output, 1, descending=True)[:, 0:num_can].contiguous()
                fcnt = (torch.isin(label, indice) == False).sum().item()
                ano_hit[num_can].append(1) if fcnt >= anomaly_rate else ano_hit[num_can].append(0)
            
    nor_label = [0]*len(nor_hit[num_candidates[0]])
    ano_label = [1]*len(ano_hit[num_candidates[0]])
    
    res = dict()
    for num_can in num_candidates:
        if num_can not in res.keys():
            res[num_can] = []
        res[num_can].append(evaluation(nor_hit[num_can] + ano_hit[num_can], nor_label + ano_label, np.average(total_loss)))
    return res


def main():
    parser = argparse.ArgumentParser(description='CSCLog Training Script')
    parser.add_argument('--data_dir', type=str, default='dataset/processed',
                       help='Directory containing preprocessed data')
    parser.add_argument('--model_dir', type=str, default='model/CSCLog',
                       help='Directory to save trained model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--window_size', type=int, default=9, help='Window size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    seed_torch(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    window_size = args.window_size
    
    # Model hyperparameters
    hidden_size = [64, 64, 64, 64, 64]  # ft_hid, lstm_hid, mlp_hid, gcn_hid, out_hid
    alpha = 0.8
    pattern = 1
    num_layers = 2
    drop = 0.1
    num_candidates = [1]
    anomaly_rate = 1
    
    # Data paths
    data_dir = args.data_dir
    train_path = os.path.join(data_dir, 'train_normal.csv')
    test_normal_path = os.path.join(data_dir, 'test_normal.csv')
    test_anomaly_path = os.path.join(data_dir, 'test_anomaly.csv')
    temp_path = os.path.join(data_dir, 'log_templates.csv')
    emb_path = os.path.join(data_dir, 'sentences_emb.json')
    com_path = os.path.join(data_dir, 'component.json')
    
    # Check if files exist
    for path in [train_path, test_normal_path, temp_path, emb_path, com_path]:
        if not os.path.exists(path):
            print(f"Error: Required file not found: {path}")
            print("Please run preprocessing first: python run_preprocessing.py")
            return 1
    
    print("="*80)
    print("CSCLog Training")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Window size: {window_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print("="*80)
    
    # Load data
    print("\nLoading training data...")
    train_dataset, attr_num, class_num, com_num = generate_train(
        'train', train_path, temp_path, emb_path, com_path, window_size
    )
    # Optimize DataLoader with num_workers and pin_memory
    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True,  # Faster CPU->GPU transfer
        num_workers=4,     # Parallel data loading
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    print("\nLoading test data...")
    # Load test_normal which may contain both normal and anomaly sequences
    test_data_all, test_len, _ = generate_pre(
        'test', test_normal_path, temp_path, emb_path, com_path, window_size
    )
    
    # Separate normal and anomaly from test set based on labels
    # test_data_all is a DataLoader, we need to separate by checking labels
    label_normal = []
    label_anomaly = []
    
    # Check if separate anomaly file exists
    if os.path.exists(test_anomaly_path):
        print("Loading separate test_anomaly.csv...")
        label_normal = test_data_all
        label_anomaly, anomal_len, _ = generate_pre(
            'test_anomaly', test_anomaly_path, temp_path, emb_path, com_path, window_size
        )
    else:
        print("No separate test_anomaly.csv found")
        print("Note: test_normal.csv should contain both normal (Label=0) and anomaly (Label=1) sequences")
        # For now, use all test data as normal for training
        # Evaluation will be done separately after training
        label_normal = test_data_all
        label_anomaly = []
    
    # Create model
    print("\nCreating model...")
    model = Model(attr_num, com_num, hidden_size, alpha, pattern,
                 num_layers, class_num, drop, True).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    best_f1 = 0
    best_epoch = 0
    best_state = None
    
    for epoch in range(num_epochs):
        train_loss = []
        model.train()
        
        pbar = tqdm(dataloader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        for step, (seq, com, quan, timp, label) in enumerate(pbar):
            optimizer.zero_grad()
            
            seq = seq.clone().detach().to(device)
            com = com.clone().detach().to(device)
            quan = quan.clone().detach().to(device)
            timp = timp.clone().detach().to(device)
            
            output = model(seq, com, quan, timp).to(device)
            loss = criterion(output, label.to(device))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
            pbar.set_postfix({'loss': f'{np.mean(train_loss):.4f}'})
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], train_loss: {np.average(train_loss):.4f}')
        
        # Validation
        if label_anomaly:
            res = eval_handle_topK(label_normal, label_anomaly, model, window_size, num_candidates, anomaly_rate, device)
            for item in res.items():
                accuracy, precision, recall, F1, _ = item[1][0]
                print(f"TopK={item[0]} | Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {F1:.3f}")
                
                if F1 > best_f1:
                    best_f1 = F1
                    best_epoch = epoch
                    best_state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': best_epoch,
                        'f1': best_f1
                    }
        else:
            print("Skipping validation (no anomaly data)")
    
    print(f"\nBest epoch: {best_epoch + 1} / Best F1: {best_f1:.3f}")
    
    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, 'CSCLog.pt')
    
    if best_state:
        torch.save(best_state, model_path)
        print(f"\nModel saved to: {model_path}")
    else:
        # Save last epoch if no validation
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': num_epochs - 1
        }, model_path)
        print(f"\nModel saved to: {model_path}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
