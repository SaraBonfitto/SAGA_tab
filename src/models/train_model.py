from logging import root
import torch
from re import I
from sklearn.linear_model import OrthogonalMatchingPursuit
from torch_geometric.data import HeteroData
import torch
from torch_geometric.nn import HeteroConv, to_hetero, GATConv
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score
import numpy as np 
from torch_geometric.data import HeteroData
import configparser
import os
from langdetect import detect
from dateutil.parser import parse
import datetime, string
from torch_geometric.utils import remove_self_loops, remove_isolated_nodes
from definitions import ROOT_DIR
import src.data.utils as utils

config = configparser.ConfigParser()
config.read(os.path.join(ROOT_DIR, 'config.ini'))

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, edge_types):
        super().__init__()
        self.conv1 = HeteroConv({edge_t: GATConv((-1, -1),hidden_channels) for edge_t in edge_types})
        self.conv2 = HeteroConv({edge_t: GATConv((-1, -1),out_channels) for edge_t in edge_types})
        self.rel_weight = torch.nn.Parameter(torch.randn(len(edge_types), out_channels))

    def forward(self, x_dict, edge_index_dict, data, rel_to_index, edge_types, root_nodes_types):
        
        x_dict = self.conv1(x_dict, edge_index_dict)

        for t,v in root_nodes_types.items():
            x_dict[t] = v
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        for t,v in root_nodes_types.items():
            x_dict[t] = v 
            
        out = x_dict
    
        pred_dict = {}
        ### LINK PREDICTION ACTS HERE ###
        for edge_t in edge_types:
            #Compute link embedding for each edge type
            #for src in train_link[edge_t].edge_label_index[0]:
            out_src = out[edge_t[0]][data[edge_t].edge_label_index[0]]#embedding src nodes
            out_dst = out[edge_t[2]][data[edge_t].edge_label_index[1]] #embedding dst nodes
        
            # LINK EMBEDDING #
            # 1 - Dot Product
            #out_sim = out_src * out_dst #dotproduct
            #pred = torch.sum(out_sim, dim=-1)
        
            # 2- Distmult with random initialized rel weights
            #r = torch.Tensor([self.rel_weight[rel_to_index[edge_t]].detach().numpy() for z in range(len(out_src))])
            out_sim = torch.sum(out_src * self.rel_weight[rel_to_index[edge_t]] * out_dst, dim=-1)
            pred = out_sim
        
            pred_dict[edge_t] = pred
        return pred_dict


def get_model(train_link, rel_to_index, edge_types, root_nodes_types):
    model = HeteroGNN(hidden_channels=4, out_channels=2, 
                    edge_types=edge_types)  
         
    with torch.no_grad():  # Initialize lazy modules.
        out = model(train_link.x_dict,train_link.edge_index_dict, train_link,rel_to_index, edge_types, root_nodes_types)
        
 
    weight_decay=5e-4
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay = weight_decay)
    criterion =  torch.nn.BCEWithLogitsLoss() #change loss function
    return model, out, optimizer, criterion
    
def create_data( entity_types_count, subject_dict, object_dict, properties_and_types = {}, property_types_count = {},max_min={}):
    data = HeteroData()
    data_to_insert = {}
    for subj in list(properties_and_types.keys()):
        for class_type, prop_name, prop_type, prop_value in properties_and_types[subj]:
            if prop_type not in data_to_insert:
                data_to_insert[prop_type] = []
            data_to_insert[prop_type].append(function_build_feature(class_type, prop_name, prop_type, prop_value, max_min))

    types = list(entity_types_count.keys())
    for t in types:
        data_to_insert[t] = [[1] for i in range(entity_types_count[t])]

    for key in data_to_insert.keys():
        lists = data_to_insert[key]
        t_key = utils.get_type(key)
        if lists != '':
            data[t_key].x = torch.Tensor(lists)
            
    for triple in subject_dict.keys():  #le keys di subject_dict ed object_dict sono ==
        #ci sono 2 dizionari diversi per evitare il doppio for (complessità n^2)
        lol = [subject_dict[triple], object_dict[triple]]
        if len(lol[0]) > 10:
            data[utils.get_type(triple[0]), utils.get_type(triple[1]), utils.get_type(triple[2])].edge_index = torch.Tensor(lol).long()

    '''
    for debug purposes
    for k,v in data.edge_index_dict.items():
        max_ind_sx = int(max(v[1]))
        try:
            n1 = data[k[2]].x[max_ind_sx]
        except IndexError:
            print("Relation:", k, " node type:", k[2], " index:", max_ind_sx, f"{k[2]} matrix dimension:", len(data_to_insert[k[2]].x))
    '''    
    
    return data

def remove_isolated(data):

    for edge_type in data.edge_index_dict.keys():
        if edge_type[0] == edge_type[2]:
            new_edge_index = remove_self_loops(data[edge_type].edge_index)[0]
            data[edge_type].edge_index = new_edge_index
        new_edge_index = remove_isolated_nodes(data[edge_type].edge_index)[0]
        data[edge_type].edge_index = new_edge_index
    return data

def get_root_node_types(data, edge_types):
    root_nodes_types = {}
    for node_type in data.x_dict.keys():
        i = 0
        for edge_t in edge_types:
            if node_type == edge_t[2]: break 
            i+=1
        if i == len(edge_types):
            root_nodes_types[node_type] = data.x_dict[node_type]
    return root_nodes_types

def split_dataset(data, edge_types):
    link_split = RandomLinkSplit(num_val=0.0,
                                num_test=0.25,
                                edge_types=edge_types,
                                rev_edge_types=[None]*len(edge_types))
    train_link, val_link, test_link = link_split(data)
    return train_link, val_link, test_link

def train_hetlinkpre(model, optimizer, criterion, train_link, rel_to_index, edge_types, root_nodes_types):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    pred_dict = model(train_link.x_dict, train_link.edge_index_dict, train_link, rel_to_index, edge_types, root_nodes_types)  # Perform a single forward pass.
    edge_labels = torch.Tensor()
    preds = torch.Tensor()
    for edge_t in edge_types:
        preds = torch.cat((preds,pred_dict[edge_t]),-1)
        edge_labels = torch.cat((edge_labels,train_link[edge_t].edge_label.type_as(pred_dict[edge_t])),-1)
    #compute loss function based on all edge types
    loss = criterion(preds, edge_labels)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss



def test_hetlinkpre(model, test_link, rel_to_index, edge_types, root_nodes_types, evaluate='linkpre'):
    if evaluate not in ['linkpre','propdetection','all']:
        #linkpre: link tra entità
        #propdetection: link tra entità e prop 
        #all: entrambi 
        raise NotImplementedError()
    model.eval()
    hs_dict = model(test_link.x_dict, test_link.edge_index_dict, test_link, rel_to_index, edge_types, root_nodes_types)
    hs = torch.Tensor([])
    edge_labels = np.array([])
    ### LINK PREDICTION ACTS HERE ###
    #evaluate distincly entity-to-entity link prediction and entity-to-property(property-detection)
    prop = ['String','Integer','Double','gYear','Date'] #add other property types if used
    rel_with_prop = [edge_t for edge_t in edge_types if edge_t[2] in prop]
    if evaluate == 'linkpre':
        edge_types_to_evaluate = [edge_t for edge_t in edge_types if edge_t not in rel_with_prop]
    elif evaluate == 'propdetection':
        edge_types_to_evaluate = rel_with_prop
    else:
        edge_types_to_evaluate = edge_types
    for edge_t in edge_types_to_evaluate:
        hs = torch.cat((hs,hs_dict[edge_t]),-1)
        edge_labels = np.concatenate((edge_labels,test_link[edge_t].edge_label.cpu().detach().numpy()))
    
    toRead_preds = []
    for edge_t in test_link.edge_index_dict.keys():
        toRead_preds.append((edge_t,len(hs_dict[edge_t]))) 
        hs = torch.cat((hs,hs_dict[edge_t]),-1)
        edge_labels = np.concatenate((edge_labels,test_link[edge_t].edge_label.cpu().detach().numpy()))
    
    pred_cont = torch.sigmoid(hs).cpu().detach().numpy()
    # EVALUATION
    test_roc_score = roc_auc_score(edge_labels, pred_cont) #comput AUROC score for test set
    
    return test_roc_score, pred_cont, toRead_preds

def test_hetscores(model,data, test_link, rel_to_index, edge_types, root_nodes_types):
    model.eval()
    hs_dict = model(data.x_dict, data.edge_index_dict, data, rel_to_index, edge_types, root_nodes_types)
    edge_labels = np.array([])
    toRead_preds = []
    hs = torch.Tensor([])
    for edge_t in data.edge_index_dict.keys():
        toRead_preds.append((edge_t,len(hs_dict[edge_t]))) 
        hs = torch.cat((hs,hs_dict[edge_t]),-1)
        edge_labels = np.concatenate((edge_labels,test_link[edge_t].edge_label.cpu().detach().numpy()))
    
    pred_cont = torch.sigmoid(hs).cpu().detach().numpy()
    
    return pred_cont, toRead_preds

def train_and_save(model, optimizer, criterion, train_link, test_link, rel_to_index, edge_types, root_nodes_types):
    epochs = 500
    '''
    for epoch in range(1,epochs):
        loss = train_hetlinkpre(model, optimizer, criterion, train_link, rel_to_index, edge_types, root_nodes_types)
    '''
    #perf_train = []
    #perf_test = []
    
    for epoch in range(epochs):
        loss = train_hetlinkpre(model, optimizer, criterion, train_link, rel_to_index, edge_types, root_nodes_types)
        roc_train = test_hetlinkpre(model, train_link, rel_to_index, edge_types, root_nodes_types, evaluate='all')
        roc_test = test_hetlinkpre(model, test_link, rel_to_index, edge_types, root_nodes_types, evaluate='all')
        
        #perf_train.append(roc_train)
        #perf_test.append(roc_test)
    #torch.save(model.state_dict(), config['model']['path'])
    torch.save(model.state_dict(), ROOT_DIR+config['model']['path'])

def function_build_feature(class_type, prop_name, p_type, value, max_min):
    #return [5] così funziona perchè è numerico

    #aggiungere funzione x riconscere le date
    if p_type == 'Integer':
        try: i = int(value) 
        except: i = 0
        if (class_type, prop_name, p_type) in max_min:
            max, min = max_min[(class_type, prop_name, p_type)]
            return [i, max, min]
        return [i]
    if p_type == 'Double':
        try: d = float(value)
        except: d = float(0.0)
        if (class_type, prop_name, p_type) in max_min:
            max, min = max_min[(class_type, prop_name, p_type)]
            return [d, float(max), float(min)]
        return [d]
    if p_type == 'gYear':
        return [int(1970-value)]
    if p_type == 'String':
        count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
        a_punct = count(value, string.punctuation)
        lang = 0
        try:
            if detect(value) == 'en': lang = 1
        except:
            lang = 0
        return [len(value), value.count(" ") , value.count("(") + value.count(")"), lang, a_punct]

    if p_type == 'Date':
        days = (parse(value) - datetime.datetime(1970,1,1)).days
        weekday = parse(value).weekday()
        month = parse(value).strftime("%m")
        if (class_type, prop_name, p_type) in max_min:
            max, min = max_min[(class_type, prop_name, p_type)]
            max_days =  (max - datetime.datetime(1970,1,1)).days
            min_days =  (min - datetime.datetime(1970,1,1)).days

            return [days, weekday, int(month), max_days, min_days]
        return [days, weekday, int(month)]
    return ""

