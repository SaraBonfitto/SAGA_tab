from torch_geometric.data import HeteroData
import torch
from torch_geometric.nn import HeteroConv, to_hetero, GATConv
import configparser
import os
import numpy as np
from definitions import ROOT_DIR

config = configparser.ConfigParser()
config.read(os.path.join(ROOT_DIR, 'config.ini'))

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, edge_types, node_sizes, root_nodes_types):
        super().__init__()
        g = torch.manual_seed(0)
        self.node_sizes = node_sizes
        self.edge_types = edge_types
        self.root_nodes_types = root_nodes_types
        self.conv1 = HeteroConv({edge_t: GATConv((node_sizes[edge_t[0]], node_sizes[edge_t[2]]),
                                                 hidden_channels,add_self_loops=False) for edge_t in edge_types})
        self.conv2 = HeteroConv({edge_t: GATConv((hidden_channels if edge_t[0] not in root_nodes_types else node_sizes[edge_t[0]],
                                                  hidden_channels),
                                                 out_channels,add_self_loops=False) for edge_t in edge_types})
        self.rel_weight = torch.nn.Parameter(torch.randn(len(edge_types), out_channels, generator=g))
        
    def forward(self, x_dict, edge_index_dict, data, rel_to_index, edge_types, root_nodes_types):
        #print("In forward test")
        x_dict = self.conv1(x_dict, edge_index_dict)

        for t,v in root_nodes_types.items():
            x_dict[t] = torch.Tensor([[v[j][0] for i in range(self.hidden_channels)] for j in range(len(v))])
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        for t,v in root_nodes_types.items():
            x_dict[t] = torch.Tensor([[v[j][0] for i in range(self.out_channels)] for j in range(len(v))])
            
        out = x_dict
    
        pred_dict = {}
        
        ### LINK PREDICTION ACTS HERE ###
        for edge_t,edge_i in edge_index_dict.items():
            #Compute link embedding for each edge type
            #for src in train_link[edge_t].edge_label_index[0]:
            out_src = out[edge_t[0]][edge_i[0]]#embedding src nodes
            out_dst = out[edge_t[2]][edge_i[1]] #embedding dst nodes
        
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


def set_model(edge_types, root_nodes_types, node_sizes):
    model = HeteroGNN(hidden_channels=4, out_channels=2, edge_types=edge_types, root_nodes_types=root_nodes_types, node_sizes=node_sizes)       
    model.load_state_dict(torch.load(ROOT_DIR+config['model']['path']))
    return model

def test_hetscores(model, data, rel_to_index, edge_types, root_node_types):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict, data, rel_to_index, edge_types, root_node_types)

    ### LINK PREDICTION ACTS HERE ###
    
    hs = torch.Tensor()
    edge_labels = np.array([])
    toRead_preds = []
    ### LINK PREDICTION ACTS HERE ###
    
    for edge_t in data.edge_index_dict.keys():
        #Compute link embedding for each edge type
        #for src in train_link[edge_t].edge_label_index[0]:
        out_src = out[edge_t[0]][data[edge_t].edge_label_index[0]]#embedding src nodes
        out_dst = out[edge_t[2]][data[edge_t].edge_label_index[1]] #embedding dst nodes
        
        toRead_preds.append((edge_t,len(out_src))) 
        #col grafo semantico dovrebbe aiutarti a capire soltanto l'ordine dei tipi (ne hai solo uno)
        
        # LINK EMBEDDING #
        # 1 - Dot Product
        #out_sim = out_src * out_dst #dotproduct
        #h = torch.sum(out_sim, dim=-1)
        
        # 2 - Distmult with randn parameters
        r = torch.Tensor([model.rel_weight[rel_to_index[edge_t]].detach().numpy() for z in range(len(out_src))])
        out_sim = torch.sum(out_src * r * out_dst, dim=-1)
        h = out_sim
        
        hs = torch.cat((hs,h),-1)
        edge_labels = np.concatenate((edge_labels,data[edge_t].edge_label.cpu().detach().numpy()))
    
    
    pred_cont = torch.sigmoid(hs).cpu().detach().numpy()
    
    return pred_cont, toRead_preds

def get_relations_weights(test_data, hetero, root_nodes_types, node_sizes):
    edge_types = list(hetero.edge_index_dict.keys())
    rel_to_index = {edge_t:i for edge_t,i in zip(edge_types,range(len(edge_types)))}
    model = set_model(edge_types, root_nodes_types, node_sizes)
    weight = test_hetscores(model, hetero, rel_to_index)[0]
    #nella sd non possono esserci archi mai visti nel training

    return weight
    
def test_hetscores2(test_graph, edge_types, root_nodes_types, node_sizes):
    model = set_model(edge_types, root_nodes_types, node_sizes)
    model.eval()
    rel_to_index = {edge_t:i for edge_t,i in zip(edge_types,range(len(edge_types)))}
    hs_dict = model(test_graph.x_dict, test_graph.edge_index_dict, test_graph, rel_to_index, edge_types, root_nodes_types)
    edge_labels = np.array([])
    toRead_preds = []
    hs = torch.Tensor([]) 
    for edge_t in test_graph.edge_index_dict.keys():
        toRead_preds.append((edge_t,len(hs_dict[edge_t]))) 
        hs = torch.cat((hs,hs_dict[edge_t]),-1)
        edge_labels = np.concatenate((edge_labels,test_graph[edge_t].edge_label.cpu().detach().numpy()))
    
    pred_cont = torch.sigmoid(hs).cpu().detach().numpy()
    
    return pred_cont, toRead_preds