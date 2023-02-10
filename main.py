from cgi import test
from src.data.graph_modelling.functions import draw_result
import src.data.make_dataset as make_dataset
import src.data.utils as utils
import src.models.train_model as train_model 
import src.models.predict_model as predict_model
import src.data.graph_modelling.semantic_model_class as semantic_model_class
import src.data.graph_modelling.approximation as approximation
import os
import configparser
import networkx as nx
import pickle
import rdflib
from torch_geometric.data import HeteroData
import torch
from torch_geometric.utils import remove_self_loops, remove_isolated_nodes
from definitions import ROOT_DIR
import random
import numpy as np
 
def data_preprocessing(knowledge, use_properties = False):
    dataset = make_dataset.MakeDataset()

    #get triples from graph (properties can be included or not) and determine a data type for each node
    triples = dataset.set_entities_and_type(knowledge, use_properties)

    #removes multiple data types associated with a node 
    new_triples = dataset.clean_triples(triples)

    #returns the number of nodes per type
    entity_types_count, property_types_count = dataset.get_count()
    
    subject_dict, object_dict, properties_and_types = dataset.get_subject_object(new_triples, entity_types_count, property_types_count)

    max_min = {}
    if use_properties:
        max_min = dataset.get_max_min(properties_and_types)

    hetero_data = train_model.create_data(entity_types_count, subject_dict, object_dict, properties_and_types, property_types_count, max_min)
    
    return hetero_data, max_min

def model_training(hetero_data):
    edge_types = list(hetero_data.edge_index_dict.keys())
    hetero_data = train_model.remove_isolated(hetero_data)
    root_nodes_types = train_model.get_root_node_types(hetero_data, edge_types)
    rel_to_index = {edge_t:i for edge_t,i in zip(edge_types,range(len(edge_types)))}
    train_link, val_link, test_link = train_model.split_dataset(hetero_data, edge_types)
    
    node_types = list(hetero_data.x_dict.keys()) 
    node_sizes = {} #dictionary with mapping node type: in_channels size. Useful to not use lazy inizialization, #to allow reproducibility
    for k in node_types:
        node_sizes[k] = len(hetero_data[k].x[0])

    model, out, optimizer, criterion = train_model.get_model(train_link, rel_to_index, edge_types, root_nodes_types, node_sizes)
    train_model.train_and_save(model, optimizer, criterion, train_link, test_link, rel_to_index, edge_types, root_nodes_types)

    roc_train,pred_cont0, toRead_preds0 = train_model.test_hetlinkpre(model, train_link, rel_to_index, edge_types, root_nodes_types, 'all')
    roc_test,pred_cont, toRead_preds = train_model.test_hetlinkpre(model, test_link, rel_to_index, edge_types, root_nodes_types, 'all')
    
    relation_weights = {}
    for triple,occ in toRead_preds:
        index = toRead_preds.index((triple,occ))
        
        if occ == 0:
            relation_weights[(triple)] = 0
        else:
            relation_weights[(triple)] = pred_cont[index + (occ-1)]


    print(f'Train AUROC: {roc_train:.4f}\nTest AUROC: {roc_test:.4f}')
    return edge_types, relation_weights


def main():
    #Set all seeds to manual for reproducibility
    device = torch.device('cuda')
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    
    #import config file
    config = configparser.ConfigParser()
    config.read(os.path.join(ROOT_DIR, 'config.ini'))

    #if properties of a node should be used for training the GNN
    use_properties = config["kg"]["use_properties"]

    if use_properties.lower() in ['true', '1', 't', 'y', 'yes']: use_properties = True 
    else: use_properties = False 

    #definition of the output path
    path_image =ROOT_DIR+ config["semantic_model_csv"]["output_path"]+config["semantic_model_csv"]["num_experiment"]

    #import of the training KG
    knowledge_graph = rdflib.Graph()
    knowledge_graph.parse(ROOT_DIR+config["kg"]["path"], format=config["kg"]["format"])

    #definition of the starting semantic model
    semantic_model = semantic_model_class.SemanticModelClass()
    sm = semantic_model.parse()
    Er, Et = semantic_model.get_complete_semantic_model(sm, use_properties)

    #data preprocessing
    max_min = {}
    graph_test, not_instances = utils.triples_to_heterodata(Er,Et, max_min)
    hetero_data, max_min = data_preprocessing(knowledge_graph, use_properties)

    #save min_max in a file, should be commented after the first execution
    with open(ROOT_DIR+ config["model"]["max_min"]+'max_min.pkl', 'wb') as f:
        pickle.dump(max_min, f)
    
    #uncomment the following to load the previously saved min_max
    #with open(ROOT_DIR+ config["model"]["max_min"]+'max_min.pkl', 'rb') as f:
    #    max_min = pickle.load(f)
    

    #removing the triples not seen in the training test
    node_types = list(hetero_data.x_dict.keys())
    edge_types = list(hetero_data.edge_index_dict.keys())    
    edge_diff = set(graph_test.edge_index_dict.keys()).difference(set(hetero_data.edge_index_dict.keys()))

    graph_test_clean= HeteroData()
    for k,v in graph_test.x_dict.items():
        if k in node_types:
            graph_test_clean[k].x = v
    for k,v in graph_test.edge_index_dict.items():
        if k not in edge_diff:
            graph_test_clean[k].edge_index = v
            graph_test_clean[k].edge_label = torch.Tensor([1 for i in range(len(v[0]))])
   
    edge_types, relation_weights = model_training(hetero_data)


    root_nodes_types = {}
    for node_type in graph_test_clean.x_dict.keys():
        i = 0
        for edge_t in graph_test_clean.edge_index_dict.keys():
            if node_type == edge_t[2]: break 
            i+=1
        if i == len(graph_test_clean.edge_index_dict.keys()):
            root_nodes_types[node_type] = graph_test_clean.x_dict[node_type]
    
    #print(graph_test_clean)
    #Remove the comment to save the trained model
    #afile = open(ROOT_DIR+config["model"]["pickle_path"]+"edge_types.pkl", 'wb')
    #pickle.dump(edge_types, afile)

    #remove the comment to load the previously trained model
    #afile = open(ROOT_DIR+ config["model"]["pickle_path"]+"edge_types.pkl", 'rb')
    #edge_types = pickle.load(afile)
    #afile.close()


    for edge_t in edge_types:
        if edge_t not in graph_test_clean.edge_index_dict.keys():
            graph_test_clean[edge_t].edge_index = torch.Tensor([[],[]]).long()
            graph_test_clean[edge_t].edge_label = torch.Tensor([]).long()


    for edge_type in graph_test_clean.edge_index_dict.keys():
        if edge_type[0] == edge_type[2]:
            new_edge_index = remove_self_loops(graph_test_clean[edge_type].edge_index)[0]
            graph_test_clean[edge_type].edge_index = new_edge_index.long()
            graph_test_clean[edge_type].edge_label = torch.Tensor([1 for i in range(len(graph_test_clean[edge_type].edge_index[0]))]).long()
    
        new_edge_index = remove_isolated_nodes(graph_test_clean[edge_type].edge_index)[0]
        graph_test_clean[edge_type].edge_index = new_edge_index.long()
        graph_test_clean[edge_type].edge_label = torch.Tensor([1 for i in range(len(graph_test_clean[edge_type].edge_index[0]))]).long()
        
    node_types = list(graph_test_clean.x_dict.keys()) 
    node_sizes = {} #dictionary with mapping node type: in_channels size. Useful to not use lazy inizialization
    for k in node_types:
        node_sizes[k] = len(graph_test_clean[k].x[0])
    
    pred_cont, toRead_preds = predict_model.test_hetscores2(graph_test_clean, edge_types, root_nodes_types, node_sizes)
    additional = 0
    for triple,occ in toRead_preds:
        index = toRead_preds.index((triple,occ))
        
        if occ == 0:
            relation_weights[(triple)] = 0
            additional = 1
        else:
            print(index + (occ-1)- additional)
            relation_weights[(triple)] = pred_cont[index + (occ-1)- additional]

    print(relation_weights)
    if use_properties:
        Er, type_leafs = semantic_model.infer_property(relation_weights, Er, Et, semantic_model.get_leafs(), max_min)

    #convertion from rdf triples to a HeteroData object
    
    relation_weights = utils.get_weight_for_triples(not_instances, relation_weights)

    pred_cont, toRead_preds = predict_model.test_hetscores2(graph_test_clean, edge_types, root_nodes_types, node_sizes)
    additional = 0
    for triple,occ in toRead_preds:
        index = toRead_preds.index((triple,occ))
        
        if occ == 0:
            relation_weights[(triple)] = 0
            additional = 1
        else:
            print(index + (occ-1)- additional)
            relation_weights[(triple)] = pred_cont[index + (occ-1)- additional]


    #creating the complete SD graph
    sd = nx.MultiGraph()

    for e in Er:
        sd.add_node(e[0])
        sd.add_node(e[2])

        rel_type = utils.get_type(e[1])
        lw = rel_type + " - " + str(e[3])

        try:
            int(e[3])
        except:
            lw = rel_type + " - " + str(e[3]["weight"])
        
        sd.add_edge(e[0],e[2], label = e[1], weight = e[3], lw = lw)


    #set weights on edges
    weights = semantic_model.update_graph_weights(sd, relation_weights, False)
    
    #relation_weights = predict_model.get_relations_weights(test_data, hetero_data)

    undirected = sd.to_undirected()

    for edge in undirected.edges:
        u = edge[0]
        v = edge[1]
        relations = undirected.get_edge_data(u,v)
    
    #adding unmatched columns
    leafs =(semantic_model.get_leafs())
    for leaf in type_leafs:
        if leaf not in leafs:
            leafs.append(leaf)
    
    #STEINER TREE
    tree = approximation.steiner_tree(weights, leafs, weight='weight')

    #refined_graph = semantic_model.add_important_edges(both_weights, both_weights_tree)

    #uncomment for DRAWING CLOSURE
    #for edge in sd.edges(data=True): edge[2]['label'] = edge[2]['lw']
    #for edge in weights.edges(data=True): edge[2]['label'] = edge[2]['lw']
    #weights = nx.relabel_nodes(weights, lambda x: utils.get_type(x))

    semantic_model.draw_result(weights, path_image)

    #DRAW RESULT
    for edge in tree.edges(data=True): edge[2]['label'] = edge[2]['lw']
    tree = nx.relabel_nodes(tree, lambda x: utils.get_type(x))
    semantic_model.draw_result(tree, path_image + "_tree")

main()
