import webbrowser
from torch_geometric.data import HeteroData
import torch
import numpy as np
import src.models.train_model as train_model 
from dateutil.parser import parse
from definitions import ROOT_DIR
import configparser
import os

config = configparser.ConfigParser()
config.read(os.path.join(ROOT_DIR, 'config.ini'))

def triples_to_heterodata(Er, Et = [], max_min={}):
    data = HeteroData()
    data_to_insert ={}
    Er, entities_and_types, properties_and_types, triples, triples_prop = get_test_graph(Er, Et)
    entity_count = _entities_count(Er, entities_and_types)
    property_count = _properties_count(properties_and_types)

    triples, not_instances = get_triples_instances(triples, entities_and_types)

    subject_dict, object_dict = get_dicts(triples, triples_prop, entities_and_types, entity_count, property_count)

    data_to_insert = {}

    for subj in list(properties_and_types.keys()):
        for class_type, prop_name, prop_type, prop_value in properties_and_types[subj]:
            values = list(entities_and_types.values())
            if class_type not in values:
                data_to_insert[class_type] = [[1] for i in range(1)]

            if prop_type not in data_to_insert:
                data_to_insert[prop_type] = []
            data_to_insert[prop_type].append(train_model.function_build_feature(class_type, prop_name, prop_type, prop_value, max_min))

    types = list(entity_count.keys())
    for t in types:
        #if t not in data_to_insert:
        if entity_count[t] == 0:
            data_to_insert[t] = [[1] for i in range(1)]
        else:    
            data_to_insert[t] = [[1] for i in range(entity_count[t])]


    for key in data_to_insert.keys():
        type_key = get_type(key)
        lists = data_to_insert[key]
        if lists != '':
            data[type_key].x = torch.Tensor(lists)
            
    for triple in subject_dict.keys():  #le keys di subject_dict ed object_dict sono ==
            
        lol = [subject_dict[triple], object_dict[triple]]
        data[get_type(triple[0]), get_type(triple[1]), get_type(triple[2])].edge_index = torch.Tensor(lol).long()
    #else:
        #    data[get_type(triple[0]), get_type(triple[1]), get_type(triple[2])].edge_index =  torch.Tensor([[],[]]).long()
    '''
    for key in data_to_insert.keys():
        if key == "String":
            continue
        lists = data_to_insert[key]
        if lists != '':
            data[key].x = torch.Tensor(lists)

    print(subject_dict)
    print(object_dict)
    for triple in subject_dict.keys(): 
        if triple[0] == "String" or triple[2] == "String":
            continue
        lol = [subject_dict[triple], object_dict[triple]]
        data[triple[0], triple[1], triple[2]].edge_index = torch.Tensor(lol).long()
        data[triple[0], triple[1], triple[2]].edge_label_index = torch.Tensor(lol).long()
        data[triple[0], triple[1], triple[2]].edge_label = torch.Tensor([1 for i in range(len(lol[0]))]).long()
    '''
    return data, not_instances

def get_test_graph(Er,Et):
    new_Er = []
    triples = []

    triples_prop = []
    entities_and_types = {}
    properties_and_types = {}
    for row in Et:
        for s_er,p_er,o_er, weight in Er:
            #s_get = get_type(s_er)
            new_subj = s_er#s_get[0:len(s_get)-1]
            new_obj = o_er
            for s_et,p_et,o_et, v, id in row:     
                if id=='true' or id == 'yes' or id == True:
                    v = v.strip()
                    v = v.replace(" ","_")
                    prefix_split = s_er.split(config["relations"]["separator"])
                    prefix = '/'.join(prefix_split[0:len(prefix_split)-1])+"/"
                    if s_et == s_er:
                        new_subj = prefix+v
                        entities_and_types[prefix+v] = s_er[0:len(s_er)-1]
                        if new_subj not in properties_and_types:
                            properties_and_types[new_subj] =[]
                        if (s_et[0:len(s_et)-1],p_et,o_et, v) not in properties_and_types[new_subj]:
                            properties_and_types[new_subj].append((s_et[0:len(s_et)-1],p_et,o_et, v))
                            triples_prop.append((new_subj, p_et, o_et,v))

                    if s_et == o_er:
                        new_obj = prefix+v
                        entities_and_types[prefix+v] = o_er[0:len(o_er)-1]

                else:
                    if s_et == s_er:
                        if new_subj not in properties_and_types:
                            properties_and_types[new_subj] =[]
                        if (s_et[0:len(s_et)-1],p_et,o_et, v) not in properties_and_types[new_subj]:
                            properties_and_types[new_subj].append((s_et[0:len(s_et)-1],p_et,o_et, v))    
                            triples_prop.append((new_subj, p_et, o_et,v))

            new_Er.append((new_subj, p_er, new_obj, weight))
            triples.append((new_subj, p_er, new_obj))
    return new_Er, entities_and_types,properties_and_types, triples, triples_prop

def add_superclass_triples(Er, superclasses):
    Er_superclasses = []
    for s,p,o, weight in Er:
        Er_superclasses.append((s, p, o, weight))
        s_type = s[0:len(s)-1]
        o_type = o[0:len(o)-1]

        if s_type in superclasses.keys():
            for superclass in superclasses[s_type]:
                if (superclass+s_type, p, o, weight) not in Er_superclasses:
                    Er_superclasses.append((superclass+s_type, p, o, weight))
        if o_type in superclasses.keys():
            for superclass_o in superclasses[o_type]:
                if (s,p, superclass_o+ o_type, weight) not in Er_superclasses:
                    Er_superclasses.append((s,p, superclass_o+ o_type, weight))

        if s_type in superclasses.keys() and o_type in superclasses.keys():
            for superclass in superclasses[s_type]:
                for superclass_o in superclasses[o_type]:
                    if (superclass+s_type,p, superclass_o+ o_type, weight) not in Er_superclasses:
                        Er_superclasses.append((superclass+s_type,p, superclass_o+ o_type, weight))
    return Er_superclasses


def  get_triples_instances(triples, entities_and_types):
    not_instances = []
    new_triples = []
    for (s,p,o) in triples:
        if s in entities_and_types.keys():
            if o in entities_and_types.keys():
                if (s,p,o) not in new_triples:
                    new_triples.append((s,p,o))
            else:
                sub = get_type(entities_and_types[s])
                obj = get_type(o[0:len(o)-1])
                prop = get_type(p)
                if (sub, prop, obj) not in not_instances:
                    not_instances.append((sub, prop, obj))
        elif o in entities_and_types.keys():
                sub = get_type(s[0:len(s)-1])
                obj = get_type(entities_and_types[o]) 
                prop = get_type(p)
                if (sub, prop, obj) not in not_instances:
                    not_instances.append((sub, prop, obj))
        else:
            sub = get_type(s[0:len(s)-1])
            obj = get_type(o[0:len(o)-1])     
            prop = get_type(p)
            if (sub, prop, obj) not in not_instances:
                not_instances.append((sub, prop, obj))

    return new_triples, not_instances

def get_dicts(triples, triples_prop, entities_and_types, entity_types_count, property_types_count = {}):

    subject_dict = {}
    object_dict = {}
    not_instance =[]

    index_dict = {t:{'count': 0} for t in entity_types_count.keys()}

    for class_name,rel, p in property_types_count.keys():
        index_dict[p] = {'count':0}
        if class_name not in index_dict.keys():
            index_dict[class_name] = {'count':0}        
    for s,p,o in triples:
        if s in entities_and_types.keys():
            s_type = entities_and_types[s]
        else:
            s_type = s[0:len(s)-1]
            if s_type not in not_instance:
                not_instance.append(s_type)
        
        if o in entities_and_types.keys():
            o_type = entities_and_types[o]
        else:
            o_type = o[0:len(o)-1]
            if o_type not in not_instance:
                not_instance.append(o_type)

        if(s_type != "" and o_type != ""):
            key_t = (s_type, p, o_type)

        if key_t not in subject_dict.keys():
            subject_dict[key_t] = []
            object_dict[key_t] = []

        if s not in index_dict[s_type]:
            index_dict[s_type][s] = index_dict[s_type]['count']
            index_dict[s_type]['count'] = index_dict[s_type]['count']+1
        s_index = index_dict[s_type][s]
        
        if o not in index_dict[o_type]:
            index_dict[o_type][o] = index_dict[o_type]['count']
            index_dict[o_type]['count'] = index_dict[o_type]['count']+1
        o_index = index_dict[o_type][o]
        
        if key_t in subject_dict and key_t in object_dict:
            if s_index in subject_dict[key_t] and o_index in object_dict[key_t]:
                continue
        subject_dict[key_t].append(s_index)
        object_dict[key_t].append(o_index)
    

    for s,p,o_type,o in triples_prop:
        if s in entities_and_types.keys():
            s_type = entities_and_types[s]
        else:
            s_type = s[0:len(s)-1]
            if s_type not in not_instance:
                not_instance.append(s_type)
        
        if(s_type != "" and o_type != ""):
            key_t = (s_type, p, o_type)

        if key_t not in subject_dict.keys():
            subject_dict[key_t] = []
            object_dict[key_t] = []

        if s not in index_dict[s_type]:
            index_dict[s_type][s] = index_dict[s_type]['count']
            index_dict[s_type]['count'] = index_dict[s_type]['count']+1
        s_index = index_dict[s_type][s]
        
        if o not in index_dict[o_type]:
            index_dict[o_type][o] = index_dict[o_type]['count']
            index_dict[o_type]['count'] = index_dict[o_type]['count']+1
        o_index = index_dict[o_type][o]
        
        if key_t in subject_dict and key_t in object_dict:
            if s_index in subject_dict[key_t] and o_index in object_dict[key_t]:
                continue
        subject_dict[key_t].append(s_index)
        object_dict[key_t].append(o_index)
    
    return (subject_dict, object_dict)

def _entities_count(Er, entities_and_types):
    entities_count = {t:{'count': 0} for t in entities_and_types.values()}

    for s, p, o, w in Er:
        if s in entities_and_types.keys():
            s_type = entities_and_types[s]
        else: 
            #s_type = s[0:len(s)-1]
            continue
        
        if o in entities_and_types.keys():
            o_type = entities_and_types[o]    
        else:
            continue
    
        if s_type not in entities_count.keys():
            entities_count[s_type] = {'count': 0}

        if s not in entities_count[s_type]:
            entities_count[s_type][s] = entities_count[s_type]['count']
            entities_count[s_type]['count'] = entities_count[s_type]['count']+1

        if o_type not in entities_count.keys():
            entities_count[o_type] = {'count': 0}

        
        if o not in entities_count[o_type]:
            entities_count[o_type][o] = entities_count[o_type]['count']
            entities_count[o_type]['count'] = entities_count[o_type]['count']+1   

    no_duplicates = {}
    for key in entities_count.keys():
        no_duplicates[key] = entities_count[key]["count"]   
    return no_duplicates

def _properties_count(properties_and_types):
    properties_count = {}
    added = []
    for key in properties_and_types.keys():
        rows = properties_and_types[key]
        for s,p,o,v in rows:  
            if (s, p, o, v) not in added:
                properties_count[(s,p,o)] = properties_count.get((s,p,o),0)+1
                added.append((s, p, o, v))
    return properties_count

def get_weight_for_triples(not_instances, relation_weights):
    weights ={}
    for s,p,o in not_instances:
        for s2,p2,o2 in relation_weights.keys():
            if (s == s2 and p == p2 and o == o2) or (s == o2 and p == p2 and o == s2):
                weights[(s,p,o)] = relation_weights[(s,p,o)]
    return weights

def get_type(relation):
    r_split = relation.split(config["relations"]["separator"])
    return str(r_split[len(r_split)-1]).strip()

def get_property_type( property):
    split_p = property.split("^^")
    p_type = str(split_p[1].split('#')[1]).lower()
    
    if p_type.startswith("xsd:integer") or p_type.startswith("integer"):
        return("Integer", split_p[0])
    if p_type.startswith("xsd:string") or p_type.startswith("string"):
        return("String", split_p[0])
    if p_type.startswith("xsd:double") or p_type.startswith("double"):
        return("Double", split_p[0])
    if p_type.startswith("xsd:gYear") or  p_type.startswith("gYear"):
        return("Year",split_p[0])
    if p_type.startswith("xsd:date") or p_type.startswith("date"):
        return("Date",split_p[0])
    return ("","")

def get_ontology_type( property):
    if property.find("#") == -1: return ("")
    p_type = str(property.split("#")[1]).lower()
    
    if p_type.startswith("int"):
        return("Integer")
    if p_type.startswith("string"):
        return("String")
    if p_type.startswith("double") or p_type.startswith("decimal"):
        return("Double")
    if p_type.startswith("gYear"):
        return("Year")
    if p_type.startswith("date"):
        return("Date")
    return ("")


def dataparser(value):
    try:
        value = int(value)
        return 'Integer'
    except:
        pass 
    try:
        value = float(value)
        return 'Double'
    except:
        pass   
    try:   
        value = parse(value)
        return 'Date'
    except:
        pass
    return 'String'



    '''backup graph_test_data
   def get_test_graph(Er,Et):
    new_Er = []
    triples = []

    triples_prop = []
    entities_and_types = {}
    properties_and_types = {}
    for row in Et:
        for s_er,p_er,o_er, weight in Er:
            #s_get = get_type(s_er)
            new_subj = s_er#s_get[0:len(s_get)-1]
            new_obj = o_er
            for s_et,p_et,o_et, v, id in row:     
                if id=='true':
                    v = v.strip()
                    v = v.replace(" ","_")
                    prefix_split = s_er.split(config["relations"]["separator"])
                    prefix = '/'.join(prefix_split[0:len(prefix_split)-1])+"/"
                    if s_et == s_er:
                        new_subj = prefix+v
                        entities_and_types[prefix+v] = s_er[0:len(s_er)-1]
                        if new_subj not in properties_and_types:
                            properties_and_types[new_subj] =[]
                        if (s_et[0:len(s_et)-1],p_et,o_et, v) not in properties_and_types[new_subj]:
                            properties_and_types[new_subj].append((s_et[0:len(s_et)-1],p_et,o_et, v))
                            triples_prop.append((new_subj, p_et, o_et,v))

                    if s_et == o_er:
                        new_obj = prefix+v
                        entities_and_types[prefix+v] = o_er[0:len(o_er)-1]

                else:
                    if s_et == s_er:
                        if new_subj not in properties_and_types:
                            properties_and_types[new_subj] =[]
                        if (s_et[0:len(s_et)-1],p_et,o_et, v) not in properties_and_types[new_subj]:
                            properties_and_types[new_subj].append((s_et[0:len(s_et)-1],p_et,o_et, v))    
                            triples_prop.append((new_subj, p_et, o_et,v))

            new_Er.append((new_subj, p_er, new_obj, weight))
            triples.append((new_subj, p_er, new_obj))
    return new_Er, entities_and_types,properties_and_types, triples, triples_prop
 
    
    '''