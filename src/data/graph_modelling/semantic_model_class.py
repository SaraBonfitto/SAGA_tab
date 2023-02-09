import csv
from email import header
import os
from re import M, S, sub
from matplotlib.pyplot import cla
import matplotlib.pyplot as plt
import networkx as nx
import configparser 
from IPython.display import Image
from rdflib import RDF, Graph as RDFGraph, Literal, URIRef
from collections import defaultdict
import rdflib
import json
from networkx.readwrite import json_graph
import src.data.utils as utils
from dateutil.parser import parse
from definitions import ROOT_DIR
from datetime import datetime
from uuid import uuid4
from torch_geometric.data import HeteroData
import src.models.train_model as train_model 

class SemanticModelClass():
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(ROOT_DIR, 'config.ini'))
        self.classes = {}
        self.leafs = {}
        self.triples = []
        self.closure_classes = {}
        self.closure_graph = nx.MultiDiGraph()
        self.ontology = rdflib.Graph()
        self.ontology.parse(ROOT_DIR+self.config["ontology"]["path"])
        #self.ontology.parse("/home/sara/Desktop/fase2/git_repo/knowledge-graph-learning/data/external/dbpedia/ontologia.ttl")
        self.properties_types = self.get_properties_types()
        self.properties = {}
        self.properties_missing_relations ={}
        self.classes_no_id =[]
        self.prefixes = self.get_prefixes()

    def get_prefixes(self):
        prefixes = {}
        for key in self.config["prefixes"]:
            prefixes[key] = self.config["prefixes"][key]
        return prefixes

    def get_properties_types(self):
        result_dict ={}
        range_list = self.config["ontology"]["range"].split(",")
        for range in range_list: 
            query = "SELECT DISTINCT ?property ?type WHERE {{ ?property rdf:type {0}; {1} ?type. }}".format(
                self.config["ontology"]["properties"],
                range
            ) 
            result = self.ontology.query(query)
            for r in result:
                result_dict[utils.get_type(str(r[0]))] = utils.get_ontology_type(str(r[1]))
        return result_dict
        
    def draw_result(self,graph, filename):
        
        node_label = nx.get_node_attributes(graph,'id')
        pos = nx.spring_layout(graph)
        p=nx.drawing.nx_pydot.to_pydot(graph)
        #print(ROOT_DIR+filename+'.png')
        p.write_png(filename+'.png')
        Image(filename=filename+'.png')

    def draw_edge_labels(self, graph, filename):
        pos = nx.spring_layout(graph)
        nx.draw(
            graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='pink', alpha=0.9,
            labels={node: node for node in graph.nodes()}
        )
        labels = dict([((n1, n2), f'{n3}')
                   for n1, n2, n3 in graph.edges])

        p = nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=labels,
            font_color='red'
        )
        plt.show()

    def get_possible_classes(self, path):
        possible_classes = []
        with open(path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)  # skip the headers
            
            for row in csv_reader:
                if row[1].strip() != "" and (row[3].strip().lower() == "true" or row[3].strip().lower() == "yes"):
                    strip_row = row[1].strip().split(":")
                    node_name = self.prefixes[strip_row[0]]+ strip_row[1]
                    possible_classes.append(node_name)
        return possible_classes

    def parse(self):
        semantic_model = nx.MultiDiGraph()
        possible_classes = self.get_possible_classes(ROOT_DIR +self.config["semantic_model_csv"]["csv_path"])

        #with open("/home/sara/Desktop/fase2/git_repo/knowledge-graph-learning/data/interim/semantic_models/00/00.csv", 'r') as csv_file:
        with open(ROOT_DIR +self.config["semantic_model_csv"]["csv_path"], 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)  # skip the headers

            for row in csv_reader:
                if any(row): #to avoid empty lines
                    if row[1].strip() == "":
                        self.properties_missing_relations[row[0]] = ("", row[3].strip(), row[4].strip())
                        continue
                    strip_row = row[1].strip().split(":")
                    if row[2].strip() == "":
                        self.properties_missing_relations[row[0]] = (self.prefixes[strip_row[0]] + strip_row[1].strip(), row[3].strip(), row[4].strip())
                        continue
                                       
                    node_name = self.prefixes[strip_row[0]]+ strip_row[1]
                    #self.classes[node_name]= self.classes.get(node_name, 0)

                    #if the element is an identifier
                    splitrow = row[2].strip().split(":")
                    rel_with_prefix = self.prefixes[splitrow[0]]+splitrow[1]
                    if row[3].strip().lower() == "true" or row[3].strip().lower() == "yes":
                        self.classes[node_name]= self.classes.get(node_name, -1)+1
                        self.triples.append(
                            (node_name +str(self.classes[node_name]), 
                            row[0], 
                            rel_with_prefix)
                        )
                        #semantic_model.add_edge(node_name +str(self.classes[node_name]),row[0], label = rel_with_prefix)
                        semantic_model.add_edge(node_name +str(row[4].strip()),row[0], label = rel_with_prefix)
                    else:
                        if node_name not in possible_classes:
                            self.classes_no_id.append(row[0].strip())
                            self.classes[node_name] = 0
                            self.triples.append((
                                (node_name +"0"), 
                                row[0], 
                                rel_with_prefix))
                            semantic_model.add_edge(node_name +"0",row[0], label = rel_with_prefix)

                    '''
                    else:
                        if node_name in self.classes.keys():
                            semantic_model.add_edge(node_name +str(self.classes[node_name]),
                                                    row[0], 
                                                    label = rel_with_prefix)

                            self.triples.append((node_name +str(self.classes[node_name]), row[0], rel_with_prefix))
                        else:
                            #adding a fake identifier
                            if node_name not in possible_classes:

                                subclasses = self.get_subclasses(node_name)
                                if len(subclasses) != 0:
                                    founded = False
                                    for sub in subclasses:
                                        if sub in self.classes.keys():
                                            founded = True
                                            node_name = sub
                                            semantic_model.add_edge(sub +str(self.classes[sub]),
                                                                    row[0], 
                                                                    label = rel_with_prefix)

                                            self.triples.append((sub +str(self.classes[sub]), row[0], rel_with_prefix))
                                            break
                                    if not founded:
                                        for sub in subclasses:
                                            if sub in possible_classes:
                                                node_name = sub
                                                semantic_model.add_edge(sub +str(self.classes[sub]),
                                                                        row[0], 
                                                                        label = rel_with_prefix)

                                                self.triples.append((sub +str(self.classes[sub]), row[0], rel_with_prefix))
                                                break
                                else:
                                    superclass = self.get_superclass(node_name)
                                    if len(superclass) != 0:
                                        founded = False
                                        for super in superclass:
                                            if super in self.classes.keys():
                                                node_name = super
                                                founded = True
                                                semantic_model.add_edge(super +str(self.classes[super]),
                                                                        row[0], 
                                                                        label = rel_with_prefix)

                                                self.triples.append((super +str(self.classes[super]), row[0], rel_with_prefix))
                                                break
                                        if not founded:
                                            for super in superclass:
                                                if super in possible_classes:
                                                    node_name = super
                                                    semantic_model.add_edge(super +str(self.classes[super]),
                                                                            row[0], 
                                                                            label = rel_with_prefix)

                                                    self.triples.append((super +str(self.classes[node_name]), row[0], rel_with_prefix))
                                                    break
                            else:
                                self.classes[node_name] = 0
                                self.triples.append(
                                    (node_name +str(self.classes[node_name]), 
                                    row[0], 
                                    rel_with_prefix)
                                )
                                semantic_model.add_edge(node_name +str(self.classes[node_name]),row[0], label = rel_with_prefix)
                    '''
                    self.properties[row[0]] = (node_name +str(row[4].strip()),
                                               rel_with_prefix, row[3].strip())
                    #self.properties[row[0]] = (node_name +str(self.classes[node_name]), rel_with_prefix, row[3].strip())
                        #print(node_name +str(self.classes[node_name]))
                    
                    if row[2].strip() == "":
                        continue
                    else:
                        rowstrip = row[2].strip().split(":")
                        self.leafs[node_name + row[4].strip()] = self.prefixes[rowstrip[0]] + rowstrip[1]
        return semantic_model

    def add_data_properties(self):
        Et = []
        identifiers ={}
        with open(ROOT_DIR+self.config["ontology"]["identifiers"], 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                strip_row_0 = row[0].strip().split(":")
                key = self.prefixes[strip_row_0[0]] + strip_row_0[1]
                strip_row_1 = row[1].strip().split(":")
                value = self.prefixes[strip_row_1[0]] + strip_row_1[1]
                identifiers[key] = value

        #with open("/home/sara/Desktop/fase2/git_repo/knowledge-graph-learning/data/interim/semantic_models/00/00_data.csv", 'r') as csv_file:
        with open(ROOT_DIR+self.config["semantic_model_csv"]["data_csv"], 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)

            for row in csv_reader:
                Et_row = []
                if any(row): #to avoid empty lines

                    for col in header:
                        if col.strip() in self.classes_no_id:
                            class_name, rel_type, id = self.properties[col.strip()]
                            rel_type = identifiers[class_name[0:len(class_name)-1]]
                            datatype = utils.dataparser(row[header.index(col)].strip())

                            inserted = False
                            for (et_class_name, et_rel_type, et_datatype, value, et_id) in Et_row:
                                if et_class_name == class_name and et_rel_type == rel_type and et_datatype == datatype:
                                    inserted = True
                                    break
                            if not inserted:
                                eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
                                Et_row.append((class_name, rel_type, datatype, eventid, "yes") )
                                        
                        if col.strip() not in self.properties_missing_relations.keys():
                            class_name, rel_type, id = self.properties[col.strip()]
                            rel_type = rel_type.strip()

                            Et_row.append((class_name, 
                                        rel_type, 
                                        utils.dataparser(row[header.index(col)].strip()),
                                        row[header.index(col)], id.strip())
                                    )
                    Et.append(Et_row)
            return Et


    def infer_property(self, relation_weights, Er, Et, leafs, max_min = {}):
        if len(self.properties_missing_relations) == 0:
            return Et
        to_insert = {}
        cols = {}
        column_possible_properties = {}
        n_properties_for_type = self.get_n_properties_for_type(Et)
        sd_classes = self.get_sm_classes(Et)

        with open(ROOT_DIR+self.config["semantic_model_csv"]["data_csv"], 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)
            predicted = {}
            types = []
            new_properties = {}
            new_properties_prefix = {}

            closure_classes = self.get_complete_sm_classes(Er)
            used_properties = self.get_used_properties(Et)

            row_index = 1
            for row in csv_reader:
                for col in header:
                    #takes properties not assigned to other clases
                    if col.strip() in self.properties_missing_relations.keys():
                        if col.strip() not in types:
                            types.append(col.strip())
                        column_possible_properties[col.strip()]=[]
                        class_name,is_identifier, n_instance = self.properties_missing_relations[col.strip()]
                        property_value = row[header.index(col)].strip()
                        
                        #manages the cases where we have a class and not a property
                        if class_name != "":
                            possible_prop = self.find_property_name(relation_weights, [class_name], 
                                                                property_value,leafs,
                                                                n_properties_for_type,
                                                                    to_insert, max_min)
                            for p in possible_prop:
                                if p not in column_possible_properties[col.strip()]:
                                    column_possible_properties[col.strip].append(p)

                        #manages the cases where both classes and properties are missing
                        else:

                            #new_class, property, p_type, instance_n 
                            pool_no_prefixes, pool_with_prefixes = self.find_property_names(closure_classes, property_value)
                            pool_no_prefixes, pool_with_prefixes = self.remove_not_in_range(pool_no_prefixes, pool_with_prefixes, property_value, max_min)
                            if len(pool_no_prefixes) > 1:
                                pool_no_prefixes, pool_with_prefixes = self.remove_used_properties(pool_no_prefixes, pool_with_prefixes, used_properties)

                
                            selected_properties, selected_with_prefix = self.choose_property(pool_no_prefixes, pool_with_prefixes, relation_weights)
                            new_properties[(col.strip(), property_value)] = selected_properties
                            new_properties_prefix[(col.strip(), property_value)] = selected_with_prefix
                            
                            for col, value in list(new_properties.keys()):
                                for new_class,property, p_type, w in list(new_properties[(col, value)]):
                                    if new_class in sd_classes:
                                        n = 1000+(2-w)
                                        weight = {"weight":n, "key": col}
                                    else:
                                        weight = {"weight":1000+(2-w*0.9), "key": col}

                                    n_instance = self.get_instance_num(new_class, Er)
                                    if (new_class+str(n_instance), property, p_type, property_value, False) not in Et[row_index-1]:
                                        Et[row_index-1].append((new_class+str(n_instance), property, p_type, property_value, False))
                                    index = new_properties[(col,value)].index((new_class,property, p_type, w))
                                    c,p,t,i = new_properties_prefix[(col,value)][index]
                                    if( utils.get_type(c), property, p_type, n_instance) not in used_properties:
                                        Er.append((c+str(n_instance),p,t,weight))
                                        used_properties.append((new_class, property, p_type, n_instance ))
                row_index += 1
                

        return Er, types
    
    def get_instance_num(self, candidate_class, Er):
        n_instance = 0
        for class_name, property, type, w in Er:
            class_name = utils.get_type(class_name)
            if class_name == candidate_class:
                n = class_name[len(class_name)-1:]
                if n> n_instance:
                    n_instance = n
        return n_instance
    
    def get_n_properties_for_type(self,Et):
        result = {}
        for et in Et:
            for (class_name, property, property_type, property_value, is_id) in et:
                class_type = class_name[0:len(class_name)-1]
                instance_num = int(class_name[len(class_name)-1:])
                if class_type not in list(result.keys()):
                    result[class_type] = {'count':instance_num+1, property: 1}
                else: 
                    if instance_num+1 > result[class_type]['count']:
                        result[class_type]['count'] = instance_num+1
                    if property not in result[class_type]:
                        result[class_type][property] = 1


        return result

    def find_property_names(self, classes, property_value):
        possible_prop_no_prefixes =[]        
        possible_properties_with_prefixes =[]

        range_list = self.config["ontology"]["range"].split(",")
        domain_list = self.config["ontology"]["domain"].split(",")
        #superclasses = self.get_superclasses(classes)
        #for superclass in superclasses:
        #    if superclass not in classes:
        #        classes.append(superclass)

        for class_name in classes:
            for range_class in range_list:
                for domain in domain_list:
                    superclass = self.get_superclass(class_name)
                    if len(superclass) >0:
                        superclass = superclass[0]
                    query = """SELECT ?property ?type WHERE {{ {{ ?property {0} <{1}>. ?property {2} ?type.
                            ?property rdf:type {3}. }} UNION {{ ?property {0} <{4}>. ?property {2} ?type.
                            ?property rdf:type {3}.}} }}""".format(
                        domain,
                        class_name,
                        range_class,
                        self.config["ontology"]["properties"],
                        superclass,
                    )

                    property_type = utils.dataparser(property_value)

                    result = self.ontology.query(query)
                    for r in result:
                        r0 = str(r[0])
                        r1 = str(r[1])
                        onto_type =  utils.get_ontology_type(r1)
                        #choose all triples having the same prop type
                        if onto_type.startswith(property_type):
                            possible_prop_no_prefixes.append((utils.get_type(class_name), utils.get_type(r0),property_type,0))
                            possible_properties_with_prefixes.append((class_name,r0, property_type,0))
        return possible_prop_no_prefixes, possible_properties_with_prefixes
            
    def remove_used_properties(self,possible_properties, pool_with_prefixes, used_properties):
        #removes the existing ones if there is not another instance
        new_possible_prop = []
        new_possible_prop_prefix = []

        for i in range(len(possible_properties)):
            class_type, prop, prop_type, n_instance = possible_properties[i]
            if (class_type, prop, prop_type, n_instance) not in used_properties:
                new_possible_prop.append((class_type, prop, prop_type, n_instance))
                new_possible_prop_prefix.append((pool_with_prefixes[i]))
            '''
            #for the management of multiple instances
            if class_type in list(n_properties_for_type.keys()):
               if prop in n_properties_for_type[class_type]:
                    if n_properties_for_type[class_type][prop] >= n_properties_for_type[class_type]["count"]:
                        type_s = utils.get_type(class_type)
                        type_p = utils.get_type(prop)
                        #if col_name not in predicted_for_col.keys():
                        if (type_s, type_p, prop_type) not in delete:
                            delete.append((type_s,type_p, prop_type))
                    else:
                        c_name, p_name, p_type, occ = possible_properties_with_prefixes[i]
                        occ = n_properties_for_type[class_type][prop] +1
                        possible_properties_with_prefixes[i] =  c_name, p_name, p_type, occ
            '''
        return new_possible_prop, new_possible_prop_prefix
    
    def get_used_properties(self, Et):
        used_prop = []
        Et = Et[0]
        for i in range(len(Et)):
            class_type, property_name, property_type, property_value, isId = Et[i]
            class_type_no_prefix = utils.get_type(class_type)
            instance = class_type_no_prefix[len(class_type_no_prefix)-1:]
            class_no_instance_n = class_type_no_prefix[0:len(class_type_no_prefix)-1]
            used_prop.append((class_no_instance_n, utils.get_type(property_name), property_type, int(instance)))
        return used_prop
    
    def remove_not_in_range(self, pool_no_prefixes, pool_with_prefixes, property_value, max_min):      
        #removing the ones not in the range of max min
        new_properties = []
        new_properties_prefix =[]
        for (class_type, property_name, property_type, isId ) in pool_no_prefixes:    
            if property_type != "String": 
                if (class_type, property_name, property_type) in max_min.keys():
                    if property_type == "Integer":
                        value = int(property_value)
                    elif property_type == "Double":
                        value = float(property_value)
                    elif property_type == "Date" and type(property_value) != datetime:
                        value = parse(property_value)

                    if value >= max_min[(class_type,property_name,property_type)][1] and  value <= max_min[(class_type,property_name,property_type)][0]:
                        new_properties.append((class_type, property_name, property_type, isId ))
                        index = pool_no_prefixes.index((class_type, property_name, property_type, isId ) )
                        new_properties_prefix.append(pool_with_prefixes[index])
            else:
                new_properties.append((class_type, property_name, property_type, isId ))
                index = pool_no_prefixes.index((class_type, property_name, property_type, isId ) )
                new_properties_prefix.append(pool_with_prefixes[index])

        return new_properties,new_properties_prefix
                        
    def choose_property(self, possible_properties, pool_with_prefixes, relation_weights):
        class_max = {}
        properties = {}
        for (s,p,o, instances) in possible_properties:
            if (s,p,o) in relation_weights.keys():   
                w = relation_weights[(s,p,o)]
                max = class_max.get(s,0)
                if w > max:
                    class_max[s] = w
                    properties[s]= p,o,max
        new_properties = []
        new_properties_prefix = []

        for s in properties.keys():
            new_properties.append((s, properties[s][0], properties[s][1], properties[s][2]))
            index = possible_properties.index((s, properties[s][0], properties[s][1],0))
            c, p, v, i = pool_with_prefixes[index]
            new_properties_prefix.append((c, p, v,properties[s][2]))
        return new_properties, new_properties_prefix
        
                        
        '''
        #removing the properties already predicted for other columns
        for column_name in predicted_for_col.keys():
            if column_name != col_name:
                for (s,p,o, instance) in possible_properties2:
                    for (col_s,col_p,col_o, value) in predicted_for_col[column_name]:
                        if col_s == s and p == col_p and col_o == o:
                            type_s = utils.get_type(col_s)
                            type_p = utils.get_type(col_p)
                            if (type_s, type_p, col_o) not in delete:
                                delete.append((type_s, type_p, col_o))
    
        not_deleted_possible_prop = []
        for (s,p,o, instances) in possible_properties:
            if (s,p,o) not in delete:   
                index = possible_properties.index((s, p, o, instances))
                not_deleted_possible_prop.append(possible_properties2[index])
    '''
    def get_complete_sm_classes(self, Er):
        closure_classes =[]
        n_properties_for_type = {}
        for s,p,o,w in Er:
            if s[0:len(s)-1] not in closure_classes:
                closure_classes.append(s[0:len(s)-1])

            if o[0:len(o)-1] not in closure_classes:
                closure_classes.append(o[0:len(o)-1])

        return closure_classes

    def get_sm_classes(self, Et):
        classes =[]
        for i in range(len(Et)):
            for s,p,o,value,id in Et[i]:
                s = utils.get_type(s)
                if s[0:len(s)-1] not in classes:
                    classes.append(s[0:len(s)-1])


        return classes


    def superclasses_for_weigths(self, Er):
        superclasses = {}
        for s, p, o, w in Er:
            s_type = utils.get_type(s)
            o_type = utils.get_type(o)
            prefix = "http://dbpedia.org/ontology/"
            #prefix = self.config["ontology"]["prefixes"]["dbo"]
            o_type = o_type[0:len(o_type)-1]
            s_type = s_type[0:len(s_type)-1]
            super_s = self.get_superclass(prefix+s_type)
            super_o = self.get_superclass(prefix+o_type)

            if len(super_s) != 0:
                for superc in super_s:
                    if s_type not in list(superclasses.keys()):
                        superclasses[s_type] = []
                    if utils.get_type(superc) not in superclasses[s_type]:
                        superclasses[s_type].append(utils.get_type(superc))

            if len(super_o) != 0:
                for superc in super_o:
                    if o_type not in list(superclasses.keys()):
                        superclasses[o_type] = []
                    if utils.get_type(superc) not in superclasses[o_type]:
                        superclasses[o_type].append(utils.get_type(superc))
        return superclasses

    def get_higher_prediction(self, cols):
        count_quadruple ={}
        for quadruple in cols:
            count_quadruple[quadruple] = count_quadruple.get(quadruple,0)+1

        max = 0
        higher_quadruple = ""
        for key, value in count_quadruple.items():
            if value > max:
                higher_quadruple = key
                max = value

        return higher_quadruple

    def get_classes(self):
        return self.classes
    
    def get_leafs(self):
        return list(self.leafs.keys())

    def get_closure_classes(self):
        results = {}
        list_classes = list(self.classes.keys())
        superclasses = self.get_superclasses(list_classes)
        range_list = self.config["ontology"]["range"].split(",")
        domain_list = self.config["ontology"]["domain"].split(",")

        for class_name in list_classes:
            for domain in domain_list:
                for range in range_list:
                    query = "SELECT DISTINCT ?closure WHERE {{ {{ ?property {0} <{1}>. ?property {2} ?closures."+\
                            " ?closure a {3}. }} UNION {{ ?property {0} ?closure. ?property {2} <{1}>. ?closure a {3} .}} }}".format(
                                domain,
                                class_name,
                                range,
                                self.config["ontology"]["class"]
                            )
                    result = self.ontology.query(query)

                    if class_name + "0" not in results.keys():
                        results[class_name + "0"] = 1
                        #results.append(class_name + "0")

                    for r in result:
                        c_name = str(r.asdict()['closures'].toPython())
                        if c_name+"0" not in results.keys():
                            if c_name in superclasses:
                                results[c_name + "0"] = 10
                            else:
                                results[c_name + "0"] = 1

                            #results.append(c_name+"0")

                    num_classes = self.classes[class_name]
                    if num_classes > 0:
                        for i in range(1,num_classes+1):
                            if class_name+str(i) not in results.keys():
                                if class_name in superclasses:
                                    results[class_name + str(i)] = 10
                                else:
                                    results[class_name + str(i)] = 1

                        #results.append(class_name + str(i))

                    #for r in result:
                    #    c_name = str(r.asdict()['closures'].toPython())
                    #    while self.classes[class_name] > -1:
                    #        results.append(c_name+str(i))     
        self.closure_classes = results   
        return results

    def get_superclasses(self, classes):
        results = {}
        list_result = []

        for class_node in classes:
            query = " SELECT ?all_super_classes WHERE {{ <{0}> {1} ?all_super_classes . }}".format(
                class_node,
                self.config["ontology"]["subclass"]
            )
            #query = "SELECT ?all_super_classes WHERE { <" +class_node +"> "+\
            #" <"+self.config["ontology"]["subclass"]+ "> ?all_super_classes . }"

            result = self.ontology.query(query)
            for r in result:
                c_name = str(r.asdict()['all_super_classes'].toPython())
                if class_node in results.keys():
                    results[class_node].append(c_name)
                else:
                    results[class_node] = [c_name]
                list_result.append(c_name)

        return list_result

    def get_subclasses(self, class_node):
        subclasses =[]
        #query = "SELECT ?all_super_classes WHERE { ?all_super_classes rdfs:subClassOf <"+class_node +">.}"

        query = " SELECT ?all_super_classes WHERE {{ ?all_super_classes {0} <{1}>  . }}".format(
            self.config["ontology"]["subclass"],
            class_node
        )
        result = self.ontology.query(query)
        for r in result:
            subclasses.append(str(r[0]))
        return subclasses


    def get_superclass(self, node):
        #query = "SELECT ?super_class WHERE { <"+node +"> <"self.config["ontology"]["subclass"]+ "> ?super_class .}"
        query = "SELECT ?super_class WHERE {{ <{0}> {1} ?super_class.}}".format(node, self.config["ontology"]["subclass"])
        result = self.ontology.query(query)
        res = []
        for r in result:
            res.append(str(r[0]))
        return res


    def get_outgoing_links(self, node):
        #query = "SELECT ?rel WHERE { ?rel rdfs:domain <"+node +"> .}"
        domain_list = self.config["ontology"]["domain"].split(",")
        res = []
        for domain in domain_list:
            query = "SELECT ?rel WHERE {{ ?rel {0} <{1}>.}}".format(domain, node)
            result = self.ontology.query(query)
            for r in result:
                res.append(str(r[0]))
        return res

    def get_out_links_and_obj(self,node):
        results = []
        
        #if not node.startswith("http://schema.dig.isi.edu/ontology/"):
        #    node = self.set_prefix(str(node))
        #else:
        #    node ="<"+node+">"
        domain_list = self.config["ontology"]["domain"].split(",")
        range_list = self.config["ontology"]["range"].split(",")

        for domain in domain_list:
            for range in range_list:
                query = "SELECT ?rel ?relatedClass WHERE {{ ?rel {0} <{1}>; rdf:type {2}; {3} ?relatedClass.}}".format(
                domain,
                node,
                self.config["ontology"]["relations"],
                range
            )
                result = self.ontology.query(query)
                for r in result:
                    results.append((str(r[0]), str(r[1])))
        return results

    def get_ingoing_links_and_subj(self,node):
        if not node.startswith("http://schema.dig.isi.edu/ontology/"):
            node = self.set_prefix(str(node))
        else:
            node ="<"+node+">"
        range_list = self.config["ontology"]["domain"].split(",")
        for range in range_list:
            query ="SELECT ?rel ?relatedClass WHERE {{ ?rel {0} <{1}>; rdf:type {2}; {0} ?relatedClass. }}".format(
                range,
                node,
                self.config["ontology"]["relations"]
            )

            result = self.ontology.query(query)
            res = []
            for r in result:
                res.append((str(r[1]), str(r[0])))
        return res

    def check_relation_exists(self, us,r,ut):
        us = us[0:len(us)-1]
        ut = ut[0:len(ut)-1]
        outgoing_links = []
        superclasses1 = self.get_superclass(us)
        superclasses2 = self.get_superclass(ut)

        outgoing_links.extend(self.get_out_links_and_obj(us))
        for superclass1 in superclasses1:
            outgoing_links.extend(self.get_out_links_and_obj(superclass1))
        
        for rel, obj in outgoing_links:
            if rel == r:
                if obj == ut or obj in superclasses2:
                    return True
        return False

    def is_subclass(self, candidate, superclass):
        #superclass = superclass[0: len(superclass)-1]
        #query = " SELECT ?subclass  WHERE { ?subclass rdfs:subClassOf <" +superclass +">. }"
        query = "SELECT ?subclass WHERE {{?subclass {0} <{1}>.}}".format(self.config["ontology"]["subclass"], superclass)
        result = self.ontology.query(query)
        for r in result:
            if str(r[0]) == candidate:
                return True
        return False
    
    def c_find(self, str1, str2):
        index = -1
        for i in range(len(str1)):
            if str1[i] == str2[i]:
                index = i
            else:
                break
            
        if index == len(str2)-1:
            return True
        
        return False

    def set_prefix(self, class_name):
        for key in self.prefixes.keys():
            #if self.c_find(self.prefixes[key],class_name):
            txt = self.prefixes[key]
            new = key+":"
            x = class_name.replace(txt, new)
            if len(x) < len(class_name):
                return x
        return x


    def is_superclass(self, candidate_superclass, subclass):
        candidate_superclass = self.set_prefix(str(candidate_superclass))
        subclass = self.set_prefix(str(subclass))
        query = "SELECT * WHERE {{ <{0}> {1} <{2}>.}}".format(subclass, self.config["ontology"]["subclass"], candidate_superclass)
        result = self.ontology.query(query)
        if len(result) != 0:
            return True
        return False

    def get_edges(self):
        range_list = self.config["ontology"]["range"].split(",")
        domain_list = self.config["ontology"]["domain"].split(",")
        for subj in self.closure_classes.keys():
            for obj in self.closure_classes.keys():
                for domain in domain_list:
                    for range in range_list:
                        #query = " SELECT ?direct_properties WHERE {"+\
                        #" ?direct_properties <"+ self.config["ontology"]["domain"]+ "> <" + +\
                        #">. ?direct_properties <"+ self.config["ontology"]["range"]+ "> <" +obj[:-1]+"> .}"

                        query = "SELECT ?direct_properties WHERE {{ ?direct_properties <{0}> <{1}>. "+\
                                "?direct_properties <{2} <{3}>. }}".format(
                                domain, 
                                subj[:-1],
                                range,
                                obj[:-1]
                                )
                        result = self.ontology.query(query)

                        for r in result:
                            p_name = str(r.asdict()['direct_properties'].toPython())
                            weight = max(self.closure_classes[subj], self.closure_classes[obj])
                            #label = "w:" + str(weight) + " - "+str(p_name)
                            #self.closure_graph.add_edge(subj,obj, label)
                            self.closure_graph.add_edge(subj,obj, label = str(p_name), weight = weight)
                            #self.closure_graph.add_edge(subj,obj)

        return self.closure_graph

    def get_graph_closure(self):
         self.get_closure_classes()
         return self.get_edges()

    def check_relations_with_superclasses(self, s_type, rel_type, o_type, weights, superclasses):
        max = 0

        if s_type in superclasses.keys():
            for superclass in superclasses[s_type]:
                if (superclass,rel_type,o_type) in weights.keys():
                    if max < weights[(superclass,rel_type,o_type)]:
                        max = weights[(superclass,rel_type,o_type)]
                elif (o_type,rel_type, superclass) in weights.keys():
                    if max < weights[(o_type,rel_type, superclass)]:
                        max = weights[(o_type,rel_type, superclass)]        
        if o_type in superclasses.keys():
            for superclass_o in superclasses[o_type]:
                if (s_type,rel_type,superclass_o) in weights.keys():
                    if max < weights[(s_type,rel_type,superclass_o)]:
                        max = weights[(s_type,rel_type,superclass_o)]
                elif (superclass_o, rel_type, s_type) in weights.keys():
                    if max < weights[(superclass_o, rel_type, s_type)]:
                        max = weights[(superclass_o, rel_type, s_type)]

        if s_type in superclasses.keys() and o_type in superclasses.keys():
            for superclass in superclasses[s_type]:
                for superclass_o in superclasses[o_type]:
                    if (superclass,rel_type,superclass_o) in weights.keys():
                        max = weights[(superclass,rel_type,superclass_o)]
                    elif (superclass_o,rel_type,superclass) in weights.keys():
                        max = weights[(superclass_o,rel_type,superclass)]
        
        #weights[(s_type,rel_type,o_type)] = max*epsilon
        return max

    def update_graph_weights(self, sd, weights, set = True, superclasses = []):
        new_graph = nx.MultiGraph()
        list_weights = []
        added_triples = []
        added_nodes = []
        epsilon = 10
        flag = 0
        norm = False
        for edge in sd.edges:
            column_name =""
            u = edge[0]
            v = edge[1]
            relations = sd.get_edge_data(u,v)
            if (u,v) not in added_nodes:
                for i in range(0, len(relations)):
                    u_type = utils.get_type(str(u)[:-1])
                    v_type = utils.get_type(v)
                    try:
                        int(v_type[len(v_type)-1:])
                        v_type = v_type[0:len(v_type)-1]
                    except:
                        v_type = v_type

                    rel_type = utils.get_type(relations[i]['label'])
                    try:
                        rgcn_weight = weights[(u_type,rel_type,v_type)]
                    except KeyError:
                        try: 
                            rgcn_weight = weights[(v_type,rel_type,u_type)]
                        except KeyError:
                            #try:
                            if len(superclasses) != 0:
                                rgcn_weight = self.check_relations_with_superclasses(u_type,rel_type,v_type, weights, superclasses)
                                flag = 1
                            else:
                                rgcn_weight = 0
                            #except KeyError:
                            #rgcn_weight = 0
                    try: 
                        v_type = utils.get_type(v)
                        int(v_type[len(v_type)-1:])
                        v_type = v_type[0:len(v_type)-1]
                    except:
                        v_type = relations[i]["weight"]["key"]
                        column_name = relations[i]["weight"]["key"]
                    if (u,rel_type,v) not in added_triples:
                        if set:
                            w = round(abs(1-rgcn_weight),4)
                            #if flag == 1:
                            #    w *= epsilon
                            #    flag = 0
                            lw = rel_type + " - " + str(w)
                            if column_name != "":
                                new_graph.add_edge(u,column_name, label = rel_type, weight = w, lw = lw)
                                added_triples.append((u, rel_type, column_name))  
                                if(u,v) not in added_nodes:
                                    added_nodes.append((u,v)) 
                            else:
                                new_graph.add_edge(u,v, label = rel_type, weight = w, lw = lw)
                                added_triples.append((u, rel_type, v))
                        else:
                            if column_name != "":
                                w = round(relations[i]["weight"]["weight"],4)

                                if(u,v) not in added_nodes:
                                    added_nodes.append((u,v)) 
                            else:
                                w = round(abs((1-rgcn_weight)*relations[i]["weight"]),4)
                            
                            if flag == 1:
                                w *= epsilon
                                flag = 0
                            list_weights.append(w)
                            lw = rel_type + " - " + str(w)

                            if column_name != "":
                                new_graph.add_edge(u,column_name, label = rel_type,
                                                    weight = w,
                                                    lw = lw)
                                added_triples.append((u, rel_type, column_name))
                            else:
                                new_graph.add_edge(u,v, label = rel_type,
                                weight = w,
                                lw = lw)
                                added_triples.append((u, rel_type, v))
                    if column_name == "":
                        added_nodes.append((u,v))     
   
        '''
        if not set and norm:
            w_min = min(list_weights)
            w_max = max(list_weights)
            added = []
            for edge in new_graph.edges:
                u = edge[0]
                v = edge[1]
                relations = new_graph.get_edge_data(u,v)
                for i in range(0, len(relations)):
                    rel_type = utils.get_type(relations[i]['label'])
                    if (u,rel_type,v) not in added:
                        w = relations[i]['weight']
                        if w > 0:
                            try:
                                w_norm = round(((w - w_min)/(w_max-w_min)),4)
                            except ZeroDivisionError:
                                w_norm = 0
                        else:
                            w_norm = 0
                        relations[i]["weight"] = w_norm
                        relations[i]["lw"] = rel_type + " - " + str(w_norm)
                        added.append((u,rel_type,v))
                '''

        return new_graph

    def add_important_edges(self, graph, tree):
        new_graph = nx.MultiGraph()
        added = []

        new_graph = tree.copy()
        for edge in graph.edges:
            u = edge[0]
            v = edge[1]
            relations = graph.get_edge_data(u,v)
            for i in range(0, len(relations)):
                rel_type = utils.get_type(relations[i]['label'])
                if relations[i]["weight"] < 0.0001:
                    if tree.has_node(u) and tree.has_node(v):
                        tree_edge = tree.get_edge_data(u,v)
                        if len(tree_edge) == 0 :
                            #tree_edge = tree.get_edge_data(u,v)
                            #key = list(tree_edge.keys())
                            #tree_edge[key[0]]["label"]
                            if (u,rel_type,v) not in added:
                                lw = relations[i]["label"] + " - " + str(relations[i]["weight"])
                                new_graph.add_edge(u,v, label = relations[i]["label"], weight = relations[i]["weight"], lw = lw)
                                added.append((u,rel_type, v))
        return new_graph


    def graph_to_json(self,graph):
        data1 = json_graph.node_link_data(graph)
        s2 = json.dumps(
            data1
        )
        return s2

    def compute_closure_graph(self,semantic_model):
        closure_graph = nx.MultiDiGraph()
        superclass_subclass = {}

        tot_classes = []
        tot_instances = []
        for node in semantic_model.nodes:
            if node[0:4].startswith("http"):
                closure_graph.add_node(node)
                superclasses = self.get_superclass(node[0:len(node)-1])
                tot_instances.append(node)
                tot_classes.append(node[0:len(node)-1])
                for superclass in superclasses:
                    tot_classes.append(superclass)
                    superclass = superclass+"0"
                    tot_instances.append(superclass)
                    if superclass not in list(superclass_subclass.keys()):
                        superclass_subclass[superclass] = []
                    superclass_subclass[superclass].append(node) 
        
        for node in tot_instances:
            node = str(node)[0:len(node)-1]
            #query = " SELECT DISTINCT ?property ?class WHERE { ?property rdfs:domain <"+ node +">."+\
            #" ?property rdfs:range ?class . ?class a " + self.config["ontology"]["class"]+".}"
            range_list = self.config["ontology"]["range"].split(",")
            domain_list = self.config["ontology"]["domain"].split(",")
            for domain in domain_list:
                for range in range_list:
                    query = "SELET DISTINCT ?property ?class WHERE {{ ?property {0} <{1}> . ?property {2} ?class. ?class a {3}. }}".format(
                        domain,
                        node,
                        range,
                        self.config["ontology"]["class"]
                    )
                    result = self.ontology.query(query)
                    for r in result:
                        rel = str(r[0])
                        obj = str(r[1])

                        node_istances = self.classes.get(node, 0)
                        obj_istances = self.classes.get(obj, 0)

                        for h in range(0, node_istances+1):
                            for h2 in range(0, obj_istances+1):
                                if node +str(h) in list(superclass_subclass.keys()) and obj+str(h2) in list(superclass_subclass.keys()):
                                    weight = 10
                                elif node +str(h) in list(superclass_subclass.keys()) or obj+str(h2) in list(superclass_subclass.keys()):
                                    weight = 10
                                else:
                                    weight = 1
                                
                                if not self.exists_edge(closure_graph, node +str(h), obj+str(h2), rel):
                                    closure_graph.add_edge(node +str(h), obj+str(h2), label = rel, weight = weight)
        return closure_graph

    def compute_closure_node(self,node):
        closure_node = nx.MultiDiGraph()
        superclass_subclass = {}

        tot_classes = []

        if node[0:4].startswith("http"): #non è una proprietà
            superclasses = self.get_superclass(node)
            tot_classes.append(node)
            for superclass in superclasses:
                tot_classes.append(superclass)

                if superclass not in list(superclass_subclass.keys()):
                    superclass_subclass[superclass] = []
                superclass_subclass[superclass].append(node) 
        
        for node in tot_classes:
            node = str(node)
            #query = " SELECT DISTINCT ?property ?class WHERE {"+\
            #" ?property rdfs:domain <"+ node +">."+\
            #" ?property rdfs:range  ?class . ?class a " + self.config["ontology"]["class"]+".} "
            range_list = self.config["ontology"]["range"].split(",")
            domain_list = self.config["ontology"]["domain"].split(",")
            for domain in domain_list:
                for range in range_list:
                    query = " SELECT DISTINCT ?property ?class WHERE {{ ?property {0} <{1}>. ?property {2} ?class . ?class a {3}.}}".format(
                        domain,
                        node,
                        range,
                        self.config["ontology"]["class"]   
                    )
                    result = self.ontology.query(query)
                    for r in result:
                        rel = str(r[0])
                        obj = str(r[1])

                        if node in list(superclass_subclass.keys()) and obj in list(superclass_subclass.keys()):
                            weight = 10
                        elif node in list(superclass_subclass.keys()) or obj in list(superclass_subclass.keys()):
                            weight = 10
                        else:
                            weight = 1
                        
                        if not self.exists_edge(closure_node, node, obj, rel):
                            closure_node.add_edge(node, obj, label = rel, weight = weight)

                    #query = " SELECT DISTINCT ?property ?class WHERE { ?property rdfs:range <"+ node +">."+\
                    #" ?property rdfs:domain  ?class . ?class a " + self.config["ontology"]["class"]+".} "
                    
                    query = "SELECT DISTINCT ?property ?class WHERE {{ ?property {0} <{1}>. ?property {2} ?class. ?class a {3}. }}".format(
                        range,
                        node,
                        domain,
                        self.config["ontology"]["class"]
                    )

                    result = self.ontology.query(query)
                    for r in result:
                        rel = str(r[0])
                        subj = str(r[1])

                        if node in list(superclass_subclass.keys()) and subj in list(superclass_subclass.keys()):
                            weight = 10
                        elif node in list(superclass_subclass.keys()) or subj in list(superclass_subclass.keys()):
                            weight = 10
                        else:
                            weight = 1
                        
                        #if not self.exists_edge(closure_node, subj, node, rel):
                        closure_node.add_edge(subj, node, label = rel, weight = weight)
        return closure_node


    def exists_edge(self,graph, u, v, label):
        edges = graph.get_edge_data(u, v)

        if edges == None:
            return False

        for i in range(0, len(edges)):
            if label == edges[i]["label"]:
                return True
        return False


    def get_distance(self, C1, C2):
        if C1 == C2:
            return 0
        superclass = self.get_superclasses([C2])
        if len(superclass) == 0:
            return 0
        if C1 in superclass:
            return 1
        
        return 1+ self.get_distance(C1, superclass[0])

    def get_distance_undirected(self, C1,C2):
        n = self.get_distance(C1, C2)
        m = self.get_distance(C2, C1)
        print("n ", n, "m ", m)
        return max(n,m)
    
    def class_exists_instances(self, class_name, instances):
        for instance in instances:
            if class_name == instance[0: len(instance)-1]:
                return True
        return False

    def homogenize_lists(self, us_list, ut_list):
        if len(us_list) > len(ut_list):
            for i in range(len(ut_list), len(us_list)):
                ut_list.append(ut_list[len(ut_list)-1])
        if len(us_list) < len(ut_list):
            for i in range(len(us_list), len(ut_list)):
                us_list.append(us_list[len(us_list)-1])
        return (us_list, ut_list)


    '''
        rimettere use_properties = False
        epsilon = 30 #discriminate on inherited relations
        delta = 20 #discriminate on different instances (e.g. Person0 - City1)
        gamma = 10 #discriminate between same labels
        sigma = 15 #nodes without values in the csv
    '''
    def get_complete_semantic_model(self,semantic_model,use_properties):
        #closure = self.compute_closure_node("http://dbpedia.org/ontology/Director")
        #return closure
        Uc_occurrences = {}

        Uc = [] 
        Ut = []
        Er = []
        Uc_ini = []
        
        epsilon = 10 #discriminate on inherited relations
        delta = 8 #discriminate on different instances (e.g. Person0 - City1)
        gamma = 4 #discriminate between same labels
        sigma = 10 #nodes without values in the csv
        
        #init UC and Ut
        if use_properties:
            Et = self.add_data_properties()
        else:
            Et = []
        for node in semantic_model.nodes:
            if node[0:4].startswith("http"):
                Uc.append(node)
                Uc_ini.append(node)
                Uc_occurrences[node[0:len(node)-1]] = Uc_occurrences.get(node[0:len(node)-1],0)+1
            else:
                Ut.append(node)

                #print(edge[0],edge[1], closure_C.get_edge_data(edge[0],edge[1]))
        closure_classes = []
        
        for uc in Uc_ini:
            C = uc[0: len(uc)-1]
            if C not in closure_classes:
                closure_classes.append(C)
            closure_graph = self.compute_closure_node(C)

            for node in closure_graph.nodes:
                #if len(self.get_superclass(node)) == 0:
                if node not in closure_classes:
                    closure_classes.append(node)
            '''
            for edge in closure_C.out_edges:
                if len(self.get_superclass(edge[0])) != 0 and not self.class_exists_instances(edge[0], Uc_ini):
                    continue
                if len(self.get_superclass(edge[1])) != 0 and not self.class_exists_instances(edge[1], Uc_ini):
                    continue
                if edge[0] not in closure_classes:
                    closure_classes.append(edge[0])
                if edge[1] not in closure_classes:
                    closure_classes.append(edge[1])
            '''

        for uc in Uc_ini:
            us = ""
            C = uc[0: len(uc)-1]

            closure_C = self.compute_closure_node(C)

            for edge in closure_C.out_edges:
                #print(edge[0],edge[1], closure_C.get_edge_data(edge[0],edge[1]))
            
                C1 = edge[0]
                C2 = edge[1]
                relations=[]
                rel = closure_C.get_edge_data(C1,C2)
                if rel != None:
                    for i in range(len(rel)):
                        relations.append(rel[i]["label"])

                    us_list =[]
                    ut_list =[]
                    if self.is_subclass(C, C1) or C==C1:
                        if C1 != C2:
                            us_list.append(uc)
                        else:
                            for u in Uc:
                                u_class = u[0:len(u)-1] 
                                if (u_class == C1 or self.is_subclass(u_class, C1)
                                    or self.is_superclass(u_class, C1)):
                                    us_list.append(u)
                    else:
                        uc1 = C1+"0"
                        if uc1 not in Uc:
                            if not self.is_superclass_or_subclass_of(uc1, Uc):
                                #if len(self.get_superclass(C1)) == 0:
                                if C1 not in Uc_occurrences.keys():
                                    Uc_occurrences[C1] = 1
                                us_list.append(uc1)
                                Uc.append(uc1)
                            else:
                                subclasses = self.get_subclasses(C1)
                                if len(subclasses)!= 0:
                                    for subclass in subclasses:
                                        k = Uc_occurrences.get(subclass,0)
                                        for i in range(k):
                                            us = subclass+str(i)
                                            #verifico che tra tutte le sottoclassi da aggiungere
                                            #non ce ne sia una che non appare nella closure
                                            if subclass in closure_classes:
                                                us_list.append(us)
                                        if k == 0 and subclass in closure_classes:
                                            us_list.append(subclass+"0")
                                
                                superclass = self.get_superclass(C1)
                                if len(superclass)!= 0:
                                    superclass = superclass[0]
                                    if superclass+"0" not in Uc_ini:
                                        k = Uc_occurrences.get(superclass,0)
                                        for i in range(k):
                                            us = C1+str(i)
                                            Uc = [ut if superclass+str(i) in s else s for s in Uc]
                                            Er = self.substitute(Er, superclass+str(i), us)
                                            if superclass in closure_classes:
                                                us_list.append(us)
                                        if k == 0 and superclass in closure_classes:
                                            Uc = [ut if superclass+"0" in s else s for s in Uc]
                                            Er = self.substitute(Er, superclass+"0", us)
                                            us_list.append(C1+"0")   
                                    
                        else:
                            us_list.append(uc1)

                    if self.is_subclass(C, C2) or C == C2: 
                        if C1 != C2:
                            ut_list.append(uc)
                        else:
                            for u in Uc:
                                u_class = u[0:len(u)-1] 
                                if (u_class == C2 or self.is_subclass(u_class, C2)
                                    or self.is_superclass(u_class, C2)):
                                    ut_list.append(u)
                    else:
                        uc2 = C2+"0"
                        if uc2 not in Uc:
                            if not self.is_superclass_or_subclass_of(uc2, Uc):
                                #if len(self.get_superclass(C2)) == 0:
                                if C2 not in Uc_occurrences.keys():
                                    Uc_occurrences[C2] = 1
                                ut_list.append(uc2)
                                Uc.append(uc2)

                            else:
                                subclasses = self.get_subclasses(C2)
                                if len(subclasses)!= 0 :
                                    for subclass in subclasses:
                                        k = Uc_occurrences.get(subclass,0)
                                        for i in range(k):
                                            ut = subclass+str(i)
                                            if subclass in closure_classes:
                                                ut_list.append(ut)
                                        if k == 0 and subclass in closure_classes:
                                            ut_list.append(subclass+"0")

                                superclass = self.get_superclass(C2)
                                if len(superclass)!= 0:
                                    superclass = superclass[0]
                                    if superclass+"0" not in Uc_ini:
                                        k = Uc_occurrences.get(superclass,0)
                                        for i in range(k):
                                            ut = C2+str(i)
                                            Uc = [ut if superclass+str(i) in s else s for s in Uc]
                                            Er = self.substitute(Er, superclass+str(i), ut)
                                            if superclass in closure_classes:
                                                ut_list.append(ut)
                                        if k == 0 and superclass in closure_classes:
                                            Uc = [ut if superclass+"0" in s else s for s in Uc]
                                            Er = self.substitute(Er, superclass+"0", ut)
                                            ut_list.append(C2+"0")   
                        else:
                            ut_list.append(uc2)

                    if len(us_list) == 0 or len(ut_list) == 0:
                        continue

                    us_list, ut_list = self.homogenize_lists(us_list, ut_list)

                    for r in relations:
                        for i in range(len(us_list)):
                            us = us_list[i]
                            for j in range(len(us_list)):
                                ut = ut_list[j]

                                H = Uc_occurrences.get(us[0:len(us)-1],0)
                                K = Uc_occurrences.get(ut[0:len(ut)-1],0)
                                h = int(us[len(us)-1:])
                                k = int(ut[len(ut)-1:])
                                #se la classe ha proprietà atomiche, annullo la distanza Pr source
                                # e Prdest. (dist 0, pr=1)
                                Pr_source = self.get_distance(C1,us[0:len(us)-1])
                                Pr_dest = self.get_distance(C2,ut[0:len(ut)-1])
                                Pr = 1+max(Pr_source, Pr_dest)*epsilon
                                #Pr = 1+(Pr_source + Pr_dest)*epsilon

                                if h != k:
                                    Pr += delta
                                
                                if self.relation_label_exists(Er,us,r,ut):
                                    Pr += gamma

                                if not self.has_properties(us[0:len(us)-1], Et) or not self.has_properties(ut[0:len(ut)-1], Et):
                                    Pr += sigma

                                if us != ut and (us, r, ut, Pr) not in Er and (ut, r, us, Pr) not in Er:
                                    if ((us[0:len(us)-1] == ut[0:len(ut)-1]) or ( h == k) 
                                         or (H <= K and h == H-1 and k > h) or (K-1 == k and h > k)):

                                        if self.check_relation_exists(us,r,ut):
                                            Er.append((us,r,ut, Pr))

        return (Er, Et)

    def substitute(self,list, old, new):
        new_Er = []
        for us, r, ut, Pr in list:
            if us == old:
                new_Er.append((new, r, ut, Pr))
            elif ut == old:
                new_Er.append((us,r,new,Pr))
            else:
                new_Er.append((us,r,ut,Pr))
        return new_Er

    def is_superclass_or_subclass_of(self, uc, Uc):
        for uq in Uc:
            if self.is_subclass(uq[0:len(uq)-1], uc[0:len(uc)-1]):
                return True

        for uq in Uc:
            if self.is_subclass(uc[0:len(uc)-1], uq[0:len(uq)-1]):
                return True
        
        return False

    def relation_label_exists(self, Er, e_us,e_r,e_ut):   
        for (us, rel, ut, pr) in Er:
            if (e_r == rel and ((us == e_us and ut == e_ut) 
                or (us == e_ut and ut == e_us))):
                return False
            elif (e_r == rel and ((us != e_us and ut == e_ut) 
                or (us == e_us and ut != e_ut))):
                return True
        return False

    def has_properties(self, concept, Et):
        for element in Et:
            element[0]
            if concept == element[0]:
                return True
        return False