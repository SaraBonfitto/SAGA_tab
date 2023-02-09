from asyncio import proactor_events
from base64 import decode
from posixpath import split
from re import sub
from tkinter import S
import rdflib
from rdflib import URIRef, RDF
from rdflib.namespace import Namespace
import configparser
import os
import src.data.utils as utils
from definitions import ROOT_DIR
import urllib.parse
from dateutil.parser import parse

class MakeDataset():
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(ROOT_DIR, 'config.ini'))
        self.possible_types = {}
        self.ontology = rdflib.Graph()
        self.ontology.parse(ROOT_DIR+ self.config['ontology']['path'], format=self.config['ontology']['format'])
        #self.ontology.bind("dbo", Namespace("http://dbpedia.org/ontology/"))
        #self.ontology.bind("dbr", Namespace("http://dbpedia.org/resource/"))
        #self.ontology.bind("rdfs", Namespace("http://www.w3.org/2000/01/rdf-schema#"))
        #self.ontology.bind("owl", Namespace("http://www.w3.org/2002/07/owl#"))
        #self.ontology.bind("rdf", Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#"))
        self.entities_and_type = {}
        self.properties_and_type ={}

    def get_superclasses(self, class_name):
        list_result = []

        query = " SELECT ?all_super_classes WHERE {{ <{0}> {1} ?all_super_classes . }}".format(
            class_name,
            self.config["ontology"]["subclass"]
        )

        result = self.ontology.query(query)
        for r in result:
            c_name = str(r.asdict()['all_super_classes'].toPython())
            list_result.append(c_name)

        return list_result


    def set_entities_and_type(self, graph, use_properties = False):
        triples = []

        for s, p, o in graph:
            str_s = urllib.parse.unquote(str(s))
            str_p = urllib.parse.unquote(str(p))
            str_o = urllib.parse.unquote(str(o))

            if p != RDF.type:
                if not str_s in self.entities_and_type.keys():
                    self.entities_and_type[(str_s)] =[]

                if type(o) != rdflib.term.Literal:
                    if not str_o in self.entities_and_type.keys():
                        self.entities_and_type[str_o]=[]
                    triples.append((s,p,o))
                else:
                    if use_properties:
                        if str_s not in self.properties_and_type.keys():
                            self.properties_and_type[str_s] =[]
                        p_type, p_value = utils.get_property_type(str_o)
                        if (str_s,p_type, p_value) not in self.properties_and_type[str_s]:
                            self.properties_and_type[str_s].append((str_p, p_type, p_value))
                        triples.append((s,p,o))
            else:
                if str_s not in self.entities_and_type.keys():
                    self.entities_and_type[str_s] =[]
                triples.append((s,p,o))
                self.entities_and_type[str_s].append(str_o)
        
        for e in self.entities_and_type:
            self.entities_and_type[e].sort()

        return triples


    def get_possible_types(self,subj_type, obj_type):
        if (subj_type,obj_type) not in self.possible_types:
            range_list = self.config["ontology"]["range"].split(",")
            domain_list = self.config["ontology"]["domain"].split(",")

            subj_superclasses = self.get_superclasses(subj_type)
            obj_superclasses = self.get_superclasses(obj_type)
            subj_superclasses.append(subj_type)
            obj_superclasses.append(obj_type)

            for range in range_list:
                for domain in domain_list:
                    q = """SELECT DISTINCT ?property WHERE {{?property {0} <{1}>. ?property {2} <{3}>. }}""".format(
                        domain,
                        subj_type,
                        range,
                        obj_type
                    )   

                    result = self.ontology.query(q)
                    results = []
                    for res in result:
                        results.append(str(res[0]))
                    
                    for subj_class in subj_superclasses:
                        for obj_class in obj_superclasses:
                            q2 = """SELECT DISTINCT ?property WHERE {{ ?property {0} <{1}>. ?property {2} <{3}>. }}""".format(
                                    domain,
                                    subj_class,
                                    range,
                                    obj_class
                                )
                            result = self.ontology.query(q2)
                            for res in result:
                                if (str(res[0])) not in results:
                                    results.append(str(res[0]))
                    
                    self.possible_types[(subj_type,obj_type)] = results
            return results
        return self.possible_types[(subj_type,obj_type)]
    '''
    def get_class_from_property(self):
        Q = " SELECT ?property ?class WHERE {?property rdfs:domain ?class; rdf:type owl:DatatypeProperty. }"
        results = {}
        result = self.ontology.query(Q)
        for res in result:
            results[self.get_type(str(res[0]))]= self.get_type(str(res[1]))
        return results
    '''
    def get_classes_types(self):
        new_properties_and_types = {}
        for s in list(self.properties_and_type.keys()):
            for element in self.properties_and_type[s]:
                s_class = self.entities_and_type[s]

                if s not in new_properties_and_types:
                    new_properties_and_types[s] = []

                new_properties_and_types[s].append((s_class[0], element[0], element[1], element[2]))
        
        self.properties_and_type = new_properties_and_types

    def disambiguate_multiple_types(self, s,p,o): 
        for subtype_subj in self.entities_and_type[str(s)]:
            if len(self.entities_and_type[str(o)]) > 1:
                for subtype_obj in self.entities_and_type[str(o)]:
                    possible_rels = self.get_possible_types( subtype_subj, subtype_obj)

                    if len(possible_rels) == 0:
                        continue   
                    
                    for rel in possible_rels:
                        if rel == p:
                            return (subtype_subj,subtype_obj)
            else:
                subtype_obj = self.entities_and_type[str(o)][0]
                possible_rels = self.get_possible_types(subtype_subj, subtype_obj)
                if len(possible_rels) == 0:
                        continue
                for rel in possible_rels:
                    if rel == p:
                        return (subtype_subj,  subtype_obj)
            
        return ("","")      

    def get_count(self):
        self.get_classes_types()
        entity_types_count = {}
        property_types_count = {}
        added = []

        for entity in self.entities_and_type.keys():
            type = self.entities_and_type[entity][0]
            if type != "":
                entity_types_count[type] = entity_types_count.get(type, 0)+1

        for subj in self.properties_and_type.keys():
            for class_name, prop_name, prop_type, prop_value in self.properties_and_type[subj]:
                key = (class_name, prop_name, prop_type)
                                
                if (class_name, prop_name, prop_type, prop_value) not in added:
                    property_types_count[key] = property_types_count.get(key, 0)+1
                    added.append((class_name, prop_name, prop_type, prop_value))
        return entity_types_count, property_types_count

    def clean_triples(self, triples):
        new_triples = []
        added_types = []
        
        for s,p,o in sorted(triples):
            str_s = urllib.parse.unquote(str(s))
            str_p = urllib.parse.unquote(str(p))
            str_o = urllib.parse.unquote(str(o))
            if p != RDF.type:

                if type(o) != rdflib.term.Literal:
                    new_triples.append((s, p, o))
                if str_s in list(self.entities_and_type.keys()) and str_o in list(self.entities_and_type.keys()):
                    #if the relation is between classes (not a property)

                    #if the subject or object type are multiple, it chooses only one, moreover it removes the relations not
                    #compliant with the structure of the ontology
                    if len(self.entities_and_type[str_s]) > 1 or len(self.entities_and_type[str_o]) > 1:
                        new_subj_type, new_obj_type = self.disambiguate_multiple_types(s,p,o)
                        if((new_subj_type, new_obj_type) == ("","") 
                            or new_subj_type == "" 
                            or new_obj_type ==""):
                            #remove from entites_and_type
                            continue

                
                        self.entities_and_type[str_s] = [new_subj_type]
                        self.entities_and_type[str_o] = [new_obj_type]

                        #adding the triple defining the node type to the new_triples set
                        if s not in added_types:
                            new_triples.append((s, 
                                                RDF.type,
                                                URIRef(new_subj_type)
                                                ))

                            added_types.append(s)

                        if o not in added_types:
                            new_triples.append((o,
                                                RDF.type,
                                                URIRef(new_obj_type)
                                                ))
                            added_types.append(o)
                    else: 
                        #adding the triple defining the node type to the new_triples set
                        if s not in added_types:
                            new_triples.append((s, 
                                                RDF.type,
                                                URIRef(self.entities_and_type[str_s][0] )))
                            added_types.append(s)
                        if o not in added_types and type(o) != rdflib.term.Literal:
                            new_triples.append((o, 
                                                RDF.type,
                                                URIRef(self.entities_and_type[str_o][0])))
                            added_types.append(o)
                    if(s,p,o) not in new_triples:
                        new_triples.append((s, p, o))
            '''
            else:
                if s not in added_types and len(self.entities_and_type[str_s]) == 1: 
                #controllo solo s perché la relazione p indica che o è il tipo, 
                # verifico che non ci sia piu di un tipo altrimenti rimando l'aggiunta a quando viene
                #disambiguato
                    new_triples.append((s, p, o))
                    added_types.append(s)
            '''
        for e in self.entities_and_type.keys():
            if len(self.entities_and_type[e])>1:
                self.entities_and_type[e] = [self.entities_and_type[e][0]]

        return new_triples

   
    def get_subject_object(self, triples, entity_types_count, property_types_count = {}):
        subject_dict = {}
        object_dict = {}

        index_dict = {t:{'count': 0} for t in entity_types_count.keys()}

        for class_name,rel, p_type in property_types_count.keys():
            index_dict[p_type] = {'count':0}
            if class_name not in index_dict.keys():
                index_dict[class_name] = {'count':0}  
                
        triples.sort()

        for s,p,o in triples:
            str_s = urllib.parse.unquote(str(s))
            str_p = urllib.parse.unquote(str(p))
            str_o = urllib.parse.unquote(str(o))
            
            if p != RDF.type:
                s_type = self.entities_and_type[str_s][0] 

                if type(o) != rdflib.term.Literal:
                    o_type = self.entities_and_type[str_o][0]
                else: 
                    o_type = utils.get_property_type(str_o)[0]
                
                if(s_type != "" and o_type != ""):
                    key_t = (s_type, str_p, o_type)
                    
                    if key_t not in subject_dict.keys():
                        subject_dict[key_t] = []
                        object_dict[key_t] = []
                        
                    if str_s not in index_dict[s_type]:
                        index_dict[s_type][str_s] = index_dict[s_type]['count']
                        index_dict[s_type]['count'] = index_dict[s_type]['count']+1
                    s_index = index_dict[s_type][str_s]
                        
                    if str_o not in index_dict[o_type]:
                        index_dict[o_type][str_o] = index_dict[o_type]['count']
                        index_dict[o_type]['count'] = index_dict[o_type]['count']+1
                    o_index = index_dict[o_type][str_o]
                        
                if key_t in subject_dict and key_t in object_dict:
                    if s_index in subject_dict[key_t] and o_index in object_dict[key_t]:
                        continue

                    subject_dict[key_t].append(s_index)
                    object_dict[key_t].append(o_index)
                
                #data[s_type, p, o_type].edge_index[0].append(entities.index(str(s)))
                #data[s_type, p, o_type].edge_index[1].append(entities.index(str(o)))
        
    
        with open('subject.txt', 'w') as f:
            for key, value in subject_dict.items(): 
                f.write('%s:%s\n' % (key, value))
        
            
        with open('object.txt', 'w') as f:
            for key, value in object_dict.items(): 
                f.write('%s:%s\n' % (key, value))
        
        return subject_dict, object_dict, self.properties_and_type

    
    def get_max_min(self, properties_and_types):
        max_min = {}
        class_prop_type_value = list(properties_and_types.values())
        for quadruple in class_prop_type_value:
            for (class_name, property, prop_type, v) in quadruple:
                class_name = utils.get_type(class_name)
                property = utils.get_type(property)
                if prop_type != "String":
                    if prop_type == "Integer":
                        value = int(v)
                    elif prop_type == "Double":
                        value = float(v)
                    elif prop_type == "Date":
                        value = parse(v)
                    if (class_name, property, prop_type) in max_min:
                        if type(max_min[(class_name, property, prop_type)][0]) == type(value):
                            if max_min[(class_name, property, prop_type)][0] < value:
                                max_min[(class_name, property, prop_type)][0] = value
                        
                            if max_min[(class_name, property, prop_type)][1] > value:
                                max_min[(class_name, property, prop_type)][1] = value
                    else:
                        max_min[(class_name, property, prop_type)] = [value, value]
        return max_min


