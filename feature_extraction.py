import stanza
import json
from nltk.stem import PorterStemmer 
from nltk.corpus import wordnet as wn

'''Event class'''
class Event:
    def __init__(self,doc_id,sent_id,event_id,event_type,trigger,arguments,tokens):
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.event_id = event_id # not being extracted feature from
        self.event_type = event_type
        self.trigger = trigger
        self.arguments = arguments
        self.tokens = tokens

    
    def get_type(self):
        return self.event_type.split(':')[0].strip()

    def get_subtype(self):
        return self.event_type.split(':')[1].strip()

    def get_trigger_text(self):
        return self.trigger['text']

    def get_trigger_pos(self,nlp_pipeline):
        doc = nlp_pipeline([self.tokens])
        for sent in doc.sentences:
            for word in sent.words:
                if word.id == self.trigger['start']+1:
                    #pos
                    pos = word.upos
                    nom_number = None
                    
                    # if nominal
                    if pos == 'NOUN' or pos == 'PROPN':
                        is_nominal = 1
                        is_pronominal = 0
                        
                        # parse features
                        feat_list = word.feats.split('|')
                        for f in feat_list:
                            key = f.split('=')[0].strip()
                            value = f.split('=')[1].strip()
                            if key == 'Number':
                                nom_number = value
                    # if not nominal
                    else:
                        is_nominal = 0
                        if pos == 'PRON':
                            is_pronominal = 1
                        else:
                            is_pronominal = 0
                    return pos, is_nominal, is_pronominal, nom_number

    def get_trigger_stem(self,stemmer):
        return stemmer.stem(self.trigger['text'])

    def get_trigger_synset(self,trigger_pos):
        if trigger_pos == 'VERB':
            pos = 'v'
        elif trigger_pos == 'ADJ':
            pos = 'a'
        elif trigger_pos == 'ADV':
            pos = 'r'
        else:
            pos = 'n'
        # print(wn.synsets(self.trigger['text'],pos = pos)[0])
        return wn.synsets(self.trigger['text'],pos = pos)[0]


"""load data"""
# data loader
    # Note: this is for ACE05 oneie.json data
def create_Event_from_line(line):
    events = []
    if line['event_mentions'] != []:
        for e in line['event_mentions']:
            events.append(Event(line['doc_id'],line['sent_id'],e['id'],e['event_type'],e['trigger'],e['arguments'],line['tokens']))
    return events 

# create doc_sent_token_em_len_dict from json data:
    # Note: this is for ACE05 oneie.json data
def create_doc_sent_token_em_len_dict(jsonfile):

    doc_sent_token_em_len_dict = {}
    with open(jsonfile) as f:
        for line in f:
            line_item = json.loads(line)
            # if line_item['event_mentions'] != []:
            doc_id = line_item['doc_id']
            sent_id = line_item['sent_id']
            tokens = line_item['tokens']
            ems = line_item['event_mentions']
        
            len_tokens = len(tokens)
            len_ems = len(ems)
            # if len_ems > 1:
            #     print(ems)
            #     # quit()
            #     print('')
            if doc_id not in doc_sent_token_em_len_dict:
                doc_sent_token_em_len_dict[doc_id] = {}
            doc_sent_token_em_len_dict[doc_id][sent_id] = (len_tokens,len_ems)
    return doc_sent_token_em_len_dict



'''feature extraction functions'''
def get_baseline_feats(AEM,prior_event_mentions,nlp_pipeline,stemmer):
    # prior_event_mentions: a list of Event object

    # the last event mention in a prior event
    LEM = prior_event_mentions[-1]

    # pair
    type_pair = (AEM.get_type(),LEM.get_type())
    subtype_pair = (AEM.get_subtype(),LEM.get_subtype())
    trigger_pair = (AEM.get_trigger_text(),LEM.get_trigger_text())
    pos_AEM, is_nominal_AEM, is_pronominal_AEM, nom_number_AEM = AEM.get_trigger_pos(nlp_pipeline)
    pos_LEM, _, _, _ = LEM.get_trigger_pos(nlp_pipeline)
    pos_pair = (pos_AEM,pos_LEM)
    
    # match
    exact_match = 0
    stem_match = 0
    for pe in prior_event_mentions:
        if AEM.get_trigger_text() == pe.get_trigger_text():
            exact_match = 1
        if AEM.get_trigger_stem(stemmer) == pe.get_trigger_stem(stemmer): # not clear accroding to the paper
            stem_match = 1

    ## trigger sim
    ## path_similarity:
    # trigger_sim = AEM.get_trigger_synset(pos_AEM).path_similarity(LEM.get_trigger_synset(pos_LEM))
    ## Leacock-Chodorow Similarity:
    trigger_sim = AEM.get_trigger_synset(pos_AEM).lch_similarity(LEM.get_trigger_synset(pos_LEM))
    ## Wu-Palmer Similarity:
    # trigger_sim = AEM.get_trigger_synset(pos_AEM).wup_similarity(LEM.get_trigger_synset(pos_LEM))
    
    return  type_pair, subtype_pair, trigger_pair, pos_pair, is_nominal_AEM, nom_number_AEM, is_pronominal_AEM, exact_match, stem_match, trigger_sim

def get_distance_feats(AEM,LEM,doc_sent_token_size_dict):
    assert AEM.doc_id == LEM.doc_id # should be within document?
    sent_idx_AEM = int(AEM.sent_id.replace(AEM.doc_id+'-','').strip())
    sent_idx_LEM = int(LEM.sent_id.replace(LEM.doc_id+'-','').strip())
    
    sent_dist = abs(sent_idx_AEM-sent_idx_LEM)
    
    event_dist = 0 
    token_dist = 0
    if sent_idx_AEM == sent_idx_LEM:
        event_dist = 0 #don't know how to caculate event dist if they are in the same sentence
        token_dist = abs(AEM.trigger['start']-LEM.trigger['start'])
    elif sent_idx_AEM < sent_idx_LEM:
        if sent_idx_LEM - sent_idx_AEM == 1:
            event_dist = 1
            token_dist = len(AEM.tokens)-AEM.trigger['end']+LEM.trigger['start']
        else:
            count_event = 0
            count_token = 0
            for i in range(sent_idx_AEM+1,sent_idx_LEM):
                sent_key = AEM.doc_id+'-'+str(i)
                count_event += doc_sent_token_size_dict[AEM.doc_id][sent_key][1] # event size in this sentence
                count_token += doc_sent_token_size_dict[AEM.doc_id][sent_key][0]
            event_dist = count_event + 1
            token_dist = count_token+len(AEM.tokens)-AEM.trigger['end']+LEM.trigger['start']
    else:
        if sent_idx_AEM - sent_idx_LEM == 1:
            event_dist = 1
            token_dist = len(LEM.tokens)-LEM.trigger['end']+AEM.trigger['start']
        else:
            count_event = 0
            count_token = 0
            for i in range(sent_idx_LEM+1,sent_idx_AEM):
                sent_key = AEM.doc_id+'-'+str(i)
                count_event += doc_sent_token_size_dict[AEM.doc_id][sent_key][1] # event size in this sentence
                count_token += doc_sent_token_size_dict[AEM.doc_id][sent_key][0]
            event_dist = count_event + 1
            token_dist = count_token + len(LEM.tokens)-LEM.trigger['end']+AEM.trigger['start']
    
    return token_dist,sent_dist,event_dist

# def get_arguments_feats(AEM,prior_event)
def get_arguments_feats(AEM,LEM):
    argument_entity_ids_roles_AEM = []
    time_AEM = ''
    place_AEM = ''
    for arg in AEM.arguments:
        arg_en_id = AEM.doc_id + '-' + arg['entity_id'].split('-')[1].strip()
        role = arg['role']
        argument_entity_ids_roles_AEM.append((arg_en_id,role)) # entity id not mention id
        if 'Time' in role:
            time_AEM = arg['text']
        if 'Place' in role:
            place_AEM = arg['text']
            

    time_LEM = ''
    place_LEM = ''
    argument_entity_ids_roles_LEM = []
    for arg in LEM.arguments:
        arg_en_id = LEM.doc_id + '-' + arg['entity_id'].split('-')[1].strip()
        role = arg['role']
        argument_entity_ids_roles_LEM.append((arg_en_id,role)) # entity id not mention id
        if 'Time' in role:
            time_LEM = arg['text']
        if 'Place' in role:
            place_LEM = arg['text']
    time_conflict = 0
    place_conflict = 0
    if time_AEM != '' and time_LEM != '':
        if time_AEM != time_LEM:
            time_conflict = 1
    if place_AEM != '' and place_LEM != '':
        if place_AEM != place_LEM:
            place_conflict = 1


    overlap_num = 0
    overlap_roles = []
    prior_num = 0
    prior_roles = []
    act_num = 0
    act_roles = []
    coref_num = 0
    
    for arg_aem in argument_entity_ids_roles_AEM:                                           
        flag = 0
        for arg_lem in argument_entity_ids_roles_LEM:
            
            if arg_lem[0] == arg_aem[0] and arg_lem[1] == arg_aem[1]:
                overlap_num += 1
                overlap_roles.append(arg_aem[1])
                flag = 1
            
            if arg_lem[0] == arg_aem[0] and arg_lem[1] != arg_aem[1]:
                coref_num += 1
        
        if flag == 0:    
            act_num += 1
            act_roles.append(arg_aem[1])

    for arg_lem in argument_entity_ids_roles_LEM:                                           
        flag = 0
        for arg_aem in argument_entity_ids_roles_AEM:
            if arg_lem[0] == arg_aem[0] and arg_lem[1] == arg_aem[1]:
                flag = 1
        if flag == 0:
            prior_num += 1
            prior_roles.append(arg_lem[1])
    
    return overlap_num,overlap_roles,prior_num,prior_roles,act_num,act_roles,coref_num,time_conflict,place_conflict




# set up components for feature extraction
doc_sent_token_em_len_dict = create_doc_sent_token_em_len_dict('./ACE05_data/train.oneie.json')
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos',tokenize_pretokenized=True) # Build the pipeline, specify part-of-speech processor, use our own tokens
stemmer = PorterStemmer() # for stem match using nltk.stem




# example usage

"""three feature function output format:

get_baseline_feats(AEM,prior_event_mentions,nlp_pipeline,stemmer) -> (type_pair, subtype_pair, trigger_pair, pos_pair, is_nominal_AEM, nom_number_AEM, is_pronominal_AEM, exact_match, stem_match, trigger_sim)

get_distance_feats(AEM,LEM,doc_sent_token_size_dict) -> (token_dist,sent_dist,event_dist)

get_arguments_feats(AEM,LEM) -> (overlap_num,overlap_roles,prior_num,prior_roles,act_num,act_roles,coref_num,time_conflict,place_conflict)

"""

line1 = {"doc_id": "CNN_CF_20030303.1900.00", "sent_id": "CNN_CF_20030303.1900.00-15", "entity_mentions": [{"id": "CNN_CF_20030303.1900.00-E4-69", "start": 2, "end": 3, "entity_type": "PER", "entity_subtype": "Group", "mention_type": "NOM", "text": "bozos"}, {"id": "CNN_CF_20030303.1900.00-E98-70", "start": 6, "end": 7, "entity_type": "PER", "entity_subtype": "Group", "mention_type": "NOM", "text": "Cubans"}, {"id": "CNN_CF_20030303.1900.00-E99-71", "start": 10, "end": 11, "entity_type": "LOC", "entity_subtype": "Region-General", "mention_type": "NOM", "text": "shores"}, {"id": "CNN_CF_20030303.1900.00-E78-155", "start": 19, "end": 20, "entity_type": "PER", "entity_subtype": "Indeterminate", "mention_type": "NOM", "text": "terrorist"}], "relation_mentions": [{"relation_type": "PHYS", "relation_subtype": "Located", "id": "CNN_CF_20030303.1900.00-R14-1", "arguments": [{"entity_id": "CNN_CF_20030303.1900.00-E98-70", "text": "Cubans", "role": "Arg-1"}, {"entity_id": "CNN_CF_20030303.1900.00-E99-71", "text": "shores", "role": "Arg-2"}]}],"event_mentions": [{"event_type": "Movement:Transport", "id": "CNN_CF_20030303.1900.00-EV1-2", "trigger": {"start": 7, "end": 8, "text": "land"}, "arguments": [{"entity_id": "CNN_CF_20030303.1900.00-E4-69", "text": "bozos", "role": "Agent"}, {"entity_id": "CNN_CF_20030303.1900.00-E98-70", "text": "Cubans", "role": "Artifact"}, {"entity_id": "CNN_CF_20030303.1900.00-E99-71", "text": "shores", "role": "Destination"}]}], "tokens": ["And", "these", "bozos", "let", "four", "armed", "Cubans", "land", "on", "our", "shores", "when", "they", "'re", "trying", "to", "make", "a", "high", "terrorist", "alert", "."], "pieces": ["And", "these", "b", "##oz", "##os", "let", "four", "armed", "Cuban", "##s", "land", "on", "our", "shores", "when", "they", "'", "re", "trying", "to", "make", "a", "high", "terrorist", "alert", "."], "token_lens": [1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1], "sentence": "And these bozos let four armed Cubans land on our shores when they 're trying to make a high terrorist alert ."}
line2 = {"doc_id": "CNN_CF_20030303.1900.00", "sent_id": "CNN_CF_20030303.1900.00-17", "entity_mentions": [{"id": "CNN_CF_20030303.1900.00-E6-79", "start": 1, "end": 2, "entity_type": "PER", "entity_subtype": "Group", "mention_type": "NOM", "text": "professionals"}, {"id": "CNN_CF_20030303.1900.00-E2-81", "start": 3, "end": 4, "entity_type": "PER", "entity_subtype": "Individual", "mention_type": "NOM", "text": "Mr."}, {"id": "CNN_CF_20030303.1900.00-E2-80", "start": 4, "end": 5, "entity_type": "PER", "entity_subtype": "Individual", "mention_type": "NOM", "text": "President"}], "relation_mentions": [], "event_mentions": [{"event_type": "Personnel:Start-Position", "id": "CNN_CF_20030303.1900.00-EV4-1", "trigger": {"start": 0, "end": 1, "text": "Hire"}, "arguments": [{"entity_id": "CNN_CF_20030303.1900.00-E6-79", "text": "professionals", "role": "Person"}]}], "tokens": ["Hire", "professionals", ",", "Mr.", "President", "."], "pieces": ["Hi", "##re", "professionals", ",", "Mr", ".", "President", "."], "token_lens": [2, 1, 1, 2, 1, 1], "sentence": "Hire professionals , Mr. President ."}

events1 = create_Event_from_line(line1) # one line can have multiple event mentions
events2 = create_Event_from_line(line2)

test_event_1 = events1[0]
test_event_2 = events2[0]

print(get_baseline_feats(test_event_1,[test_event_2],nlp,stemmer))
print(get_distance_feats(test_event_1,test_event_2,doc_sent_token_em_len_dict))
print(get_arguments_feats(test_event_1,test_event_2))

# test_event_1 = Event("CNN_CF_20030303.1900.00","CNN_CF_20030303.1900.00-15","CNN_CF_20030303.1900.00-EV1-2","Movement:Transport",{"start": 7, "end": 8, "text": "land"},[{"entity_id": "CNN_CF_20030303.1900.00-E4-69", "text": "bozos", "role": "Agent"}, {"entity_id": "CNN_CF_20030303.1900.00-E98-70", "text": "Cubans", "role": "Artifact"}, {"entity_id": "CNN_CF_20030303.1900.00-E99-71", "text": "shores", "role": "Destination"}],["And", "these", "bozos", "let", "four", "armed", "Cubans", "land", "on", "our", "shores", "when", "they", "'re", "trying", "to", "make", "a", "high", "terrorist", "alert", "."])
# test_event_2 = Event("CNN_CF_20030303.1900.00","CNN_CF_20030303.1900.00-17","CNN_CF_20030303.1900.00-EV4-1","Personnel:Start-Position",{"start": 0, "end": 1, "text": "Hire"},[{"entity_id": "CNN_CF_20030303.1900.00-E6-79", "text": "professionals", "role": "Person"}],["Hire", "professionals", ",", "Mr.", "President", "."])

# t_e_1 = Event('1','1-1','','Die',{},[{"entity_id": "1-E1-1", "text": "mike", "role": "Person"},{"entity_id": "1-E2-1", "text": "ivy", "role": "Person"},{"entity_id": "1-E4-1", "text": "ivy_tree", "role": "Time"}],[])
# t_e_2 = Event('1','1-2','','Die',{},[{"entity_id": "1-E1-2", "text": "mike", "role": "Person"},{"entity_id": "1-E3-1", "text": "ivy2", "role": "Place"},{"entity_id": "1-E2-3", "text": "ivy_clock", "role": "Time"}],[])

# print(test_event_1.get_trigger_stem(stemmer))
# print(test_event_1.get_trigger_pos(nlp))
# print(get_arguments_feats(t_e_1,t_e_2))

# tokens = ["He", "lost", "an", "election", "to", "a", "dead", "man", "."]
# doc = nlp([tokens]) # Run the pipeline on the input text
# print(doc) # Look at the result