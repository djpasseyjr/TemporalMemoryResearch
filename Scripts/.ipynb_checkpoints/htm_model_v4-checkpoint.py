# This is the first code on  the htm model
# This written by Dipak Ghosal, UC Davis
# This code  started by following the method  outlined in the the following paper
#   Why Neurons Have Thousands of Synapses: a Theory of Sequence Memory in Neocortex in Frontiers in Neural Circuits


# Importing requred libraries

import os 
#import simpy
import random
import numpy as np
#import networkx as nx
#import matplotlib
#matplotlib.use('Gtk3Agg')
#import matplotlib.pyplot as plty
import collections
import re 
import string
import os 
import pickle

# The global parameters
class G:
    # WORKING_DIRECTORY = "/Users/dghosal/Documents/HTM-Work"
    WORKING_DIRECTORY = os.getcwd()

    # Network paramaters
    number_of_columns = 1024
    cells_per_column = 16
    
    
    list_of_column_ids = list(range(1,number_of_columns+1))
    list_of_ids_in_a_column = list(range(1,cells_per_column+1))
    
    # cell_id = (colum_id, cell_in_column_id)
    
    
    # Cell parameters
    number_of_segments_min = 64
    number_of_segments_max = 64
    fixed_number_of_segments = 64
    
    
    # segment parameters
    max_num_synapses = 30
    min_num_synapses = 20
    fixed_num_synapses = 25
    
    theta = 16
    
     
    # synapse parameter
    
    max_p_value = 1
    min_p_value = 0

    alpha = 0.3 
    
    
    # synaptic strength update parameters 
    
    small_decrement = -0.05
    very_small_decrement = -0.01
    large_increment = 0.36
    
    
    # encoding parameters  
    encoding_length = 32  # number of columns activated for every input. 
    encoding_range = number_of_columns # 
    
    
    # This file will contain the encodings
    encodings_filename = "encodings_file.txt"
    
    
    
    # epoch length
    
    epoch_length = 12
    
    # training file
    pre_training_filename = "tkamb_training.txt"
    #training_filename = "tkamb_training.txt"
    #training_filename_long = "tkamb_training_long.txt"
    #training_filename = "tkamb.txt"
    
    #training_filename = "training_file-two-inputs.txt"

    training_filename = "mapped_data_22093.txt"
    
    # testing file
    testing_filename = "testing_file.txt"
    
    # output file
    output_filename = "output_file_v4_sm00_22093.txt"
    
    # pickle file to store the network
    pickle_filename = "my_network_v4_sm00.p"

    

class sparse_encoder_decoder(object):
    
    def __init__(self, encoding_length):
        
        self.encoding_length = encoding_length
        self.encoding_range = G.encoding_range
        self.encodings = {}  #his is a dictionary of the encodings 
        self.population = list(range(1, self.encoding_range +1))
    
    def sparse_encode(self, input_char):
        
        if input_char in self.encodings.keys():
            return(self.encodings[input_char])
        else:
            # give an input char it will generate an encoding and store the encodings 
            # the dictionary 
            # We will assume that we will prepopulate this dictionary
            # First generater sample of length K 
            
            temp = random.sample(self.population, self.encoding_length)
            e = set(sorted(temp))
        
            while e in self.encodings.values():
                temp = random.sample(self.population, self.encoding_length)
                e = set(sorted(temp))
            
            self.encodings.update({input_char:e})
            return(e)

    def sparse_decode(self, input_char):
        # given an input_char retiurs the tuple
        return(self.encodings[input_char])
    
  
    def load_current_encodings_from_dictionary(self, filename):
        file = open(filename, "r")
        lines = file.readlines()
        for line in lines: 
            c = line.split(":")
            key = c[0]
            value = eval(c[1])
            self.encodings.update({key:value})
        file.close()
        
    def store_current_encoding_in_dictionary(self,filename):
        file = open(filename, "w")
        for key in self.encodings.keys():
            line = str(key) + ":" + str(self.encodings[key]) + "\n"
            file.write(line)
        file.close()
        
    def generate_encodings(self, input_filename):
        
        # takes a filenname which is plain text 
        # removes all spaces and punctuations and get all the words and then the characters
        # generates the encodings
        inputfile = open(input_filename, 'r')
        lines = inputfile.readlines()
        for line in lines:
            words = re.findall(r'\w+', line) 
            #print(words)
            for word in words:
                for character in list(word):
                    self.sparse_encode(character.lower()) 
        
        inputfile.close()

        
   
    def print_encodings(self):
        
        # print the encoded dioctionary
        for key in self.encodings.keys():
            print(key, self.encodings[key])
            

class synapse(object):
    
    def __init__(self, synapse_id, segment_id, local_cell_id): 
        self.id = synapse_id                           # an integer
        self.segment_id = segment_id                   # which segment does the synapse belong to
        self.local_cell_id = local_cell_id                   # which cell does it belong to. This is a tuple
        self.remote_cell_id = ()                       # which cell does it connect to 
                                                       #   we assume each synapse connect to a single cell which is a tuple
        self.p_value = np.random.uniform(0,1)          # the permanance value which is initialized to a value between 0 and 1
        
    
    def get_p_value(self):
        return(self.p_value)
    
    def update_p_value(self, p_value_change):
        old_p_value = self.p_value
        self.p_value = min(max(self.p_value + p_value_change,0),1)
        #print("Updated synapse {0} in segment {1} in cell {2}  with remote cell_id {3} with current p_Value {4} updated to new p_value {5}".format(self.id, self.segment_id, self.local_cell_id, self.remote_cell_id, old_p_value, self.p_value))
        
    
    def get_id(self): 
        return(self.id)
        
    def get_segment_id(self):
        return(self.segment_id)
    
    def get_local_cell_id(self):                       # which cell does this synapse belong to
        return(self.cell_id)
    
    def assign_remote_cell_id(self, remote_cell_id):
        self.remote_cell_id = remote_cell_id
    
    def get_remote_cell_id(self):             # Which cell does this connect to 
        return(self.remote_cell_id)
        


class segment(object):
    
    def __init__(self, segment_id, cell_id, num_synapses):
        self.segment_id  = segment_id                # an integer id
        self.local_cell_id = cell_id               # which cell does this segmeent belong to
        self.num_synapses = num_synapses     # number of synapses
        self.segment_synapses = {}           # dictionary of synapses of this segment. key is the synapse id  value is the synapse
        self.dij = {}                        # dictionary key synapseid value is the permanence value
        self.remote_cell_ids = set()            # a set containing all the remote cell ids of the segment
        self.tildedij = {}                    # dictionary key synapseid value remote-cell-id for synapses with p_value > G.alpha
        self.tildedij_set = set()                 # This is just the set consisting of the remote cellids of connected synapse
        self.dotdij = {}                     # dictionary key synapseid value remote-cell-id for synapses with p_value > 0
        self.dotdij_set = set()                  # This is just the set consisting of the remote cellids of synapses with +ve p_value
    
    def create_synapses(self): 
        
        # this function will create the synapses 
        # initilize dij, tildedij, and dotdij the dictionaries and the sets
                
        for i in list(range(1,self.num_synapses+1)):
            
            # create the synapse
            s = synapse(i, self.segment_id, self.local_cell_id)
            
            # store the synapse
            self.segment_synapses[i] = s
            
            # connect the synapse to a remote cell
            # generate a random cell_id 
            # we want to makem sure that all the synapses of the segment connect to 
            #   distinct cells 

            x = np.random.randint(1, G.number_of_columns+1)
            y = np.random.randint(1, G.cells_per_column+1)
            remote_cell_id = (x,y)
            while (remote_cell_id in self.remote_cell_ids) or (remote_cell_id == self.local_cell_id): 
                x = np.random.randint(1, G.number_of_columns+1)
                y = np.random.randint(1, G.cells_per_column+1)
                remote_cell_id = (x,y)
            
            #print(remote_cell_id)
            # assign the remoote cell-id to the synapse
            s.assign_remote_cell_id(remote_cell_id)
            
            p_value = s.get_p_value()
            
            # add it to dij 
            #k = str(i) + ':' + str(remote_cell_id)
            self.dij.update({i:p_value})
            
            # add the remote cell_id to the set self.remote_cell_ids
            self.remote_cell_ids.add(remote_cell_id) 
            
            # add to dotdij 
            if p_value > 0:
                self.dotdij.update({i:remote_cell_id})
                self.dotdij_set.add(remote_cell_id)

            
            # add to tildedij 
            if p_value > G.alpha:
                self.tildedij.update({i:remote_cell_id})
                self.tildedij_set.add(remote_cell_id)
 
            
                       
    def update_synapse(self, synapse_id, p_value_change, new_remote_cell_id): 
        # This is the main function. iut 
        
        current_remote_cell_id = self.segment_synapses[synapse_id].get_remote_cell_id()
        

        self.segment_synapses[synapse_id].update_p_value(p_value_change)
        new_synapse_p_value = self.segment_synapses[synapse_id].get_p_value()
 
        #print("Updated synapse {0} in segment {1} in cell {2} with current rcell id {3} with new rcell {4} and new p_value: {5}".format(synapse_id, self.segment_id, self.local_cell_id, current_remote_cell_id, new_remote_cell_id, new_synapse_p_value))
        
        self.dij.update({synapse_id:new_synapse_p_value})
        
        if (new_remote_cell_id != ()):
            
            #print("Reconnecting a synapse")
            # yes update the remote cell id 
            self.segment_synapses[synapse_id].assign_remote_cell_id(new_remote_cell_id)
            
            # remove the current rempote cell id in the set remote cell ids
            self.remote_cell_ids.remove(current_remote_cell_id)
            # add  the new new_rempte cell id
            self.remote_cell_ids.add(new_remote_cell_id)
        
        remote_cell_id = self.segment_synapses[synapse_id].get_remote_cell_id()

        if (new_synapse_p_value > 0):
            self.dotdij.update({synapse_id:remote_cell_id})
            #get the remote_cell_id 
            if (remote_cell_id not in self.dotdij_set):
                self.dotdij_set.add(remote_cell_id)
        
        if (new_synapse_p_value > G.alpha):
            self.tildedij.update({synapse_id:remote_cell_id})
        
            if (remote_cell_id not in self.tildedij_set):
                self.tildedij_set.add(remote_cell_id)
            
            
 
    def update_all_synapses_in_dotdij(self, p_value_change):

        # Since we will be mpdifying dotdij we wiill first copy synapse_id
        
        segment_synapse_ids = self.dotdij.keys()
        
        for synapse_id in segment_synapse_ids:
            self.update_synapse(synapse_id, p_value_change, ())
    


    def update_all_synapses_with_active_presynaptic_cell(self, at, p_value_change):
        
        # This function will increase the p_value of all the synapses with active
        # presynaptic cell 
        overlapp_cells = self.dotdij_set.intersection(at)
        
        # We need to find the synapses correspoinding to these remote cells
        #  given an value find the key. We are sure to find different values 
        #  since we made sure of  that when generating
        
        overlapp_synapse_ids = []
        for remote_cell_id in overlapp_cells: 
            for synapse_id in self.dotdij.keys(): 
                if remote_cell_id == self.segment_synapses[synapse_id].get_remote_cell_id(): 
                    overlapp_synapse_ids.append(synapse_id)
        
        for synapse_id in overlapp_synapse_ids:
            self.update_synapse(synapse_id, p_value_change, ())
            

            

    def compute_overlapp_with_tidledij(self, at):

        # computes overlapp with at
        # returns overlapp
        overlapp = self.tildedij_set.intersection(at)
        return(overlapp)
        

    def compute_overlapp_with_dotdij(self, at):
        # computes overlapp with at
        # returns overlapp
        
        overlapp = self.dotdij_set.intersection(at)
        
        return(overlapp)
        
    def update_and_reconnect_synapses(self, number_of_synapses_to_reconnect, overlapp, active_cells_previous):
        
        #print("Max Cell-id: {0}, Number of Synapses to reconnect: {1}, Overlapp: {2}".format(self.local_cell_id, number_of_synapses_to_reconnect, overlapp))
        
        
        # Overlapp contaons remote cellids that are axctive
        # First we will reward the synapses with pre-active cells 
        
        for remote_cell_id in overlapp:
            for synapse_id in self.segment_synapses.keys():
                if self.segment_synapses[synapse_id].get_remote_cell_id() == remote_cell_id: 
                    self.update_synapse(synapse_id, G.large_increment, ()) 
        
        
        # We are here because the strength was not enough
        # We will reconnect the missing number of synapses. This is perhaps too aggresive. 
 
        
        set_of_available_remote_cell_id = set(active_cells_previous).difference(overlapp)
        
        if (len(set_of_available_remote_cell_id) < number_of_synapses_to_reconnect):
            print("Error reconnecting synapses: len of remote  cell ids {0} < number of synapses to reconnect {1}".format(len(set_of_available_remote_cell_id), number_of_synapses_to_reconnect))
 
        else:
            # find the synapses witb the lowest p_values
            synapses_sorted_by_p_value = sorted(self.dij.items(), key=lambda x: x[1])
            #print(synapses_sorted_by_p_value)
            # now for the number of synapses
            #potential_synapse_ids = synapses_sorted_by_p_value.keys()
            
            for i in list(range(number_of_synapses_to_reconnect)):
                # resassgn the remote_cell_id randomly to one 
                
                if (len(set_of_available_remote_cell_id) > 0): 
                    new_remote_cell_id = set_of_available_remote_cell_id.pop()
                
                    while (new_remote_cell_id in self.remote_cell_ids) and (len(set_of_available_remote_cell_id) > 0):
                        new_remote_cell_id = set_of_available_remote_cell_id.pop()
                # update synapse
                
                    self.update_synapse(synapses_sorted_by_p_value[i][0],G.large_increment, new_remote_cell_id)
                else: 
                    break
                
        
    def print_segment_synapse_connection_strenght(self):
         
         for synapse_id in self.segment_synapses.keys():
             print(self.segment_id, synapse_id, self.segment_synapses[synapse_id].p_value, self.segment_synapses[synapse_id].remote_cell_id)
             
    
class cell(object):
    
    def __init__(self, cell_id, num_segments):
        self.cell_id = cell_id                 # this is a tuple
        self.num_segments = num_segments       # how many segments does the cell have
        self.cell_segments = {}                # dictionary of all the segments in the cell
                                               #   the segment-id is the key and segment object is the value 

        self.activity_state = 0
        self.predictive_state = 0
        
        self.active_segments = []             # this is a set of segment ids that have been found to be active                                   # 
        self.segment_strengths = {}                                # create dictionary segment_id: strength wrt to at

        
    def create_segments(self):
        
        for i in list(range(1, self.num_segments+1)):
            #num_synapses = np.random.randint(G.min_num_synapses, G.max_num_synapses)   
            num_synapses = G.fixed_num_synapses  
            s = segment(i, self.cell_id, num_synapses)
            s.create_synapses()
            self.cell_segments[i] = s

            
            
    def get_activity_state(self):
        
        return(self.activity_state)
        
        
    def get_predictive_state(self):
        
        return(self.predictive_state)
        
        
    def update_activity_state(self, value):
        
        # recompute the activity state of the cell   
        
        self.activity_state = value
              

    def compute_predictive_state(self, at):
        
        # this function will compute the predictive state given the currrent feedforward input 
        # We will break as soon as we find the first active segment
        
        count = 0
        
        for segment_id in list(range(1, self.num_segments+1)):
            
            # for this segment find the overlapp with at 
            overlapp = self.cell_segments[segment_id].compute_overlapp_with_tidledij(at)

            if len(overlapp) > G.theta:
                count += 1
                break     
            
        if count == 0:
            self.predictive_state = 0
            return(0)
        else:
            self.predictive_state = 1
            return(1)
    
    
    
    def determine_segment_id_causing_predictive_state(self, at):
        
        # given a set of cell ids in at
        # generates a dictionary of  segment_id and strengths that caused this cell to be in the predictive state  
        
        self.segment_ids_causing_predictive_state = {}
        for segment_id in self.cell_segments.keys(): 
            overlapp = self.cell_segments[segment_id].tildedij_set.intersection(at)
            if len(overlapp) > G.theta:
                self.segment_ids_causing_predictive_state.update({segment_id:len(overlapp)})
        #print("Segments causing predictive state: {0}".format(self.segment_ids_causing_predictive_state))
        

    def determine_segment_id_strengths(self, at):
        
        # given a set of cell ids in at
        # returns a dictionaries of  segment_id and their overlapps
        
        self.segment_id_strengths = {}
        
        for segment_id in self.cell_segments.keys(): 
            overlapp = self.cell_segments[segment_id].dotdij_set.intersection(at)
            self.segment_id_strengths.update({segment_id:overlapp})
        
        
class network(object):
    
    def __init__(self, num_columns, cells_per_column):
        
        self.num_columns = num_columns
        self.cells_per_column = cells_per_column

        self.active_cells_previous_ = []         # a list of cell-ids that were active in the previous timestep
        self.active_cells_current = []          # a list of cell-ids that are active in the current timestep
        self.active_cells_predicted =[]         # list of active cells that were predicted
        self.active_cell_notpredicted_column_ids = ()  # this is a set of column ids
        self.active_cells_notpredicted = []     # this is a lidt of cell_ids
        self.cells_not_activated =[]            # cells in wining columns not activated 
        self.predictive_state_of_cells = {}     # a dictionary tuple values (cell_id) to predictive state
        self.cell_objects = {}                  # a dictionary  key cell_id value cell object
        
        self.list_of_cells_predicted = []
        self.list_of_cells_predicted_previous = []
        self.feed_forward_input = []
        self.prediction_accuracy = []
    
    def initialize_network(self):
        
        for i in list(range(1, self.num_columns + 1)):
            for j in list(range(1, self.cells_per_column + 1)):
                cell_id = (i,j)
                #num_segments = np.random.randint(G.number_of_segments_min,G.number_of_segments_min+1)
                num_segments = G.fixed_number_of_segments
                c = cell(cell_id, num_segments)
                c.create_segments()
                self.cell_objects.update({cell_id:c})
                
                # initialize cell predictive state to 0 
                self.predictive_state_of_cells.update({cell_id:0})
                

        
    def compute_cell_activity_state(self, feed_forward_input):
        
        # First save the current set of active cells 
        self.active_cells_previous = self.active_cells_current
        
        # Next we reset the cells that were active in  the previous time step
        for cell_id in self.active_cells_previous: 
            self.cell_objects[cell_id].update_activity_state(0)
        
        
        # reset the data structures for the three types of cells to be generated by the feedforward input 
        self.active_cells_current = []
        self.active_cells_predicted =[]
        self.active_cells_notpredicted = []
        self.active_cell_notpredicted_column_ids = set()
        self.cells_not_activated =[]
        
        # The feed forward input is a set of column id 
        # Determine the active cell in  the current time step
        
        for i in feed_forward_input: 
            
            cu_sum = 0
            for j in list(range(1, self.cells_per_column+1)):
        
                cell_id = (i,j)

                # check the predictive states of the cell 
 
                if (self.cell_objects[cell_id].get_predictive_state() == 1): 
                    cu_sum += 1
                    if (cu_sum == 1):
                        self.cell_objects[cell_id].update_activity_state(1)
                        #Add the cell-id the list of active cells
                        self.active_cells_current.append(cell_id)
                        self.active_cells_predicted.append(cell_id)
                    elif (cu_sum > 1):
                        self.cell_objects[cell_id].update_activity_state(0)
                        self.cells_not_activated.append(cell_id)
            
            # if none of the cells were in  the preditive state 
            if (cu_sum == 0):  # none of the cells in  the winning column were in the predictive state
                self.active_cell_notpredicted_column_ids.add(i)   
                for j in G.list_of_ids_in_a_column:
                    cell_id = (i,j)
                    self.cell_objects[cell_id].update_activity_state(1)
                    self.active_cells_current.append(cell_id)
                    self.active_cells_notpredicted.append(cell_id)
            
            
            # There are cells that were predicted but not activated
            cell_predicted_but_not_activated = set(self.list_of_cells_predicted).difference(set(self.active_cells_predicted))
            self.cells_not_activated.extend(list(cell_predicted_but_not_activated))
    
    def compute_cell_predictive_state(self, feed_forward_input):
                
         self.list_of_cells_predicted_previous = self.list_of_cells_predicted
        
         # reset the list of cells predicted
         self.list_of_cells_predicted = []
         
         # We determine the predictive state of each cell. 
         # We will go through each cell and compute the predictive state
         for i in G.list_of_column_ids:
            for j in G.list_of_ids_in_a_column:
                cell_id = (i,j)
                # the following
                result = self.cell_objects[cell_id].compute_predictive_state(self.active_cells_current)
                if (result == 1):
                    self.list_of_cells_predicted.append(cell_id)
                     
    
    def print_predicted_cell(self): 
        print("List of cell predcited {0}".format(self.list_of_cells_predicted))
    
    def update_segment_and_synapse(self, feed_forward_input):
        
        
        # Consider the three cases here all with respect to the set of cell in the columns 
        # correspoinding to the feedforward input
        
        # Case 1: cell was in predictive state and got activated
        #         These cell ids are in self.active_cells_predicted = []
        
        # Consider each cell at a time
        #print("List of active cells predicted {0}".format(self.active_cells_predicted))
        for cell_id in self.active_cells_predicted:
            
            # get all the segments that caused the cell to be in the predictive state in the previous
            #  timestep


            # To do this consider the previous active list and find all the active segments 
            # for which the intersection with tildeD is greater than theta
            self.cell_objects[cell_id].determine_segment_id_causing_predictive_state(self.active_cells_previous)
            
            
            # Find the man
            
            for segment_id in self.cell_objects[cell_id].segment_ids_causing_predictive_state.keys():
                
                # For each segment we update the synaptic strengths as follows 
                # self.cell_objects[cell_id].cell_segments[segment_id].print_segment_synapse_connection_strenght()
                # We decrement the p_value of all the synapses in dotdij
                self.cell_objects[cell_id].cell_segments[segment_id].update_all_synapses_in_dotdij(G.small_decrement)
                
                # next we find the overlapp of a(t-1) with dotdij and increment the  corresponding
                # synapses by a large value
                self.cell_objects[cell_id].cell_segments[segment_id].update_all_synapses_with_active_presynaptic_cell(self.active_cells_previous, G.large_increment)
                
        
        # Case 2: No cell in these winning columns were in a predictive state
        # Hence all the cell in the columns became active
        
        # This is a set of column_ids 

        # for each column in the set 
        #print(self.active_cell_notpredicted_column_ids)
        #print("List of cells not-predicted column ids {0}".format(self.active_cell_notpredicted_column_ids))
        for i in self.active_cell_notpredicted_column_ids:
            max_cell_id = ()
            max_segment_id = -1
            max_segment_overlapp = set()
            for j in G.list_of_ids_in_a_column:
                cell_id = (i,j)
                # get a list of segment id and their overlapps with the previous active cells
                self.cell_objects[cell_id].determine_segment_id_strengths(self.active_cells_previous)
                for segment_id in self.cell_objects[cell_id].segment_id_strengths.keys():
                    if len(self.cell_objects[cell_id].segment_id_strengths[segment_id]) > len(max_segment_overlapp):
                        max_cell_id = cell_id
                        max_segment_id = segment_id
                        max_segment_overlapp = self.cell_objects[cell_id].segment_id_strengths[segment_id]
                
            # At this point we have the following case
                        
            # Case 1: We have found a segment that has some strength max_strength > 0 
            #         In thism case we find the number of additional synapses and update
            #         their strengths
            
            #print("Max cell_id: {0}, max_segment_id: {1}, len of overlap: {2}".format(max_cell_id, max_segment_id, len(max_segment_overlapp)))
            
            if (len(max_segment_overlapp) > 0):
                 num_synapses_to_reconnect = G.theta - len(max_segment_overlapp) + 1
                 self.cell_objects[max_cell_id].cell_segments[max_segment_id].update_and_reconnect_synapses(num_synapses_to_reconnect, max_segment_overlapp, self.active_cells_previous)
            
            # # Case 2: max_strength > 0 
            # #     In this case we consider a random segment and consider the lowest G.theta + synapases 
            # #     and reassign remote id's and update strengths
            elif (len(max_segment_overlapp) == 0):
                 j = np.random.randint(1, G.cells_per_column+1)
                 max_cell_id = (i,j)
                 max_segment_id = np.random.randint(1, self.cell_objects[max_cell_id].num_segments+1) 
                 num_synapses_to_reconnect = G.theta + 1
                 self.cell_objects[max_cell_id].cell_segments[max_segment_id].update_and_reconnect_synapses(num_synapses_to_reconnect, max_segment_overlapp, self.active_cells_previous)
                
        
            
        
        # Case 3: cells that did not  become active but had active dendritic segments 
        #         For these segments we decay the synaptic strenghts
        for cell_id in self.cells_not_activated:
            
            # We find the strengths of the segments 
            self.cell_objects[cell_id].determine_segment_id_strengths(self.active_cells_previous)
            for segment_id in self.cell_objects[cell_id].segment_strengths.keys(): 
                if self.cell_objects[cell_id].segment_strengths[segment_id] > G.theta:
                    # For the active segments decrement p_value and dotdij
                    for synapse_id in self.cell_objects[cell_id].segment_strengths[segment_id].segment_synapses.keys():
                        self.cell_objects[cell_id].segment_strengths[segment_id].update_synapse(synapse_id, G.very_small_decrement, ())
 
                
        
    def train_network(self, sdr, input_filename, output_filename):
        #print(filename)
        input_file = open(input_filename, 'r')
        output_file = open(output_filename, 'w')
        lines = input_file.readlines()
        epoch_number = 1
        index_within_epoch = 0
        for line in lines:
            #print("Line: {0}:".format(line), file=output_file)
            #while num_words < 10000:
            #words = line.rsplit()
            words = re.findall(r'\w+', line) 
            #print(words)
            for word in words:
                #print("Word: {0}:".format(word), file=output_file)
                n = len(word)
                for i in list(range(n)):
                    input_char = word[i].lower()
                    #print("List of cells activated: {0}".format(self.active_cells_current))
                    #print("List of cells predicted: {0}".format(self.list_of_cells_predicted))      
                    # determine the feed forward input
                    self.feed_forward_input = sdr.sparse_encode(input_char)
                    #print(word[i], feed_forward_input)
    
                    #print("Feed froward input: {0}".format(sorted(self.feed_forward_input)))
                    
                    # compute_cell_actovity_state
                    self.compute_cell_activity_state(self.feed_forward_input)
                    # print("List of cells activated: {0}".format(self.active_cells_current))
                    
                                        
                    predicted_active_overlapp = set(self.list_of_cells_predicted).intersection(set(self.active_cells_current))
                    list_of_columns_predicted = []
                    for cell_predicted in self.list_of_cells_predicted:
                        list_of_columns_predicted.append(cell_predicted[0])   
                    self.prediction_accuracy.append(len(predicted_active_overlapp)/len(self.feed_forward_input))
                    print("{0} : {1} : {2} : {3} : {4} : {5}".format(epoch_number, index_within_epoch, input_char,len(predicted_active_overlapp)/len(self.feed_forward_input), sorted(self.feed_forward_input), sorted( list_of_columns_predicted)), file=output_file)
           
                
                    # update the synpatic strengths
                    self.update_segment_and_synapse(self.feed_forward_input)
                    
                    index_within_epoch += 1
                    
                    if index_within_epoch < G.epoch_length: 
                        
                        # update_cell_predictive_state
                        self.compute_cell_predictive_state(self.feed_forward_input)
                        #print("Next list of cell predicted: {0}".format(self.list_of_cells_predicted))     
                    else: 
                       index_within_epoch = 0 
                       epoch_number += 1
                    
                    # Done training with current input 
                    #print("Completed training for input {0} with ffinput {1}".format(word[i], self.feed_forward_input))
        input_file.close()
        output_file.close()
    
    def test_network(self, sdr, input_filename, output_filename):
        output_file = open(output_filename, 'w')
        input_file = open(input_filename, 'r')
        lines = input_file.readlines()
        for line in lines:
            #words = line.split()
            words = re.findall(r'\w+', line) 
            for word in words:
                n = len(word)
                for i in list(range(n)):
                    # determine the predicted input
                    predicted_input = self.compute_predicted_state()
                    
                    
                    # determine the feed forward input
                    feed_forward_input = sdr.sparse_decode(word[i])
                    
                    print("{0}, {1}".format(feed_forward_input, predicted_input), file=output_filename)
                    
                    # compute_cell_actovity_state
                    self.compute_cell_activity_state(feed_forward_input)
                
                    # update_cell_predictive_state
                    self.update_cell_activity_state(feed_forward_input)
            
                    # update_synaptic_strengths
                    self.update_synaptic_strength()
        
        input_file.close()
        output_file.close()
        
   
    
def main():
    os.chdir(G.WORKING_DIRECTORY)
    
    np.random.seed(1234567898)
    
    # Create the encoder decoder object and encode the alphabet set
    sdr = sparse_encoder_decoder(G.encoding_length)
    
    try: 
        #encoding_filename = open(G.encoding_filename, 'r')
        sdr.load_current_encodings_from_dictionary(G.encodings_filename)
        
    except FileNotFoundError: 
        sdr.generate_encodings(G.pre_training_filename)
    
    finally:
        sdr.print_encodings()
    
    
    try:
        pickle_file = open(G.pickle_filename, 'rb')
        my_network = pickle.load(pickle_file)
        # Found a pickle 
        print("Importing network from a file")
    
    
    except FileNotFoundError: 
        # Create the network 
        my_network = network(G.number_of_columns, G.cells_per_column) 
    
        # Initializer the network
        my_network.initialize_network()
        print("Network created and initiaized")

   
    finally: 
    
        # Reset the data collection 
        my_network.prediction_accuracy = []
    
        # Train the network
        my_network.train_network(sdr, G.training_filename, G.output_filename)
             
        # Test the network
        #my_network.test_network(sdr, G.testing_filename, G.output_filename)

        save_network_in_file = open(G.pickle_filename, 'wb')
        pickle.dump(my_network, save_network_in_file)
    
        save_network_in_file.close() 
        
        sdr.store_current_encoding_in_dictionary(G.encodings_filename)
        print("Experiment Completed")
    
if __name__ == '__main__': main()   
        