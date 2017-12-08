# MIT 6.034 Lab 6: Neural Nets
# Written by Jessica Noss (jmn), Dylan Holmes (dxh), Jake Barnwell (jb16), and 6.034 staff

from nn_problems import *
from math import e
INF = float('inf')

#### NEURAL NETS ###############################################################

# Wiring a neural net

nn_half = [1]

nn_angle = [2,1]

nn_cross = [2,2,1]

nn_stripe = [3,1]

nn_hexagon = [6,1]

nn_grid = [4,2,1]

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    return x >= threshold

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1/(1+e**(-steepness*(x-midpoint)))

              
def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    if x<0:
        return 0
    return x

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -.5*(desired_output-actual_output)**2

# Forward propagation

def node_value(node, input_values, neuron_outputs):  # STAFF PROVIDED
    """Given a node, a dictionary mapping input names to their values, and a
    dictionary mapping neuron names to their outputs, returns the output value
    of the node."""
    if isinstance(node, basestring):
        return input_values[node] if node in input_values else neuron_outputs[node]
    return node  # constant input, such as -1

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    neuron_outputs = {}
    for node in net.topological_sort():
        input_nodes = net.get_incoming_neighbors(node) #list of nodes whose outputs are connected to current node
        weighted_inputs_sum = sum([node_value(input_node,input_values,neuron_outputs)\
                                   * net.get_wires(input_node,node)[0].get_weight() \
                                   for input_node in input_nodes])
        neuron_outputs[node] = threshold_fn(weighted_inputs_sum)
        
    return (neuron_outputs[net.get_output_neuron()], neuron_outputs)
        
        
# Backward propagation warm-up
def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    max_setup = (func(*inputs), inputs[:])
    for i in range(len(inputs)):
        for j in range(len(inputs)):
            for k in range(len(inputs)):
                current_setup_vector = map(lambda x: convert_to_stepsize(x,step_size),(i,j,k))
                current_setup_args = map(sum,zip(inputs[:], current_setup_vector))
                current_setup = (func(*current_setup_args), current_setup_args)
                if current_setup[0]>max_setup[0]:
                    max_setup = current_setup
    return max_setup
                
# Helper function
def convert_to_stepsize(input, step_size):
    """Given an integer in {0,1,2} anda step_size,
    this function maps the integer as follow:
    0 ==> 0 
    1 ==> - step_size
    2 ==> + step_size
    returns the mapped number."""

    if input == 1:
        return -1* step_size
    elif input == 2:
        return step_size
    return 0
    

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    start_node, end_node = wire.startNode, wire.endNode
    output = set([start_node,end_node,wire])

    wire_list = net.get_wires(startNode = end_node)

    for wire in wire_list:
        output.add(wire)
        output.add(wire.endNode)
    
    return output
   


# Backward propagation
def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    neurons_coeffs = {}
    neurons_list = net.topological_sort()
    
    n_f = neurons_list.pop() # final layer
    o = neuron_outputs[n_f]
    neurons_coeffs[n_f] = o*(1-o)*(desired_output-o)

    while neurons_list != []:
        curr_neuron = neurons_list.pop()
        o = neuron_outputs[curr_neuron]

        summation = 0
        for neuron in net.get_outgoing_neighbors(curr_neuron):
            summation += net.get_wires(curr_neuron,neuron)[0].get_weight()*neurons_coeffs[neuron]

        neurons_coeffs[curr_neuron] = o*(1-o)*summation

    return neurons_coeffs
            
        

    

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    neuron_delta_weights = calculate_deltas(net, desired_output, neuron_outputs)
    neuron_list = net.topological_sort()

    while neuron_list != []:
        curr_neuron = neuron_list.pop()
        wires = net.get_wires(endNode = curr_neuron)
        
        for wire in wires:
            start_node = wire.startNode
            if start_node in input_values:
                wire.set_weight(wire.get_weight() + r*input_values[start_node]*neuron_delta_weights[curr_neuron])
            elif start_node in net.neurons:
                wire.set_weight(wire.get_weight() + r*neuron_outputs[start_node]*neuron_delta_weights[curr_neuron])
            else:
                wire.set_weight(wire.get_weight() +r*start_node*neuron_delta_weights[curr_neuron])
    return net



def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    n_f_o, neuron_outputs = forward_prop(net, input_values, threshold_fn=sigmoid)
    current_accuracy = accuracy(desired_output, n_f_o)
    iterations_count = 0

    while current_accuracy < minimum_accuracy:
        net = update_weights(net, input_values, desired_output, neuron_outputs,r)
        iterations_count +=1
        
        n_f_o, neuron_outputs = forward_prop(net, input_values, threshold_fn=sigmoid)
        current_accuracy = accuracy(desired_output, n_f_o)

    return (net, iterations_count)

# Training a neural net

ANSWER_1 = None
ANSWER_2 = None
ANSWER_3 = None
ANSWER_4 = None
ANSWER_5 = None

ANSWER_6 = None
ANSWER_7 = None
ANSWER_8 = None
ANSWER_9 = None

ANSWER_10 = None
ANSWER_11 = None
ANSWER_12 = None


#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None

