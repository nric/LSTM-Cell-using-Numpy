#%%
import numpy as np
from scipy.special import expit as sigmoid

def forget_gate(x,h,s,W_x_f,B_x_f,W_h_f,B_h_f):
    """Defines the operations of the forget gate.
    The output of the forget gate gets multiplied with the previous cell state and
    is a number between 0 and 1. Hence, it decides "how much is let thought". 
    params:
        x : new event
        h : previous hidden state (Short term memory)
        s : previous cell state (Long term memory)
        W_x_f : Weights for new event of foget gate
        B_x_f : Bias of the latter
        W_h_f : Weights for hidden state of foget gate
        B_h_f : Bias of the latter
    """
    forget_h = np.dot(W_h_f , h) + B_h_f
    forget_x = np.dot(W_x_f , x) + B_x_f
    return np.multiply(sigmoid(forget_h + forget_x) , s)


def input_gate(x,h,s,W_x_i,B_x_i,W_h_i,B_h_i,W_x_l,B_x_l,W_h_l,B_h_l):
    """Defines the operations of the input gate.
    Splitted into Learn Path and Input Path with seperate weights for each.
    params:
        x : new event
        h : previous hidden state (Short term memory)
        s : previous cell state (Long term memory)
        W_x_i : Weights for new event of ignore gate
        B_x_i : Bias of the latter
        W_h_i : Weights for hidden state of ignore gate
        B_h_i : Bias of the latter
        W_x_l : Weights for new event of learn gate
        B_x_l : Bias of the latter
        W_h_l : Weights for hidden state of learn gate
        B_h_l : Bias of the latter
    """
    ignore_h = np.dot(W_h_i,h) + B_h_i
    learn_h = np.dot(W_h_l,h) + B_h_l
    ignore_x = np.dot(W_x_i,x) + B_x_i
    learn_x = np.dot(W_x_l,x) + B_x_l
    ignore_result = sigmoid(ignore_h + ignore_x)
    learn_result = np.tanh(learn_h +  learn_x)
    return np.multiply(ignore_result,learn_result)


def cell_state(forget_gate_out, input_gate_out):
    """Calcultated the cell state. 
    Is carreid to the next cell state. Is also used to calc the hidden state aka output.
    params:
        forget_gate_out : Output of the forget gate
        input_gate_out : Output of the input gate
    """
    return forget_gate_out + input_gate_out


def output_gate(x,h,cell_state,W_h_o,B_h_o,W_x_o,B_x_o):
    """Defines the operations of the output gate = hidden state
    Splitted into Learn Path and Input Path with seperate weights for each.
    params:
        x : new event
        h : previous hidden state (Short term memory)
        cell_state : state of this cell
        W_x_o : Weights for new event of output gate
        B_x_o : Bias of the latter
        W_h_o : Weights for hidden state of output gate
        B_h_o : Bias of the latter
    """
    output_h = np.dot(W_h_o,h) + B_h_o
    output_x = np.dot(W_x_o,x) + B_x_o
    output_cell_state = np.tanh(cell_state)
    return np.multiply( sigmoid(output_h + output_x) , output_cell_state)

#%%
#Everything from here is just to test. It is taken from: https://towardsdatascience.com/the-lstm-reference-card-6163ca98ae87
#Set Parameters for a small LSTM network
input_size  = 2 # size of one 'event', or sample, in our batch of data.
hidden_dim  = 3 # dimensionality of the state and output. Equvalent to "units" param in keras.layers.lstm
output_size = 1 # desired model output. Not the LSTM state or LSTM output which is equal to hidden_dim

def model_output(lstm_output, fc_Weight, fc_Bias):
    '''Takes the LSTM output and transforms it to our desired 
    output size using a final, fully connected layer'''
    return np.dot(fc_Weight, lstm_output) + fc_Bias

#define test data. The weights and biases are "random".
#The data here is taken from this example:
#https://towardsdatascience.com/the-lstm-reference-card-6163ca98ae87

#Event (x) Weights and Biases for all gates
Weights_xi = [[ 0.3813, -0.4317],[ 0.4705,  0.3694],[ 0.4851, -0.4427]]  # shape  [h, x]
Weights_xf = [[-0.3875,  0.2747],[-0.5389,  0.5706],[ 0.1229,  0.0746]]  # shape  [h, x]
Weights_xl = [[-0.4937,  0.1840],[ 0.2483,  0.0916],[ 0.5553,  0.1734]]  # shape  [h, x]
Weights_xo = [[-0.5120,  0.4851],[ 0.1960, -0.2754],[-0.5303,  0.3291]] # shape  [h, x]

Bias_xi = [-0.3205, -0.3293, -0.1545]  #shape is [h, 1]
Bias_xf = [-0.1866, -0.3926,  0.4666]  #shape is [h, 1]
Bias_xl = [0.0644,  0.2632, 0.4282]  #shape is [h, 1]
Bias_xo = [-0.3741,  0.4407, -0.2892] #shape is [h, 1]

#Hidden state (h) Weights and Biases for all gates
Weights_hi =[[ 0.5487, -0.4730,  0.0316],[ 0.2071, -0.2726, -0.1263],[-0.3855, -0.2730, -0.5264]]  #shape is [h, h]
Weights_hf = [[-0.0134,  0.3423,  0.2808],[ 0.5424, -0.5071, -0.0710],[ 0.5621,  0.0945, -0.1628]]  #shape is [h, h]
Weights_hl = [[-0.5200,  0.2687,  0.4383],[ 0.4630,  0.4833,  0.1130],[ 0.4115, -0.1453,  0.4689]] #shape is [h, h]
Weights_ho = [[-0.0494, -0.1191, -0.2870],[ 0.3074,  0.2336,  0.3672],[-0.3690, -0.3070,  0.5464]] #shape is [h, h]

Bias_hi = [-0.0919,  0.4369,  0.5323]  #shape is [h, 1]
Bias_hf = [0.5068,0.3320,  0.5366]  #shape is [h, 1]
Bias_hl = [-0.2080, -0.0367,-0.1975]  #shape is [h, 1]
Bias_ho = [-0.0424, -0.0702,  0.3085] #shape is [h, 1]

#--------------------------------------------------------------------
# Final, fully connected layer Weights and Bias
fc_Weight = [ 0.3968, -0.4158, -0.3188] #shape is [h, output_size]
fc_Bias = [-0.1776] #shape is [,output_size]

#%%
data = np.array([[1,1],[2,2],[3,3]])

#Initialize cell and hidden states with zeroes
h = np.zeros(hidden_dim)
s = np.zeros(hidden_dim)

#Loop through data, updating the hidden and cell states after each pass
for eventx in data:
  f = forget_gate(eventx, h, s, Weights_xf, Bias_xf, Weights_hf, Bias_hf)
  i = input_gate(eventx, h, s, Weights_xi, Bias_xi, Weights_hi, Bias_hi, Weights_xl, Bias_xl, Weights_hl, Bias_hl)
  s = cell_state(f,i)
  h = output_gate(eventx, h, s, Weights_ho, Bias_ho, Weights_xo, Bias_xo)
  print(f"cell_state:{s} hidden_state:{h}")
print(f"model_output:{model_output(h, fc_Weight, fc_Bias)}")


"""Should be:
np Hidden State: [-0.11898849  0.47585365  0.32522364]
np Cell State: [-0.3555854   1.17887101  1.3025983 ]
"""

#%%
