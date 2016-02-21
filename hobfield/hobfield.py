#
# Leon Adams 
#
# Python Module for running a hopfield network to relocate the memory from a perturbed image.
# The raw data set is represented in png image format. This code takes the three color channels (rgb)
# Converts to a single channel gray scaled image and then transforms the output to a [-1,1] vector
# for use in calculation of a hobfield neural network.
#
# Dependencies: numpy; matplotlib
# 
# Usage
# Can use as normal python module or can be used as a python script.
# When calling from command line as script supply corruption percent at end of call
#
# Example: python hopfield.py 2 3 4
# This will produced 2, 3, and 4 percent perturbation on the image file and then
# attempt to locate closest memorized pattern using hopfield network with hebb learning rule.
# If called without perturbation parameters default to [1, 5, 10, 15, 20, 25] corruption percentages.

# Output: output of the execution is a series of images showing first the perturbed
# image with the corrupted percentages in the title. Then we show the closest memorized
# image found from the hobfield network.


# begin import needed libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# end import libraries

def rgb_to_gray_array(rgb):
    '''
    Helper function to convert from rgb tensor to matrix gray-scaled image representation.
    Input: rgb tensor matrix of the three rgb color channels.
    output: numpy array of gray-scaled numeric values.
    '''
    return np.dot(rgb[...,:3], np.array([0.299, 0.587, 0.114]))    

def read_images(filenames):
    '''
    Read images to set to memory. Convert from rgb tensor to gray scale representation.
    Takes a list of filenames in directory containing pixel images. Returns a list
    of numpy arrays converted to gray-scale.
    '''
    data = [( mpimg.imread(number) ) for number in filenames]
    return data, data[0].shape

def create_vector_image(data_array):
    ''' 
    Converts a gray-scaled image to [-1, +1] vector representation for hopfield networks.    
    '''
    data_array = np.where(data_array < 0.99, -1, 1)
    return data_array.flatten()
    
def print_unique_cnts(array):
    print( np.unique(array, return_counts=True ) )
    
def train(memories):
    '''
    Training function for hobfield neural network. Trained with Hebb update rule.
    '''
    rate, c = memories.shape
    Weight = np.zeros((c, c))
    for p in memories:
        Weight = Weight + np.outer(p,p)
        
    Weight[np.diag_indices(c)] = 0
    return Weight/rate

    
def look_up(Weight_matrix, candidate_pattern, shape, percent_corrupted, steps=5):
    '''
    Given a candidate pattern, lookup closet memorized stable state. Return the
    stable memorized state.
    '''
    sgn = np.vectorize(lambda x: -1 if x<0 else 1)
    
    img = None
    for i in range(steps):
        im = show_pattern(candidate_pattern, shape) 
        candidate_pattern = sgn(np.dot(candidate_pattern, Weight_matrix))
        if img is None:
            img = plt.imshow(im, cmap=plt.cm.binary, interpolation='nearest')
            plt.title(str(percent_corrupted) + ' percent corrupted pixels')
        else:
            img.set_data(im)
        plt.pause(.2)
        plt.draw()

    return candidate_pattern
        
def hopfield_energy(Weight, patterns):
    '''
    Calculates the current energy value for a given pattern and weight matrix.
    '''
    return np.array([-0.5*np.dot(np.dot(p.T, Weight), p) for p in patterns])

def show_img(image, shape):
    '''
    Helper function to produce visualization of an image.
    '''
    plt.imshow(image.reshape(shape), cmap=plt.cm.binary, interpolation='nearest')
    plt.show()

def show_pattern(pattern, shape):
    return np.where(pattern < 0, 0, 1).reshape(shape)

    
def corrupts(pattern, percentage):
    '''
    Helper function for deriving corrupted pattern images. Specify stable memory pattern
    and the percentage of pixels to switch.
    '''
    
    counts = int( 2*np.ceil( len(pattern) * percentage / 200 ) )
    neg_mask = np.where(pattern <= 0)[0]
    pos_mask = np.where(pattern > 0)[0]
    
    neg_corrupt_indices = np.random.choice(neg_mask, counts/2, replace = False)
    pos_corrupt_indices = np.random.choice(pos_mask, counts/2, replace = False)
    
    corrupt_pattern = np.copy(pattern)
    corrupt_pattern[neg_corrupt_indices] = 1
    corrupt_pattern[pos_corrupt_indices] = -1
    return corrupt_pattern

data, shape = read_images(['C.png', 'D.png', 'J.png'])

stable_memories = np.array([create_vector_image(rgb_to_gray_array(array)) for array in data ])
norm_weight_matrix = train(stable_memories)

def test_stable_memories(stable_memory_patterns, corrupt_perentages):
    for memory in stable_memory_patterns:
        for percent in corrupt_perentages:
            crpt_memory = corrupts(memory, percent)
            look_up(norm_weight_matrix, crpt_memory, shape[0:2], percent_corrupted = percent, steps=5)


if __name__ == "__main__":
    user_input = sys.argv
    
    if len(user_input) > 1:
        test_stable_memories(stable_memories, [float(i) for i in user_input[1:] ])
    else:
        test_stable_memories(stable_memories, [1, 5, 10, 15, 20, 25])
