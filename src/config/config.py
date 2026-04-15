TRAIN_INSTANCE = 1
POP_SIZE = 25 # the population size evaluated for each step (each individual number)
FES_MAX = 1000 # the total number of evaluations, but keeps being cut by POP_SIZE 
NUM_RUN = 25 # the number of individual times we run each, each full block in the numpy arrays

# To account for the one hot encoding, this is just the size of the state not including the space for instances
STATE_SIZE = 2 
ACTION_SIZE = 1
SIGMA_0 = 0.5
# All the episodes which will have intermediate models produced
EPISODES = [1, 1*60, 2*60, 3*60, 4*60, 5*60, 6*60, 7*60, 8*60, 9*60, 10*60, 11*60, 12*60, 13*60, 14*60, 15*60, 16*60, 17*60, 18*60, 19*60, 20*60]
EPISODES_MINI= [1, 2*60, 4*60,8*60,16*60,20*60]
STABILITY_THRESHOLD = 3



