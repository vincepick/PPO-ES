# PPO-ES + SPACE: SPACE-ES Abstract

Algorithms play a crucial role in solving a wide variety of problems
across many spaces. Most algorithms have their own parameters that
need to be tuned to achieve the best performance. In some cases, in-
stead of finding the best static parameter setting for an algorithm, it
is highly beneficial to configure the parameter values while the algo-
rithm is running. Dynamic Algorithm Configuration (DAC) is a new
area of research dedicated to solving this task in an automated and
data-driven fashion. A policy is learned to map the current state of
the algorithm, to a parameter setting that best suits it. A common
approach for learning this mapping is through deep reinforcement
learning.
This project aims to investigate the effectiveness of a novel self-paced
learning method, SPACE, for selecting instances during the training
of deep-RL algorithms in DAC contexts. Training a model to dy-
namically configure parameters across multiple problem instances is
a challenge. SPACE has the potential to significantly improve this
process, and will provide insights into the development of robust and
sample-efficient DAC methods.


For more details, please see accompanying report. 

# Package Install List 
codex 
coco-experiment
scipy
matplotlib
gymnasium
stable_baselines3

# Overall Directory Structure

Prior to submission, some of the data was aggregated into seperate folders to be clearer, and much had to be deleted to account for the submission size limit. All code components themselves are unchanged since experiment running, just moved. For example, all AUC ranges used for scaling are now located in a corresponding auc_ranges folder. 
 

 /auc_ranges        - contains the AUC ranges used for scaling data for more effective analysis. Generated using the script at src/utilities/retrieve_all_global_mins.py 
 /final_data_truncated        - contains some of the data used for analysis within the dissertation (most had to be removed for submission as it totaled 21 gigs, which far exceeded the limit)
 run.py             - entry point to run the program, udpated significantly to hanldle additional parameters, granting user more control of the experiment. 
 /src               - contains the source code for the project
    Files which contain most of the additional contributions within src:
    /analysis/comparing_algorithms.py   - used to add the additional parameters used during experimentaiton
    /callbacks/callbacks.py             - callbacks used to facilitate SPACE-ES curriculum generation and handling
    /environment/es_env.py              - refactored and updated ES environment for SPACE-ES integration. 
    /models/ppo_es_model.py             - added additional functionalities, logging, callback handling, and more robust data generation using numerous runs across multiple seeds. 
    /config/config.py                   - updated configurations for more detailed experiments 
 bbob_optima.json   - contains the optimal values (f-opt) of each of the BBOB benchmark functions. Used to calculate the AUCs. 

 /generating_scripts - scripts used for generating the graphs used in the final report evaluation and throughout development. More detail in final_graph_generation.md
 /plots: All plots used within the dissertation evaluation section, and more


# Default Configuration Used for All Runs 
For all combinations of hyper parameters used for the evaluation, please see the *experimentation* section of the accompanying report. 

- Population size: 25
- Initial step size: 0.5
- Max function evaluations: FES_MAX = 1000
- Seeds: 5 per experiment

# To activate venv

`source venv/bin/activate`

# Experiment Scripts
## For generating models in parallel
Generates all the models used within this dissertation. Specific configurations set as variables in the script itself. 
bash `generate_for_gnu_parallel.sh`

## For running testing on BBOB benchmark functions. 
Generates all the test model data used within this dissertation on the BBOB benchmark functions (dimension 40)
bash `test_for_gnu_parallel.sh`

## For running testing on lagescale BBOB benchmark functions.
Generates all the test model data used within this dissertation on the BBOB-largescale benchmark functions (dimension 80, 160, 320, 640)
bash `test_for_gnu_parallel_largescale.sh`



# For a single run of the program itself, not generating all test data
    python run.py \
        --train \
        --test_models \
        --type bbob \
        --instance 1 \
        --dim 40 \
        --experiment_name <output_dir_name> \
        --use_space <what SPACE-ES operation mode to use> \
        --instance_ordering <what instance ordering to use> \
        --num_training_functions <the number of training functions in the curriculum> \
        --num_steps_per_rollout <number of steps collected in the rollout buffer for each policy update>

# Graph Generation
For additional instructions on graph generation, see `final_graph_generation.md`


### SPACE_ES and Ordering values

SPACE-ES Operation Mode Values
    NO_SPACE = 0
    JUST_SIZES = 1
    INSTANCE_STATE = 2
    ONE_GENERATION = 3

Ordering Values
    ABSOLUTE = 0
    IMPROVEMENT = 1
    RELATIVE_IMPROVEMENT = 2
    NONE = 3



