import random

from src.environment.es_env import ES_Env
from src.callbacks.callbacks import LearningRateScheduler, SaveOnBestTrainingRewardCallback, UpdateEnvCallback
from src.config.config import NUM_RUN, TRAIN_INSTANCE, EPISODES
from src.utilities.tools import linear_schedule, save_data
from stable_baselines3 import PPO
# from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import os



def test_model(env, model_path, data_path, episode, problem_index, instance, experiment_logger, type_algorithm, num_seeds):


    experiment_logger.debug(f"About to run tests inside test_model on %d ", problem_index)


    if type_algorithm == "trpo":
    # Loading one of the saved models
        # model = TRPO.load(model_path, env=env)
        return
    else: 
        model = PPO.load(model_path, env=env)
    all_fitness_values = []

    num_each = NUM_RUN // num_seeds
    # The number of indepedent testing runs, set to 25 in the config
    for num in range(num_each):

        experiment_logger.debug(f"Attempting problem run: %d", num)
        temp = env.envs[0].reset()
    
        obs = temp[0]
        # Making it not have an instance in the observation for this prediction 
        obs[2] = 0.0
        first_value = temp[1]
        
        fitness_values = []
        fitness_values.append(first_value)

        while env.envs[0].unwrapped.countevals <= env.envs[0].unwrapped.fes_max:  # Adjust the number of steps as needed

            action, _states = model.predict(obs, deterministic=True)

            obs, rewards, dones, info = env.step(np.array([action]))
            obs = obs[0]
            if dones:
                break
            fitness_values.append(env.envs[0].unwrapped.current_best_fitness)
        all_fitness_values.append(np.array(fitness_values))


    all_fitness_values_arr = np.array(all_fitness_values)

    return all_fitness_values_arr




class PPO_ES:
    def __init__(self, base_dir, cuda_device, logger=None, config_info=None):
        self.base_dir = base_dir
        self.cuda_device = cuda_device
        self.space_logger = logger
        
        self.seeds = [42, 789, 1738, 2026, 7021]
        self.num_models_to_gen = 1
        self.results_dir = os.path.join(base_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        print(config_info)
        self.config_info = config_info

    def train_ppo_es(self, num_steps_per_rollout):
        """
        Train PPO-ES models across multiple seeds. 

        Initializes ES_Env  and trains a PPO policy for each seed, saving it as a model once training is complete. 

        Uses provided callbacks for various functionalities, such as saving models at set intervals or facilitating SPACE. 
        """
        

        self.space_logger.info(f"Starting training")
        
        # Train a model per seed
        for seed in self.seeds:
            """ 
            need to plan how im going to call each training step with each instance here 
            """
            self.space_logger.info("------------------------------")
            self.space_logger.info(f"STARTING TO TRAIN A NEW MODEL")
            self.space_logger.info(f"   training with seed: %d", seed)
            self.space_logger.info("------------------------------")
                    
            print(self.config_info)

            env = make_vec_env(lambda: ES_Env( 
                                              seed=seed, 
                                              space_logger=self.space_logger,
                                              dim=self.config_info["test_dimension"], 
                                              use_space=self.config_info["use_space"], 
                                              num_training_instances=self.config_info["num_training_instances"], 
                                              instance=self.config_info["test_instance"]), 
                                              n_envs=1)
                                                # train instance is always 1 for their experiments

            
            # Reset the model with the new environment to ensure it's training from scratch
            model = PPO(
                policy='MlpPolicy',
                env=env,                            # Passing in our environment 
                device=self.cuda_device,
                learning_rate=3e-4,                 
                verbose=1,                          # For full logging functionality
                n_steps=num_steps_per_rollout,      # The number of steps per rollout, dictates how often the policy is updated 
                batch_size=64,                      # Batch size for training
                n_epochs=10,                        # Number of epochs to run for each update
                gamma=0.99,                         # Discount factor for the reward function
                gae_lambda=0.95,                    # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
                clip_range=0.2                      # PPO clipping threshold for stability
            )


            episodes_trained_dir = os.path.join(self.results_dir, f'episodes_trained')

            # make the output directory for the model directory if its not already there 
            os.makedirs(episodes_trained_dir, exist_ok=True)

            # Monitors the mean reward during training. if a new "best" reward is reached, saves the current model weights. 
                # so you load the best checkpoint instead of the last one
                # these callbacks are defined in the callbacks class of this implementation 
            # this callback is the thing that saves the model to the .zip file
            callback = SaveOnBestTrainingRewardCallback(save_path=episodes_trained_dir, seed=seed, space_logger=self.space_logger)
            total_timesteps = 12 * 4000

            # Adjusts the learning rate linearly over time, decays from 3e-4 to 0 by the end of training 
            scheduler = linear_schedule(initial_value=3e-4)
            lr_scheduler_callback = LearningRateScheduler(
                initial_learning_rate=3e-4,
                scheduler=scheduler
            )

            lr_scheduler_callback.total_timesteps = total_timesteps

            space_callback = UpdateEnvCallback("ppo", space_logger=self.space_logger, use_space_val=self.config_info["use_space"], instance_ordering_val=self.config_info["instance_ordering"])

            # Using PPO from stable baseline 3
                # the callback returns the trained model after neraling
                # https://stable-baselines3.readthedocs.io/en/v1.0/modules/ppo.html
                # https://spinningup.openai.com/en/latest/algorithms/ppo.html

            # callbacks for custom logic             
                # lr_scheduler_callback used to save models throughout training
                # space_callback used to facilitate SPACE curriculum generation
            model.learn(total_timesteps=total_timesteps, callback=[callback, lr_scheduler_callback, space_callback]) 

            """
            # Inside learn method from stablebaselines3 
                # collects a rollout of n_steps timesteps
                # computes advantage estimates by interacting with the environment through step() in es_env
            
            # step: just takes in action 
                # applies action 
                # returns the new state
                    # reward
                    # done?
                    # info 
                    # observation
                    # reward
                    # terminated
                    # truncated
                    # {}
            """


    def test_ppo_es(self, problem_type, test_problem_dimension, problem_index, instance, experiment_logger):
        """
        Evaluate trained PPO-ES models on a test problem. 
        
        Loads saved models, searches test functions in a test ES_Env, using the policy to guide search. 
        """
         
        
        # Select a seed for testing
        # seed = random.choice(self.seeds)
        episodes_tested_dir = os.path.join(self.results_dir, f'episodes_tested', f'DIM_{test_problem_dimension}')
        os.makedirs(episodes_tested_dir, exist_ok=True)


        for episode in EPISODES:
            combined_fitness_values = []

            experiment_logger.debug(f"About to run all episodes for %d problem inside test_ppo_es", problem_index)
            
            for seed in self.seeds:
            # Make another environment for testing
                seed_env = make_vec_env(lambda: ES_Env(problem_type=problem_type, 
                                                    instance=instance,
                                                    dim=test_problem_dimension,
                                                    problem_index=problem_index,
                                                    seed=seed, debug_logger=experiment_logger
                                                    ), n_envs=1)



                
                for single_env in seed_env.envs:
                    single_env.unwrapped.set_mode('testing')

                

                model_filename = f"model_seed_{seed}_episode_{episode}.zip"
                
                # Test for a model of each of the following episodes. 
                test_model_path = os.path.join(self.results_dir, 'episodes_trained', model_filename)


                seed_results = test_model(seed_env, test_model_path, episodes_tested_dir, episode, problem_index, instance, experiment_logger, "ppo", len(self.seeds))

                
                combined_fitness_values.extend(seed_results)

            save_path = os.path.join(
                episodes_tested_dir,
                f'fitness_episode_{episode}_problem_{problem_index}_instance_{instance}.npy'
            )
            save_data(save_path, np.array(combined_fitness_values))

