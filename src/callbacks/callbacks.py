from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from stable_baselines3.common.utils import obs_as_tensor
import torch as th
import logging  
# import space_enum
from src.config.space_enum import space_operation, instance_ordering

class LearningRateScheduler(BaseCallback):
    def __init__(self, initial_learning_rate, scheduler, verbose=0):
        super(LearningRateScheduler, self).__init__(verbose)
        self.scheduler = scheduler
        self.initial_learning_rate = initial_learning_rate
        self.current_learning_rate = initial_learning_rate
        self.total_timesteps = 0  # Initialize the total_timesteps attribute

    def _on_training_start(self):
        # Update the optimizer's learning rate at the start of training
        self.update_learning_rate(self.initial_learning_rate)

    def _on_step(self):
        # Get the current progress remaining (from 1 to 0)
        progress_remaining = 1 - self.num_timesteps / self.total_timesteps
        # Calculate the current learning rate based on the progress
        self.current_learning_rate = self.scheduler(progress_remaining)
        # Update the optimizer's learning rate
        self.update_learning_rate(self.current_learning_rate)
        return True

    def update_learning_rate(self, new_learning_rate):
        # Set the new learning rate to the optimizer
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_learning_rate


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, save_path: str, seed: int, verbose=1, space_logger=None):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.save_path = save_path
        self.seed = seed
        self.last_episode = 0
        self.space_logger = space_logger

    def _on_step(self) -> bool:
        current_episode = self.training_env.envs[0].unwrapped.current_episode

        # Determines at what step increments a model is saved
        if current_episode == 1 or current_episode % (60) == 0:
            if current_episode != self.last_episode:
                self.last_episode = current_episode
                if self.verbose > 0:
                    print(f"Saving model at episode {current_episode}, seed {self.seed}")
                model_filename = f"model_seed_{self.seed}_episode_{current_episode}.zip"
                save_path = os.path.join(self.save_path, model_filename)
                print("saving at: ", save_path)
                self.space_logger.modellog.info(f"Saved model seed=%d episode=%d path=%s", self.seed, current_episode, save_path)
                os.makedirs(self.save_path, exist_ok=True)
                try:
                    self.model.save(save_path)
                except Exception as e:
                    print(f"Error saving model at episode {current_episode}: {e}")

        return True


# class space_operation(Enum):
#     NO_SPACE = 0
#     JUST_SIZES = 1
#     INSTANCE_STATE = 2
#     ONE_GENERATION = 3

    
# class instance_ordering(Enum):
#     ABSOLUTE = 0
#     IMPROVEMENT = 1
#     RELATIVE_IMPROVEMENT = 2
#     NONE = 3
# Used to update the environment curriculum at each step 
class UpdateEnvCallback(BaseCallback):
    def __init__(self, algo_name: str, space_logger, use_space_val=1, instance_ordering_val=1):
        super().__init__()
        # See use_space 
        self.algo_name = algo_name
        self.last_q = 0.0
        self.curriculum_size = 1
        self.curriculum = []
        self.space_logger = space_logger

        # For now just training on 12 total instances
        self.num_training_instances = 12
        # Following the enums commented above
        self.use_space=space_operation(use_space_val)
        self.instance_ordering=instance_ordering(instance_ordering_val)
        self.update_counter=0

        # Use a dictionary to preserve information on ordering. 
        self.last_evals = {}
        for i in range(self.num_training_instances):
            self.last_evals[i] = 0

        self.stable_streak = 0
        self.unstable_streak = 0
        self.stability_threshold = 3


    def _on_training_start(self):
        """
        At the start of training, ensure that curriculum is already set.
        """
        # If not using SPACE skip...
        if self.use_space == space_operation.NO_SPACE:
            return True
        # On training start, initialize the curriculum
        self.update_curriculum()
        

    def _on_step(self) -> bool:

        """
        Runs after each call to "step()" function in es_env.py. Checks each time to see if full curriculum
        has been exhausted, and if it has then it may increase curriculum size and will set the next curriculum. 
        
        :param self: Description
        :return: If the function completed successfully
        """

        # A rollout is collected every n_steps steps (default value: 12 * 400 steps). 
        if self.use_space == space_operation.NO_SPACE:
            # If space isn't being used, skip
            return True

        # Retrieve the current index and curriculum
        current_index = self.training_env.envs[0].unwrapped.curriculum_index
        curriculum = self.training_env.envs[0].unwrapped.curriculum

        # This means that the current curriculum has been exhausted, need to reset
        if current_index > len(curriculum):

            self.space_logger.info("Current Curriculum exhausted, transitioning...")
            self.space_logger.info(self.training_env.envs[0].unwrapped.get_sigma())

            # Update the curriculum size
            self.update_curriculum_size(curriculum)

            # if self.use_space != (space_operation.NO_SPACE):
                # Update the curriculum itself
            self.update_curriculum()

        return True

    def _on_rollout_end(self):
        """
        Function for logging when a rollout is over, and the policy is therefore updated
        Also used for keeping track of once the policy has been updated at least once
        """
        
        self.update_counter += 1
        # Total timesteps is set to 12 * 4000 by default, the program keeps training until this number of steps is reached for each model.
        # each timestep is one call to the step.env, which is done within the env_es
        # each rollout is 12 * 400 steps (by default)
        self.space_logger.info(f"Collected rollout:")
        self.space_logger.info(f"                  about to do update [%d] to the policy", self.update_counter)
        self.space_logger.info(f"                  at [%d] model timesteps so far", self.num_timesteps)

        # Sets after the first rollout, signifying the policy has been updated at least once. 
        self.training_env.envs[0].unwrapped.before_first_rollout = False

        return True

    def update_curriculum_size(self, curriculum):
        """
        Helper function used to update the size of the curriculum

        If learning sufficiently converged, we can add more instances. 
        """
        STEP_SIZE_CONST = 1 # Change in size of curriculum

        # Calculate mean_q
        eval_env = self.training_env.envs[0].unwrapped 
        mean_q = self.get_mean_q(self.model, eval_env, curriculum)
        delta_q = np.abs(np.abs(mean_q) - np.abs(self.last_q))

        # Constant used in sizing calculation, explained further in corresponding report
        eta_const = .1
        is_stable = delta_q <= eta_const * np.abs(self.last_q) 

        # If condition passes, then increase size by STEP_SIZE_CONST
        if (is_stable):
            self.stable_streak+=1
            self.unstable_streak = 0
        else:
            self.stable_streak = 0
            self.unstable_streak += 1

        self.last_q = mean_q

        # Don't reset streaks based on if you update, can keep streak going continually either way 
        if self.stable_streak >= (self.stability_threshold+1): # this is equal to 4 
            curric_size = self.curriculum_size
            # Added guardrail, can't go outside bounds
            self.curriculum_size = min(curric_size + STEP_SIZE_CONST, self.num_training_instances)
            self.space_logger.info(f"Increasing size from: [%d] to [%d]", curric_size, self.curriculum_size)

        elif self.unstable_streak >= (self.stability_threshold-1): # this is equal to 2)
            curric_size = self.curriculum_size
            # Added guardrail, can't go outside bounds
            self.curriculum_size = max(curric_size - STEP_SIZE_CONST, 1)
            self.space_logger.info(f"Decreasing size from: [%d] to [%d]", curric_size, self.curriculum_size)
        



    def update_curriculum(self):
        """
        Helper function used to update the curriculum
        """

        # Set the env
        eval_env = self.training_env.envs[0].unwrapped 
        temp = []

        # The first curriculum should be randomly sampled
        if self.use_space == space_operation.JUST_SIZES or self.instance_ordering == instance_ordering.NONE:
            temp = list(range(self.num_training_instances))
        else:

            if eval_env.before_first_rollout:
                self.space_logger.info("Before first policy update, random sample curriculum")
                temp = self.random_sample_of_instances()

            else: 

                if self.instance_ordering == instance_ordering.ABSOLUTE: 
                    self.space_logger.info("After first policy update, absolute space ordering")
                    temp = self.order_instances_qvals(self.model, eval_env, self.num_training_instances)

                elif self.instance_ordering == instance_ordering.IMPROVEMENT: 
                    self.space_logger.info("After first policy update, improvement space ordering")
                    temp = self.order_instances_improvement(self.model, eval_env, self.num_training_instances, self.last_evals)
                
                elif self.instance_ordering == instance_ordering.RELATIVE_IMPROVEMENT:
                    self.space_logger.info("After first policy update, relative improvement space ordering")
                    temp = self.order_instances_relative_improvement(self.model, eval_env, self.num_training_instances, self.last_evals)

        
        self.space_logger.info(f"New curriculum is: {temp}")

        self.curriculum = temp
        new_curriculum = self.curriculum[:self.curriculum_size]
        
        # Set the updated size, and the updated curriculum
        self.training_env.envs[0].unwrapped.set_curriculum_size(self.curriculum_size)
        self.training_env.envs[0].unwrapped.set_curriculum(new_curriculum)
        self.training_env.envs[0].unwrapped.reset_curriculum_index()
        self.space_logger.info(f"Updating curriculum to: %s", new_curriculum)

        # This is so that it ignores the first one, which is already set as callbacks run after the step
        self.training_env.envs[0].unwrapped.set_curriculum_index(1) 
    
    # Returns indices in ascending order, used for "absolute" ordering
    def order_instances_qvals(self,learner, env, num_instances):
        # Order the instances by q value
        evals = self.get_instance_evals(learner, env, num_instances)
        self.last_evals = evals
        return np.argsort(evals)
    
    # Computes absolute improvement in value since last time, used for "improvement" ordering
    def order_instances_improvement(self, learner, env, num_instances, last_evals):
        evals = self.get_instance_evals(learner, env, num_instances)
        self.space_logger.info(f"In order instances improvement, old evals are: {last_evals}")
        self.space_logger.info(f"In order instances improvement, current evals are: {evals}")
        improvement = {}
        evals_dict = {}

        for i in range(len(evals)):
            improvement[i] = evals[i] - self.last_evals[i]
            evals_dict[i] = evals[i]

        if all(imp == 0 for imp in improvement.values()):

            self.space_logger.info(f"No Improvements, keeping curriculum the same")
            # Don't change the curriculum if nothing improved 
            ordered_instances = self.curriculum
        else:

            self.space_logger.info(f"Improvements are: {improvement}")
            ordered_instances = sorted(improvement, key=improvement.get, reverse=True)
            self.last_evals = evals_dict
            
        return ordered_instances


    # Computes relative improvement in value since last time, used for "relative improvement" ordering
    def order_instances_relative_improvement(
        self, learner, env, num_instances, last_evals
    ):
        evals = self.get_instance_evals(learner, env, num_instances)
        self.space_logger.info(f"In order instances relative improvement, old evals are: {last_evals}")
        self.space_logger.info(f"In order instances relative improvement, current evals are: {evals}")

        relative_improvement = {}
        evals_dict = {}
    
        for i in range(len(evals)):
            old_eval = self.last_evals[i]
            new_eval = evals[i]
            evals_dict[i] = new_eval

            if old_eval == 0:
                # Avoid division by zero
                old_eval = 0.00001

            relative_improvement[i] = (new_eval - old_eval) / old_eval

        if all(imp == 0 for imp in relative_improvement.values()):
            self.space_logger.info("No relative improvements, keeping curriculum the same")
            # Don't change the curriculum if nothing improved 
            ordered_instances = self.curriculum
        else:
            self.space_logger.info(f"Relative improvements are: {relative_improvement}")
            ordered_instances = sorted(relative_improvement,key=relative_improvement.get,reverse=True)
            self.last_evals = evals_dict

        return ordered_instances


    # Returns a numpy array of value estimates 
    # This is the one used for ordering the instances. 
    def get_instance_evals(self,learner, env, num_instances):

        # Retrieve the instance evaluations for every instance being trained on
        evals = []
        prev_set = env.get_curriculum()
        obs_list = []

        for i in range(num_instances):

            # When you set the curriculum in the environment, it correctly sets its own problem as the first item in that curriclumu
            env.set_curriculum([i])
            # Rest and go to starting state for this instance
            obs, info = env.reset()

            if self.use_space == space_operation.ONE_GENERATION:
                mini_rollout = env.poll_env()

                self.space_logger.info(f"Polled environment to get {mini_rollout}")

                obs[0] = mini_rollout[0]
                obs[1] = mini_rollout[1]
            
            obs_t = obs_as_tensor(obs, learner.device)
            obs_t = obs_t.unsqueeze(0)
            obs_list.append(obs_t)
            val = 0

            # Maintaining backwards compatibility
            if self.algo_name == "trpo":
                # value is the network's estimate of the expected discounted return from that initial state 
                val = learner.policy_pi.policy.predict_values([obs_as_tensor(obs, self.model.device)])
            else:
                val_t = learner.policy.predict_values(obs_t)
                val = float(val_t.detach().cpu().numpy().squeeze())
            evals.append(val)

        # Set the environment curriculum back to what it was before
        env.set_curriculum(prev_set)

        self.space_logger.info("Collected instance evals: %s", evals)
        return np.array(evals)



    def get_mean_q(self, learner, eval_env, curriculum):
        """
        Get the mean_q for every instance, and return the mean. 

        Adjusted from the SPACE implementation, to account for differences with SB1 and SB3
        """
        qs = []
        n_insts = len(curriculum)
        prev_set = eval_env.get_curriculum()

        env = self.training_env.envs[0].unwrapped 

        for i in range(n_insts):
            env.set_curriculum([i])
            obs, first_val = env.reset()

            if self.use_space == space_operation.ONE_GENERATION:
                mini_rollout = env.poll_env()
                self.space_logger.info(f"Polled environment to get {mini_rollout}")
                obs[0] = mini_rollout[0]
                obs[1] = mini_rollout[1]

            val = 0
            obs_t = obs_as_tensor(obs, learner.device)

            # This is needed to fix the out of range eror 
            obs_t = obs_t.unsqueeze(0)

            if self.algo_name == "trpo":
                # val = self.model.policy_pi.value([obs])
                # I made it not have gradients since its evaluating not training here
                # predict_values Get the estimated values according to the current policy given the observations.
                with th.no_grad():
                    val = self.model.policy.predict_values(obs_as_tensor(obs, self.model.device))
                val = val.cpu().numpy()
            # this is for PPO 

            elif self.algo_name == "ppo":

                val_t = learner.policy.predict_values(obs_t)
                val = float(val_t.detach().cpu().numpy().squeeze())

            else:
                print("Algo name not recognized")
            qs.append(val)
            # its over a flattened array 

        # Set the environment curriculum back to what it was before
        env.set_curriculum(prev_set)
        return np.mean(qs)

    
    def random_sample_of_instances(self):
        """Return a random curriculum of instances, for curriculum generation before a policy update"""
        sampled = np.random.permutation(self.num_training_instances).tolist()
        return sampled