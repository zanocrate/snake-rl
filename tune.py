from grpc import method_handlers_generic_handler
from torch.utils.data import DataLoader
from torch.optim import Adam
from src.env import SnakeEnv
from src.replaymemory import ReplayMemory

import json
import numpy as np
import torch
import tqdm
import os

from copy import deepcopy
from itertools import count

import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler


def train_loop_per_worker(
    config : dict,
    env_kwargs : dict,
    model , 
    load_buffer_path : str = None,
    n_episodes : int = 500,
    seed = None,
    buffer_size : int = 12000
        ):

    """
    Plays N episodes, report metrics (return and duration) for each.


    Config contains:

    batch_size
    tau
    gamma
    epsilon
    lr

    Wrappable arguments are:
    env_kwargs : dict of kwargs to pass to the env
    load_buffer_path : str, path to saved replay buffer samples
    n_episodes : number of episodes per run
    seed : if None, random starting states; else int
    model : the nn.Module subclass to optimize
    """

    # threshold for gradient clipping
    max_norm = 100

    # env init
    env = SnakeEnv(**env_kwargs)

    state_shape = (env_kwargs['history_length'],env_kwargs['width'],env_kwargs['height'])
    replay_buffer = ReplayMemory(buffer_size,state_shape)
    if load_buffer_path is not None: replay_buffer.load(load_buffer_path)

    # initialize Net
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_net = deepcopy(model)
    target_net = deepcopy(model)

    # optimizer
    optimizer = Adam(policy_net.parameters(), lr = config['lr'])

    ############################################################ EPISODES LOOP

    for i_episode in range(n_episodes):
        
        # play the episode
        total_return = 0
        state,_ = env.reset(seed)

        for t in count():

            #################################################### EPSILON GREEDY POLICY

            n_actions = policy_net.output_layer.out_features 
            if np.random.rand() < config['epsilon']:
                action = np.random.randint(n_actions)
            else:
                with torch.no_grad(): # disable gradient calculations. useful when doing inference to skip computation
                    # find the maximum over the action dim for each batch sample
                    # but batch_size is one in inference so expand dims
                    q = policy_net(np.expand_dims(state,axis=0)) # q is shaped like (1,n_actions)
                    action = q.argmax(1).item() # get the argmax along the actions axis

        
            next_state, reward, done, _ = env.step(action)

            total_return+=reward
            
            # Store the transition in memory
            replay_buffer._add_sample(state,action,reward,next_state,done)

            # Move to the next state
            state = next_state

            #################################################### TRAINING
            
            # only if we have enough samples
            if replay_buffer.__len__() >= config['batch_size']:

                data_loader = DataLoader(replay_buffer,config['batch_size'],shuffle=True)
                
                # batch is a dict of tensors for s,a,r,s2,terminal
                batch=next(iter(data_loader))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.logical_not(batch['terminal']).to(device)
                non_final_next_states = batch['s2'][non_final_mask].to(device)
                state_batch = batch['s'].to(device)
                action_batch = batch['a'].to(device)
                reward_batch = batch['r'].to(device)

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = policy_net(state_batch.numpy()).gather(1,action_batch[:,None])

                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1)[0].
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(config['batch_size'], device=device)
                with torch.no_grad():
                    # compute
                    next_state_values[non_final_mask] = target_net(non_final_next_states.numpy()).max(1)[0]
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * config['gamma']) + reward_batch

                # Compute loss
                criterion = torch.nn.MSELoss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # In-place gradient clipping
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), max_norm)
                optimizer.step()


            #################################################### UPDATE Q_i -> Q_(i-1)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*config['tau'] + target_net_state_dict[key]*(1-config['tau'])
            target_net.load_state_dict(target_net_state_dict)
            
            #################################################### SAVE METRICS

            if done:

                ############################################# REPORT AND CHECKPOINT

                # Here we save a checkpoint. It is automatically registered with
                # Ray Tune and can be accessed through `session.get_checkpoint()`
                # API in future iterations.
                # os.makedirs("dqn", exist_ok=True)
                # torch.save((policy_net.state_dict(), optimizer.state_dict()), "dqn/checkpoint.pt")
                # checkpoint = Checkpoint.from_directory("dqn")
                session.report({"episode_T": t,
                            "episode_G": total_return}
                            # , checkpoint=checkpoint
                            )
                
                break

    print("Done training")



if __name__ == '__main__':

    # BEFORE RUNNING THIS, INITIALIZE RAY CLUSTER WITH 
    # ray start --head 
    # on the head node

    # init ray to head
    ray.init('localhost:6379')
    
    # number of trials i guess
    num_samples = 16

    # hyperparameters space
    config = {
                "batch_size": tune.choice([256, 512,1024]),
                "lr": tune.loguniform(1e-5, 1e-3),
                "gamma": tune.uniform(0.5,0.75),
                "tau": tune.uniform(0,0.3),
                "epsilon" : tune.uniform(0.1,0.2)
            }

    # scheduler = ASHAScheduler(
    # # max_t=10, # isnt this the same as setting it in the trainable function? idk
    # grace_period=3,
    # reduction_factor=2)

    # load env configurations
    with open('config.json') as f:
        env_kwargs = json.load(f)['env']
    
    # model
    from src.models.CNN2_relative import DQN
    input_shape = (env_kwargs['history_length'],env_kwargs['width'],env_kwargs['height'])

    

    model = DQN(input_shape)    
    
    tuner = tune.Tuner(
        # wrap the training loop in this
        tune.with_parameters(
            train_loop_per_worker,
            # parameters to pass to the training loop
            env_kwargs=env_kwargs,
            load_buffer_path=None,
            n_episodes=50000,
            seed=None,
            model=model,
            buffer_size = 12000
                ),
        # tune configurations
        tune_config=tune.TuneConfig(
            metric="episode_G",
            mode="max",
            num_samples=num_samples,
            # max_concurrent_trials=8 # maybe this is the number of processes it spawns?
        ),
        # run configuration; we disable sync because NFS
        run_config=ray.air.RunConfig(
        name="RL_snake_more",
        local_dir="/dataNfs/snake/",
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
        )),
        # search space
        param_space=config,
    )

    results = tuner.fit()

    # best_result = results.get_best_result("val_loss", "min")

   
    # save results

    df = results.get_dataframe()

    df.to_csv('/home/ubuntu/snake-rl/ray_results.csv')

