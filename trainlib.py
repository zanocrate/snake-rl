from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ReplayMemory(Dataset):

    def __init__(self,max_capacity):
        
        # init max_capacity
        self.max_capacity = max_capacity
        # init samples as empty lists
        self.s = []
        self.a = []
        self.r = []
        self.s2 = []
        self.terminal = []
    
    def __len__(self):
        return len(self.s) # return length of one of the lists
    
    def __getitem__(self,idx):
        
        sample = {
            's' : self.s[idx],
            'a' : self.a[idx],
            'r' : self.r[idx],
            's2' : self.s2[idx],
            'terminal' : self.terminal[idx]
        }

        return sample

    def _add_sample(self,s,a,r,s2,terminal):
        """
        s and s2 are numpy arrays of shape (history_len,width,height)
        a is an int
        r is float
        """
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s2.append(s2)
        self.terminal.append(terminal)
        
        if self.__len__() > self.max_capacity:
            random_drop = np.random.randint(0,self.max_capacity)
            self.s.pop(random_drop)
            self.a.pop(random_drop)
            self.r.pop(random_drop)
            self.s2.pop(random_drop)


def random_policy():
    return np.random.randint(4)


def epsilon_greedy_policy(state,epsilon,net):
    if isinstance(state,dict):
        raise NotImplementedError()
    else:
        if np.random.rand() < epsilon:
            return random_policy()
        else:
            with torch.no_grad(): # disable gradient calculations. useful when doing inference to skip computation
                # find the maximum over the action dim for each batch sample
                # but batch_size is one in inference so expand dims
                q = net(np.expand_dims(state,axis=0))
                a = q.argmax(1).item() 
                return a # is an int
            

def optimize_model():
    if REPLAY_MEMORY.__len__() < BATCH_SIZE:
        # do not optimize if we do not have enough samples
        return
    # data loader
    data_loader = DataLoader(REPLAY_MEMORY,BATCH_SIZE,True)

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
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # compute
        next_state_values[non_final_mask] = target_net(non_final_next_states.numpy()).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
