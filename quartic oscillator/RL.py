import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from numba import njit
import layers
from optimizer import LaProp
from multiprocessing.sharedctypes import Value, RawValue, RawArray
import multiprocessing as mp
import time, random
from termcolor import colored
import os, sys
from arguments import args

t_max = 100.
F_max = 5.
def set_parameters(**kwargs):
    if 'read_step_length' in kwargs:
        global read_step_length
        read_step_length = kwargs["read_step_length"]
    if 't_max' in kwargs:
        global t_max
        t_max = kwargs['t_max']
    if 'failing_reward' in kwargs:
        global failing_reward
        failing_reward = kwargs['failing_reward']
    if 'F_max' in kwargs:
        global F_max
        F_max = kwargs['F_max']

class DQN_measurement(nn.Module):
    num_of_control_resolution_oneside = 10
    def __init__(self, inputs):
        self.control_force_spacing = F_max/self.num_of_control_resolution_oneside
        super(DQN_measurement, self).__init__()
        self.inputs=inputs
        self.Batch_Normalize = False
        k1 = 13; s1 = 5
        k2 = 11; s2 = 4
        k3 = 9; s3 = 4
        filters = (32, 64, 64)
        self.conv1=layers.Conv1d_weight_normalize(2, filters[0], kernel_size = k1, stride = s1, padding = 0, bias = not self.Batch_Normalize)
        if self.Batch_Normalize: self.bn1=layers.BatchRenorm1d(filters[0])
        self.conv2=layers.Conv1d_weight_normalize(filters[0], filters[1], kernel_size = k2, stride = s2, padding = 0, bias = not self.Batch_Normalize)
        if self.Batch_Normalize: self.bn2=layers.BatchRenorm1d(filters[1])
        self.conv3=layers.Conv1d_weight_normalize(filters[1], filters[2], kernel_size = k3, stride = s3, padding = 0, bias = not self.Batch_Normalize)
        if self.Batch_Normalize: self.bn3=layers.BatchRenorm1d(filters[2])
        def calculate_next_layer_dim(number, k, s):
            return (number-(k-1)+(s-1)) // s
        number = calculate_next_layer_dim(inputs, k1, s1)
        number = calculate_next_layer_dim(number, k2, s2)
        number = calculate_next_layer_dim(number, k3, s3)
        self.fc1=nn.Linear(number * filters[2], 256)
        self.fc21=layers.FactorizedNoisy(256, 256)
        self.fc31=layers.FactorizedNoisy(256, self.num_of_control_resolution_oneside*2 + 1) # left and right means doubling, 1 extra for no control
        self.fc22=layers.Linear_weight_normalize(256, 128)
        self.fc32=layers.Linear_weight_normalize(128, 1) # and 1 for mean prediction
        print(' + Number of params: {}'.format(sum([p.data.nelement() for p in self.parameters()])))
    def forward(self, x, noise=(None, None)):
        x = self.conv1(x)
        if self.Batch_Normalize: x = self.bn1(x)
        x = self.conv2(F.relu(x))
        if self.Batch_Normalize: x = self.bn2(x)
        x = self.conv3(F.relu(x))
        if self.Batch_Normalize: x = self.bn3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # we do not throw away the generated noise and reuse them when comparing the evaluation of different networks
        action, noise_1 = self.fc21(x, noise[0])
        action, noise_2 = self.fc31(F.relu(action), noise[1])
        mean = F.relu(self.fc22(x))
        mean = self.fc32(mean).squeeze(dim=1)
        return action, mean, (noise_1, noise_2)
    def convert_to_force(self, n):
    ### !!! when change the total number of controls, change the number of outputs at the final fc layer !!! ###
        return round(n-self.num_of_control_resolution_oneside)*self.control_force_spacing

class direct_DQN(nn.Module):
    num_of_control_resolution_oneside = 10
    def __init__(self, data_length, noisy_layers = 2):
        self.control_force_spacing = F_max/self.num_of_control_resolution_oneside
        super(direct_DQN, self).__init__()
        Linear = layers.Linear_Default #layers.Linear_Initialized #layers.Linear_weight_normalize # 
        self.inputs={'data_length':data_length, 'noisy_layers':noisy_layers}
        self.fc1=Linear(data_length, 512)
        self.fc2=Linear(512, 512)
        if noisy_layers >= 2:
            self.fc31=layers.FactorizedNoisy(512, 256)
        else:
            self.fc31=Linear(512, 256)
        self.fc32=Linear(512, 128)
        if noisy_layers >= 1:
            self.fc41=layers.FactorizedNoisy(256, self.num_of_control_resolution_oneside*2 + 1) # left and right, and 1 extra for no control
        else:
            self.fc41=Linear(256, self.num_of_control_resolution_oneside*2 + 1)
        self.fc42=Linear(128, 1) # mean prediction
        print(' + Number of params: {}'.format(sum([p.data.nelement() for p in self.parameters()])))
    def forward(self, x, noise=(None, None)):
        x = F.relu(self.fc1(x)[0])
        x = F.relu(self.fc2(x)[0])
        action, noise_1 = self.fc31(x, noise[0])
        action, noise_2 = self.fc41(F.relu(action), noise[1])
        mean = F.relu(self.fc32(x)[0])
        mean = self.fc42(mean)[0].squeeze(dim=1)
        return action, mean, (noise_1, noise_2)
    def convert_to_force(self, n):
        if type(n) == int:
            return round(n-self.num_of_control_resolution_oneside)*self.control_force_spacing
        if type(n) == torch.Tensor:
            return (torch.round(n.float())-self.num_of_control_resolution_oneside).float()*self.control_force_spacing

class TrainDQN(object):
    def __init__(self, net, memory, batch_size, gamma=0.99, backup_period = 200, args={}):
        torch.backends.cudnn.benchmark = True
        # take the input arguments
        if net.__class__ == direct_DQN:
            self.net = net.__class__(**net.inputs)
            self.target_net = net.__class__(**net.inputs)
            self.measurement = False
            self.input_length = net.inputs['data_length']
        else:
            self.net = net.__class__(net.inputs)
            self.target_net = net.__class__(net.inputs)
            self.measurement = True
            self.input_length = net.inputs
        self.net.load_state_dict(net.state_dict())

        self.gamma = gamma
        self.memory = memory
        self.batch_size = batch_size

        self.backup_counter = 0
        self.backup_period = backup_period

        # set report print()
        self.report_period = 30
        self.clear_report()

        # prepare to train
        self.net, self.target_net = self.net.cuda(), self.target_net.cuda()
        # the optimizer ***
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr, betas = (0.9, 0.9995), eps=1e-10) #LaProp(self.net.parameters(), centered=False, lr=args.lr, betas = (0.9, 0.9995) ,amsgrad=True)# # ,amsgrad=True, centered=True
        self.net.train()
        self.target_net.train()
        self.CDQN = args.CDQN

        # prepare the shared memory to do sampling
        self.transitions_storage = RawArray('f', self.batch_size*self.memory.tree.data_size)
        self.transitions = torch.from_numpy( np.frombuffer(self.transitions_storage,dtype='float32').reshape((self.batch_size, self.memory.tree.data_size)) )
        if self.measurement: # avoid reallocating memory
            self.next_states_storage = torch.empty(self.batch_size,2,self.input_length,device='cuda')
            self.previous_states_storage = torch.empty(self.batch_size,2,self.input_length,device='cuda')
    def clear_report(self):
        self.report_i = 0
        self.accu_err = 0.
    def __call__(self):
        # when a sampling is too large to obtain from the current replay memory, directly ignore
        if len(self.memory) < self.batch_size: return

        # update the target network
        if self.backup_counter == 0: 
            self.target_net.load_state_dict(self.net.state_dict())
            for p in self.target_net.parameters(): p.requires_grad=False
            print('\nReload target_net')
        self.backup_counter += 1
        if self.backup_counter >= self.backup_period: self.backup_counter=0

        net, transitions = self.net, self.transitions
        indices, ISWeights = self.memory.obtain_sample(self.batch_size)
        # prepare the input data.
        transitions = transitions.cuda()
        next_states = transitions[:, self.input_length:2*self.input_length]
        previous_states = transitions[:, 0:self.input_length]
        # start evaluation
        if not self.CDQN:
            with torch.no_grad():
                _values, _means, noise = net(next_states)
                next_actions = _values.max(1)[1]
            action_values, avg_value, _noise = net(previous_states, noise)#
            # dim=0 is sample dimension, while dim=1 at output is action dimension
            # call by index will reduce the number of dim by 1 at the called dimension
            state_action_values = action_values.gather(1, transitions[:,-2].cuda().unsqueeze(1).long()).squeeze() + avg_value - action_values.mean(dim=1)
            # rescale the rewards by (1-\gamma_r)
            rewards = (1-self.gamma)*transitions[:,-1].cuda()
            fail_mask = (transitions[:,-1]<=failing_reward).cuda()

            next_action_values, next_avg_value, _noise = self.target_net(next_states, noise)#
            next_state_values = next_action_values.gather(1, next_actions.unsqueeze(1)).squeeze() + next_avg_value - next_action_values.mean(dim=1)
            expected_state_action_values = (next_state_values * self.gamma) + rewards
            expected_state_action_values[fail_mask] = transitions[:,-1][fail_mask]
            # the use of data in array "transitions" is over, so it can be updated by
            # the next batch data using a separate process to enhance the parallelization 
            self.memory.request_sample(self.batch_size)

            unweighted_loss = F.mse_loss(state_action_values, expected_state_action_values, reduction='none')
            ISWeights = torch.from_numpy(ISWeights).to(unweighted_loss)
            loss = (unweighted_loss*ISWeights).mean()
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        else:
            #print(torch.mean(torch.sum(previous_states*previous_states,axis=1).sqrt(),axis=0))
            next_action_values_current, next_avg_value_current, noise = net(next_states)
            with torch.no_grad():
                next_actions = next_action_values_current.max(1)[1]
            action_values, avg_value, _noise = net(previous_states, noise)#
            state_action_values = action_values.gather(1, transitions[:,-2].cuda().unsqueeze(1).long()).squeeze() + avg_value - action_values.mean(dim=1)
            # rescale the rewards by (1-\gamma_r)
            rewards = (1-self.gamma)*transitions[:,-1].cuda()
            fail_mask = (transitions[:,-1]<=failing_reward).cuda()
            with torch.no_grad():
                next_action_values_target, next_avg_value_target, _noise = self.target_net(next_states, noise)#
                next_state_values_target = next_action_values_target.gather(1, next_actions.unsqueeze(1)).squeeze() + next_avg_value_target - next_action_values_target.mean(dim=1)
                expected_state_action_values_target = (next_state_values_target * self.gamma) + rewards
            
            next_state_values_current = next_action_values_current.gather(1, next_actions.unsqueeze(1)).squeeze() + next_avg_value_current - next_action_values_current.mean(dim=1)
            expected_state_action_values_current = (next_state_values_current * self.gamma) + rewards
            with torch.no_grad():
                target_mask = (torch.abs(state_action_values - expected_state_action_values_target) >= torch.abs(state_action_values - expected_state_action_values_current))
            expected_state_action_values = torch.where(target_mask, expected_state_action_values_target, expected_state_action_values_current) 
            expected_state_action_values[fail_mask] = transitions[:,-1][fail_mask]
            self.memory.request_sample(self.batch_size)

            unweighted_loss = F.mse_loss(state_action_values, expected_state_action_values, reduction='none')
            ISWeights = torch.from_numpy(ISWeights).to(unweighted_loss)
            loss = (unweighted_loss*ISWeights).mean()
            loss.backward()
            #if torch.any(torch.isnan(loss)): 
            #    print(unweighted_loss, expected_state_action_values, expected_state_action_values_target, expected_state_action_values_current, state_action_values, next_state_values_current, next_avg_value_current, next_action_values_current)
            #    for p in net.parameters(): print(p)
            #    assert False
            self.optim.step()
            self.optim.zero_grad()
        
        # update the statisics for prioritized sampling
        surprise = unweighted_loss.detach_().cpu().numpy()
        success = not np.any(np.isnan(surprise))
        if not success: 
            print(surprise, next_states, previous_states)
            for para in net.parameters(): print(para)
        assert(success)
        self.memory.batch_update(indices, surprise)
        self.accu_err += loss.item()
        # print deviation error
        self.report_i += 1
        if self.report_i >= self.report_period:
            assert not np.isnan(self.accu_err), '"NAN" encountered in loss values. Numerical error has appeared.'
            print('RMS error {:.3g}\t'.format(math.sqrt(self.accu_err/self.report_i)), flush=True)
            self.clear_report()            
        sys.stdout.flush()
        return loss.item()


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0
    def __init__(self, capacity, data_size, data_tree = None, policy = 'random', passes_before_random = 0.):
        if data_tree == None:
            self.capacity = capacity  # for all priority values
        # calculate singular layers where it cannot be fully devided by 2:
            width = 1; self.num_of_nodes = 0
            while width < capacity:
                self.num_of_nodes += width
                width *= 2

            self.tree_buffer = RawArray('d', self.num_of_nodes + capacity)
            self.tree = np.frombuffer(self.tree_buffer,dtype='float64')
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: self.num_of_nodes                   size: capacity
            self.data_buffer = RawArray('f', capacity*data_size)
            self.data = np.frombuffer(self.data_buffer,dtype='float32').reshape((capacity,data_size))  # for all transitions
            self.data_size = data_size
        # [--------------data frame-------------]
        #             size: capacity
            self.len = RawValue('i',0)
            self.passes = - passes_before_random
            assert self.passes <= 0

            #self.childrens = []
            if policy == 'sequential': self.sequential = True
            elif policy == 'random': self.sequential = False

        else:
            self.capacity, self.len, self.passes, self.sequential, self.data_size, self.num_of_nodes = capacity, data_tree.len, data_tree.passes, data_tree.sequential, data_tree.data_size, data_tree.num_of_nodes
            self.data = np.frombuffer(data_tree.data_buffer,dtype='float32').reshape((self.capacity,self.data_size))
            self.tree = np.frombuffer(data_tree.tree_buffer,dtype='float64')
            #data_tree.childrens.append(self)

    def add(self, p, data):
        if self.sequential or self.passes < 1.: 
            # We start selection once the memory is full
            tree_idx = self.data_pointer + self.num_of_nodes
            self.data[self.data_pointer] = data  # update transition data
            self.update(tree_idx, p)  # update tree_frame
            self.data_pointer += 1
            self.passes += 1./self.capacity
            if self.len.value != self.capacity: self.len.value += 1
            if self.data_pointer >= self.capacity:  # replace when exceed the capacity
                self.data_pointer = 0
        elif self.sequential == False:
            self.data_pointer = random.randrange(self.capacity)
            tree_idx = self.data_pointer + self.num_of_nodes
            self.data[self.data_pointer] = data  # update transition data
            self.update(tree_idx, p)

    def update(self, tree_idx, p):
        compiled_update(tree_idx, p, self.tree)

    def recalculate_structure(self):
        compiled_recalculate_structure(self.tree, self.capacity)

    def get_leaf(self, v):
        return compiled_get_leaf(v, self.tree, self.capacity, self.data)

    @property
    def total_p(self):
        return self.tree[0]

@njit(parallel=False)
def compiled_update(tree_idx, p, tree):
    tree[tree_idx] = p
    # then propagate the change through tree
    while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
        _sum = tree[tree_idx] + tree[tree_idx+1 if tree_idx % 2 else tree_idx-1]
        tree_idx = (tree_idx - 1) // 2
        tree[tree_idx] = _sum

@njit(parallel=False)
def compiled_recalculate_structure(tree, capacity):
    num_of_nodes = len(tree) - capacity
    width = 1; num_of_nodes = 0
    while width < capacity:
        print(np.sum(tree[num_of_nodes:num_of_nodes+width]))
        num_of_nodes += width
        width *= 2
    for i in range(num_of_nodes):
        parent_idx = num_of_nodes-1-i
        cl_idx = 2 * parent_idx + 1; cr_idx = cl_idx + 1
        if cl_idx >= len(tree):
            left, right = 0., 0.
        else:
            left = tree[cl_idx]
            if cr_idx >= len(tree):
                right = 0.
            else:
                right = tree[cr_idx]
        tree[parent_idx] = left + right
    print('memory cleaned')

@njit(parallel=False)
def compiled_get_leaf(v, tree, capacity, data):
    """
    Tree structure and array storage:
    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions
    Array type for storing:
    [0,1,2,3,4,5,6]
    """
    parent_idx = 0
    while True:     # the while loop is faster than the method in the reference code
        cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
        cr_idx = cl_idx + 1
        if cl_idx >= len(tree):        # reach bottom, end search
            leaf_idx = parent_idx
            break
        else:       # downward search, always search for a higher priority node
            if v <= tree[cl_idx]:
                parent_idx = cl_idx
            else:
                v -= tree[cl_idx]
                parent_idx = cr_idx

    data_idx = leaf_idx - (len(tree)-capacity)
    if data_idx < 0:
        return leaf_idx, 0., None
    else:
        return leaf_idx, tree[leaf_idx], data[data_idx]

class Memory(object):  # stored as ( s, action, reward ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.00001  # small amount to avoid zero priority
    alpha = 0.3  # [0~1] convert the importance of TD error to priority
    beta = 0.2  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 10.  # clipped abs error

    def __init__(self, capacity, data_size, data_tree=None, policy = 'sequential', passes_before_random = 0.):
        self.inputs = {'capacity':capacity, 'data_size':data_size, 'policy':policy}

        if data_tree == None:
            self.tree = SumTree(capacity, data_size, policy = policy, passes_before_random = passes_before_random)
        else:
            assert(issubclass(data_tree.__class__, SumTree))
            self.tree = SumTree(data_tree.capacity, data_tree.data_size, data_tree)
        self.max=0.
        self.start = False
    def start_proxy_process(self, shared_data, transitions_sampling_memory, array_shape):
        ctx=mp.get_context('fork')
        #self.end = ctx.Event()
        self.batch_update_queue = ctx.Queue()
        sampling_queue = ctx.Queue()
        conn3, conn4 = ctx.Pipe(False)
        conn_recver, self.conn_sender = ctx.Pipe(False)
        # **********
        # it is strange that a parent process can manage the data passing between recver and sender in itself correctly,
        # but if the recver and sender are both passed to a subprocess, it no longer works there.
        # **********
        self.loader = ctx.Process(target=Load, args=(conn_recver, shared_data, transitions_sampling_memory, array_shape, self.batch_update_queue, self.inputs, self.tree, (sampling_queue, conn3), random.randrange(0,2**32-1)))
        self.sampling_recv = sampling_queue
        self.sampling_request = conn4
        self.sampling_request_recv_head = conn3
        self.loader.start()

    def set_memory_source(self, queue, shared_things):
        self.conn_sender.send((queue, shared_things))

    def set_transition_sampling_storage(self, array):
        self.transition_storage = array

    def __del__(self):
        if hasattr(self,'loader'):
            #self.end.set()
            self.loader.join()
        return
    def __len__(self):
        return self.tree.len.value
    def store(self, transition):
        if self.max == 0.:
            self.tree.add(self.abs_err_upper, transition)
        else: self.tree.add(self.max, transition)   # set the max p for new p
    def clean(self):
        self.tree.recalculate_structure() # machine error accumulates
    def obtain_sample(self, n):
        if self.__len__() < n: return
        if hasattr(self,'loader'):
            if not self.start: self.request_sample(n); self.start = True
            data = self.sampling_recv.get()
            return data
        else:
            self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
            return compiled_sampling(n, self.tree.data_size, self.tree.total_p, self.beta,\
                    self.__len__(),self.tree.tree, self.tree.capacity, self.tree.data, self.transition_storage)

    def request_sample(self, n):
        if not self.sampling_request_recv_head.poll():
            self.sampling_request.send(n)

    def batch_update(self, tree_idx, abs_errors):
        if hasattr(self,'batch_update_queue'):
            self.batch_update_queue.put((tree_idx, abs_errors))
            return
        self.epsilon = 0.0001 * self.max
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        success = np.all(clipped_errors!=0.)
        if not success: print(self.max, self.abs_err_upper, abs_errors)
        assert(success)
        compiled_batch_update(tree_idx,clipped_errors,self.alpha,self.tree.tree)
        self.max = 0.95 * max(self.max, np.max(clipped_errors))

@njit(parallel=False)
def compiled_sampling(n, data_size, total_p, beta, length, tree, capacity, tree_data, transition_storage):
    pri_seg = total_p / n
    b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, ), dtype=np.float32)
    v_rand = np.random.rand(n)
    for i in range(n):
        #a , b = pri_seg * i , pri_seg * (i + 1) 
        v = (i+v_rand[i])*pri_seg #np.random.uniform(a, b)
        idx, p, data = compiled_get_leaf(v, tree, capacity, tree_data)
        # sometimes it errors, so we need to check whether the "p" sampled out is valid
        if p == 0.: 
            compiled_recalculate_structure(tree, capacity)
            a, b = pri_seg * i, pri_seg * (i + 1)
            while p == 0.:
                v = (b - a) * np.random.random_sample() + a
                idx, p, data = compiled_get_leaf(v, tree, capacity, tree_data)
        prob = p / total_p
        ISWeights[i] = np.power(length*prob, -beta)
        b_idx[i], transition_storage[i] = idx, data
    # the sampled transitions from memory is already updated in the given transition_storage array
    # !!! take care of the asynchronous access to the "transitions" array !!!
    return b_idx, ISWeights#, b_memory

@njit(parallel=False)
def compiled_batch_update(tree_idx,clipped_errors,alpha,tree):
    ps = np.power(clipped_errors, alpha)
    for ti, p in zip(tree_idx, ps):
        compiled_update(ti, p, tree)

def Load(LoadPipe, shared_data, Transitions_Sampling_Memory, Memory_Shape, batch_update_queue, inputs, tree_data, sampling_conns, seed):
    random.seed(seed); np.random.seed(seed)
    memory_in = Memory(**inputs, data_tree=tree_data)
    memory_in.set_transition_sampling_storage(\
                    np.frombuffer(Transitions_Sampling_Memory,dtype='float32').reshape(Memory_Shape))
    sampling_queue, sampling_request_recv = sampling_conns
    pending_training_updates, episode, t_done, last_achieved_time = shared_data
    pause = False
    last_idle_time=0.
    last_time = time.time()
    loaded = False
    while not loaded:
        if LoadPipe.poll():
            MemoryQueue, shared_things = LoadPipe.recv()
            loaded = True
        else: time.sleep(0.05)
    pause_event, end_event, learning_in_progress_event = shared_things
    os.environ["NUMBA_NUM_THREADS"]="1"; numerical_failure_count = 0; numerical_failure_episode_count = 0
    while not end_event.is_set():
        something_done = False
        last_episode = episode.value
        while not MemoryQueue.empty():
            # for each episode data put into the Queue:
            something_done = True; episode.value += 1
            experience, t, avg_phonon, Fail = MemoryQueue.get()
            num_of_sample = len(experience)
            if not args.train and not learning_in_progress_event.is_set(): learning_in_progress_event.set()
            for i, array in enumerate(experience):
                memory_in.store(array)
                # This block may take a lot of time. We hope it can do the sampling used for training simultaneously.
                if i%32==0:
                    if sampling_request_recv.poll() and not sampling_queue.full():
                        n = sampling_request_recv.recv()
                        sampling_queue.put(memory_in.obtain_sample(n))
                    if not batch_update_queue.empty():
                        memory_in.batch_update(*batch_update_queue.get())
            with t_done.get_lock():
                t_done.value += t
            with pending_training_updates.get_lock():
                pending_training_updates.value += num_of_sample * 8. / 256.
            # This above line indicates we use each sample for the standard 8 times by default,
            # which can be effectively rescaled in "main*.py".

            # decide whether the training is just starting
            if last_achieved_time.value != t_max:
                if last_achieved_time.value < t:
                    if t < t_max:
                        last_achieved_time.value = t
                    else:
                        print(colored('\nReset the counting of Episodes', 'yellow',attrs=['bold']))
                        episode.value = 1; failure_counter = 0; numerical_failure_count = 0
                        last_achieved_time.value = t
                        last_episode = 1
                        learning_in_progress_event.set()
            elif t < t_max:
                # If last_achieved_time.value == t_max but the new experience fails, we check whether it continues failing,
                # and if so, we reset the "last_achieved_time" and go back 
                failure_counter += 1
                # we customarily let the counter reset when 10 successive failures occur 
                if failure_counter == 40: last_achieved_time.value = t; learning_in_progress_event.clear()
            elif failure_counter != 0: failure_counter = 0

            if t != t_max:
                if Fail: numerical_failure_count = numerical_failure_count + 1
                if (last_achieved_time.value==t_max and episode.value%2==0) or episode.value%10==0:
                    Fail_reason = 'border error halt' if Fail else 'high energy halt'
                    print('Episode {}\tt = {:.2f}\t{}'.format(episode.value, t, Fail_reason))
            elif episode.value%2==0:
                string = 'Episode {}\tavg energy = {:.5f}'.format(episode.value, avg_phonon)
                if episode.value == 2: string += ' \t(t = 100)'
                print(string, flush=True)

            if episode.value % 1000 == 0:
                numerical_failure_episode_count += 1000
                if numerical_failure_count != 0:
                    print(colored('boundary numerical imprecision failure {}/{}'.format(numerical_failure_count, numerical_failure_episode_count), 'red'))
                    numerical_failure_count = 0; numerical_failure_episode_count = 0

            if episode.value > last_episode + 50 and not pause: pause_event.set(); pause=True

            while not batch_update_queue.empty():
                memory_in.batch_update(*batch_update_queue.get())
            sys.stdout.flush()
        if pause and not pending_training_updates.value >= 30.: pause=False; pause_event.clear()
        while not batch_update_queue.empty():
            memory_in.batch_update(*batch_update_queue.get())
            something_done = True
        if sampling_request_recv.poll() and not sampling_queue.full():
            n = sampling_request_recv.recv()
            sampling_queue.put(memory_in.obtain_sample(n))
            something_done = True
        if not something_done:
            time.sleep(0.01); last_idle_time+=0.01 # if not something done, pause for 0.01 second
            # if this sleep time is too large (e.g. 0.2), it will become a bottleneck of the whole program because this is in the main loop.
            # Take care.
        elif last_idle_time != 0. and time.time() - last_time > 50.: 
            print('loader pending for {:.1f} seconds out of {:.1f}'.format(last_idle_time, time.time() - last_time))
            last_idle_time = 0.
            last_time = time.time()
    # if the queues are not empty, the threads managing the queues will not exit, and the process will hang on.
    # So we add the following codes to ensure that when the process exit, it clears all the queues.
    while not batch_update_queue.empty(): batch_update_queue.get()
    while not MemoryQueue.empty(): MemoryQueue.get()
    del sampling_queue, MemoryQueue
    return

