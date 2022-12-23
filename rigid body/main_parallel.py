#!/usr/bin/env python3
import os, sys
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from math import pi, sqrt, exp
from scipy.sparse import csr_matrix as csr
from scipy import linalg
import random
import torch
import time, datetime
from termcolor import colored
import copy
from control_config import *
if __name__ == '__main__':
    from math import *
    import torch.multiprocessing as mp
    import fnmatch
    import argparse
    from multiprocessing.sharedctypes import Value, RawValue
    #import matplotlib.animation as animation


# the commandline arguments are detailed in "arguments.py" because there are too many
from arguments import args
torch.cuda.set_device(args.gpu_id)
torch.set_num_threads(1) # this is for CPU usage
if args.seed != -1: 
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

set_control_forces(args.F_choices, args.F_max)
num_of_controls = len(control_force_list)


time_step = args.time_step

control_time = args.control_time
control_interval = round(args.control_time/time_step)
controls_per_unit_time = round(1./args.control_time)
assert min(abs(args.control_time%time_step), abs(time_step-args.control_time%time_step)) < 0.1*time_step, 'We require that control_time {} to be fully divided by time_step {}.'.format(args.control_time, time_step)


num_of_episodes = args.num_of_episodes
reward_multiply = args.reward_multiply
failing_reward = -(args.energy_cutoff)*reward_multiply

t_max = args.t_max
num_of_saves = args.num_of_saves

###########################################
x_max=args.x_max
grid_size = args.grid_size

ground_state_size = args.ground_state_size
omega = math.pi
m_times_omega = 1./(2* ground_state_size**2)
I_xy = m_times_omega/omega
k = (omega**2) * I_xy

Q_z = args.Q_z

real_k = k + (Q_z**2)/4/I_xy
real_omega = sqrt(real_k/I_xy)
real_sigma = sqrt(1./(I_xy*real_omega) /2.)

gamma_factor = args.gamma_factor
gamma = k*gamma_factor

energy_shift = real_omega/pi+ 5.


data_size = 125

def data_augmentation(transition, data_length):
    if random.random()<0.5:
        states = transition[:,:2*data_length]
        action = transition[:,-2].astype(int)
        transition[:,:2*data_length] = states * flip_xy
        action = action_flip_xy[action]
        transition[:,-2] = action.astype(transition.dtype)



# set the reinforcement learning settings

failing_reward = -(args.energy_cutoff)*reward_multiply
reward_shift = energy_shift*reward_multiply
if __name__ == '__main__':
    import RL
    RL.set_parameters(t_max=t_max, failing_reward=failing_reward, reward_shift=reward_shift, data_augmentation=data_augmentation if not args.no_augmentation else None)

################################## end learning setting


# Below is the worker function for subprocesses, which carries out the control simulations and pushes the experiences and records to queues that are collected and handled by other processes. (Quantum simulation is implemented in a compiled C module)
# Because too many processes using CUDA will occupy a huge amount of GPU memory, we avoid using CUDA in these workers. Instead, these workers ask a manager process when they want to evaluate the neural network, and only the manager process is allowed to use CUDA to evaluate the neural network for the controls.
def Control(net, pipes, shared_buffer, seed, EPS_START, idx):
    simulation = __import__('simulation')
    # seeding
    random = np.random.RandomState(seed)
    simulation.set_seed(random.randint(0,999999))
    # preparing pipes
    MemoryQueue, ResultsQueue, ActionPipe, EndEvent, PauseEvent = pipes
    state_data_to_manager = np.frombuffer(shared_buffer,dtype='float32')
    # data input for the neural network
    __data__ = np.zeros((data_size,))
    def get_data(state):
        simulation.get_moments(state, __data__)
        return __data__
    # random action decision hyperparameters
    EPS_END = 0.005
    EPS_DECAY = controls_per_unit_time*t_max*1200*args.train_episodes_multiplicative
    # initialization
    steps_done = 0
    def call_force(data):
        if args.LQG:
            if random.uniform() < EPS_END:
                shift_x, shift_y = random.normal(0.,1., 2) * 0.3
                (shift_x, shift_y), action = map_to_discrete_forces(shift_x, shift_y)
                return (shift_x, shift_y), action, True
            data = data / args.input_scaling
            (shift_x, shift_y), action = map_to_discrete_forces(*LQG_bounded(data[0]/sqrt(k/2.), data[1]/sqrt(k/2.), data[2]*sqrt(2.*I_xy), data[3]*sqrt(2.*I_xy), control_time, k, I_xy, Q_z, args.F_max))
            return (shift_x, shift_y), action, False
        
        nonlocal steps_done
        # apply an \epsilon-greedy strategy:
        eps_threshold = (EPS_START-EPS_END) * exp(-1. * steps_done / EPS_DECAY)
        eps_threshold += EPS_END
        steps_done += args.num_of_actors # this approximates the total steps_done of all the actors
        if random.uniform() < eps_threshold and not args.test:
            shift_x, shift_y = random.normal(0.,1., 2) * 0.3
            (shift_x, shift_y), action = map_to_discrete_forces(shift_x, shift_y)
            rnd = True
        else:
            # copy data to 
            state_data_to_manager[:]=data
            while ActionPipe.poll(): ActionPipe.recv() # ensure that no data remain in the recv pipe
            ActionPipe.send(idx)
            action = ActionPipe.recv()
            shift_x, shift_y = control_force_list[action]
            rnd = False
        return (shift_x, shift_y), action, rnd
    def Gaussian_packet(x, wavelength, mean, std):
        return np.exp(2.j*math.pi*(x-mean)/wavelength)*np.exp(-(x-mean)*(x-mean)/(4.*std*std))/sqrt(sqrt(2*math.pi)*std)
    x_array = ( np.arange(2*x_max+1).astype(np.complex128) - x_max ) * grid_size
    # do one episode
    n_steps = 20
    assert control_interval % n_steps == 0
    def do_episode():
        t = 0.
        # prepare the quantum state
        y_start, x_start = random.uniform(0.,1., 2) * args.init_dist
        state = np.outer(Gaussian_packet(x_array, -2.*math.pi/(Q_z*x_start/2.), y_start, real_sigma), Gaussian_packet(x_array, 2.*math.pi/(Q_z*y_start/2.), x_start, real_sigma))
        # force is the parameter before -\pi\hat{x}, which is the physical force divided by \pi
        force = (0., 0.)
        last_action = no_action_choice
        # start the simulation loop
        i = 0
        experience = []
        accu_energy = 0.; accu_counter = 0; to_stop = False
        while not t >= t_max-0.01*time_step:
            if i % control_interval == 0:
                energy = simulation.energy(state)/omega
                data = get_data(state)*args.input_scaling # the multiplication "args.input_scaling" ensures that "data" is a copy of "__data__"
                if args.train and i != 0:
                    reward = -energy*reward_multiply + reward_shift
                    if energy >= args.energy_cutoff: reward += failing_reward
                    if to_stop: reward -= 1
                    experience.append(np.hstack(( last_data, data, 
                        np.array([last_action],dtype=np.float32),  
                        np.array([reward],dtype=np.float32) )) )
                if energy >= args.energy_cutoff or to_stop:
                    break
                (force_x, force_y), last_action, rnd = call_force(data) 
                last_data = data
                if t>30-0.01*time_step: accu_energy += energy; accu_counter += 1
            #print(i, t)
            Fail, boundary_prob = simulation.simulate_n_steps(state, time_step, force_x, force_y, gamma, n_steps)
            i += n_steps
            t += time_step*n_steps
            if boundary_prob>1.5e-3 and not to_stop: to_stop = True
        #print("an episode ends")
        # push experience into the main process and push results to the manager
        if t>= t_max-0.01*time_step: t=t_max
        avg_energy = accu_energy/accu_counter if accu_counter>0 else args.energy_cutoff
        if not EndEvent.is_set():
            MemoryQueue.put( (experience, t, avg_energy, steps_done) )
            ResultsQueue.put((t, avg_energy))
        return avg_energy
    while True:
        # whether to end the program
        if EndEvent.is_set():
            break
        do_episode()
        while PauseEvent.is_set():
            time.sleep(1.)
            # to avoid an endless loop
            if EndEvent.is_set():
                break
    while ActionPipe.poll(): ActionPipe.recv() # ensure that no data remain in the recv pipe
    ActionPipe.send(None) # tell the manager that the worker has ended
    return


# the manager process for workers. It is used to organise all neural network evaluations into one single process in order to save GPU memory.
# It also monitors the current performance and saves the models.
def worker_manager(net, pipes, num_of_processes, seed, others):
    # initialize
    MemoryQueue, ActorPipe, EndEvent, PauseEvent = pipes
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    # prepare the path
    if not os.path.isdir(args.folder_name): os.makedirs(args.folder_name, exist_ok=True)
    training_data_file = os.path.join(args.folder_name, datetime.datetime.now().strftime("%Y_%m_%d %H_%M_%S")+".txt")
    # prepare workers
    import multiprocessing as mp
    from multiprocessing.sharedctypes import RawArray
    fork = mp.get_context('forkserver')
    results_queue = fork.Manager().Queue()
    processes = []
    message_conn = []; message_worker_conn = []
    worker_data = []
    for n in range(num_of_processes):
        conn1, conn2 = fork.Pipe(True)
        message_conn.append(conn1); message_worker_conn.append(conn2)
        shared_buffer = RawArray('f', data_size)
        np_memory = np.frombuffer(shared_buffer,dtype='float32')
        worker_data.append(torch.from_numpy(np_memory))
        seed = random.randrange(0,2**31 - 1)
        EPS_START = 0.3 - n*(0.3-0.1)
        processes.append( fork.Process( target=Control, args=(copy.deepcopy(net).cpu(), (MemoryQueue, results_queue, conn2, EndEvent, PauseEvent), shared_buffer, seed, EPS_START, n) ) )
    net=net.cuda()
    net.eval()
    # prepare to save
    good_actors=[(args.energy_cutoff/2.,0.) for i in range(num_of_saves)]
    simulated_T = 0.
    performances = []
    episode_passed = 0
    # when receiving a net, check whether the previous net should be stored
    def receive_net():
        nonlocal simulated_T
        if args.test or args.LQG:
            if not results_queue.empty():
                while not results_queue.empty():
                    result = results_queue.get()
                    simulated_T += result[0]/2.
                    performances.append(result[1])
                    with open(os.path.join(args.folder_name, others+'_record.txt'),'a') as f:
                        f.write('{}, {}\n'.format(simulated_T, result[1]))
            return
        if ActorPipe.poll():
            nonlocal net
            if not results_queue.empty():
                nonlocal episode_passed
                while not results_queue.empty():
                    result = results_queue.get()
                    simulated_T += result[0]/2.
                    performances.append(result[1]) # get the avg_energy in the result tuple
                    episode_passed += 1
                    if args.write_training_data:
                        with open(training_data_file,'a') as f:
                            f.write('{}, {}\n'.format(simulated_T, result[1]))
                # only if 50 additional episodes have passed, do we consider saving the next model
                if episode_passed > 50 and len(performances) >= 8: 
                    new_avg_energy = np.mean(np.array(performances[-8:]))
                    if new_avg_energy < good_actors[-1][0]:
                        good_actors[-1] = (new_avg_energy, copy.deepcopy(net).cpu())
                        good_actors.sort(key=lambda p: p[0], reverse=False) # sort in increasing order; "reverse" is False, in fact unnecessary
                        print(colored('new avg energy record: {:.5f}'.format(new_avg_energy), 'green',attrs=['bold']))
                        for idx, actor in enumerate(good_actors):
                            if type(actor[1]) != float:
                                #torch.save(actor[1].state_dict(), os.path.join(args.folder_name,'{}.pth'.format(idx+1)))
                                existing_record_name = os.path.join(args.folder_name,'{}_record.txt'.format(idx+1))
                                if os.path.isfile(existing_record_name): os.remove(existing_record_name)
                        episode_passed = 0
            performances.clear()
            net.load_state_dict(ActorPipe.recv())
            while ActorPipe.poll():
                net.load_state_dict(ActorPipe.recv())
            net = net.cuda()
            net.eval()
            if args.show_actor_recv: print(colored('new model received', 'yellow'))

    network_input = torch.empty((num_of_processes,data_size), device='cuda')

    for proc in processes:
        proc.start()
    process_ended = 0
    while process_ended!=num_of_processes:
        # receive network input data (confirm that there are data, and copy them from the shared buffer to GPU)
        num_of_data = 0
        data_received_id = []
        for i in range(num_of_processes):
            if message_conn[i].poll():
                idx=message_conn[i].recv()
                if idx!=None:
                    data_received_id.append(i) # the data sent through pipe is exactly the id
                    network_input[num_of_data]=worker_data[i][:]
                    num_of_data += 1
                else: process_ended += 1

                while message_conn[i].poll(): # For safety; when False, this while loop is ignored
                    message_conn[i].recv()

        # process the received data
        if num_of_data == 0:
            time.sleep(0.0005) # if no data, wait for 0.5 milisecond
            if args.LQG: time.sleep(0.1)
        else:
            action_values, avg_value, _noise = net(network_input[:num_of_data])
            actions = action_values.max(1)[1].cpu()
            for i,idx in enumerate(data_received_id):
                message_conn[idx].send(actions[i].item())
        # update the network
        receive_net()
    # end, if have left the while loop
    if args.test or args.LQG:
        with open(os.path.join(args.folder_name, others+'.txt'),'w') as f:
            performances = np.array(performances)
            f.write('{} +- {}\n'.format(np.mean(performances), np.std(performances, ddof=1)/np.sqrt(len(performances)) ))
    for proc in processes:
        proc.join()

if __name__ == '__main__':
    class Main_System(object):
        # the Main_System does not need to keep a copy of network
        # only the copy of network inside TrainDQN class is modified by training, so we pass its state_dict to subprocesses
        def __init__(self, train, num_of_processes, others=''):
            self.train = train
            self.processes = []
            self.actor_update_time = 5.
            self.lr_step = -1
            self.pending_training_updates = Value('d',0.,lock=True)
            # somehow RawValue also needs us to call ".value", otherwise it says the type is c_double or c_int
            self.episode = RawValue('i',0)
            self.t_done = Value('d',0.,lock=True)
            self.last_achieved_time = RawValue('d',0.)
            # set the data going to subprocesses:
            self.train.memory.start_proxy_process((self.pending_training_updates, self.episode, self.t_done, self.last_achieved_time), self.train.transitions_storage, (self.train.batch_size, self.train.memory.tree.data_size))
            # the following will create threads, which not end and cause error (not exiting) 
            spawn=mp.get_context('spawn')
            self.manager = spawn.Manager()
            self.MemoryInputQueue = self.manager.Queue()
            self.end_event = self.manager.Event()
            self.pause_event = self.manager.Event()
            self.learning_in_progress_event = self.manager.Event()
            # actors
            self.ActorReceivePipe, self.ActorUpdatePipe = spawn.Pipe(False) # unidirectional pipe that send message from conn2 to conn1
            seed = random.randrange(0,9999999)
            self.worker_manager = spawn.Process( target=worker_manager, args=(copy.deepcopy(train.net).cpu(), (self.MemoryInputQueue, self.ActorReceivePipe, self.end_event, self.pause_event), num_of_processes, seed, others) )
            # store and manage experience (including updating priority and potentially sampling out replays)
            # all the arguments passed into it are used (**by fork initialization in RL module**).
            # somehow RawValue also needs us to call ".value" ? Otherwise it says the type is c_double / c_int
            self.train.memory.set_memory_source(self.MemoryInputQueue, (self.pause_event, self.end_event, self.learning_in_progress_event))
            self.backup_period = self.train.backup_period
            self.train.backup_period = 100
        def __call__(self, num_of_episodes):
            started = False
            self.worker_manager.start()
            last_time = time.time()
            last_idle_time = 0.
            updates_done = 0.
            # We assume batch_size is 256 and each experience is learned 8 times in RL.py, and when we change, we use the rescaling factor below to implement.
            # If we disable training, we use 'inf' instead to make the condition of training always False.  
            downscaling_of_default_num_updates = (8./args.n_times_per_sample)*(args.batch_size/256.) if args.train else float('inf')

            while self.episode.value < num_of_episodes or (self.episode.value < args.maximum_trails_before_giveup and not self.learning_in_progress_event.is_set()):
                something_done = False # check whether nothing is done in one event loop
                remaining_updates = self.pending_training_updates.value - updates_done
                if remaining_updates >= 1. *downscaling_of_default_num_updates:
                    if remaining_updates >= 150. *downscaling_of_default_num_updates and not self.pause_event.is_set():
                        self.pause_event.set(); print('Wait for training')
                    loss = self.train()
                    # if we parallelize the training as a separate process, the following block should be deleted
                    if loss!=None:
                        updates_done += 1.*downscaling_of_default_num_updates
                        something_done = True # one training step is done
                        if not started: started = True
                        # to reduce the frequency of calling "get_lock()", we only periodically reset the shared data "pending_training_updates"  
                        if updates_done >= 200.*downscaling_of_default_num_updates or self.pause_event.is_set():
                            with self.pending_training_updates.get_lock():
                                self.pending_training_updates.value -= updates_done
                                updates_done = 0.
                if remaining_updates < 50. *downscaling_of_default_num_updates and self.pause_event.is_set():
                    self.pause_event.clear()
                if self.t_done.value >= self.actor_update_time:
                    self.scale_up_actor_update_time(self.last_achieved_time.value, self.learning_in_progress_event.is_set())
                    if not self.ActorReceivePipe.poll() and started and not args.LQG:
                        self.ActorUpdatePipe.send(self.train.net.state_dict())
                        with self.t_done.get_lock():
                            self.t_done.value = 0.
                        something_done = True
                if something_done:
                    # print out how much time the training process has been idle for
                    if last_idle_time != 0. and time.time() - last_time > 50.: 
                        print('trainer pending for {:.1f} seconds out of {:.1f}'.format(last_idle_time, time.time() - last_time))
                        last_idle_time = 0.
                        last_time = time.time()
                # if nothing is done, wait.
                if not something_done: time.sleep(0.01); last_idle_time += 0.01
                self.adjust_learning_rate()
            self.end_event.set()
            self.worker_manager.join()
            return
        def scale_up_actor_update_time(self, achieved_time, stabilized):
            changed = False
            # if it becomes unstable, we reset "self.actor_update_time"
            if self.actor_update_time == 300. and not stabilized:
                self.actor_update_time = 50; changed = True
                self.train.report_period = 30
            if achieved_time>=100. and self.actor_update_time<=150.:
                self.actor_update_time = 300.; changed = True
                self.train.report_period = 100
            elif achieved_time>30. and self.actor_update_time<=25.:
                self.actor_update_time = 50.; changed = True
            elif achieved_time>15. and self.actor_update_time<=10.:
                self.actor_update_time = 25.; changed = True
            if changed and args.train: print('actor_update_time adjusted to {:.1f}'.format(self.actor_update_time))
        def adjust_learning_rate(self):
            if self.train.backup_period != self.backup_period and self.learning_in_progress_event.is_set():
                self.train.backup_period = self.backup_period
            # the learning rate schedule is written in "arguments.py"
            if self.episode.value >= args.lr_schedule[self.lr_step+1][0] and self.last_achieved_time.value == t_max:
                self.lr_step += 1
                if args.lr_schedule[self.lr_step][1] < args.lr:
                    args.lr = args.lr_schedule[self.lr_step][1]
                    if args.train:
                        for param_group in self.train.optim.param_groups: param_group['lr'] = args.lr
                        print(colored('learning rate set to {:.2g}'.format(args.lr),attrs=['bold']))
                


    # system settings, checks and the framework
    def check_C_module_and_compile():
        if not args.compile:
            try:
                simulation = __import__('simulation')
                (c_x_n, c_grid_size, c_k, c_I, c_Q_z, c_moment_order, c_error_boundary_length) = simulation.check_settings()
                default_str = ' of the existing C module ({}) does not match the current task ({}). Recompile.\n'
                x_n = 2*x_max + 1
                if c_x_n != x_n:
                    print(colored(('X_N'+default_str).format(c_x_n, x_n), 'yellow',attrs=['bold']))
                    args.compile = True
                elif c_grid_size != grid_size:
                    print(colored(('Grid_size'+default_str).format(c_grid_size, grid_size), 'yellow',attrs=['bold']))
                    args.compile = True
                elif c_k != k:
                    print(colored(('potential strength K'+default_str).format(c_k, k), 'yellow',attrs=['bold']))
                    args.compile = True
                elif c_I != I_xy:
                    print(colored(('moment of intertial I_XY'+default_str).format(c_I, I_xy), 'yellow',attrs=['bold']))
                    args.compile = True
                elif c_Q_z != args.Q_z:
                    print(colored(('angular momentum Q_z'+default_str).format(c_Q_z, args.Q_z), 'yellow',attrs=['bold']))
                    args.compile = True
            except (ModuleNotFoundError, AttributeError) as e:
                args.compile = True
            if args.compile: time.sleep(1)
        if args.compile:
            code = os.system('python{} setupC.py --ground_state_size {} --x_max {} --grid_size {} --Q_z {} '.format(sys.version[:3], ground_state_size, x_max, grid_size, Q_z)) 
            if code != 0:
                raise RuntimeError('Compilation Failure')

if __name__ == '__main__':
    time_of_start = time.time()
    # set the title of the terminal so that what the terminal is doing is clear
    print('\33]0;{}\a'.format(' '.join(sys.argv)), end='', flush=True)
    print(args)

    # compile the simulation module in C
    check_C_module_and_compile()

    # set the replay memory
    capacity = round(args.size_of_replay_memory*controls_per_unit_time*t_max) if args.train else 1
    memory = RL.Memory(capacity = capacity, data_size = data_size * 2 + 2, policy = 'sequential', passes_before_random = 2)
    # define the neural network
    net = RL.direct_DQN(data_size, num_of_controls, noisy_layers = 0).cuda()
    # set the task
    if args.train or args.LQG:
        train = RL.TrainDQN(net, memory, batch_size = args.batch_size, gamma=args.gamma_r, backup_period = args.target_network_update_interval, args=args)
        del net
        # the main function of training
        if args.seed != -1: 
            random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
        if args.train: 
            main = Main_System(train, num_of_processes=args.num_of_actors)
            main(num_of_episodes)
        # when we do not train and we test the result of analytic strategies
        elif args.LQG: 
            main = Main_System(train, num_of_processes=args.num_of_actors, others="LQG" if args.LQG else "")
            main(args.num_of_test_episodes)
    # if we test existing models, we use a loop to iterate over the models
    else:
        # find all models to test that end with no extension or '.pth' in the given directory
        import glob
        test_nets = []
        for name in glob.glob(os.path.join(args.folder_name,'*')):
            file_name, ext = os.path.splitext(os.path.basename(name))
            if (ext=='.pth' or ext=='') and os.path.isfile(name): test_nets.append((file_name, torch.load(name)))
        assert len(test_nets)!=0, 'No model found to test'
        from utilities import isfloat, isint
        test_nets = sorted([t for t in test_nets if isfloat(t[0])], key = lambda t: float(t[0])) + sorted([t for t in test_nets if not isfloat(t[0])])
        # for each model we run the main loop once
        for test_net in test_nets:
            net.load_state_dict(test_net[1])
            train = RL.TrainDQN(net, memory, batch_size = args.batch_size, gamma=0.99, backup_period = args.target_network_update_interval, args=args)
            main = Main_System(train, num_of_processes=args.num_of_actors, others=test_net[0])
            main(args.num_of_test_episodes)
        del net
        # organize all test results into one file
        with open(os.path.join(args.folder_name,'test_result.txt'),'w') as test_result:
            for test_net in test_nets:
                with open(os.path.join(args.folder_name,test_net[0]+'.txt')) as f:
                    result = f.readline()
                    test_result.write('{}:\t'.format(test_net[0])+result)
                    print('{}:\t'.format(test_net[0])+result, end='')
                os.remove(os.path.join(args.folder_name,test_net[0]+'.txt'))
    del main
    del memory

    from timer import print_elapsed_time
    print_elapsed_time(time_of_start)
