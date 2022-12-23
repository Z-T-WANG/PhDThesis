import argparse, datetime

parser = argparse.ArgumentParser()

# setting of quantum simulation (compiled in a C module which is importable by Python)
parser.add_argument('--ground_state_size', default = 0.1 , type=float,
                    help='the size (the standard deviation) of the ground state Gaussian wave packet of the trapping potential, ignoring all high order terms as well as the angular momentum')
parser.add_argument('--x_max', default = 47, type=int,
                    help='the number of discretization sites in the simulation space, in both x and y direction, counted from the center to the boundary')
parser.add_argument('--Q_z', default = 5, type=int,
                    help='the angular momentum in the local z direction, along the axis of the rotating quantum rod')
parser.add_argument('--init_dist', default = 0.4, type=float, # 0.4
                    help='the maximal distance from the initialized state to the center of the trap, in both x and y coordinates')
parser.add_argument('--gamma_factor', default = 3. , type=float,          # from 0.5 to 4. ?   Beyond 4. it tends to diverge.
                    help=r'the measurement strength \gamma divided by the potential strength k')
parser.add_argument('--grid_size', default = 0.03 , type=float,
                    help='the grid size of the discretized space of x and y')
parser.add_argument('--time_step', default = 0.00125 , type=float,
                    help='the time step used in the simulation of the time evolution')
parser.add_argument('--control_time', default = 0.2, type=float,
                    help="the time step of the controller, during which the control field is constant")
parser.add_argument('--t_max', default = 100., type=float,
                    help='the time of a complete episode in simulation, or, the maximal time that is continuously simulated')
parser.add_argument('--F_choices', default = 5, type=int, metavar='F_{max}',
                    help='the number of allowed choices of the control force from 0 (exclusive) to +F_max in the x and the y direction, which specifies the discretization of the control')
parser.add_argument('--F_max', default = 1., type=float, metavar='F_{max}',
                    help='the maximum control force allowed, measured in terms of sqrt(E_x^2 + E_y^2)/E_z')
parser.add_argument('--energy_cutoff', default = 30., type=float,          # 20 ? 25 ? 30 ?
                    help='the maximum energy allowed, beyond which we terminate the simulation and end the episode, in which case the Q value is set to be a constant')
parser.add_argument('-c','--compile', action='store_true',
                    help='whether to force a compilation if a existing file exist')

# where to store models, whether to test models or LQG control
parser.add_argument('--save_dir', default='', type=str,
                    help='the directory to save trained models. It defaults to a conventional naming style that is "inputType_omega_gamma".')
parser.add_argument('--LQG', action='store_true',
                    help='whether to use the LQG control without training')
parser.add_argument('--test', action='store_true',
                    help='whether to test existing trained models rather than to train')
parser.add_argument('--load_dir', default='', type=str,
                    help='the directory of models to test')
parser.add_argument('--num_of_test_episodes', default=1000, type=int,
                    help='the number of episodes to test and collect performance data for each model')

# training settings
parser.add_argument('--gamma_r', default = 0.96, type=float,            # 0.96 ? (25 controls) 0.98 ? (50 controls)
                    help='the discount factor gamma in reinforcement learning')
parser.add_argument('--batch_size', default = 256, type=int,
                    help='the sampled minibatch size per update step in training')
parser.add_argument('--n_times_per_sample', default = 8, type=int,
                    help='the number of times each experience is sampled and learned')
parser.add_argument('--size_of_replay_memory', default = 2000, type=int,
                    help='the size of the replay memory that stores the accumulated experiences, in units of full-length episodes.\nIts default value for "xp" and "wavefunction" input is 5000, and is 1000 for "measurements" input. When this argument receives a non-zero value, the default is overridden.')
parser.add_argument('--target_network_update_interval', default = 300, type=int,
                    help='the number of performed gradient descent steps before updating the target network. \nThe target network is a lazy copy of the currently trained network, i.e., it is updated to the current network only after sufficiently many gradient descent steps are done. It is used in DQN training to provide a more stable evaluation of the current Q value. The number of the gradient descent steps is this "target_network_update_interval" parameter.')
parser.add_argument('--train_episodes_multiplicative', default = 1., type=float,
                    help=r'the multiplicative factor that rescales the default number of simulated episodes (9000), each of time 100, i.e. \frac{100}{\omega_c}. The counting of episodes will be reset to 1 when the controller achieves time 100 for the first time, so it corresponds to the number of episodes after learning has started. This rescaling factor also rescales the learning rate schedule.')
parser.add_argument('--maximum_trails_before_giveup', default = 200000, type=int,
                    help=r'the maximal number of simulated episodes when the learning does not proceed. If the simulated episodes exceed this value, we give up training.')
parser.add_argument('--init_lr', default = 4e-4, type=float,
                    help='the initial learning rate. The learning rate will be decayed to 4e-5 at episode 1000, 8e-6 at 3000, 2e-6 at 5000, 4e-7 at 6500 and 1e-7 at 8000 when the current learning rate is higher.')
parser.add_argument('--reward_multiply', default = 1., type=float,
                    help='a multiplicative factor of the reward for the AI')
parser.add_argument('--input_scaling', default = 1., type=float,
                    help='a multiplicative factor of the input data to the AI. This is to avoid a possibly different scale between the input and the output of the AI, which may require too many unnecessary update steps during learning. This feature is set to 1 and thus disabled by default.')
parser.add_argument('--num_of_actors', default = 10, type=int,
                    help='the number of actors, i.e. the number of working processes that repeatedly do the control to accumulate experiences.')
parser.add_argument('--show_actor_recv', action='store_true',
                    help='to signify when a new model is received by the actors during training')
parser.add_argument('--num_of_saves', default = 20, type=int,
                    help='the number of models to save. Models with higher training performances are saved in order.')

parser.add_argument('--no_augmentation', action = 'store_true',
                    help='whether not to use the data augmentation (x and y flip)')
#parser.add_argument('--write_training_data', action = 'store_true',
#                    help='whether to store the data that are used to plot training curves')

parser.add_argument('--CDQN', action = 'store_true',
                    help='whether to use the convergent DQN')
parser.add_argument('--comment', default = "", type=str)
parser.add_argument('--folder', default = "", type=str)

# system config
parser.add_argument('--gpu_id', default = 0, type=int,
                    help='the index of the GPU to use')
parser.add_argument('--seed', default = -1, type=int,
                    help='the random seed. When not set, it is random by default.')

#parser.add_argument('--eta', default = 1., type=float,
#                    help='the measurement efficiency. \eta being smaller than 1 means that there are additional measurements performed by the environment but are ignored.')

args = parser.parse_args()

args.write_training_data = True

#assert 0. < args.eta <= 1., "The measurement efficiency should be larger than 0 and equal to or smaller than 1. It is currently {:.3g}.".format(args.eta)

args.num_of_episodes = round(9000*args.train_episodes_multiplicative)
args.lr_schedule = [(round(t[0]*args.train_episodes_multiplicative) if t[0]!=float('inf') else t[0], t[1]) for t in [(0,3e-4), (1000,1e-4), (3000,4e-5), (5000,2e-5), (6500,8e-6), (8000,2e-6), (9000, 0.)]]
# set the default learning rate
args.lr = args.init_lr

# decide whether to train based on the commandline arguments
if args.test: args.train = False
elif args.LQG: args.train = False
else: args.train = True

if not args.train: args.n_times_per_sample = 0.0000001

# prepare the path name
args.folder_name = "{}_{}/{:.1f}_{}{}".format(args.ground_state_size, args.grid_size, args.gamma_factor, args.Q_z, args.comment) if args.save_dir == '' else args.save_dir
if args.test:
    if args.load_dir != '':
        args.folder_name = args.load_dir # "load_dir" overrides "save_dir"
    else: 
        args.load_dir = args.folder_name # if "save_dir" is provided but "load_dir" is not, we assume that "load_dir" is the same as "save_dir"
elif args.load_dir != '' and args.save_dir == '': args.folder_name = args.load_dir


