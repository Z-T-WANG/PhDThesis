#!/usr/bin/env python3
import simulation
import numpy as np
from math import *
from scipy.sparse import csr_matrix as csr
from scipy.sparse import csc_matrix as csc
from scipy import linalg
from time import time
import random, os
simulation.set_seed(9999)
import torch
if __name__ == '__main__':
    import torch.multiprocessing as mp
    from matplotlib.animation import FFMpegWriter
    import matplotlib.animation as animation

if __name__ != '__main__':
    torch.cuda.set_device(0)

################################### x space start (for display)

x_max = 11.; x_n = 250
x = np.linspace(-x_max,x_max,x_n, dtype=np.float64)

################################### x space end

################################### energy space start

omega = pi # don't ingore this energy multiplier !!!

def probability(state):
    return np.real(np.conj(state)*state)

def common_factor_of_1Dharmonics(n):
    return 1./np.sqrt(np.float128(repr(2**n))*np.float128(repr(factorial(n))))  *  sqrt(sqrt(1./pi)) * np.exp((-x*x/2).astype(np.float128))

def adjust_n_max(new_n_max):
    # \hbar = m\omega = 1; k = m\omega^2 = \omega
    global n_max
    n_max = new_n_max
    global n_phonon
    n_phonon=np.array([i for i in range(n_max+1)],dtype=np.float64)
    global sqrt_n
    sqrt_n = np.array([sqrt(i) for i in range(1,n_max+1)])
    global annihilation, creation
    annihilation=csr(np.diag(sqrt_n, k=1))
    creation=csr(np.diag(sqrt_n, k=-1))
    annihilation.prune(); creation.prune()
    global x_hat, p_hat
    x_hat = sqrt(1/2)*(creation + annihilation)
    p_hat = 1.j*sqrt(1/2)*(creation - annihilation)
    x_hat.prune(); p_hat.prune()
    global x_hat_2, p_hat_2, xp_px_hat
    x_hat_2 = x_hat.dot(x_hat); p_hat_2 = np.real(p_hat.dot(p_hat))
    xp_px_hat = x_hat.dot(p_hat)+p_hat.dot(x_hat)
    x_hat_2.prune(); p_hat_2.prune(); xp_px_hat.prune()
    global harmonic_Hamil
    harmonic_Hamil = omega *np.diag(1/2 + n_phonon)
    harmonic_Hamil = csr(harmonic_Hamil)
    harmonic_Hamil.prune()
    global eigen_states
    eigen_states=[]
    for i in range(new_n_max + 1):
        eigen_states.append(common_factor_of_1Dharmonics(i)*np.polynomial.hermite.hermval(x.astype(np.float128, order='C'), np.array([0. for j in range(i)]+[1.],dtype=np.float128)))
    eigen_states=np.array(eigen_states).transpose().astype(np.float64, order='C')
    print('n_max adjusted to {}'.format(new_n_max))

adjust_n_max(130)

def normalize(vector):
    p=np.sum(probability(vector))
    return vector / sqrt(p)

def phonon_number(state):
    return np.sum(probability(state)*n_phonon)

def x_expct(state):
    return np.real(np.conj(state).dot(x_hat.dot(state)))

def p_expct(state):
    return np.real(np.conj(state).dot(p_hat.dot(state)))

def expct(state, hermitian_operator):
    return np.real(np.conj(state).dot(hermitian_operator.dot(state)))

def spatial_repr(state):
    mask = np.abs(state) > 1e-4
    return eigen_states[:, :state.size][:, mask].dot(state[ mask ])

def get_data(state):
    x_expc, p_expc = x_expct(state), p_expct(state)
    return x_expc, p_expc, expct(state, x_hat_2)-x_expc**2, expct(state, p_hat_2)-p_expc**2, expct(state, xp_px_hat)/2-x_expc*p_expc
################################### energy space end

################################## start learning setting

half_period_steps = 360*4 # 360
time_step = 1 / half_period_steps # ~= 0.003

controls_per_half_period = 18
control_interval = round(half_period_steps / controls_per_half_period) # 

num_of_episodes = 20000
reward_multiply = 0.1
failing_reward = -(80-20)*reward_multiply

n_periods_to_read = 5
read_length = n_periods_to_read * 2*half_period_steps # 3600
n_actions_to_read = n_periods_to_read * 2*controls_per_half_period


if __name__ == '__main__':
    import plot
    plot.set_parameters(x=x, x_max=x_max, dt=time_step, num_of_episodes=num_of_episodes, probability=probability, 
        reward_multiply=reward_multiply, read_length=read_length, controls_per_half_period=controls_per_half_period)

import RL
RL.set_parameters(control_interval=control_interval, reward_multiply=reward_multiply, failing_reward=failing_reward)

################################## end learning setting

EPS_START = 0.4
EPS_END = 0.0002
EPS_DECAY = 1200

class Control(object):
    """
    this class is used as a central controller that distributes information to workers, i.e. actions,
    and organizes when to call train, and record training loss.
    
    The structure of this program is:
    
    --> Control <--> {Do_Episode}_n   (multiprocessing actor, 
    |                            |    including respective plots, and also Step instances for caching
    |                            |
    | call & update actor        | store
    |                            |
    --> RL.Train <-- RL.Memory <--
    it contains the data that are related to AI's decisions and parameters for the epsilon strategy (EPS_ s),
    and a mutable gamma that controls the continuous measurement strength, which is used in Step.__call__
    because it remembers the statistics of the control of one episode, it needs to clear its statistics when 
    reinitialized, otherwise it would continue using its last action at the start of an episode
    """
    gamma = 1.*omega
    update_time_interval = 10 # how many seconds before updating the network once
    def __init__(self, net, steps_done=0):
        self.steps_done = steps_done
        self.net = net
        self.no_action_choice = net.num_of_control_resolution_oneside
        self.clear()
    def clear(self):
        self.accu_Loss = 0.
        self.cache = 0.
        self.last_action = self.no_action_choice
    def __call__(self, data):
        #data = torch.from_numpy(data).float().unsqueeze(0).cuda()
        #with torch.no_grad():
        #    action_values, avg_value = self.net(data)
        #    action_values = action_values - action_values.mean(dim=1,keepdim=True) + avg_value.unsqueeze(1)
        #    self.value, self.last_action =action_values.max(1)
        #    self.last_action = self.last_action.item()
        #force = self.net.convert_to_force(self.last_action)
        x=data[0]; p=data[1]
        dt = 1./controls_per_half_period
        force_max = self.net.convert_to_force(2*self.no_action_choice)
        F = ((x+p)+(p-x)*omega*dt)/dt
        force = F / omega
        force = min(force, force_max)
        force = max(force, -force_max)
        force = float(round(force/force_max*self.no_action_choice))/self.no_action_choice*force_max
        self.cache = - force
        return False
    def get_loss(self):
        return None
    def get_force(self):
        return self.cache

class Do_Episode(object):
    t_max = 100.
    def __init__(self, controller, state=None, training_time = None):
        self.step = Step()
        self.control = controller
        if __name__ == '__main__':
            self.plot = plot.Plot(self.t_max)
            self.to_plot = True # switch of whether to plot or not
            self.xlims = self.plot.ax3.get_xlim()
            self.graph3_xdata = np.copy(self.plot.graph3_xdata)
            if training_time != None:
                self.plot.ax2.set_xlabel(r'training time $\approx$ {:.1f}s'.format(training_time), rotation=0)
        else: self.to_plot = False
        self.i_episode=0
        self.training_time = training_time
        if state.__class__ != type(None):
            self.state = state
            self.clear()
        else: self.reinit_state()
    def clear(self):
        self.measurement_results = np.array([],dtype=np.float32)
        print('clear state: Episode {}'.format(self.i_episode))
        self.cache_q = [0. for i in range(read_length)]
        self.t = 0.
        self.x_mean = [0. for i in range(read_length)]
        self.forces = [0. for i in range(read_length)]
        self.last_actions = [0. for i in range(n_actions_to_read)]
        self.last_action = self.control.no_action_choice
        self.to_stop = False
    def reinit_state(self):
        self.state = np.zeros((n_max+1,), dtype=np.complex128)
        self.state[0]=1.
        self.i_episode += 1
        self.clear()
        self.last_data = np.array(get_data(self.state))
    def __call__(self, n):
        if hasattr(self.step, 'ab'):
            self.step.ab = self.step.get_tridiagonal_matrix()
        if self.i_episode <= num_of_episodes:
            fail_counter = 0
            j=0
            self.to_stop = False
            stop = False
            force = self.control.get_force()
            gamma = self.control.gamma
            while j < n:
                m = self.measurement_results
                if m.size + len(self.cache_q) >= read_length + control_interval:
                    self.do_plot()
                    self.data = np.array(get_data(self.state))
                    m = np.hstack((m, np.array(self.cache_q, dtype=np.float32)))
                    self.last_actions.append(-1.*force)
                    if not self.to_stop: 
                         pass
                    else: 
                        stop = True
                    m = m[-read_length:] # reload the measurement data results
                    self.measurement_results = m
                    self.last_actions = self.last_actions[-n_actions_to_read:]
                    self.cache_q=[]
                    # from this line
                    rnd = self.control(self.data)
                    force = self.control.get_force()
                    self.last_action = self.control.last_action
                    self.last_data = self.data
                    # till this line, it needs to be synchronized by pipe & queue
                    # deliver estimations and current energies to plotting
                    self.clear_recorded_Xmean_forces()
                #elif len(self.cache_q)%control_interval==round(control_interval/2): # plot the evolution per half control instant
                #    self.do_plot()
                q, x_mean, Fail = simulation.step(self.state, time_step, force, gamma)
                self.cache_q.append(q)
                self.x_mean.append(x_mean)
                self.forces.append(force)
                if Fail and not self.to_stop : self.to_stop = True 
                self.t += time_step
                if stop or self.t >= self.t_max:
                    if not stop: self.do_plot(); print('\nSucceeded')
                    else: 
                        stop = False; print('\nFailed \t t = {:.2f}'.format(self.t)); fail_counter += 1
                    self.reinit_state()
                    self.control.clear()
                    j += 1
        print('{}/{}'.format(fail_counter,n))
        return (fail_counter,n)
    def clear_recorded_Xmean_forces(self):
        if len(self.x_mean) > read_length: self.x_mean = self.x_mean[-read_length:]
        if len(self.forces) > read_length: self.forces = self.forces[-read_length:]
    def do_plot(self):
        self.clear_recorded_Xmean_forces()
        if self.to_plot: 
            artists=self.plot(spatial_repr(self.state), np.hstack(( self.measurement_results, np.array( self.cache_q ) ))[-read_length:], self.x_mean, self.forces)
            return artists
    def frame(self, N):
        time_per_N = 8.* 1./controls_per_half_period / 30 # 8 controls per second (30 frames)
        if hasattr(self.step, 'ab'):
            self.step.ab = self.step.get_tridiagonal_matrix()
        stop = False
        force = self.control.get_force()
        gamma = self.control.gamma
        xlims = self.plot.ax3.get_xlim()
        self.plot.ax3.set_xlim(self.xlims[0]+N*time_per_N, self.xlims[1]+N*time_per_N)
        self.plot.graph3_xdata = self.graph3_xdata+N*time_per_N
        while self.t < N*time_per_N:
            m = self.measurement_results
            if m.size + len(self.cache_q) >= read_length + control_interval:
                data = np.array(get_data(self.state))
                m = np.hstack((m, np.array(self.cache_q, dtype=np.float32)))
                if not self.to_stop:
                    pass
                else: 
                    stop = True
                m = m[-read_length:] # reload the measurement data results
                self.measurement_results = m
                self.cache_q=[]
                self.control(data)
                force = self.control.get_force()
                # deliver estimations and current energies to plotting
                self.clear_recorded_Xmean_forces()
            #elif len(self.cache_q)%control_interval==round(control_interval/2): # plot the evolution per half control instant
            #    self.do_plot()
            q, x_mean, Fail = simulation.step(self.state, time_step, force, gamma)
            self.cache_q.append(q)
            self.x_mean.append(x_mean)
            self.forces.append(force)
            if Fail and not self.to_stop : self.to_stop = True 
            self.t += time_step
            if stop or self.t >= self.t_max:
                if not stop: self.do_plot(); print('\nSucceeded')
                else: 
                    stop = False; print('\nFailed \t t = {:.2f}'.format(self.t))
                self.reinit_state()
                self.control.clear()
        artists = self.do_plot()
        return artists

if True:
    class Step(object):
        def get_tridiagonal_matrix(self):
            temp_array = self.dt_cache*(0.5j)*omega*self.force_cache*sqrt(1/2)*sqrt_n
            upper_diag = np.hstack(([0.],temp_array))
            lower_diag = np.hstack((temp_array,[0.]))
            diag = np.full_like(upper_diag, 1.) + self.dt_cache*(0.5j)*np.diag(harmonic_Hamil.toarray())
            Hamiltonian = (harmonic_Hamil+omega*self.force_cache*x_hat)
            self.Hamiltonian_square = Hamiltonian.dot(Hamiltonian)
            return np.vstack((upper_diag,diag,lower_diag))
        def D1(self, state, x_avg=None):
            if x_avg==None: x_avg=x_expct(state)
            x_hat_state=x_hat.dot(state)
            relative_state=x_hat_state-x_avg*state
            return (-1.j)*(harmonic_Hamil.dot(state) + omega*self.force_cache*x_hat_state) - self.gamma/4*(x_hat.dot(relative_state)-x_avg*relative_state), relative_state
        def D1ImRe(self, state, x_avg=None):
            if x_avg==None: x_avg=x_expct(state)
            x_hat_state=x_hat.dot(state)
            relative_state=x_hat_state-x_avg*state
            return (-1.j)*(harmonic_Hamil.dot(state) + omega*self.force_cache*x_hat_state), - self.gamma/4*(x_hat.dot(relative_state)-x_avg*relative_state), relative_state
        def D2(self, state, relative_state=None):
            if relative_state.__class__==type(None): relative_state = x_hat.dot(state) - x_expct(state)*state
            return sqrt(self.gamma/2)*relative_state
        def __call__(self, state, dt, force, gamma): 
            # [Kloeden and Platen, Numerical Solution of Stochastic Differential Equations, p.408, (3.4), implicit strong order 1.5 for the imaginary part, and p.378, (2.1)]
            if not hasattr(self, 'dt_cache'):
                self.dt_cache=dt; self.force_cache=force
                self.ab = self.get_tridiagonal_matrix()
            elif dt != self.dt_cache or force != self.force_cache:
                self.dt_cache=dt; self.force_cache=force
                self.ab = self.get_tridiagonal_matrix()
            self.gamma = gamma

            # Eq. (10.4.3): initialize random variables
            U1,U2=np.random.normal(size=(2,))
            dW = sqrt(dt)*U1; dZ = (sqrt(dt)*dt)*0.5*(U1+U2/sqrt(3))

            x_mean = x_expct(state)
            q = x_mean + dW/sqrt(2*gamma)/dt
            D1_state, relative_state = self.D1(state, x_mean)
            D2_state = self.D2(state, relative_state)
            D2_state_dW, D2_state_drt = D2_state*dW, D2_state*sqrt(dt)
            Y = state + D1_state*dt
            Y_plus, Y_minus = Y + D2_state_drt, Y - D2_state_drt # these terms have normalization error up to 1st order of dt
            D1_Y_plusIm, D1_Y_plusRe, relative_Y_plus = self.D1ImRe(Y_plus)
            D1_Y_minusIm, D1_Y_minusRe, relative_Y_minus = self.D1ImRe(Y_minus)
            D2_Y_plus, D2_Y_minus = self.D2(Y_plus,relative_Y_plus), self.D2(Y_minus,relative_Y_minus)
            Phi_plus, Phi_minus = Y_plus + sqrt(dt)*D2_Y_plus, Y_plus - sqrt(dt)*D2_Y_plus
            D2_Phi_plus, D2_Phi_minus = self.D2(Phi_plus), self.D2(Phi_minus)
            D1_Y_plusIm_substract_D1_Y_minusIm = D1_Y_plusIm-D1_Y_minusIm

            # Start Eq. (11.2.1)
            # The 6th line is the 4th line of Eq. (12.3.4). This term cancels the b*dW term in the implicit term of state', as an 3/2 order correction, since state' appears as state' * dt
            state = state + D2_state_dW + 0.5/sqrt(dt)*dZ*(D1_Y_plusIm_substract_D1_Y_minusIm + D1_Y_plusRe-D1_Y_minusRe) + \
                0.25*dt*(D1_Y_plusRe+2*D1_state+D1_Y_minusRe) + \
                0.25/sqrt(dt)*(dW*dW-dt)*(D2_Y_plus - D2_Y_minus) + \
                0.5/dt*(dW*dt-dZ)*(D2_Y_plus + D2_Y_minus - 2*D2_state) + \
                0.25/dt*(dW*dW/3-dt)*dW*(D2_Phi_plus - D2_Phi_minus - D2_Y_plus + D2_Y_minus) \
              - 0.25*sqrt(dt)*dW*(D1_Y_plusIm_substract_D1_Y_minusIm) \
              + dt*dt*dt/12.*self.Hamiltonian_square.dot(D1_state)
            # a third order correction term of Hamiltonian*dt, since its error results in decrease of high energy components
            # the coefficient -1/12 comes from the fact that implicit term D1Im(state') includes a1'*(a'a*(dt^2/2)), which becomes
            # (dt^3/4)(a1'*a'*a), larger than (dt^3/6)(a1'*a1'*a) by a factor of (dt^3/12)
            if np.absolute(state[-11]) < 1e-5: Fail = False
            else: Fail = True
            state = linalg.solve_banded((1,1), self.ab, state, overwrite_ab=False, overwrite_b=True)
            state = normalize(state)
            return state, q, x_mean, Fail

def test(args):
    net, steps_done, training_time = data
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    call_control=Control(net.cuda(), steps_done)
    do_episode = Do_Episode(call_control)
    do_episode.to_plot = False
    return do_episode(n)

def record_animation(net,i):
    #net, steps_done, training_time = data
    call_control=Control(net.cuda()) # , steps_done
    do_episode = Do_Episode(call_control) #, training_time=training_time
    #do_episode(10)
    #writer = FasterFFMpegWriter(fps=30, bitrate=-1, codec='h264') 
    # https://stackoverflow.com/questions/22010586/matplotlib-animation-duration
    # it explains that "fps" controls the saved animation, but "interval" controls the displayed
    ani = animation.FuncAnimation(do_episode.plot.figure, do_episode.frame, 500, interval=1000/30, blit=False)
    writer = animation.FFMpegWriter(fps=30, bitrate=-1, codec='h264')#_nvenc
    #os.makedirs('./videos', exist_ok=True)
    ani.save('./animation_test.mp4', writer=writer)

if __name__ == '__main__':
    #mp.set_start_method('forkserver')
    datas = []
    #for i in range(1,16): 
    datas.append( torch.load( './DQN_save/1'))#.format(i)) )
    #with mp.Pool(processes=5) as pool:
    #    results = list(pool.imap(test, [(net,1000,random.randrange(0,9999999)) for net in nets]))
    for i,data in enumerate(datas):
        record_animation(data,i)
        #print('number {}: {}/{}'.format(i, result[0],result[1]))
    #record_animation(torch.load('./DQN_save/2', map_location='cpu'))
    #print('time spent = {:.3f}s'.format(time()-last_time))
