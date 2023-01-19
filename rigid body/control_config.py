import numpy as np
import math
from math import sqrt, pi
def LQG(x, y, px, py, dt, k, I, Q_z, displacement_max):
    vx = (px - Q_z * y / 2.) / I
    vy = (py + Q_z * x / 2.) / I
    target_vx = -sqrt(k/I)*x
    target_vy = -sqrt(k/I)*y
    F_x = (target_vx-vx)*I / dt
    F_y = (target_vy-vy)*I / dt
    
    if(abs(F_x/k)<=displacement_max and abs(F_y/k)<=displacement_max):
        # include first order correction (not all?)
        # for change in x and y
        avg_vx = (target_vx+vx)/2.
        avg_vy = (target_vy+vy)/2.
        dx = avg_vx*dt
        dy = avg_vy*dt
        F_x += (dx*(-sqrt(k/I))*I)/dt
        F_y += (dy*(-sqrt(k/I))*I)/dt
        # for cancelling the force of the potential
        F_x += k*(x+dx/2.)
        F_y += k*(y+dy/2.)
        # for cancelling the force of the magnetic field
        F_x -= (-avg_vy*Q_z)
        F_y -= ( avg_vx*Q_z)
    displacement_x = F_x/k
    displacement_y = F_y/k
    displacement_x = max(min(displacement_x, displacement_max), -displacement_max)
    displacement_y = max(min(displacement_y, displacement_max), -displacement_max)
    return displacement_x, displacement_y 

def LQG_bounded(x, y, px, py, dt, k, I, Q_z, displacement_max):
    vx = (px - Q_z * y / 2.) / I
    vy = (py + Q_z * x / 2.) / I
    target_vx = -sqrt(k/I)*x
    target_vy = -sqrt(k/I)*y
    F_x = (target_vx-vx)*I / dt
    F_y = (target_vy-vy)*I / dt
    
    if(abs(F_x/k)<=displacement_max and abs(F_y/k)<=displacement_max):
        # include first order correction (not all?)
        # for change in x and y
        avg_vx = (target_vx+vx)/2.
        avg_vy = (target_vy+vy)/2.
        dx = avg_vx*dt
        dy = avg_vy*dt
        F_x += (dx*(-sqrt(k/I))*I)/dt
        F_y += (dy*(-sqrt(k/I))*I)/dt
        # for cancelling the force of the potential
        F_x += k*(x+dx/2.)
        F_y += k*(y+dy/2.)
        # for cancelling the force of the magnetic field
        F_x -= (-avg_vy*Q_z)
        F_y -= ( avg_vx*Q_z)
    displacement_x = F_x/k
    displacement_y = F_y/k
    displacement_x = max(min(displacement_x, displacement_max), -displacement_max)
    displacement_y = max(min(displacement_y, displacement_max), -displacement_max)
    bound_dist = pi/4.
    displacement_x = max(min(displacement_x, x+bound_dist), x-bound_dist)
    displacement_y = max(min(displacement_y, y+bound_dist), y-bound_dist)
    return displacement_x, displacement_y 

control_force_list = []
no_action_choice = None

grid_indices_of_force_actions = []


action_flip_x = np.array([-1 for i in range(81)])
action_flip_y = np.array([-1 for i in range(81)])
action_flip_xy = np.array([-1 for i in range(81)])

def map_to_discrete_forces(displacement_x, displacement_y):
    min_dist = float("inf")
    index = -1
    for i, (F_x, F_y) in enumerate(control_force_list):
        dist = (F_x-displacement_x)**2 + (F_y-displacement_y)**2
        if dist < min_dist:
            index = i 
            min_dist = dist
    return control_force_list[index], index

def set_control_forces(discrete_values_on_one_side, maximum):
    assert type(discrete_values_on_one_side) is int
    one_D_forces = [i*maximum/discrete_values_on_one_side for i in range(-discrete_values_on_one_side, discrete_values_on_one_side+1)]
    control_force_list.clear()
    grid_indices_of_force_actions.clear()
    global no_action_choice
    grid_index = 0
    for i in one_D_forces:
        for j in one_D_forces:
            if i*i + j*j <= maximum:
                control_force_list.append((j,i)) # in the order of x, y
                grid_indices_of_force_actions.append(grid_index)
                if i == 0. and j == 0.: 
                    no_action_choice = len(control_force_list) - 1
            grid_index += 1
    global action_flip_x, action_flip_y, action_flip_xy
    action_flip_x = np.zeros(len(control_force_list), dtype=int)
    action_flip_y = np.zeros(len(control_force_list), dtype=int)
    action_flip_xy = np.zeros(len(control_force_list), dtype=int)
    for i, (force_x, force_y) in enumerate(control_force_list):
        action_flip_x[i] = map_to_discrete_forces(-force_x, force_y)[1]
        action_flip_y[i] = map_to_discrete_forces(force_x, -force_y)[1]
        action_flip_xy[i] = map_to_discrete_forces(-force_x, -force_y)[1]
    return len(control_force_list)


set_control_forces(5, 1.)


flip_x = np.ones(250)
flip_y = np.ones(250)
flip_xy = np.ones(250)

with open("moments.txt") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        moments = (line.strip()[1:-1]).split(',')
        moments = [int(moment) for moment in moments]
        if moments[0]%2==1:
            flip_x[i] *= -1; flip_xy[i] *= -1
        if moments[1]%2==1:
            flip_y[i] *= -1; flip_xy[i] *= -1
        if moments[2]%2==1:
            flip_x[i] *= -1; flip_xy[i] *= -1
        if moments[3]%2==1:
            flip_y[i] *= -1; flip_xy[i] *= -1
    flip_x[125:] = flip_x[:125]
    flip_y[125:] = flip_y[:125]
    flip_xy[125:] = flip_xy[:125]



