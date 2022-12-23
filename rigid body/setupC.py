from distutils.core import setup, Extension
from math import pi
import numpy as np 
import os, sys, shutil, glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--omega', default= pi, type=float, metavar='\omega',
                    help='the angular frequency of the potential in terms of the quadratic term')
parser.add_argument('--x_max', default=75, type=float, metavar='x_{max}',
                    help='the number of grid points from the center to the border of the simulated space')
parser.add_argument('--ground_state_size', default= 0.02, type=float, metavar='\sigma',
                    help='suppose that the ground state size is 0.02 ??')
#parser.add_argument('--I_xy', default= 1., type=float, metavar='I_xy',
#                    help='moment of inertia for the x and y directions')
parser.add_argument('--Q_z', default= 50, type=int, metavar='Q_z',
                    help='the angular momentum in the local Z direction')
parser.add_argument('--grid_size', default = 0.002, type=float, metavar='h',
                    help='the distance between grid points in simulation') # suppose that the ground state size is 0.02 ??
parser.add_argument('--moment', default = 5, type=int,
                    help='the order of the distribution moments to compute in the compiled function "get_moments"')
args = parser.parse_args()




# Please rewrite the following arguments based on your OS and your prescription of compilation if necessary
# Please refer to https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor . Usually Python uses GCC as the default compiler, and then GNU compiler should be selected. The arguments starting with "-I" mean to "include" those directories.

os.environ["MKLROOT"] = "/opt/intel/compilers_and_libraries/linux/mkl"


link_options = ['-Wl,--start-group', os.environ['MKLROOT']+'/lib/intel64/libmkl_intel_ilp64.a', os.environ['MKLROOT']+'/lib/intel64/libmkl_intel_thread.a', os.environ['MKLROOT']+'/lib/intel64/libmkl_core.a', '-Wl,--end-group', '-liomp5', '-lpthread', '-lm', '-ldl']

compiler_options = ['-DMKL_ILP64','-m64']

##############################################################################
# The following is the compilation program. 

def compile(k, x_max, grid_size, I_xy, Q_z, moment):
    assert k> 0, 'the coefficient of the potential must be positive'
    assert grid_size> 0., 'the gird size should be positive'
    assert x_max> 0., 'the size of the simulation space (2 * x_max + 1) should be positive'
    assert I_xy > 0., 'the moment of inertia must be larger than zero'# the angular frequency is fixed to pi, and I_xy is ground_state_size/pi
    assert moment >= 1, 'the order of distribution moments should be larger than 1'

    # It invokes the native "distutils.core" of Python by setting the commandline arguments stored in sys.argv to the desired one ("build")

    # set the "build" command
    original_args_exist = False
    if len(sys.argv)>=2:
        original_args=sys.argv[1:]
        sys.argv = [sys.argv[0], "build"]
        original_args_exist = True
    else: sys.argv.append("build")

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    package_name = 'simulation'

    module1 = Extension(package_name,language='c++',
                    define_macros = [('V_K', str(k)), ('X_MAX', str(x_max)), ('I_XY',str(I_xy)), ('Q_Z',str(float(Q_z))), ('MOMENT', str(moment)), ('GRID_SIZE', str(grid_size))], # pass the defining parameters
                    include_dirs = [np.get_include(), os.path.join(os.environ['MKLROOT'],'include')],
                    sources = ['simulation_r_xy_exact.cpp'],  # 
                    extra_compile_args = compiler_options+['-Ofast','-funroll-loops', '-march=native', '-flto','-fuse-linker-plugin','--param', 'ipcp-unit-growth=2000', '-std=c++14','-fno-stack-protector','-fmerge-all-constants'], 
                    extra_link_args = link_options+['-Ofast','-fdelete-null-pointer-checks','-funroll-loops', '-march=native', '-fwhole-program','-flto','-fuse-linker-plugin','--param', 'ipcp-unit-growth=2000','-std=c++14','-fno-stack-protector','-fmerge-all-constants']) #Ofast

    setup (name = package_name,
       version = '1.0',
       description = 'do simulation steps',
       author = 'Wang Zhikang',
       ext_modules = [module1])

    # copy the compiled C module to the root to import
    compiled_files = glob.glob('build/**/*')
    for compiled_file in compiled_files:
        if 'temp' not in compiled_file:
            shutil.move(compiled_file, os.path.basename(compiled_file), copy_function=shutil.copy2)

    # restore the original commandline arguments
    if original_args_exist: sys.argv = [sys.argv[0]]+original_args
    else: sys.argv.pop(1)

m_times_omega = 1./(2* args.ground_state_size**2)
I_xy = m_times_omega/args.omega
k = (args.omega**2) * I_xy
compile(k=k, x_max=args.x_max, grid_size=args.grid_size, I_xy=I_xy, Q_z=args.Q_z, moment=args.moment)



