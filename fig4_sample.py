import argparse
import numpy as np

from libdai import run_libdai
    

def main(args_dict):
    # Extract model parameters
    topology = 'grid'
    A = args_dict['A']
    B = args_dict['B']
    n = A * B
    R = args_dict['R']
    f = args_dict['f']
    cs = args_dict['cs']

    # Extract computation parameters
    MK = args_dict['MK']
    
    # Run all desired models
    for c in cs:
        print('c = %g' % (c))
        
        potentials_type = 'attractive'
        MAP_solver = 'JT_BP'
        run_libdai(topology, n, A, B, R, f, c, potentials_type, MK, MAP_solver)

        potentials_type = 'mixed'
        MAP_solver = 'JT'
        run_libdai(topology, n, A, B, R, f, c, potentials_type, MK, MAP_solver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--A', default=10, type=int, help='spin glass model width')
    parser.add_argument('--B', default=10, type=int, help='spin glass model height')
    parser.add_argument('--R', default=2, type=int, help='number of different labels a variable can take')
    parser.add_argument('--f', default=1.0, type=float, help='range of unary potentials')
    parser.add_argument('--cs', default=(3.0 * np.array([0, 1, 3, 5, 7, 9]) / 9), nargs='+', type=float, help='coupling strengths')
    parser.add_argument('--MK', default=100000, type=int, help='number of MAP samples to obtain (M*K)')
    args_dict = vars(parser.parse_args())
    main(args_dict)
