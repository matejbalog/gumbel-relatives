import os
import subprocess

from utils import json_load


PATH_CPP = os.path.expanduser('~/Programs/libdai/examples/spin_glass')


def get_filename(topology, n, A, B, K, f, c, potentials_type, M, MAP_solver):
    """ Compute libdai output filename string from model and inference parameters """
    text_size = '%dx%d' % (A, B) if topology == 'grid' else '%d' % (n)
    return 'data/%s_%s_K%d_f%g_c%g_%s_M%d_%s' % (topology, text_size, K, f, c, potentials_type, M, MAP_solver)


def load_libdai_results(topology, n, A, B, K, f, c, potentials_type, M, MAP_solver):
    """ Load libdai output in JSON format for given model and inference parameters """
    filename = get_filename(topology, n, A, B, K, f, c, potentials_type, M, MAP_solver)
    data_json = json_load(filename + '.json')
    return data_json


def run_libdai(topology, n, A, B, K, f, c, potentials_type, M, MAP_solver):
    """ Run libdai on model with specified parameters, using specified MAP solver """
    path_stdout = get_filename(topology, n, A, B, K, f, c, potentials_type, M, MAP_solver) + '.json'
    cmd = [PATH_CPP, topology, str(n), str(A), str(K), str(f), str(c), potentials_type, str(M), MAP_solver, str(0)]
    with open(path_stdout, 'w') as stdout:
        p = subprocess.Popen(cmd, stdout=stdout)
        p.wait()
