from enum import Enum


class Device(Enum):
    CPU = 0
    CUDA = 1

class config:
    netname = None
    dataset = None
    output_file = None
    timeout = 3600
    max_iterations = 10000
    milp_num_of_neurons = 200
    milp_time_limit = 100
    lr = 0.1
    lambda_ = 0.99
    lambda_alpha = 0.75
    lambda_min = 0.01
    parallelization = True
    samples_num = 10
    t_size = 1e-4
    device = Device.CPU
