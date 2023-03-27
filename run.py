import pickle
import sys
import os
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, './util/')
sys.path.insert(0, './util/eran')
from eran import ERAN
import argparse
from pprint import pprint
from refine_gpupoly import *
from datasets import *
from nn_models import *
from MaRVeL import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MaRVeL Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--netname', type=str, default=config.netname, help='the network name, the extension can be only .onnx')
    parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, fmnist, cifar10, contagio, or syn')
    parser.add_argument('--timeout', type=float, default=config.timeout, help='the maximal certification time')
    parser.add_argument('--max_iterations', type=float, default=config.max_iterations, help='the maximal number of optimization steps')
    parser.add_argument('--milp_num_of_neurons', type=float, default=config.milp_num_of_neurons, help='the number of milp neurons per layer')
    parser.add_argument('--milp_time_limit', type=float, default=config.milp_time_limit, help='the time limit to solve a single milp problem')
    parser.add_argument('--lr', type=float, default=config.lr, help='the learning rate')
    parser.add_argument('--lambda', type=float, default=config.lambda_, help='the trade-off parameter')
    parser.add_argument('--lambda_alpha', type=float, default=config.lambda_alpha, help='the trade-off parameter decay rate')
    parser.add_argument('--lambda_min', type=float, default=config.lambda_min, help='the trade-off parameter minimal value')
    parser.add_argument('--parallelization', type=float, default=config.parallelization, help='run milp problems in parallel 0/1')
    parser.add_argument('--output_file', type=str, default=config.output_file, help='the output file ')
    parser.add_argument('--samples_num', type=float, default=config.samples_num, help='the number of samples to analyze')
    parser.add_argument('--t_size', type=float, default=config.t_size, help='the minimal difference between two optimization steps')
    args = parser.parse_args()

    for k, v in vars(args).items():
        setattr(config, k, v)
    config.json = vars(args)
    pprint(config.json)
    os.sched_setaffinity(0, cpu_affinity)
    netname = config.netname
    timeout = config.timeout
    output_file = config.output_file
    max_iterations = config.max_iterations
    samples_num = config.samples_num

    results_to_save = []

    assert config.netname, 'a network has to be provided for analysis.'
    assert os.path.isfile(netname), f"Model file not found. Please check \"{netname}\" is correct."
    filename, file_extension = os.path.splitext(netname)
    assert file_extension == ".onnx", "file extension not supported"

    model, is_conv = read_onnx_net(netname)
    eran = ERAN(model, is_onnx=True)
    dataset = Datasets(config.dataset)
    nn_model = NN_Model(netname, dataset, is_conv)
    marvel = MaRVeL(dataset.num_pixels, config)
    [images, labels, w, h, c, num_pixels, ub_limit, lb_limit, m, means, stds] = dataset.get_dataset_attributes()
    for i in range(len(images)):
        if i >= samples_num:
            break
        image = np.float64(images[i]).reshape(num_pixels) / np.float64(m)
        label = int(labels[i])
        plabel, nn, nlb, nub, _, _ = eran.analyze_box(dataset.normalize(image), dataset.normalize(image), "deeppoly", 1, 1, True)
        if plabel == label:
            start_ = time.time()
            opt_step = 0
            while opt_step < max_iterations:
                opt_step += 1
                fast_certification_result, target_label, nlb, nub, specLB, specUB = marvel.fast_step(image, label, dataset.normalize, lb_limit, ub_limit, eran)
                milp_certification_result, milp_images, milp_objs, milp_target, milp_adv_images = marvel.milp_step(label, target_label, specLB, specUB, nn, nlb, nub)
                verified_eps_avg = marvel.progress_step(fast_certification_result, milp_certification_result, image, lb_limit, ub_limit, milp_adv_images, dataset.dswap)
                if marvel.terminate(num_pixels) or time.time()-start_ > timeout:
                    break
                marvel.optimize(milp_objs, milp_images, nn_model, label, num_pixels)

                print("---------------------", opt_step ,"---------------------")
                color_ = '\x1b[6;30;42m' if milp_certification_result else '\x1b[0;30;41m'
                print("sample:", i, "label:", label, "verified average diameter", verified_eps_avg, "time", time.time()-start_, "certification results:", color_ + str(milp_certification_result) + '\x1b[0m')
            marvel.reset( num_pixels, config )

            if not (output_file is None):
                results_to_save.append([i, label, verified_eps_avg, time.time()-start_])
                with open(output_file, 'wb') as f:
                    pickle.dump(results_to_save, f)

        else:
            print("img", i, "not considered, incorrectly classified")
            end = time.time()
    if len(results_to_save) == 0:
        print("No correctly classfied samples")
    else:
        results_to_report = np.array(results_to_save)
        print("Average of verified diameters:", np.mean(results_to_report[:, 2]), "Average of certification time[s]:", np.mean(results_to_report[:, 3]))
