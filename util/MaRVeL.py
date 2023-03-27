import csv
import numpy as np
import os.path
import pickle
from constraint_utils import *
from refine_gpupoly import *
import multiprocessing as mp
import torch
import time


def compute_milp(nn, specLB, specUB, nlb, nub, num_of_nuerons, time_limit_, time_stamp, target_label, label):
    constraints = get_constraints_for_dominant_label(label, [target_label])
    verified_flag, milp_images, milp_objs, milp_target = \
        verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints, num_of_nuerons, time_limit_)
    pickle.dump([verified_flag, milp_images, milp_objs, milp_target], open("/tmp/"+str(target_label) + "_" +str(time_stamp)+ "_res.p", "wb"))


class MaRVeL:
    def __init__(self, num_pixels, config):

        self.num_of_neurons = config.milp_num_of_neurons
        self.lambda_ = config.lambda_
        self.lambda_min = config.lambda_min
        self.lambda_alpha = config.lambda_alpha
        self.etta_ = config.lr
        self.time_limit_ = config.milp_time_limit
        self.parallelize = bool(config.parallelization)
        self.t_size = config.t_size
        self.precision_constant = 1e-6
        self.back_off = 0.99
        self.epsilon_up_v = np.zeros(num_pixels)
        self.epsilon_down_v = np.zeros(num_pixels)
        self.epsilon_up_ = np.zeros(num_pixels)
        self.epsilon_down_ = np.zeros(num_pixels)
        self.prevent_up = []
        self.prevent_down = []
        self.last_mean_v = -1
        self.first_fail_mean = 0
        self.first_fail_time = 0
        self.time_stamp = int(time.time() * 1000000)

    def fast_step(self, image, label, normalize, lb_limit, ub_limit, eran):
        specLB = normalize(np.clip(image - self.epsilon_down_, lb_limit, ub_limit))
        specUB = normalize(np.clip(image + self.epsilon_up_, lb_limit, ub_limit))
        perturbed_label, _, nlb, nub, failed_labels, x = eran.analyze_box(specLB, specUB, "deeppoly", 1,
                                                                          1,
                                                                          True, label=label,
                                                                          prop=-1, K=0, s=0,
                                                                          timeout_final_lp=100,
                                                                          timeout_final_milp=100,
                                                                          use_milp=False, complete=False,
                                                                          terminate_on_failure = False,
                                                                          partial_milp=0, max_milp_neurons=0,
                                                                          approx_k=0)
        if perturbed_label == label:
            ub_bounds = np.copy(nub[-1])
            ub_bounds[label] = -np.infty
            target_label = list(set([np.argmax(ub_bounds)]))
        else:
            target_label = list(set(failed_labels))
        return (perturbed_label == label), target_label, nlb, nub, specLB, specUB

    def milp_step(self, label, target_label, specLB, specUB, nn, nlb, nub):
        if self.parallelize == True:
            processes = [
                mp.Process(target=compute_milp, args=(nn, specLB, specUB, nlb, nub, self.num_of_neurons, self.time_limit_, self.time_stamp, t, label)) for
                t in target_label]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            verified_flag = True
            milp_images = []; milp_objs = []
            milp_target = []; milp_adv_images = []
            for t in target_label:
                [verified_flag_, milp_images_, milp_objs_, milp_target_] = pickle.load(open("/tmp/"+str(t) + "_" + str(self.time_stamp) + "_res.p", "rb"))
                os.remove("/tmp/"+str(t) + "_" + str(self.time_stamp) + "_res.p")
                verified_flag = verified_flag and verified_flag_
                if not verified_flag_ and milp_images_ is not None:
                    milp_adv_images += milp_images_
                if milp_images_ is not None:
                    milp_images += milp_images_
                    milp_objs += milp_objs_
                    milp_target += milp_target_
        else:
            constraints = get_constraints_for_dominant_label(label, target_label)
            verified_flag, milp_images, milp_objs, milp_target = verify_network_with_milp(nn, specLB, specUB, nlb, nub,
                                                                                          constraints, self.num_of_nuerons,
                                                                                          self.time_limit_)
            milp_adv_images = []
            for i, m_obj in enumerate(milp_objs):
                if m_obj <= 0:
                    milp_adv_images.append(milp_images[i])
        return [verified_flag, milp_images, milp_objs, milp_target, milp_adv_images]

    def progress_step(self, fast_certification_result, milp_certification_result, image, lb_limit, \
                      ub_limit, milp_adv_images,dswap):
        specLB = np.clip(image - self.epsilon_down_, lb_limit, ub_limit)
        specUB = np.clip(image + self.epsilon_up_, lb_limit, ub_limit)
        specLB_v = np.clip(image - self.epsilon_down_v, lb_limit, ub_limit)
        specUB_v = np.clip(image + self.epsilon_up_v, lb_limit, ub_limit)
        if fast_certification_result or milp_certification_result:
            self.epsilon_up_v = np.copy(self.epsilon_up_)
            self.epsilon_down_v = np.copy(self.epsilon_down_)
            mean_eps_v = np.mean(np.array(specUB) - np.array(specLB))
            if np.abs(self.last_mean_v - mean_eps_v) <= self.t_size:
                self.lambda_ = self.lambda_ * self.lambda_alpha
            self.last_mean_v = mean_eps_v
        else:
            self.lambda_ = self.lambda_ * self.lambda_alpha
            self.CEGIS(specLB, specUB, specLB_v, specUB_v, milp_adv_images,dswap)
            self.epsilon_up_ = np.copy(self.epsilon_up_v)
            self.epsilon_down_ = np.copy(self.epsilon_down_v)

        return self.last_mean_v

    def CEGIS(self, specLB, specUB, specLB_v, specUB_v, milp_adv_images, dswap):
        problomatic_pixels_up = np.copy(specLB) * 0
        problomatic_pixels_down = np.copy(specLB) * 0

        for cnt_adv_, adv_i in enumerate(milp_adv_images):
            adv_i = dswap(adv_i)
            ind_up = [p_ for p_ in range(len(specLB)) if
                      adv_i[p_] <= specUB[p_] + self.precision_constant and adv_i[p_] >= specUB_v[p_] - self.precision_constant]
            problomatic_pixels_up[ind_up] += 1
            ind_down = [p_ for p_ in range(len(specLB)) if
                        adv_i[p_] <= specLB_v[p_] + self.precision_constant and adv_i[p_] >= specLB[p_] - self.precision_constant]
            problomatic_pixels_down[ind_down] += 1

        pp_up = []
        for i_adv_ind in range(len(milp_adv_images)):
            pp_up = [p_ for p_ in range(len(specLB)) if problomatic_pixels_up[p_] == len(milp_adv_images) - i_adv_ind]
            if len(pp_up) > 0:
                break
        pp_down = []
        for i_adv_ind in range(len(milp_adv_images)):
            pp_down = [p_ for p_ in range(len(specLB)) if
                       problomatic_pixels_down[p_] == len(milp_adv_images) - i_adv_ind]
            if len(pp_down) > 0:
                break
        self.prevent_up += pp_up
        self.prevent_up = list(set(self.prevent_up))
        self.prevent_down += pp_down
        self.prevent_down = list(set(self.prevent_down))
        modefiy_up = []
        modefiy_down = []
        for cnt_adv_, adv_i in enumerate(milp_adv_images):
            modefiy_up += [p_ for p_ in range(len(specLB)) if
                           (specUB_v[p_] - adv_i[p_]) <= self.precision_constant]
            modefiy_down += [p_ for p_ in range(len(specLB)) if
                             (adv_i[p_] - specLB_v[p_]) <= self.precision_constant]
            modefiy_up = list(set(modefiy_up))
            modefiy_down = list(set(modefiy_down))
        for mup_ in modefiy_up: self.epsilon_up_v[mup_] = self.back_off * self.epsilon_up_v[mup_]
        for mlp_ in modefiy_down: self.epsilon_down_v[mlp_] = self.back_off * self.epsilon_down_v[mlp_]
        self.prevent_up += modefiy_up
        self.prevent_up = list(set(self.prevent_up))
        self.prevent_down += modefiy_down
        self.prevent_down = list(set(self.prevent_down))

    def optimize(self, milp_objs, milp_images, model, label, num_pixels):
        data_grads = []
        milp_objs_softmax = np.exp(-np.array(milp_objs)) / np.sum(np.exp(-np.array(milp_objs)))
        for cnt_, imgs_ in enumerate(milp_images):
            image_for_gradient = np.copy(np.array(imgs_))
            data_grad = model.compute_gradient_to_input(torch.from_numpy((np.array([image_for_gradient]))), int(label))
            data_grads.append(milp_objs_softmax[cnt_] * data_grad)
        data_grad = np.squeeze(np.mean(np.array(data_grads), axis=0))
        epsilon_grad_up = np.sign(self.epsilon_up_)
        epsilon_grad_up[epsilon_grad_up == 0] = 1
        epsilon_grad_down = np.sign(self.epsilon_down_)
        epsilon_grad_down[epsilon_grad_down == 0] = 1
        lambda_up = self.lambda_ * np.linalg.norm(epsilon_grad_up) / np.linalg.norm(data_grad)
        epsilon_up_total = epsilon_grad_up + lambda_up * data_grad
        epsilon_up_total[epsilon_up_total < 0] = 0
        epsilon_up_total = (epsilon_up_total / np.linalg.norm(epsilon_up_total)).reshape(num_pixels)
        lambda_down = self.lambda_ * np.linalg.norm(epsilon_grad_down) / np.linalg.norm(data_grad)
        epsilon_down_total = epsilon_grad_down - lambda_down * data_grad
        epsilon_down_total[epsilon_down_total < 0] = 0
        epsilon_down_total = (epsilon_down_total / np.linalg.norm(epsilon_down_total)).reshape(num_pixels)
        self.epsilon_up_ += self.etta_ * epsilon_up_total
        self.epsilon_down_ += self.etta_ * epsilon_down_total
        self.epsilon_up_[self.prevent_up] = np.copy(self.epsilon_up_v[self.prevent_up])
        self.epsilon_down_[self.prevent_down] = np.copy(self.epsilon_down_v[self.prevent_down])

    def terminate(self, num_pixels):
        if (len(self.prevent_up) == num_pixels and len(self.prevent_down) == num_pixels) or self.lambda_ < self.lambda_min:
            return True
        return False

    def reset(self, num_pixels, config):
        self.num_of_neurons = config.milp_num_of_neurons
        self.lambda_ = config.lambda_
        self.lambda_min = config.lambda_min
        self.lambda_alpha = config.lambda_alpha
        self.etta_ = config.lr
        self.time_limit_ = config.milp_time_limit
        self.parallelize = bool(config.parallelization)
        self.t_size = config.t_size
        self.precision_constant = 1e-6
        self.back_off = 0.99
        self.epsilon_up_v = np.zeros(num_pixels)
        self.epsilon_down_v = np.zeros(num_pixels)
        self.epsilon_up_ = np.zeros(num_pixels)
        self.epsilon_down_ = np.zeros(num_pixels)
        self.prevent_up = []
        self.prevent_down = []
        self.last_mean_v = -1
        self.first_fail_mean = 0
        self.first_fail_time = 0
