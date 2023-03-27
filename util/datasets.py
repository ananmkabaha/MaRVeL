import csv
import numpy as np
import os.path
import pickle

class Datasets:
    def __init__(self, dataset, means=[], stds=[]):
        assert os.path.exists("./data/"+dataset+"_test.p"), dataset+" is not suppourted"
        self.dataset =  dataset

        [self.images, self.labels] = pickle.load(open("./data/"+dataset+"_test.p", "rb"))

        if self.dataset == 'mnist' or self.dataset == 'fmnist':
            self.w, self.h, self.c = 28, 28, 1
            self.u, self.l = 1, 0
            self.num_pixels = 784
            self.m = 255.0
            self.classes_num = 10
        elif self.dataset == 'cifar10':
            self.w, self.h, self.c = 32, 32, 3
            self.u, self.l = 1, 0
            self.num_pixels = 3072
            self.m = 255.0
            self.classes_num = 10
        elif self.dataset == 'contagio':
            self.w, self.h, self.c = 135, 1, 1
            self.num_pixels = 135
            self.u, self.l = np.inf, -np.inf
            self.m = 1
            self.classes_num = 2
        elif self.dataset == 'syn':
            self.w, self.h, self.c = 2, 1, 1
            self.num_pixels = 2
            self.u, self.l = 1, -1
            self.m = 1
            self.classes_num = 10

        if len(means)==0:
            if dataset == 'mnist' or dataset == 'fmnist' or dataset == 'contagio' or dataset == 'syn':
                means = [0]
                stds = [1]
            elif dataset == "cifar10":
                means = [0, 0, 0]
                stds = [1.0, 1.0, 1.0]
        self.means = means
        self.stds = stds

    def get_samples(self):
        return [self.images, self.labels]

    def get_dataset_attributes(self):
        return [self.images, self.labels, self.w, self.h, self.c, self.num_pixels, self.u, self.l, self.m, self.means, self.stds]

    def normalize(self,image_in):
        image = np.copy(image_in)
        if self.dataset == 'mnist' or self.dataset == 'fmnist' or self.dataset == 'syn' or self.dataset == 'contagio':
            for i in range(len(image)):
                image[i] = (image[i] - self.means[0])/self.stds[0]
        elif self.dataset=='cifar10':
            tmp = np.zeros(len(image))
            for i in range(len(image)):
                tmp[i] = image[i]
            count = 0
            for i in range(int(len(image)/3)):
                image[i] = tmp[count]
                count = count + 1
                image[i + int(len(image)/3)] = tmp[count]
                count = count + 1
                image[i + 2*int(len(image)/3)] = tmp[count]
                count = count + 1
        return image

    def swap(self, image):
        if self.dataset == 'cifar10':
            return image
        count = 0
        image_ = np.copy(image)
        for i in range(1024):
            image_[i] = image[count]
            count = count + 1
            image_[i + 1024] = image[count]
            count = count + 1
            image_[i + 2048] = image[count]
            count = count + 1
        return image_

    def dswap(self, image):
        if self.dataset != 'cifar10':
            return image
        count = 0
        image_ = np.copy(image)
        for i in range(1024):
            image_[count] = image[i]
            count = count + 1
            image_[count] = image[i + 1024]
            count = count + 1
            image_[count] = image[i + 2048]
            count = count + 1
        return image_
