
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import torch
import os


class MODEL1(torch.nn.Module):
    def __init__(self, num_pixels, c):
        # call constructor from superclass
        super().__init__()
        self.num_pixels = num_pixels
        self.fc1 = torch.nn.Linear(num_pixels, 50)
        self.fc2 = torch.nn.Linear(50, 50)
        self.fc3 = torch.nn.Linear(50, c)

    def forward(self, x):
        x = x.view(-1, self.num_pixels).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MODEL2(torch.nn.Module):
    def __init__(self, num_pixels, c):
        super().__init__()
        self.num_pixels = num_pixels
        self.fc1 = torch.nn.Linear(num_pixels, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100, c)

    def forward(self, x):
        x = x.view(-1, self.num_pixels).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MODEL3(torch.nn.Module):
    def __init__(self,num_pixels, input_channels):
        super().__init__()
        self.num_pixels = num_pixels
        self.c = input_channels
        self.conv1 = torch.nn.Conv2d(1, 16, 4, stride=(2, 2))
        self.conv2 = torch.nn.Conv2d(16, 16, 4, stride=(4, 4))
        self.flatten1 = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(144, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, self.c, int(np.sqrt(self.num_pixels/self.c)), int(np.sqrt(self.num_pixels/self.c)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MODEL4(torch.nn.Module):
    def __init__(self,num_pixels, input_channels):
        super().__init__()
        self.num_pixels = num_pixels
        self.c = input_channels
        self.conv1 = torch.nn.Conv2d(1, 16, 4, stride=(2, 2))
        self.conv2 = torch.nn.Conv2d(16, 32, 4, stride=(2, 2))
        self.conv3 = torch.nn.Conv2d(32, 32, 4, stride=(2, 2))
        self.flatten1 = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, self.c, int(np.sqrt(self.num_pixels/self.c)), int(np.sqrt(self.num_pixels/self.c)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MODEL5(torch.nn.Module):
    def __init__(self,num_pixels, input_channels):
        super().__init__()
        self.num_pixels = num_pixels
        self.c = input_channels
        self.conv1 = torch.nn.Conv2d(input_channels, 16, 4, stride=(4, 4))
        self.conv2 = torch.nn.Conv2d(16, 16, 4, stride=(4, 4))
        self.flatten1 = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, self.c, int(np.sqrt(self.num_pixels/self.c)), int(np.sqrt(self.num_pixels/self.c)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MODEL6(torch.nn.Module):
    def __init__(self,num_pixels, input_channels):
        super().__init__()
        self.num_pixels = num_pixels
        self.c = input_channels
        self.conv1 = torch.nn.Conv2d(input_channels, 16, 4, stride=(2, 2))
        self.conv2 = torch.nn.Conv2d(16, 16, 4, stride=(2, 2))
        self.conv3 = torch.nn.Conv2d(16, 16, 4, stride=(2, 2))
        self.flatten1 = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, self.c, int(np.sqrt(self.num_pixels/self.c)), int(np.sqrt(self.num_pixels/self.c)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MODEL7(torch.nn.Module):
    def __init__(self,num_pixels):
        super().__init__()
        self.num_pixels = num_pixels
        self.fc1 = torch.nn.Linear(num_pixels, 10)
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.num_pixels).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MODEL8(torch.nn.Module):
    def __init__(self,num_pixels):
        super().__init__()
        self.num_pixels = num_pixels
        self.fc1 = torch.nn.Linear(num_pixels, 250)
        self.fc2 = torch.nn.Linear(250, 250)
        self.fc3 = torch.nn.Linear(250, 10)

    def forward(self, x):
        x = x.view(-1, self.num_pixels).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NN_Model:
    def __init__(self, path, dataset, is_conv_network):
        self.dataset = dataset
        self.is_conv_network = is_conv_network
        assert os.path.exists(path), path + " can not be found"
        pth_path = path.replace(".onnx",".pth")
        assert os.path.exists(pth_path), pth_path + " can not be found"
        print(pth_path)
        if "MNIST1" in path or "FMNIST1" in path or "CONTAGIO1" in path:
            self.model = MODEL1(dataset.num_pixels, dataset.classes_num)
            self.model.load_state_dict(torch.load(pth_path))
            self.model.eval()
        elif "FMNIST2" in path:
            self.model = MODEL8(dataset.num_pixels)
            self.model.load_state_dict(torch.load(pth_path))
            self.model.eval()
        elif "MNIST2" in path or "CONTAGIO2" in path:
            self.model = MODEL2(dataset.num_pixels, dataset.classes_num)
            self.model.load_state_dict(torch.load(pth_path))
            self.model.eval()
        elif "MNIST3" in path or "FMNIST3" in path:
            self.model = MODEL3(dataset.num_pixels, dataset.c)
            self.model.load_state_dict(torch.load(pth_path))
            self.model.eval()
        elif "FMNIST4" in path:
            self.model = MODEL4(dataset.num_pixels, dataset.c)
            self.model.load_state_dict(torch.load(pth_path))
            self.model.eval()
        elif "CIFAR101" in path:
            self.model = MODEL5(dataset.num_pixels, dataset.c)
            self.model.load_state_dict(torch.load(pth_path))
            self.model.eval()
        elif "CIFAR102" in path:
            self.model = MODEL6(dataset.num_pixels, dataset.c)
            self.model.load_state_dict(torch.load(pth_path))
            self.model.eval()
        elif "SYN1" in path:
            self.model = MODEL7(dataset.num_pixels)
            self.model.load_state_dict(torch.load(pth_path))
            self.model.eval()

    def compute_gradient_to_input(self, x, label):
        image_for_gradient = np.copy(np.array(x))
        image_for_gradient = self.dataset.swap(image_for_gradient[0])
        data = torch.from_numpy((np.array([image_for_gradient]))).float()
        data.requires_grad = True
        output = self.model.forward(data)
        indices = (-output[0].cpu().detach().numpy()).argsort()
        for ind_ in indices:
            if ind_ != label:
                loss = output[0][label] - output[0][ind_]
                break
        self.model.zero_grad()
        loss.backward()
        data_grad = data.grad.data.cpu().detach().numpy()

        if self.is_conv_network:
            data_grad = data_grad.reshape(-1, self.dataset.num_pixels)
            return self.dataset.dswap(data_grad[0])
        else:
            return data_grad


