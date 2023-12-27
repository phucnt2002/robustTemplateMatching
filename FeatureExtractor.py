import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, utils
import numpy as np
import copy

class FeatureExtractor():
    def __init__(self, model, use_cuda=True, padding=True):
        self.model = copy.deepcopy(model)
        self.model = self.model.eval()
        self.use_cuda = use_cuda
        self.feature_maps = []

        if self.use_cuda:
            self.model = self.model.cuda()

        self.index = []
        self.f = []
        self.stride = []
        for i, module in enumerate(self.model.children()):
            if isinstance(module, nn.Conv2d):
                self.index.append(i)
                self.f.append(module.kernel_size[0])
                self.stride.append(module.stride[0])
            if isinstance(module, nn.MaxPool2d):
                if padding:
                    module.padding = 1
                self.index.append(i)
                self.f.append(module.kernel_size)
                self.stride.append(module.stride)

        self.rf = np.array(self.calc_rf(self.f, self.stride))

    def save_template_feature_map(self, module, input, output):
        self.template_feature_map = output.detach()

    def save_image_feature_map(self, module, input, output):
        self.image_feature_map = output.detach()

    def calc_rf(self, f, stride):
        """
        This function is responsible for calculating the receptive field of a neuron in a convolutional neural network.
        The receptive field is the portion of the input space that a particular CNN's feature is looking at.
        """
        rf = []
        for i in range(len(f)):
            if i == 0:
                rf.append(3)
            else:
                rf.append(rf[i-1] + (f[i]-1)*self.product(stride[:i]))
        return rf

    def product(self, lis):
        if len(lis) == 0:
            return 0
        else:
            res = 1
            for x in lis:
                res *= x
            return res

    def calc_l_star(self, template, k=3):
        """
        This function is responsible for calculating 'l_star', which could be an optimal value or a target value 
        that a model is trying to achieve. The exact meaning of 'l_star' could vary depending on the context.
        """
        l = np.sum(self.rf <= min(list(template.size()[-2:]))) - 1
        l_star = max(l - k, 1)
        return l_star
    
    def calc_NCC(self, F, M):
        c, h_f, w_f = F.shape[-3:]
        tmp = np.zeros((c, M.shape[-2] - h_f, M.shape[-1] - w_f, h_f, w_f))
        for i in range(M.shape[-2] - h_f):
            for j in range(M.shape[-1] - w_f):
                M_tilde = M[:, :, i:i+h_f, j:j+w_f][:, None, None, :, :]
                eps = 1e-12
                tmp[:, i, j, :, :] = M_tilde / (np.linalg.norm(M_tilde) + eps)
        NCC = np.sum(tmp*F.reshape(F.shape[-3], 1, 1, F.shape[-2], F.shape[-1]), axis=(0, 3, 4))
        return NCC
    def save_image(self, index):

        # Convert the image_feature_map tensor to a numpy array
        image_feature_map_np = self.image_feature_map.squeeze().numpy()

        # If the tensor has more than one channel, take the first channel
        if image_feature_map_np.shape[0] > 1:
            image_feature_map_np = image_feature_map_np[0]

        # Normalize pixel values to the range [0, 255]
        image_feature_map_np = (image_feature_map_np - image_feature_map_np.min()) / (
            image_feature_map_np.max() - image_feature_map_np.min()) * 255.0

        # Convert to an unsigned 8-bit integer array
        image_feature_map_np = image_feature_map_np.astype(np.uint8)

        # Save the resulting array as an image
        cv2.imwrite("./imageFea/image_map"+str(index)+".jpg", image_feature_map_np)
        print("./imageFea/image_map"+str(index)+".jpg")


    def save_template(self, index):
    
        # Convert the image_feature_map tensor to a numpy array
        image_feature_map_np = self.template_feature_map.squeeze().numpy()

        # If the tensor has more than one channel, take the first channel
        if image_feature_map_np.shape[0] > 1:
            image_feature_map_np = image_feature_map_np[0]

        # Normalize pixel values to the range [0, 255]
        image_feature_map_np = (image_feature_map_np - image_feature_map_np.min()) / (
            image_feature_map_np.max() - image_feature_map_np.min()) * 255.0

        # Convert to an unsigned 8-bit integer array
        image_feature_map_np = image_feature_map_np.astype(np.uint8)

        # Save the resulting array as an image
        cv2.imwrite("./templateFea/image_map"+str(index)+".jpg", image_feature_map_np)
        print("./templateFea/image_map"+str(index)+".jpg")


    def __call__(self, template, image, threshold=None, use_cython=True, index=0):
        if self.use_cuda:
            template = template.cuda()
            image = image.cuda()

        self.l_star = self.calc_l_star(template)

        print("save features...")

        # save template feature map (named F in paper)
        template_handle = self.model[self.index[self.l_star]].register_forward_hook(
            self.save_template_feature_map)
        self.model(template)
        template_handle.remove()
        self.save_template(index)

        # save image feature map (named M in papar)
        image_handle = self.model[self.index[self.l_star]].register_forward_hook(
            self.save_image_feature_map)
        self.model(image)
        image_handle.remove()
        self.save_image(index)

        if self.use_cuda:
            self.template_feature_map = self.template_feature_map.cpu()
            self.image_feature_map = self.image_feature_map.cpu()

        print("calc NCC...")

        if use_cython:
            F = self.template_feature_map.numpy()[0].astype(np.float32)
            M = self.image_feature_map.numpy()[0].astype(np.float32)
            import cython_files.cython_calc_NCC as cython_calc_NCC
            self.NCC = np.zeros(
                (M.shape[1] - F.shape[1]) * (M.shape[2] - F.shape[2])).astype(np.float32)
            cython_calc_NCC.c_calc_NCC(M.flatten().astype(np.float32), np.array(M.shape).astype(
                np.int32), F.flatten().astype(np.float32), np.array(F.shape).astype(np.int32), self.NCC)
            self.NCC = self.NCC.reshape(
                [M.shape[1] - F.shape[1], M.shape[2] - F.shape[2]])
        else:
            self.NCC = self.calc_NCC(
                self.template_feature_map.numpy(), self.image_feature_map.numpy())
        max_i = self.NCC.shape[0] - 1
        max_j = self.NCC.shape[1] - 1

        # Lower -> usually lower threshold is more relaxed
        if threshold is None:
            threshold = 0.95 * np.max(self.NCC)
        else:
            threshold = threshold * np.max(self.NCC)

        max_indices = np.array(np.where(self.NCC > threshold)).T
        print("detected boxes: {}".format(len(max_indices)))

        boxes = []
        centers = []
        scores = []

        for max_index in max_indices:
            i_star, j_star = max_index

            # Avoids overflow
            if i_star >= max_i:
                i_star -= 1
            if j_star >= max_j:
                j_star -= 1

            # Broadcasting fails if NCC_part is not (3,4) of shape
            NCC_part = np.zeros([3,4])
            if i_star>=1 and j_star>=2:
                NCC_part = self.NCC[i_star-1:i_star+2, j_star-2:j_star+2]
            elif i_star == 0 and j_star>=2:
                NCC_part[1:3,:] = self.NCC[0:i_star+2, j_star-2:j_star+2]
            elif i_star == 0 and j_star == 1:
                NCC_part[1:3,1:4] = self.NCC[0:i_star+2, j_star-1:j_star+2]
            elif i_star == 0 and j_star == 0:
                NCC_part[1:3,1:4] = self.NCC[0:i_star+2, j_star:j_star+3]
            elif i_star>=1 and j_star == 1:
                NCC_part[:,1:4] = self.NCC[i_star-1:i_star+2, j_star-1:j_star+2]
            elif i_star>=1 and j_star == 0:
                NCC_part[:,2:4] = self.NCC[i_star-1:i_star+2, j_star:j_star+2]

            x_center = (j_star + self.template_feature_map.size()
                        [-1]/2) * image.size()[-1] // self.image_feature_map.size()[-1]
            y_center = (i_star + self.template_feature_map.size()
                        [-2]/2) * image.size()[-2] // self.image_feature_map.size()[-2]

            x1_0 = x_center - template.size()[-1]/2
            x2_0 = x_center + template.size()[-1]/2
            y1_0 = y_center - template.size()[-2]/2
            y2_0 = y_center + template.size()[-2]/2
            stride_product = self.product(self.stride[:self.l_star])
            eps = 1e-12
            x1 = np.sum(
                NCC_part * (x1_0 + np.array([-2, -1, 0, 1]) * stride_product)[None, :]) / (np.sum(NCC_part) + eps)
            x2 = np.sum(
                NCC_part * (x2_0 + np.array([-2, -1, 0, 1]) * stride_product)[None, :]) / (np.sum(NCC_part) + eps)
            y1 = np.sum(
                NCC_part * (y1_0 + np.array([-1, 0, 1]) * stride_product)[:, None]) / (np.sum(NCC_part) + eps)
            y2 = np.sum(
                NCC_part * (y2_0 + np.array([-1, 0, 1]) * stride_product)[:, None]) / (np.sum(NCC_part) + eps)
            x1 = int(round(x1))
            x2 = int(round(x2))
            y1 = int(round(y1))
            y2 = int(round(y2))
            x_center = int(round(x_center))
            y_center = int(round(y_center))

            boxes.append([(x1, y1), (x2, y2)])
            centers.append((x_center, y_center))
            scores.append(np.sum(NCC_part))

        return boxes, centers, scores
