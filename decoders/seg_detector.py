from collections import OrderedDict

import torch
import torch.nn as nn
BatchNorm2d = nn.BatchNorm2d

import cv2
import numpy as np
from scipy.spatial import KDTree

from .merge import merge_cat
from .CDAN import PSA_dif

class SegDetector(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector, self).__init__()
        self.first_train_epoch = 0
        self.psa = PSA_dif()
        self.merge = merge_cat()
        self.min_are = 1.5
        self.k = k
        self.serial = serial
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                    inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, prompt_points, epoch, gt=None, masks=None, training=False):
        pred_points = []

        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1) # B, 256, 160, 160

        loss_out = self.psa(fuse)

        with torch.no_grad():

            for i in range(fuse.shape[0]):
                if self.training:
                    if epoch < self.first_train_epoch:
                        pred_points = [] 
                        break 
                if self.first_train_epoch == -1:
                    pred_points = [] 
                    break    
                else:
                    bacth_pred_points = []
                    feature_map_np = loss_out[i].squeeze().detach().cpu().numpy()
                    threshold = 0.1
                    _, binary_map = cv2.threshold(feature_map_np, threshold, 1, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    num_contours = min(len(contours), 1000)
                    contours = contours[:num_contours]
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x+w/2.0
                        center_y = y+h/2.0
                        are = w*h
                        if are > self.min_are:
                            bacth_pred_points.append([center_x/16.0,center_y/16.0, w/16.0, h/16.0])
                    del contours 
                    pred_points.append(bacth_pred_points)

            gaussian_map = self.generate_gaussian_map(prompt_points, pred_points, fuse.shape[2]//4, fuse.shape[3]//4, epoch)

        fuse = self.merge(fuse, gaussian_map)
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
    
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary, gauss_map=loss_out)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def generate_gaussian_map(self, prompt_points_tensor, pred_points=None, Height=None, Width=None, epoch=0):
        prompt_points = prompt_points_tensor.cpu().numpy()
        prompt_points[:, :, 0] = (prompt_points[:, :, 0]*Width)
        prompt_points[:, :, 1] = (prompt_points[:, :, 1]*Height)
        Y, X = np.meshgrid(np.arange(Height), np.arange(Width), indexing='ij')
        gaussian_maps = []
        if self.training and epoch < self.first_train_epoch:
            for i in range(len(prompt_points)):
                valid_mask = np.any(prompt_points[i] != 0, axis=1)
                filtered_points = prompt_points[i][valid_mask]
                if len(filtered_points) != 0:
                    sigma_x = np.array([2])
                    sigma_y = np.array([1])
                    gaussian = np.exp(-((Y - filtered_points[:, 1, np.newaxis, np.newaxis]) ** 2 / (2 *  sigma_y ** 2)
                                + (X - filtered_points[:, 0, np.newaxis, np.newaxis]) ** 2 / (2 * sigma_x ** 2)))
                    gaussian_maps.append(np.sum(gaussian, axis=0)[np.newaxis, :])
                else:
                    gaussian_maps.append(np.zeros((Height, Width))[np.newaxis, :])
                continue
        
        else:
            if len(pred_points)==0:
                for i in range(len(prompt_points)):
                    valid_mask = np.any(prompt_points[i] != 0, axis=1)
                    filtered_points = prompt_points[i][valid_mask]
                    if len(filtered_points) != 0:
                        sigma_x = np.array([2])
                        sigma_y = np.array([1])
                        gaussian = np.exp(-((Y - filtered_points[:, 1, np.newaxis, np.newaxis]) ** 2 / (2 *  sigma_y ** 2)
                                    + (X - filtered_points[:, 0, np.newaxis, np.newaxis]) ** 2 / (2 * sigma_x ** 2)))
                        gaussian_maps.append(np.sum(gaussian, axis=0)[np.newaxis, :])
                    else:
                        gaussian_maps.append(np.zeros((Height, Width))[np.newaxis, :])
                    continue
            else:
                for i in range(len(prompt_points)):
                    pred_points_i = np.array(pred_points[i])
                    valid_mask = np.any(prompt_points[i] != 0, axis=1)
                    filtered_points = prompt_points[i][valid_mask]
                    if len(filtered_points) == 0 or len(pred_points_i) == 0:
                        if len(filtered_points) != 0:
                            sigma_x = np.array([2])
                            sigma_y = np.array([1])
                            gaussian = np.exp(-((Y - filtered_points[:, 1, np.newaxis, np.newaxis]) ** 2 / (2 *  sigma_y ** 2)
                                        + (X - filtered_points[:, 0, np.newaxis, np.newaxis]) ** 2 / (2 * sigma_x ** 2)))
                            gaussian_maps.append(np.sum(gaussian, axis=0)[np.newaxis, :])
                        else:
                            gaussian_maps.append(np.zeros((Height, Width))[np.newaxis, :])
                        continue

                    tree = KDTree(pred_points_i[:,:2])
                    _, indices = tree.query(filtered_points)
                    nearest_points = np.array(pred_points[i])[indices].astype(np.float16)

                    nearest_points[:, 2] = nearest_points[:, 2]/3.0
                    nearest_points[:, 3] = nearest_points[:, 3]/3.0

                    sigma_x = np.array([2])
                    sigma_y = np.array([1])

                    gaussian = (np.exp(-((Y - nearest_points[:, 1, np.newaxis, np.newaxis]) ** 2 / (2 * nearest_points[:, 3, np.newaxis, np.newaxis] ** 2+10e-8)
                                + (X - nearest_points[:, 0, np.newaxis, np.newaxis]) ** 2 / (2 * nearest_points[:, 2, np.newaxis, np.newaxis] ** 2+10e-8))))\
                                + (np.exp(-((Y - filtered_points[:, 1, np.newaxis, np.newaxis]) ** 2 / (2 *  sigma_y ** 2)
                                        + (X - filtered_points[:, 0, np.newaxis, np.newaxis]) ** 2 / (2 * sigma_x ** 2))))
                    
                    gaussian_maps.append(np.sum(gaussian, axis=0)[np.newaxis, :])
                    del pred_points_i, tree, indices, nearest_points

        gaussian_map = np.concatenate(gaussian_maps, axis=0)
        gaussian_map = torch.as_tensor(gaussian_map).to(prompt_points_tensor.device)

        return gaussian_map.unsqueeze(1) 
    
if __name__ == "__main__":

    prompts = torch.Tensor([[[0.1,0.1],[0.24,0.25],[0.66,0.77]]])
    pred = torch.Tensor([[[1,1,10,10],[25,24,10,10],[67,78,10,10]]])
    seg = SegDetector()
    map = seg.generate_gaussian_map(prompt_points_tensor=prompts, pred_points=pred, Height=40, Width=40)
    print(map)
