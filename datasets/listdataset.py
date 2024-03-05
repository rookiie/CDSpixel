import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data



class ListDataset(data.Dataset):
    def __init__(self, root, dataset, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=None, datatype=None, scale=True, use_MI=True):

        self.root = root
        self.dataset = dataset
        self.img_path_list =path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader
        self.datatype = datatype
        self.scale = scale
        self.use_MI = use_MI

    def __getitem__(self, index):
        img_path = self.img_path_list[index][:-1]
        # We do not consider other datsets in this work
        assert self.dataset == 'bsd500'
        assert (self.transform is not None) and (self.target_transform is not None)

        inputs, label = self.loader(img_path, img_path.replace('_img.jpg', '_label.png'))
        image_hsv = cv2.cvtColor(inputs, cv2.COLOR_RGB2LAB)

        if self.scale:
            inputs, label, image_hsv = self.generate_scale_label(inputs, label, image_hsv)
            
        if self.co_transform is not None:
            inputs, label = self.co_transform([inputs.astype(np.float32), image_hsv.astype(np.float32)], label)

        if self.transform is not None:
            image = self.transform(inputs[0])
        
        if self.target_transform is not None:
            #print(label.shape)
            label = self.target_transform(label[:,:,:1])
            image_hsv = self.target_transform(inputs[1])
        if self.use_MI:
            return image, label, image_hsv, self._get_sobel(image)
        else:
            return image, label, image_hsv

    def generate_scale_label(self, image, label, edge):
        f_scale = 0.7 + np.random.randint(0, 8) / 10.0
        image = cv2.resize(image, None,fx=f_scale, fy=f_scale, interpolation = cv2.INTER_CUBIC)
        label = cv2.resize(label, None,fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        edge = cv2.resize(edge, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label, edge
        #return image, label
        
    def generate_scale_label_v2(self, image, label, edge):
        f_scale = 0.3 + np.random.randint(0, 12) / 10.0
        image_ = cv2.resize(image, None,fx=f_scale, fy=f_scale, interpolation = cv2.INTER_CUBIC)
        label_ = cv2.resize(label, None,fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        edge_ = cv2.resize(edge, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_CUBIC)
        if f_scale >= 0.7:    
            return image_, label_, edge_
        else:
            image_bg = cv2.resize(image, None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
            label_bg = cv2.resize(label, None,fx=0.7, fy=0.7, interpolation = cv2.INTER_NEAREST)
            edge_bg = cv2.resize(edge, None, fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
            H, W, _ = image_bg.shape
            h, w, _ = image_.shape
            x = np.random.randint(0, W - w)
            y = np.random.randint(0, H - h)
            image_bg[y:y+h, x:x+w] = image_
            label_bg[y:y+h, x:x+w] = label_
            edge_bg[y:y+h, x:x+w] = edge_
            return image_bg, label_bg, edge_bg
             
        #return image, label

    def __len__(self):
        return len(self.img_path_list)

    def _get_sobel(self, x):
        kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
        kernels = np.asarray(kernels) #2,3,3
        kernels = np.expand_dims(kernels, 1).repeat(3, axis=1)
        kernels = torch.from_numpy(kernels.astype(np.float32))
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        sob = F.conv2d(x, kernels, stride=1, padding=1)
        return torch.sum(sob, dim=1, keepdim=True).repeat(1,3,1,1).squeeze()

        