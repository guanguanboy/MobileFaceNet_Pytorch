import numpy as np
import scipy.misc
import os
import torch

class CASIA_Face(object):
    def __init__(self, root):
        self.root = root

        img_txt_dir = os.path.join(root, 'CASIA-WebFace-112X96.txt')
        image_list = []
        label_list = []
        with open(img_txt_dir) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_dir, label_name = info.split(' ')
            image_list.append(os.path.join(root, 'CASIA-WebFace-112X96', image_dir))
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = scipy.misc.imread(img_path) #read a image from a file as an array将图片读取出来为array类型，即numpy类型

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        flip = np.random.choice(2)*2-1 #np.random.choice(2) 产生一个ndarray的数组[0,1]
        img = img[:, ::flip, :]
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float() ## numpy 转为 pytorch格式

        return img, target

    def __len__(self):
        return len(self.image_list)



if __name__ == '__main__':
    data_dir = 'E:\\CodeFromGitHub\\MobileFaceNet_Pytorch\\testdata\\CASIA_Test'
    dataset = CASIA_Face(root=data_dir)
    #trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)
    print(len(dataset))
    #for data in trainloader:
        #print(data[0].shape)
