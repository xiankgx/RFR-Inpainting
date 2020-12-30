import glob
import os
import random

import albumentations as A
import cv2
import numpy as np
import scipy
import torch
import torchvision.transforms.functional as F
# from scipy.misc import imread
from imageio import imread
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, mask_mode, target_size, augment=True, training=True, mask_reverse=False):
        super(Dataset, self).__init__()

        self.training = training
        self.augment = augment
        self.data = self.load_list(image_path)
        self.mask_data = self.load_list(mask_path)
        self.target_size = target_size
        self.mask_type = mask_mode
        self.mask_reverse = mask_reverse

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_item(self, index):
        img = imread(self.data[index])
        if self.training:
            img = self.resize(img)
        else:
            img = self.resize(img, True, True, True)

        # load mask
        mask = self.load_mask(img, index)

        # augment data
        if self.training:
            # random horizontal flip
            if self.augment and np.random.binomial(1, 0.5) > 0:
                img = img[:, ::-1, ...]
            if self.augment and np.random.binomial(1, 0.5) > 0:
                mask = mask[:, ::-1, ...]

        # handle 2-dimensional grayscale image
        if img.ndim == 2:
            img = np.stack([img, ] * 3, axis=-1)

        return self.to_tensor(img), self.to_tensor(mask)

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]

        # external mask, random order
        if self.mask_type == 0:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            # threshold due to interpolation
            mask = (mask > 0).astype(np.uint8)
            mask = self.resize(mask, False)
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255

        # generate random mask
        if self.mask_type == 1:
            mask = 1 - \
                generate_stroke_mask([self.target_size, self.target_size])
            mask = (mask > 0).astype(np.uint8) * 255
            mask = self.resize(mask, False)
            return mask

        # external mask, fixed order
        if self.mask_type == 2:
            mask_index = index
            mask = imread(self.mask_data[mask_index])
            # threshold due to interpolation
            mask = (mask > 0).astype(np.uint8)
            mask = self.resize(mask, False)
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255

    def resize(self, img, aspect_ratio_kept=True, fixed_size=False, centerCrop=False):
        if aspect_ratio_kept:
            imgh, imgw = img.shape[0:2]
            side = np.minimum(imgh, imgw)

            if fixed_size:
                if centerCrop:
                    # center crop

                    j = (imgh - side) // 2
                    i = (imgw - side) // 2
                    img = img[j:j + side, i:i + side, ...]

                else:
                    # random crop

                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side,
                              w_start:w_start + side, ...]

            else:
                if side <= self.target_size:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side,
                              w_start:w_start + side, ...]

                else:
                    side = random.randrange(self.target_size, side)
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = random.randrange(0, j)
                    w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side,
                              w_start:w_start + side, ...]

        # img = scipy.misc.imresize(img, [self.target_size, self.target_size])
        img = np.array(Image.fromarray(img)
                       .resize(size=(self.target_size, self.target_size)))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def load_list(self, path):
        if isinstance(path, str):
            if path[-3:] == "txt":
                line = open(path, "r")
                lines = line.readlines()
                file_names = []
                for line in lines:
                    file_names.append(
                        "../../Dataset/Places2/train/data_256"+line.split(" ")[0])
                return file_names

            if os.path.isdir(path):
                path = list(glob.glob(path + '/**/*.jpg', recursive=True)) + \
                    list(glob.glob(path + '/**/*.png', recursive=True))
                path.sort()
                return path

            if os.path.isfile(path):
                try:
                    return np.genfromtxt(path, dtype=np.str, encoding='utf-8')
                except:
                    return [path]
        return []


class DatasetV2(torch.utils.data.Dataset):

    def __init__(self, image_path, mask_path,
                 mask_mode, target_size=256,
                 augment=True,
                 training=True,
                 mask_reverse=False):
        super(DatasetV2, self).__init__()

        self.image_path = image_path
        self.mask_path = mask_path
        self.target_size = target_size if isinstance(target_size, (list, tuple)) \
            else (target_size, target_size)
        self.mask_type = mask_mode
        self.augment = augment
        self.training = training
        self.mask_reverse = mask_reverse

        print(f"target_size : {self.target_size}")
        print(f"training    : {self.training}")
        print(f"mask_mode   : {self.mask_type}")
        print(f"mask_reverse: {self.mask_reverse}")

        self._list_files()
        self._get_transform()

    def _list_files(self):
        img_exts = [".jpg", ".jpeg", ".jfif", ".png"]

        images = glob.glob(self.image_path + "/**/*",
                           recursive=True)
        self.images = list(filter(lambda p: os.path.splitext(p)[-1] in img_exts,
                                  images))
        print(f"Found {len(self.images)} images in {self.image_path}.")

        if self.mask_type == 1:
            self.masks = []
        else:
            masks = glob.glob(self.mask_path + "/**/*",
                              recursive=True)
            self.masks = list(filter(lambda p: os.path.splitext(p)[-1] in img_exts,
                                     masks))
            print(f"Found {len(self.masks)} images in {self.mask_path}.")

    def _get_transform(self):
        self.transform = self._get_train_transform() if self.training \
            else self._get_test_transform()

    def _get_train_transform(self):
        print("Using train transform")
        return A.Compose([
            A.HorizontalFlip(p=0.5),

            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                          hue=0,
                          p=0.5),

            A.OneOf([
                A.Compose([
                    A.SmallestMaxSize(max_size=self.target_size[0],
                                      interpolation=cv2.INTER_LINEAR,
                                      p=1.0),
                    A.RandomCrop(height=self.target_size[0], width=self.target_size[0],
                                 p=1.0)
                ], p=1.0),

                A.RandomResizedCrop(height=self.target_size[0], width=self.target_size[1],
                                    scale=(0.25, 1.0), ratio=(3./4., 4./3.),
                                    interpolation=cv2.INTER_LINEAR,
                                    p=1.0),

                A.Resize(height=self.target_size[0], width=self.target_size[1],
                         interpolation=cv2.INTER_LINEAR,
                         p=1.0)
            ], p=1.0)
        ])

    def _get_test_transform(self):
        print("Using test transform")
        return A.RandomResizedCrop(height=self.target_size[0], width=self.target_size[1],
                                   scale=(0.75, 1.0), ratio=(3./4., 4./3.),
                                   interpolation=cv2.INTER_LINEAR,
                                   p=1.0)
        # return A.Compose([
        #     A.SmallestMaxSize(max_size=self.target_size[0],
        #                       interpolation=cv2.INTER_LINEAR,
        #                       p=1.0),
        #     A.RandomCrop(height=self.target_size[0], width=self.target_size[0],
        #                  p=1.0)
        # ], p=1.0)
        # return A.Resize(height=self.target_size[0], width=self.target_size[1],
        #                 interpolation=cv2.INTER_LINEAR,
        #                 p=1.0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_p = self.images[idx]
        img = np.array(Image.open(img_p).convert("RGB"))

        # External mask, random order
        if self.mask_type == 0:
            mask_p = self.masks[np.random.randint(len(self.masks))]
            mask = np.array(Image.open(mask_p).convert("RGB"))

        # External mask, fixed order
        elif self.mask_type == 2:
            mask_p = self.masks[idx]
            mask = np.array(Image.open(mask_p).convert("RGB"))

        # Generate random mask
        elif self.mask_type == 1:
            # [0, 1] -> [0, 255]
            mask = (generate_stroke_mask(self.target_size) * 255) \
                .astype(np.uint8)
            mask = np.array(Image.fromarray(mask).convert("RGB"))

        assert img.ndim == 3
        assert img.dtype == np.uint8
        assert mask.ndim == 3
        assert mask.dtype == np.uint8

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, img.shape[:2][::-1])

        # Binarization to handle interpolation
        mask = (mask > 0).astype(np.uint8) * 255

        if self.mask_reverse:
            mask = 255 - mask

        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        assert img.shape[:2] == self.target_size
        assert mask.shape[:2] == self.target_size
        assert all([el in [0, 255] for el in np.unique(mask)])

        # return img, mask
        return F.to_tensor(img), F.to_tensor(mask)


def generate_stroke_mask(im_size, max_parts=15, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(1, max_parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength,
                                        maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.concatenate([mask, mask, mask], axis=2)
    return mask


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds = DatasetV2("/home/gx/datasets/coco/test2017",
                   "/home/gx/datasets/mask",
                   target_size=384,
                   mask_mode=0,
                   training=False,
                   mask_reverse=True)

    img, mask = ds[np.random.randint(len(ds))]

    alpha = mask/255.0
    masked = (img * alpha).astype(np.uint8)
    print(f"img - {img.dtype}, {img.min()}, {img.max()}")
    print(f"mask - {mask.dtype}, {mask.min()}, {mask.max()}")
    print(f"alpha - {alpha.dtype}, {alpha.min()}, {alpha.max()}")
    print(f"masked - {masked.dtype}, {masked.min()}, {masked.max()}")
    print(f"np.unique(mask): {np.unique(mask)}")
    plt.imshow(np.concatenate([img, mask, masked], axis=1))
    plt.show()
