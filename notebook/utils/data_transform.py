import numpy as np
import torch


class PlanetTransform():
    def __init__(self,spatial_encoder=True, image_size=32):
        self.spatial_encoder = spatial_encoder
        self.image_size=image_size

    def transform(self,image_stack, mask):
        if self.spatial_encoder == False:  # average over field mask: T, D = image_stack.shape
            image_stack = image_stack[:, :, mask > 0].mean(2)
            mask = -1  # mask is meaningless now but needs to be constant size for batching
        else:  # crop/pad image to fixed size + augmentations: T, D, H, W = image_stack.shape
            if image_stack.shape[2] >= self.image_size and image_stack.shape[3] >= self.image_size:
                image_stack, mask = random_crop(image_stack, mask, self.image_size)

            image_stack, mask = crop_or_pad_to_size(image_stack, mask, self.image_size)

            # rotations
            rot = np.random.choice([0, 1, 2, 3])
            image_stack = np.rot90(image_stack, rot, [2, 3])
            mask = np.rot90(mask, rot)

            # flip up down
            if np.random.rand() < 0.5:
                image_stack = np.flipud(image_stack)
                mask = np.flipud(mask)

            # flip left right
            if np.random.rand() < 0.5:
                image_stack = np.fliplr(image_stack)
                mask = np.fliplr(mask)

        image_stack = image_stack * 1e-4

        # z-normalize
        image_stack -= 0.1014 + np.random.normal(scale=0.01)
        image_stack /= 0.1171 + np.random.normal(scale=0.01)

        return torch.from_numpy(np.ascontiguousarray(image_stack)).float(), torch.from_numpy(np.ascontiguousarray(mask))

#class Sentinel1Transform():
#    raise NotImplementedError

#class Sentinel2Transform():
#    raise NotImplementedError

def random_crop(image_stack, mask, image_size):
    H, W = image_stack.shape[2:]

    # skip random crop is image smaller than crop size
    if H - image_size // 2 <= image_size:
        return image_stack, mask
    if W - image_size // 2 <= image_size:
        return image_stack, mask

    h = np.random.randint(image_size, H - image_size // 2)
    w = np.random.randint(image_size, W - image_size // 2)

    image_stack = image_stack[:, :, h - int(np.floor(image_size // 2)):int(np.ceil(h + image_size // 2)),
                  w - int(np.floor(image_size // 2)):int(np.ceil(w + image_size // 2))]
    mask = mask[h - int(np.floor(image_size // 2)):int(np.ceil(h + image_size // 2)),
           w - int(np.floor(image_size // 2)):int(np.ceil(w + image_size // 2))]

    return image_stack, mask

def crop_or_pad_to_size(image_stack,  mask, image_size):
    T, D, H, W = image_stack.shape
    hpad = image_size - H
    wpad = image_size - W

    # local flooring and ceiling helper functions to save some space
    def f(x):
        return int(np.floor(x))
    def c(x):
        return int(np.ceil(x))

    # crop image if image_size < H,W
    if hpad < 0:
        image_stack = image_stack[:, :, -c(hpad) // 2:f(hpad) // 2, :]
        mask = mask[-c(hpad) // 2:f(hpad) // 2, :]
    if wpad < 0:
        image_stack = image_stack[:, :, :, -c(wpad) // 2:f(wpad) // 2]
        mask = mask[:, -c(wpad) // 2:f(wpad) // 2]
    # pad image if image_size > H, W
    if hpad > 0:
        padding = (f(hpad / 2), c(hpad / 2))
        image_stack = np.pad(image_stack, ((0, 0), (0, 0), padding, (0, 0)))
        mask = np.pad(mask, (padding, (0, 0)))
    if wpad > 0:
        padding = (f(wpad / 2), c(wpad / 2))
        image_stack = np.pad(image_stack, ((0, 0), (0, 0), (0, 0), padding))
        mask = np.pad(mask, ((0, 0), padding))
    return image_stack, mask
