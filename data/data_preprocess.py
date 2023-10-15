import cv2
import copy
import random
import numpy as np
import albumentations as A

def load_img_mask(image_path, mask_path, size=256, only_img=False):
    if only_img:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size))

        return image
    
    else:
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path) 
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (size, size))
        mask = cv2.resize(mask, (size, size))

        return image, mask


def encode_mask(mask):  
    hsv_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])

    lower_mask = cv2.inRange(hsv_mask, lower1, upper1)
    upper_mask = cv2.inRange(hsv_mask, lower2, upper2)
    red_mask = lower_mask + upper_mask;
    red_mask[red_mask != 0] = 2

    # boundary RED color range values; Hue (36 - 70)
    green_mask = cv2.inRange(hsv_mask, (36, 25, 25), (70, 255,255))
    green_mask[green_mask != 0] = 1
    full_mask = cv2.bitwise_or(red_mask, green_mask)
    full_mask = full_mask.astype(np.uint8)

    return full_mask


def decode_mask(pred_mask):
        decoded_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        decoded_mask[pred_mask == 0] = [0, 0, 0]
        decoded_mask[pred_mask == 1] = [0, 255, 0] ## Green
        decoded_mask[pred_mask == 2] = [255, 0, 0] ## Red
        
        return decoded_mask


def train_img_mask_transform(transform, image, mask): 
    x, y = copy.deepcopy(image), copy.deepcopy(mask)
    transformed = transform(image=x, mask=y)
    transformed_image, transformed_mask = transformed["image"], transformed["mask"]

    return transformed_image, transformed_mask


def mosaic_augmentation(piecies, size):
    h, w = size, size
    mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
    mosaic_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
    cx, cy = w // 2, h // 2
    
    indices = [0, 1, 2, 3]
    random.shuffle(indices)
    for i, index in enumerate(indices):
        piece_image, piece_mask = piecies[index][0], piecies[index][1]
        
        if i == 0:
            mosaic_img[:cy, :cx] = cv2.resize(piece_image, (cx, cy))
            mosaic_mask[:cy, :cx] = cv2.resize(piece_mask, (cx, cy))
        elif i == 1:
            mosaic_img[:cy, cx:] = cv2.resize(piece_image, (w-cx, cy))
            mosaic_mask[:cy, cx:] = cv2.resize(piece_mask, (w-cx, cy))
        elif i == 2:
            mosaic_img[cy:, :cx] = cv2.resize(piece_image, (cx, h-cy))
            mosaic_mask[cy:, :cx] = cv2.resize(piece_mask, (cx, h-cy))
        elif i == 3:
            mosaic_img[cy:, cx:] = cv2.resize(piece_image, (w-cx, h-cy))
            mosaic_mask[cy:, cx:] = cv2.resize(piece_mask, (w-cx, h-cy))
    
    return mosaic_img, mosaic_mask


def rand_bbox(size, lam):
    W = size[1]
    H = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_augmentation(image1, mask1, image2, mask2):
    i1, i2, m1, m2 = copy.deepcopy(image1), copy.deepcopy(image2), copy.deepcopy(mask1), copy.deepcopy(mask2)
    lam = np.clip(np.random.beta(1.0, 1.0), 0.2, 0.8)
    bbx1, bby1, bbx2, bby2 = rand_bbox(i1.shape, lam)

    i1[bbx1:bbx2, bby1:bby2] = i2[bbx1:bbx2, bby1:bby2]
    m1[bbx1:bbx2, bby1:bby2] = m2[bbx1:bbx2, bby1:bby2]

    return i1, m1


def spatially_exclusive_pasting(image, mask, alpha=0.7, iterations=10):
    target_image, target_mask = copy.deepcopy(image), copy.deepcopy(mask)
    L_gray = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)

    hs, ws = np.where(L_gray == 1)
    if not hs.any() or not ws.any():
        return target_mask

    he, we = hs.max(), ws.max()
    hs, ws = hs.min(), ws.min()
    
    Lf_gray = L_gray[hs:he, ws:we]
    If = target_image[hs:he, ws:we]
    Lf_color = target_mask[hs:he, ws:we]
    
    M = np.random.rand(*target_image.shape[:2])
    M[L_gray == 1] = float('inf')
    
    height, width = he - hs, we - ws

    for _ in range(iterations):
        px, py = np.unravel_index(M.argmin(), M.shape)        
        candidate_area = (slice(px, px + height), slice(py, py + width))
        
        if candidate_area[0].stop > target_image.shape[0] or candidate_area[1].stop > target_image.shape[1]:
            M[px, py] = float('inf')
            continue
        
        if np.any(L_gray[candidate_area] & Lf_gray):
            M[candidate_area] = float('inf')
            continue
        
        target_image[candidate_area] = alpha * target_image[candidate_area] + (1 - alpha) * If
        target_mask[candidate_area] = alpha * target_mask[candidate_area] + (1 - alpha) * Lf_color
        L_gray[candidate_area] = cv2.cvtColor(target_mask[candidate_area], cv2.COLOR_BGR2GRAY)
        
        M[candidate_area] = float('inf')
        
        kernel = np.ones((3, 3), np.float32) / 9
        M = cv2.filter2D(M, -1, kernel)

    return target_image, target_mask


def mixup(foreground_image, background_image, alpha):
    image1, image2 = copy.deepcopy(foreground_image), copy.deepcopy(background_image)

    height, width = image1.shape[:2]
    background_transform = A.Compose([A.RandomRotate90(p=0.5),
                                      A.VerticalFlip(p=0.3),
                                      A.HorizontalFlip(p=0.3),
                                      A.RandomBrightnessContrast(p=0.6),
                                      A.Resize(height=height, width=width, p=1)])
    
    transformed = background_transform(image=image2)
    transformed_image = transformed["image"]

    mixed_image = cv2.addWeighted(image1, alpha, transformed_image, 1 - alpha, 0)   
    
    return mixed_image