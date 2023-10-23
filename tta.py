import os
import warnings
import cv2
import yaml
import torch
import numpy as np
import pandas as pd
import albumentations as A

from tqdm import tqdm
from glob import glob
from mmseg.models import build_segmentor
from albumentations.pytorch import ToTensorV2

from model_confing import get_model_cfg
from utils import mask_to_rgb, color_dict

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 225] = 255
    pixels[pixels <= 225] = 0
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)


def rle2mask(mask_rle, shape=(3,3)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        
        img = cv2.imread(path)[:, :, ::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)

    r = {'ids': ids, 'strings': strings,}
    return r


def apply_tta(model, image, batch_transform, num_augments=10):
    outputs_list = []

    # Original image prediction
    transformed = batch_transform(image=image)
    input_img = transformed["image"]
    input_img = input_img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model.forward_dummy(input_img).squeeze(0).cpu().numpy()
    outputs_list.append(outputs)
    
    for _ in range(num_augments):
        for transform in tta_transforms:
            aug_image = transform(image=image)['image']
            transformed = batch_transform(image=aug_image)
            input_img = transformed["image"]
            input_img = input_img.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model.forward_dummy(input_img).squeeze(0).cpu().numpy()

            if isinstance(transform, A.HorizontalFlip):
                output = np.flip(output, axis=2)
            elif isinstance(transform, A.VerticalFlip):
                output = np.flip(output, axis=1)
            elif isinstance(transform, A.Rotate):
                output = cv2.warpAffine(output, cv2.getRotationMatrix2D((output.shape[1] / 2, output.shape[0] / 2), -45, 1), (output.shape[1], output.shape[0]))

            outputs_list.append(output)

    averaged_outputs = np.mean(outputs_list, axis=0)
    mask = np.argmax(averaged_outputs, axis=0)

    return mask


def evaluate(config, model):
    model.eval()
    test_dir = config["data_dir"] + "/" + config["test"]
    save_dir = '/'.join(config["weight_dir"].split('/')[:-2])
    date = save_dir.split('/')[-1]

    print(save_dir)
    if not os.path.isdir(f"{save_dir}/test_result"):
        os.makedirs(f"{save_dir}/test_result")

    files = sorted(glob(f"{test_dir}/*"))
    for idx in tqdm(range(len(files))):
        file = files[idx]
        file_name = file.split("/")[-1].split('.')[0]
        ori_img = cv2.imread(file)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori_w, ori_h = ori_img.shape[0], ori_img.shape[1]
        img = cv2.resize(ori_img, (config["img_size"], config["img_size"]))

        output_mask = apply_tta(model, img, batch_transform, num_augments=config["num_augment"])
        mask_rgb = mask_to_rgb(output_mask, color_dict)
        mask_rgb = cv2.resize(mask_rgb, (ori_h, ori_w))
        
        cv2.imwrite(f"{save_dir}/test_result/{file_name}.jpeg", mask_rgb)

    MASK_DIR_PATH = f"{save_dir}/test_result"
    dir = MASK_DIR_PATH
    res = mask2string(dir)
    df = pd.DataFrame(columns=['Id', 'Expected'])
    df['Id'] = res['ids']
    df['Expected'] = res['strings']
    df.to_csv(f'output_{date}_tta.csv', index=False)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    batch_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                 ToTensorV2()])
    
    tta_transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.4),
        A.RandomGamma(gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(p=0.4),
        A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
        A.OneOf([A.Blur(), A.GaussianBlur(), A.GlassBlur(), A.MotionBlur(), A.GaussNoise(), A.Sharpen(), A.MedianBlur(), A.MultiplicativeNoise()]),
        A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.15, brightness_coeff=1.5, p=0.09),
        A.RandomShadow(p=0.1)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmentor(get_model_cfg(config)).to(device)
    model.load_state_dict(torch.load(config["weight_dir"]), strict=False)
    evaluate(config, model)
