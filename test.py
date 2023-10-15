import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import yaml
import torch
import numpy as np
import pandas as pd
import albumentations as A

from glob import glob
from mmseg.models import build_segmentor
from albumentations.pytorch import ToTensorV2

from model_confing import get_model_cfg
from utils import mask_to_rgb, color_dict


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


def evaluate(config, model):
    model.eval()

    test_dir = config["data_dir"] + "/" + config["test"]
    save_dir = '/'.join(config["weight_dir"].split('/')[:-2])

    print(save_dir)
    if not os.path.isdir(f"{save_dir}/test_result"):
        os.makedirs(f"{save_dir}/test_result")

    for file in sorted(glob(f"{test_dir}/*")):
        file_name = file.split("/")[-1].split('.')[0]
        
        ori_img = cv2.imread(file)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori_w, ori_h = ori_img.shape[0], ori_img.shape[1]
        img = cv2.resize(ori_img, (config["img_size"], config["img_size"]))

        transformed = batch_transform(image=img)
        input_img = transformed["image"]
        input_img = input_img.unsqueeze(0).to(device)

        with torch.no_grad():
            output_mask = model.forward_dummy(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
            mask = cv2.resize(output_mask, (ori_h, ori_w))
            mask = np.argmax(mask, axis=-1)
            
            mask_rgb = mask_to_rgb(mask, color_dict)
            cv2.imwrite(f"{save_dir}/test_result/{file_name}.jpeg", mask_rgb)

    return f"{save_dir}/test_result"


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    batch_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
                                 ToTensorV2()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmentor(get_model_cfg(config)).to(device)
    model.load_state_dict(torch.load(config["weight_dir"]), strict=False)
    result_dir = evaluate(config, model)

    MASK_DIR_PATH = result_dir
    dir = MASK_DIR_PATH
    res = mask2string(dir)
    df = pd.DataFrame(columns=['Id', 'Expected'])
    df['Id'] = res['ids']
    df['Expected'] = res['strings']

    df.to_csv(r'output.csv', index=False)