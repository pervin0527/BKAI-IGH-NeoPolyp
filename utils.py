import yaml
import numpy as np

color_dict = {0: (0,   0, 0),
              1: (0, 255,   0),
              2: (0, 0,   255)}


def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))
    for k in color_dict.keys():
        output[mask==k] = color_dict[k]
    return np.uint8(output)


def save_config_to_yaml(config, save_dir):
    with open(f"{save_dir}/params.yaml", 'w') as file:
        yaml.dump(config, file)


def decode_mask(pred_mask):
        decoded_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        decoded_mask[pred_mask == 0] = [0, 0, 0]
        decoded_mask[pred_mask == 1] = [0, 255, 0] ## Green
        decoded_mask[pred_mask == 2] = [255, 0, 0] ## Red
        
        return decoded_mask


def decode_image(image):
    # image = (1 + image) * 127.5
    # image = image * 255
    # image = image.astype(np.uint8)

    image = np.transpose(image, (1, 2, 0))
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    image = (image * std) + mean
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    return image
