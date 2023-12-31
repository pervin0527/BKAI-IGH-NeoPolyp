import cv2
import random
import albumentations as A

from glob import glob
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

from data.data_preprocess import load_img_mask, encode_mask, train_img_mask_transform, mosaic_augmentation, spatially_exclusive_pasting, get_bg_image, mixup

class BKAIDataset(Dataset):
    def __init__(self, config):
        self.data_dir = config["data_dir"]
        self.img_size = config["img_size"]
        self.spatial_alpha = config["spatial_alpha"]
        self.mixup_alpha = config["mixup_alpha"]

        self.image_dir = f"{self.data_dir}/train"
        self.mask_dir = f"{self.data_dir}/train_gt"

        self.image_files = sorted(glob(f"{self.image_dir}/*.jpeg"))
        self.mask_files = sorted(glob(f"{self.mask_dir}/*.jpeg"))
        
        self.total_files = list(zip(self.image_files, self.mask_files))

        self.train_transform = A.Compose([A.HorizontalFlip(p=0.5),
                                          A.VerticalFlip(p=0.5),
                                          A.RandomGamma (gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
                                          A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                                          A.OneOf([A.Blur(), A.GaussianBlur(), A.GlassBlur(), A.MotionBlur(), A.GaussNoise(), A.Sharpen(), A.MedianBlur(), A.MultiplicativeNoise()]),
                                          A.CoarseDropout(p=0.2, max_height=35, max_width=35, fill_value=0, mask_fill_value=0),
                                          A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.15, brightness_coeff=1.5, p=0.09),
                                          A.RandomShadow(p=0.1),
                                          A.ShiftScaleRotate(p=0.45, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.15, scale_limit=0.15, rotate_limit=180),
                                          A.RandomCrop(self.img_size, self.img_size)])
        
        self.batch_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
                                          ToTensorV2()])
        
        self.background_dir = config["background_dir"]
        # self.background_files = sorted(glob(f"{self.background_dir}/*"))
        self.background_files = []
        for folder in ["train", "val", "test"]:
            files = sorted(glob(f"{self.background_dir}/{folder}/1_ulcerative_colitis/*"))
            self.background_files.extend(files)


    def __len__(self):
        return len(self.total_files)
    

    def __getitem__(self, index):
        prob = random.random()

        if prob <= 0.3:
            image_path, mask_path = self.total_files[index]
            image, mask = load_img_mask(image_path, mask_path, size=self.img_size)

            augment_image, augment_mask = train_img_mask_transform(self.train_transform, image, mask)
            
            if random.random() > 0.7:
                background_image = get_bg_image(self.background_files)
                augment_image = mixup(augment_image, background_image, alpha=random.uniform(self.mixup_alpha, self.mixup_alpha + 0.3))


        elif 0.3 < prob <= 0.6:
            piecies = []
            while len(piecies) < 4:
                i = random.randint(0, len(self.total_files)-1)
                image_path, mask_path = self.total_files[i]
                image, mask = load_img_mask(image_path, mask_path, size=self.img_size)

                if random.random() > 0.5:
                    piece_image, piece_mask = train_img_mask_transform(self.train_transform, image, mask)
                else:
                    piece_image, piece_mask = spatially_exclusive_pasting(image, mask, alpha=random.uniform(self.spatial_alpha, self.spatial_alpha + 0.2))

                if random.random() > 0.7:
                    background_image = get_bg_image(self.background_files)
                    piece_image = mixup(piece_image, background_image, alpha=random.uniform(self.mixup_alpha, self.mixup_alpha + 0.3))

                piecies.append([piece_image, piece_mask])

            augment_image, augment_mask = mosaic_augmentation(piecies, size=self.img_size)


        elif 0.6 < prob <= 1:
            image_path, mask_path = self.total_files[index]
            image, mask = load_img_mask(image_path, mask_path, size=self.img_size)
            augment_image, augment_mask = spatially_exclusive_pasting(image, mask, alpha=random.uniform(self.spatial_alpha, self.spatial_alpha + 0.2))

            if random.random() > 0.7:
                background_image = get_bg_image(self.background_files)
                augment_image = mixup(augment_image, background_image, alpha=random.uniform(self.mixup_alpha, self.mixup_alpha + 0.3))

        encoded_mask = encode_mask(augment_mask)
        batch_image, batch_mask = train_img_mask_transform(self.batch_transform, augment_image, encoded_mask)

        return batch_image, batch_mask


if __name__ == "__main__":
    import yaml
    from torch.utils.data import DataLoader

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_dataset = BKAIDataset(config)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for image, mask in train_dataloader:
        print(image.shape, mask.shape)

        break