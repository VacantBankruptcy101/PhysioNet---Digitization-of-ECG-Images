import cv2
import numpy as np
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ECGImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (512, 1024)):
        self.target_size = target_size
        
    def preprocess(self, image: np.ndarray, for_training: bool = False) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        processed = self.denoise(image)
        processed = self.correct_rotation(processed)
        processed = self.normalize_contrast(processed)
        processed = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        processed = self.normalize_image(processed)
        
        return processed
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised
    
    def correct_rotation(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines[:20]:
                rho, theta = line[0]
                angle = np.degrees(theta) - 90
                if -45 < angle < 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, float(median_angle), 1.0)
                    image = cv2.warpAffine(image, M, (w, h), 
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def normalize_contrast(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        return image


def get_training_augmentation(img_size: Tuple[int, int] = (512, 1024)):
    return A.Compose([
        A.RandomRotate90(p=0.1),
        A.Flip(p=0.1),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
        ], p=0.3),
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_validation_augmentation(img_size: Tuple[int, int] = (512, 1024)):
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
