import os
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from typing import Dict, List, Tuple
from preprocessing import ECGImagePreprocessor, get_training_augmentation, get_validation_augmentation


class ECGDataset(Dataset):
    def __init__(self, metadata_path: str, data_dir: str, 
                 transform=None, is_training: bool = True,
                 img_size: Tuple[int, int] = (512, 1024)):
        self.metadata = pd.read_csv(metadata_path)
        self.data_dir = data_dir
        self.is_training = is_training
        self.img_size = img_size
        
        if transform is None:
            if is_training:
                self.transform = get_training_augmentation(img_size)
            else:
                self.transform = get_validation_augmentation(img_size)
        else:
            self.transform = transform
        
        self.preprocessor = ECGImagePreprocessor(target_size=img_size)
        
        image_variants = ['-0001', '-0003', '-0004', '-0005', '-0006', 
                         '-0009', '-0010', '-0011', '-0012']
        self.valid_samples = []
        
        for idx, row in self.metadata.iterrows():
            ecg_id = row['id']
            for variant in image_variants:
                img_path = os.path.join(data_dir, str(ecg_id), f"{ecg_id}{variant}.png")
                csv_path = os.path.join(data_dir, str(ecg_id), f"{ecg_id}.csv")
                
                if os.path.exists(img_path) and os.path.exists(csv_path):
                    self.valid_samples.append({
                        'id': ecg_id,
                        'img_path': img_path,
                        'csv_path': csv_path,
                        'fs': row['fs'],
                        'sig_len': row['sig_len']
                    })
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        image = cv2.imread(sample['img_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = self.preprocessor.preprocess(image, for_training=self.is_training)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        signal_df = pd.read_csv(sample['csv_path'])
        
        fs = sample['fs']
        lead_ii_len = int(10 * fs)
        other_leads_len = int(2.5 * fs)
        
        signals = {}
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        for lead_name in lead_names:
            if lead_name in signal_df.columns:
                signal_data = signal_df[lead_name].values.astype(np.float32)
                
                if lead_name == 'II':
                    target_len = lead_ii_len
                else:
                    target_len = other_leads_len
                
                if len(signal_data) > target_len:
                    signal_data = signal_data[:target_len]
                elif len(signal_data) < target_len:
                    signal_data = np.pad(signal_data, (0, target_len - len(signal_data)), 
                                        mode='edge')
                
                signals[lead_name] = torch.from_numpy(signal_data).float()
        
        return {
            'image': image,
            'signals': signals,
            'fs': fs,
            'id': sample['id'],
            'lead_ii_len': lead_ii_len,
            'other_leads_len': other_leads_len
        }


class ECGTestDataset(Dataset):
    def __init__(self, metadata_path: str, data_dir: str,
                 img_size: Tuple[int, int] = (512, 1024)):
        self.metadata = pd.read_csv(metadata_path)
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = get_validation_augmentation(img_size)
        self.preprocessor = ECGImagePreprocessor(target_size=img_size)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        ecg_id = row['id']
        
        img_path = os.path.join(self.data_dir, f"{ecg_id}.png")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = self.transform(image=image)
        image = augmented['image']
        
        fs = row['fs']
        lead_ii_len = int(10 * fs)
        other_leads_len = int(2.5 * fs)
        
        return {
            'image': image,
            'fs': fs,
            'id': ecg_id,
            'lead_ii_len': lead_ii_len,
            'other_leads_len': other_leads_len
        }


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    
    all_signals = {}
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    for lead_name in lead_names:
        if lead_name in batch[0]['signals']:
            signals = [item['signals'][lead_name] for item in batch]
            max_len = max(s.shape[0] for s in signals)
            
            padded_signals = []
            for s in signals:
                if s.shape[0] < max_len:
                    padded = torch.nn.functional.pad(s, (0, max_len - s.shape[0]))
                    padded_signals.append(padded)
                else:
                    padded_signals.append(s)
            
            all_signals[lead_name] = torch.stack(padded_signals)
    
    return {
        'images': images,
        'signals': all_signals,
        'fs': [item['fs'] for item in batch],
        'ids': [item['id'] for item in batch],
        'lead_ii_len': [item['lead_ii_len'] for item in batch],
        'other_leads_len': [item['other_leads_len'] for item in batch]
    }
