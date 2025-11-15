import torch
import json
import os
from itertools import product
from typing import Dict, List, Any, Callable
import numpy as np
from tqdm import tqdm


class HyperparameterTuner:
    def __init__(self, param_grid: Dict[str, List[Any]], save_dir: str = 'tuning_results'):
        self.param_grid = param_grid
        self.save_dir = save_dir
        self.results = []
        os.makedirs(save_dir, exist_ok=True)
        
    def generate_configs(self) -> List[Dict[str, Any]]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        configs = []
        for combination in product(*values):
            config = dict(zip(keys, combination))
            configs.append(config)
        
        return configs
    
    def evaluate_config(self, config: Dict[str, Any], 
                       model_creator: Callable, 
                       trainer_creator: Callable,
                       train_loader, val_loader,
                       epochs: int = 10) -> Dict[str, Any]:
        print(f"\nEvaluating config: {config}")
        
        model = model_creator(config)
        trainer = trainer_creator(model, config)
        
        best_val_snr = -np.inf
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_snr': [],
            'val_snr': []
        }
        
        for epoch in range(epochs):
            train_loss, train_snr = trainer.train_epoch(train_loader)
