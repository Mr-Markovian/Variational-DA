# dataset.py
import torch
from functools import reduce
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class QGDataset(Dataset):
    def __init__(self, qg_data_cfg, device):
        self.device = device
        ds = xr.open_dataset(qg_data_cfg.path,decode_times=False)
        self.vorticity = ds[qg_data_cfg.state].to_numpy()
        self.stream_f = ds[qg_data_cfg.tgt].to_numpy()
        self.sf_obs = ds[qg_data_cfg.obs].to_numpy()
        self.masks =ds[qg_data_cfg.ob_masks].to_numpy()
        self.sparsity=qg_data_cfg.sparsity
        self.num_timesteps = self.vorticity.shape[0]
        self.chunk_size = 10  # Non-overlapping chunks of 10 time steps

    def __len__(self): 
        return self.num_timesteps // self.chunk_size 

    def __getitem__(self, idx): 
        start_idx = idx * self.chunk_size 
        end_idx = start_idx + self.chunk_size 
        vorticity_chunk = torch.from_numpy(self.vorticity[start_idx:end_idx]).to(self.device) 
        stream_f_chunk = torch.from_numpy(self.stream_f[start_idx:end_idx]).to(self.device)
        sf_obs_chunk = torch.from_numpy(self.sf_obs[start_idx:end_idx]).to(self.device) 
        #-torch.normal(mean=0.0, std=fourDvar_cfg.noise_std, size=X_sf_torch[Masks_torch].shape).to(params.device)
        if self.sparsity=='default':
            mask_chunk = torch.from_numpy(self.masks[start_idx:end_idx]).bool().to(self.device)
        else:
            s_factor=self.sparsity
            # Initialize an empty list to collect results
            or_results = []

            # Loop through the tensor in chunks of 5 time points
            for i in range(self.chunk_size):
                # Take 5 consecutive time points
                masks_ = torch.from_numpy(self.masks[start_idx:start_idx+s_factor]).bool().to(self.device)
                
                # Apply element-wise OR across the 5 time points
                or_result = reduce(torch.logical_or, masks_)
                or_results.append(or_result)
            
            # Stack the results into a new tensor of shape [10, 128, 128]
            mask_chunk = torch.stack(or_results)
            
            
        return vorticity_chunk, stream_f_chunk, sf_obs_chunk, mask_chunk
