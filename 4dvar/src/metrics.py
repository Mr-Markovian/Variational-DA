import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import QGDataset

def compute_rmse(true, pred):
    """
    Compute RMSE between true and predicted values.
    Args:
        true (torch.Tensor or np.ndarray): Ground truth values with shape [batch, days, H, W].
        pred (torch.Tensor or np.ndarray): Predicted values with shape [batch, days, H, W].
    
    Returns:
        float: RMSE across all spatial and temporal dimensions.
    """
    true = torch.tensor(true) if isinstance(true, np.ndarray) else true
    pred = torch.tensor(pred) if isinstance(pred, np.ndarray) else pred
    se_sum = torch.sum((true - pred) ** 2)
    return torch.sqrt(se_sum)

def fourDvar_score(QG_ODE_model, qg, cfg, state, X_sf_torch, Masks_torch, batch_idx):
    """For computing the cost over a da-window, we want to compute the cost"""
    days = cfg.fourd_var.days
    alpha_obs=cfg.fourd_var.alpha_obs
    alpha_dyn=cfg.fourd_var.alpha_dyn
    X_torch=state.squeeze(0)
    X_pred = QG_ODE_model(X_torch[0:days - 1])
    X_sf_torch=X_sf_torch.squeeze(0)

    Masks_torch=Masks_torch.squeeze(0)
    sf_pred = qg.get_streamfunction(X_pred)   # Stream function from past iteration (day 2 to day 10)
    sf_torch = qg.get_streamfunction(X_torch) # Stream function from day 2 to day 10

    # Compute dynamical loss
    loss_dyn = torch.sum((sf_torch[1:] - sf_pred) ** 2)

    # observation noise
    torch.manual_seed(cfg.fourd_var.seed1+batch_idx)
    
    # Compute observational loss with Gaussian noise
    noise = torch.normal(mean=0.0, std=cfg.fourd_var.noise_std, 
                          size=sf_torch[Masks_torch].shape).to(cfg.params.device)
    
    loss_obs = torch.sum((sf_torch[Masks_torch] - X_sf_torch[Masks_torch] - noise) ** 2)

    # Total loss
    loss = alpha_obs * loss_obs + alpha_dyn * loss_dyn
    return loss

def compute_metrics(QG_ODE_model, qg, cfg):
        
    optimal_states=torch.from_numpy(np.load(cfg.output.solution_file)).view(cfg.fourd_var.max_batch,10,128,128).to(cfg.params.device)  
    
    """Multiple 4dvar problems for different initial conditions"""
    dataset = QGDataset(cfg.qg_data, cfg.params.device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    optimal_states_scores = []   # Score of the optimal 4dvar states.
    true_states_scores = []      # Score of the true states as the I.C.
    optimal_states_rmse = []     # RMSE of the optimal 4dvar states.
    for batch_id,batch in enumerate(dataloader):
        if batch_id >= cfg.fourd_var.max_batch:  # Stop after processing max_batches
            break    
        X_torch, X_sf_torch, YObs_torch, Masks_torch = batch
        score1 = fourDvar_score(QG_ODE_model, qg, cfg, X_torch, X_sf_torch, Masks_torch, batch_id)
        score2= fourDvar_score(QG_ODE_model, qg, cfg, optimal_states[batch_id], X_sf_torch, Masks_torch, batch_id)
        rmse = compute_rmse(optimal_states[batch_id], X_torch)
        optimal_states_scores.append(score2)
        true_states_scores.append(score1)
        optimal_states_rmse.append(rmse)

    return torch.stack(optimal_states_scores), torch.stack(true_states_scores), torch.stack(optimal_states_rmse)
