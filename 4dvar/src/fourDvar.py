# fourDvar_solver.py
import torch
from torch.utils.data import DataLoader
from dataset import QGDataset
from utils import extract_states_from_batch, initialize_optimizer

def optimize(QG_ODE_model, qg, params, qg_data_cfg, batch_idx, batch, fourDvar_cfg):
    """perform the 4dvar optimization for a specific number of iterations and then compute the rmse of the obtained solutions"""

    X_torch, X_sf_torch, YObs_torch, Masks_torch= extract_states_from_batch(batch, fourDvar_cfg, params) 
    # Re-initialize the field to have a gradient tensor..
    X_torch = torch.autograd.Variable(X_torch, requires_grad=True)
    X_torch.retain_grad()
    
    # Initialize the optimizer, only three are possible: SGD, ADAM, LBFGS
    optimizer= initialize_optimizer(X_torch, fourDvar_cfg)
    optimizer.zero_grad()

    # fix the seed to the first one...
    torch.manual_seed(fourDvar_cfg.seed1+batch_idx)
    noise = torch.normal(mean=0.0, std=fourDvar_cfg.noise_std, size=X_sf_torch[Masks_torch].shape).to(params.device)
    losses = torch.zeros(fourDvar_cfg.NIter, 3)
    for iter in range(fourDvar_cfg.NIter):
        with torch.set_grad_enabled(True):
            if fourDvar_cfg.optimizer == 'LBFGS':
            # Define the closure for LBFGS
                def closure():
                    optimizer.zero_grad()  # Reset gradients
                    loss, loss_dyn, loss_obs = compute_loss(QG_ODE_model, 
                                                            qg, 
                                                            X_torch, 
                                                            X_sf_torch, 
                                                            Masks_torch, 
                                                            noise,
                                                            fourDvar_cfg)
                    print(f"iter {iter}: loss {loss.item():.3f}, dyn_loss {loss_dyn.item():.3f}, obs_loss {loss_obs.item():.3f}")
                    loss.backward()  # Compute gradients
                    return loss

        # Perform an optimization step
                optimizer.step(closure)
            else:
                loss, loss_dyn, loss_obs =compute_loss(QG_ODE_model, 
                                                       qg, 
                                                       X_torch, 
                                                       X_sf_torch, 
                                                       Masks_torch, 
                                                       noise, 
                                                       fourDvar_cfg)

                losses[iter, :] = torch.tensor([loss.item(), loss_dyn.item(), loss_obs.item()])
                print(f"iter {iter}: loss {loss.item():.3f}, dyn_loss {loss_dyn.item():.3f}, obs_loss {loss_obs.item():.3f}")
                loss.backward() 
                optimizer.step()
                optimizer.zero_grad()
        
    return losses, X_torch 


def compute_loss(QG_ODE_model, qg, X_torch, X_sf_torch, Masks_torch, noise, fourDvar_cfg):
    """Compute the total loss, including dynamical and observational losses."""
    X_pred = QG_ODE_model(X_torch[0:fourDvar_cfg.days - 1])
    sf_pred = qg.get_streamfunction(X_pred)
    sf_torch = qg.get_streamfunction(X_torch)

    loss_dyn = torch.sum((sf_torch[1:] - sf_pred) ** 2)
    loss_obs = torch.sum((sf_torch[Masks_torch] - X_sf_torch[Masks_torch] - noise) ** 2)

    total_loss = fourDvar_cfg.alpha_obs * loss_obs + fourDvar_cfg.alpha_dyn * loss_dyn
    return total_loss, loss_dyn, loss_obs


def solve_fourDvar(QG_ODE_model, qg, params, qg_data_cfg, fourDvar_cfg):
    """Multiple 4dvar problems for different batch of initial conditions, optimizing each batch separately.
    Returns the losses and the optimized solutions."""
    dataset = QGDataset(qg_data_cfg, params.device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_losses = []
    all_X_torch=[]
    for i,batch in enumerate(dataloader):
        if i >= fourDvar_cfg.max_batch:  # Stop after processing max_batches
            break    
        losses, X_solution = optimize(QG_ODE_model, 
                                      qg, 
                                      params, 
                                      qg_data_cfg, 
                                      i, 
                                      batch, 
                                      fourDvar_cfg)
        all_losses.append(losses)
        all_X_torch.append(X_solution)

    return torch.cat(all_losses), torch.cat(all_X_torch)

