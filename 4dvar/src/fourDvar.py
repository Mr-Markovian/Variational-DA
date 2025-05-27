# fourDvar_solver.py
import torch
from torch.utils.data import DataLoader
from dataset import QGDataset
from misc import apply_gaussian_smoothing, generate_correlated_fields, warp_field

def optimize(QG_ODE_model, qg, params, qg_data_cfg, batch_idx, batch, fourDvar_cfg):
    """perform the 4dvar optimization for a specific number of iterations and then compute the rmse of the obtained solutions"""
    days = fourDvar_cfg.days
    NIter=fourDvar_cfg.NIter
    alpha_obs=fourDvar_cfg.alpha_obs
    alpha_dyn=fourDvar_cfg.alpha_dyn
    delta=fourDvar_cfg.delta
    
    if fourDvar_cfg.ic_type=='true field':
        X_torch, X_sf_torch, YObs_torch, Masks_torch = batch
    
    if fourDvar_cfg.ic_type=='blurred_ic':
        X_torch, X_sf_torch, YObs_torch, Masks_torch = batch
        X_torch= apply_gaussian_smoothing(X_torch.squeeze(0),fourDvar_cfg.kernel_size,fourDvar_cfg.sigma).unsqueeze(0) 

    if fourDvar_cfg.ic_type=='coherent-space-shifted':
        X_true, X_sf_torch, YObs_torch, Masks_torch = batch
        displacement_x = generate_correlated_fields(params.Nx, fourDvar_cfg.l_scale, fourDvar_cfg.t_scale, fourDvar_cfg.sigma_field, device=params.device,seed=fourDvar_cfg.seed1)
        displacement_y = generate_correlated_fields(params.Nx, fourDvar_cfg.l_scale, fourDvar_cfg.t_scale, fourDvar_cfg.sigma_field, device=params.device,seed=fourDvar_cfg.seed2)
        # Perform warping with mean_x and mean_y
        X_torch = warp_field(X_true.squeeze(0).unsqueeze(1), displacement_x+fourDvar_cfg.mean_x, displacement_y+fourDvar_cfg.mean_y).squeeze(1)

    X_torch=X_torch.squeeze(0)
    YObs_torch=YObs_torch.squeeze(0)
    Masks_torch=Masks_torch.squeeze(0)
    X_sf_torch=X_sf_torch.squeeze(0)
     
    # Re-initialize the field to have a gradient tensor..
    X_torch = torch.autograd.Variable(X_torch, requires_grad=True)
    X_torch.retain_grad()
    losses = torch.zeros(NIter, 3)
    
    if fourDvar_cfg.optimizer=='SGD':
        optimizer = torch.optim.SGD([X_torch], lr=delta, momentum=0.5,nesterov=True)
        
    if fourDvar_cfg.optimizer=='ADAM':
        optimizer = torch.optim.Adam([X_torch], lr=delta, eps=1e-3, amsgrad=True)

    if fourDvar_cfg.optimizer=='LBFGS':
        optimizer = torch.optim.LBFGS([X_torch], lr=1.0,history_size=20,line_search_fn='strong_wolfe')
    
    optimizer.zero_grad()

    # fix the seed to the first one...
    torch.manual_seed(fourDvar_cfg.seed1+batch_idx)
    noise = torch.normal(mean=0.0, std=fourDvar_cfg.noise_std, size=X_sf_torch[Masks_torch].shape).to(params.device)

    for iter in range(NIter):
        with torch.set_grad_enabled(True):
            if fourDvar_cfg.optimizer == 'LBFGS':
            # Define the closure for LBFGS
                def closure():
                    optimizer.zero_grad()  # Reset gradients
                    X_pred = QG_ODE_model(X_torch[0:days - 1])
                    sf_pred = qg.get_streamfunction(X_pred)   # Stream function from past iteration (day 2 to day 10)
                    sf_torch = qg.get_streamfunction(X_torch) # Stream function from day 2 to day 10

                    # Compute dynamical loss
                    loss_dyn = torch.sum((sf_torch[1:] - sf_pred) ** 2)

                    # Compute observational loss with Gaussian noise
                    noise_sgd= torch.normal(mean=0.0, std=fourDvar_cfg.noise_std, size=X_sf_torch[Masks_torch].shape).to(params.device)
                    loss_obs = torch.sum((sf_torch[Masks_torch] - X_sf_torch[Masks_torch] - noise) ** 2)
        
                    # Total loss
                    loss = alpha_obs * loss_obs + alpha_dyn * loss_dyn
                    #losses[iter, :] = torch.tensor([loss.item(), loss_dyn.item(), loss_obs.item()])
                    #if iter % 10 == 0:
                    print(f"iter {iter}: loss {loss.item():.3f}, dyn_loss {loss_dyn.item():.3f}, obs_loss {loss_obs.item():.3f}")
                    loss.backward()  # Compute gradients
                    return loss

        # Perform an optimization step
                optimizer.step(closure)
            else:
    
                X_pred = QG_ODE_model(X_torch[0:days-1])
                sf_pred = qg.get_streamfunction(X_pred)   # Stream function from past iteration from day 2 to day 10
                sf_torch = qg.get_streamfunction(X_torch) # Stream function from day 2 to day 10
    
                loss_dyn = torch.sum((sf_torch[1:] - sf_pred) ** 2)
                # Question: Should the dynamical loss be in the space of vorticity or the space of stream function.
                #loss_obs = torch.sum((sf_torch - YObs_torch) ** 2 * Masks_torch)
                noise_sgd= torch.normal(mean=0.0, std=fourDvar_cfg.noise_std, size=X_sf_torch[Masks_torch].shape).to(params.device)

                loss_obs=torch.sum((sf_torch[Masks_torch]- X_sf_torch[Masks_torch] - noise)**2)

                loss = alpha_obs * loss_obs + alpha_dyn * loss_dyn
    
                losses[iter, :] = torch.tensor([loss.item(), loss_dyn.item(), loss_obs.item()])
                #if iter % 10 == 0:
                print(f"iter {iter}: loss {loss.item():.3f}, dyn_loss {loss_dyn.item():.3f}, obs_loss {loss_obs.item():.3f}")
    
                loss.backward() 
                optimizer.step()
                optimizer.zero_grad()
                # if fourDvar_cfg.gradient_smoothing=='True':
                #     grad_blur=apply_gaussian_smoothing(X_torch.grad.data,kernel_size=3, sigma=1.0)
                #     X_torch = X_torch - delta * grad_blur
                # else:
                #     X_torch = X_torch - delta * X_torch.grad.data 
                #X_torch = torch.autograd.Variable(X_torch, requires_grad=True)
        
    return losses, X_torch 

def solve_fourDvar(QG_ODE_model, qg, params, qg_data_cfg, fourDvar_cfg):
    """Multiple 4dvar problems for different initial conditions"""
    dataset = QGDataset(qg_data_cfg, params.device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_losses = []
    all_X_torch=[]
    for i,batch in enumerate(dataloader):
        if i >= fourDvar_cfg.max_batch:  # Stop after processing max_batches
            break    
        losses, X_solution = optimize(QG_ODE_model, qg, params, qg_data_cfg, i, batch, fourDvar_cfg)
        all_losses.append(losses)
        all_X_torch.append(X_solution)

    return torch.cat(all_losses), torch.cat(all_X_torch)

# To compute the metric here and store the same.
