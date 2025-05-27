# main.py
import hydra
import os
import numpy as np
from omegaconf import DictConfig
from fourDvar import solve_fourDvar
from metrics import compute_metrics
from qg_dynamics import QG, ODE_Block

@hydra.main(version_base=None, config_path="../config")
def main(cfg: DictConfig, xp: DictConfig = None) -> None:
      
    qg = QG(cfg.params)
    # print some useful info

    print(cfg.fourd_var)

    # The above is just an RHS model, we use it with an ODE solver with the help of neural ODE package
    QG_ODE_model = ODE_Block(qg, cfg.qg_solver).to(cfg.params.device)
            
    # Solve multiple 4dvar problems, ( the data loader is inside this big function)
    all_losses, all_X_ = solve_fourDvar(QG_ODE_model, qg, cfg.params , cfg.qg_data , cfg.fourd_var)
    
    # Compute the 4dvar loss for the optimal solution:
    
    # Ensure the output directory exists
    os.makedirs(cfg.output.dir_structure, exist_ok=True)
    np.save(cfg.output.losses_file, all_losses.cpu().numpy())
    np.save(cfg.output.solution_file, all_X_.detach().cpu().numpy())

    # Compute the 4dvar scores and RMSE:
    scores_optimal, scores_true, rmse_optimal = compute_metrics(QG_ODE_model, qg, cfg)
    np.save(cfg.output.rmse_file, rmse_optimal.cpu().numpy())
    np.save(cfg.output.true_state_scores_file, scores_true.cpu().numpy())
    np.save(cfg.output.optimal_state_scores_file, scores_optimal.cpu().numpy())


if __name__ == "__main__":
    main()
