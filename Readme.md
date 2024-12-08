## Variational Data Assimilation

This repositiory contains a new code for weak-constraint 4dvar or simply weak-4dvar 
data assimilation that is implemented in pytorch and hydra.

Problem statement: 
Given the sequence of observations Observations sequence $Y^i=\left(y^i_0,y^i_1,...y^i_n\right)$ on $\left(\Omega_i\right) \in \Omega$  find the optimal trajectory  $X^i=\left(x^i_0,x^i_1,...x^i_n\right)$ over that minimizes the following cost function. 
The weak-4dvar cost function is:

$$\mathcal{J}(X_{0:T})=\sum_{i=1} \|\|X_i - \mathcal{M}(X_{i-1})\|\|^2+ \|\|y_i-\mathcal{H}(x_i)\|\|^2$$

The dynamical systems $M$ takes the system state $X_k$ to $X_{k+1}$.
The two terms correspond to different to fitting to the obsrvations accounting for the observation error, the second term corresponds to the dynamical loss, i.e. which brings the state sequence close to being a dynamical trajectory of the system.   

The paramters for any numerical experiments is loaded as a config.yaml file and hydra is used to initialise experiments. The code can run on CPU or GPU. 

The current implementation has Quasi-geostrophic model on a 128 X 128 grid. The vorticity is the dynamical variable and the observations are in the stream function space.
The observations masks are Nadir Satelite altimetry tracks. The nature of these observations are realistic, and they are very sparse. 

[vorticity_and_masks](vort_sf_obs_128.pdf)


The weak-4dvar problem is solved for 10 day assimilation window. The  
The default implementation performs optimization of the loss function of the weak-4dvar using SGD based algorithm, although this can be easily changed to a different optimizer that is written or available in pytorch. 
We have explored different initial conditions at the moment:

1. True i.c. - just to check for consistency and stability.
2. Blurred i.c. - to test stability and convergence to the true solution.
3. Gaussian random field- random ic that with no prior choice on the state.
4. Coherent-shifted field- a i.c. to mimic psition based errors.

The folder structure is a as follows:

To experiment with new models, two things need to be worked upon-
1. The dataset and the dataloader within it for the observations of the system.
2. Create a pytorch implementation of 'your_dynamical_system.py' using pytorch's nn module.
3. The neural ode package which will be able to handle derivative computation via the adjoint implementation. 
