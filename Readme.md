## Variational Data Assimilation

This repositiory contains a new code for weak-constraint 4dvar or simply weak-4dvar 
data assimilation that is implemented in pytorch and hydra.

The weak-4dvar cost function is:


The paramters for any numerical experiments is loaded as a config.yaml file and hydra is used to initialise experiments. The code can run on CPU or GPU. 

The current implementation has Quasi-geostrophic model on a 128 X 128 grid. 
The weak-4dvar problem is solved for 10 day assimilation window. The  


The observations are Nadir Satelite altimetry tracks. 
The nature of these observations are realistic, and they are very sparse. 
The default implementation performs optimization of the loss function of the weak-4dvar using SGD based algorithm, although this can be easily changed to a different optimizer that is written or available in pytorch. 
We have explored different initial conditions at the moment:

1. True i.c. - just to check for consistency and stability.
2. Blurred i.c. - to test stability and convergence to the true solution.
3. Gaussian random field- random ic that with no prior choice on the state.
4. Coherent-shifted field- a i.c. to mimic psition based errors.

The folder structure is a as follows:



To experiment with new models, two things need to be worked upon-
1. The dataset and the dataloader within it for the observations of the system.
2. The 'your_dynamical_system.py' which needs to be implemented using pytorch nn module.
