""" This code simply runs the QG model for a fixed set of parameters taken from a dictionary
which contains all the parameter values. This will be done later on. """

import math
import torch
import torch.nn as nn
import numpy as np
from grid import TwoGrid
from torchdiffeq import odeint_adjoint, odeint

class QG(nn.Module):
        """This code accepts the stream function and computes the stream function at the next step"""
        def __init__(self, params):
            super(QG, self).__init__()
            # dimensions : 
            self.dt       = params['dt']
            self.t        = torch.tensor(0)
            self.wave_num = 4#torch.nn.Parameter(torch.randn(1,1)*0+1, requires_grad=True)
            self.mu       = params['mu']
            self.nu       = params['nu']
            self.nv       = params['nv']
            self.B        = params['B']
            self.topography= params['topography']
            self.eta = torch.zeros([int(3./2.*params['Nx']),int(3./2.* params['Ny'])], dtype=torch.float64, requires_grad=True).to(params['device'])
            self.device   = params['device']
            self.grid     = TwoGrid(params['device'], Nx=params['Nx'], Ny=params['Ny'], Lx=params['Lx'], Ly=params['Ly'])
            self.d_alias  = TwoGrid(params['device'], Nx=int((3./2.)*params['Nx']), Ny=int((3./2.)*params['Ny']), Lx=params['Lx'], Ly=params['Ly'], dealias=1/3)
            self.Lin      = self.qg_Lin()
            self.state_sequences=[]
            if (self.topography):
                self.eta=torch.sin(3*self.d_alias.y).view(self.d_alias.Ny,1)+ torch.cos(3*self.d_alias.x).view(1,self.d_alias.Nx)            
      
        def Fs(self, bs):
            y = (self.wave_num)*(torch.cos((self.wave_num) * self.grid.y).view(self.grid.Ny, 1) + torch.cos((self.wave_num) * self.grid.x).view(1, self.grid.Nx))
            y = y.unsqueeze(0).repeat(bs,1,1)
            yh = self.to_spectral(y)
            return yh 

        def Fs_t(self,t, bs):
            """We use a time dependent forcing:
            F_(t)=cos(4y)-cos(4x+ \pi sin(1.5 t))"""
            
            y = (torch.cos((self.wave_num) * self.grid.y + math.pi* torch.sin(1.4* t)).view(self.grid.Ny, 1) - 
                 torch.cos((self.wave_num) * self.grid.x + math.pi* torch.sin(1.5* t)).view(1, self.grid.Nx))
            y = y.unsqueeze(0).repeat(bs,1,1)
            yh = self.to_spectral(y)
            return yh

        def qg_Lin(self):
            Lc = -self.mu - self.nu * self.grid.krsq**self.nv + 1j * self.B * self.grid.kr * self.grid.irsq
            Lc[0, 0] = 0
            return Lc.unsqueeze(0)
        
        def to_spectral(self, x):
            return torch.fft.rfftn(x, norm='forward', dim = [-2,-1])
        
        def to_physical(self, x):
            return torch.fft.irfftn(x, norm='forward', dim = [-2,-1])
        
        def reduce(self, x):
            x_r = x.size()
            z = torch.zeros([x_r[0], self.grid.Ny, self.grid.dk], dtype=torch.complex128).to(self.device)
            z[:, :int(self.grid.Ny / 2)               , :self.grid.dk]        = x[:, :int(self.grid.Ny / 2), :self.grid.dk]
            z[:, int(self.grid.Ny / 2):self.grid.Ny, :self.grid.dk] = x[:, x_r[1] - int(self.grid.Ny / 2):x_r[1],:self.grid.dk]
            #x.data = z
            return z#x
        
        def increase(self, x):
            x_r = x.size()
            z = torch.zeros([x_r[0], self.d_alias.Ny, self.d_alias.dk], dtype=torch.complex128).to(self.device)
            z[:,  :int(x_r[1] / 2), :x_r[2]] = x[:, :int(x_r[1] / 2),        :x_r[2]]
            z[:, self.d_alias.Ny - int(x_r[1] / 2):self.d_alias.Ny,         :x_r[2]] = x[:,  int(x_r[1] / 2):x_r[1], :x_r[2]]
            #x.data = z
            return z#x

        def init_randn(self,nb_inits,energy, wavenumbers):
            K = torch.sqrt(self.grid.krsq)
            k = self.grid.kr.repeat(self.grid.Ny, 1)
            qih = torch.randn([nb_inits,self.grid.ky.shape[0],self.grid.kr.shape[-1]], dtype=torch.complex128).to(self.device)
            qih[:,K < wavenumbers[0]] = 0.0
            qih[:,K > wavenumbers[1]] = 0.0
            qih[:,k == 0.0] = 0.0

            E0 = energy
            Ei = 0.5 * (self.grid.int_sq(self.grid.kr * self.grid.irsq * qih) + self.grid.int_sq(self.grid.ky * self.grid.irsq * qih)) / (self.grid.Lx * self.grid.Ly)
            qih *= torch.sqrt(E0 / Ei)
            return self.to_physical(qih)
        
        def get_state_vars(self, q):
            qh = self.to_spectral(q)
            ph = -qh * self.grid.irsq.unsqueeze(0)
            uh = -1j * self.grid.ky.unsqueeze(0) * ph
            vh =  1j * self.grid.kr.unsqueeze(0) * ph

            # Potential vorticity
            q = self.to_physical(qh)
            # Streamfunction
            p = self.to_physical(ph)
            # x-axis velocity
            u = self.to_physical(uh)
            # y-axis velocity
            v = self.to_physical(vh)
            return q, p, u, v
        
        def get_streamfunction(self,q):
            # return stream function in physical
            qh=self.to_spectral(q)
            ph = -qh * self.grid.irsq.unsqueeze(0)
            p = self.to_physical(ph)
            return p

        def get_vorticity(self,ph):
            # return vorticity in spectral
            qh = -ph / self.grid.irsq.unsqueeze(0)
            return qh
        
        def non_linear(self, qh):
            # compute stream function and u and v in spectral domain
            ph = -qh * self.grid.irsq.unsqueeze(0)
            uh = -1j * self.grid.ky.unsqueeze(0) * ph
            vh =  1j * self.grid.kr.unsqueeze(0) * ph

            qhh = self.increase(qh)
            uhh = self.increase(uh)
            vhh = self.increase(vh)

            q = self.to_physical(qhh) + self.eta
            u = self.to_physical(uhh)
            v = self.to_physical(vhh)

            uq = u * q
            vq = v * q

            uqhh = self.to_spectral(uq)
            vqhh = self.to_spectral(vq)

            qh = self.reduce(qhh)
            uqh = self.reduce(uqhh)
            vqh = self.reduce(vqhh)

            S = -1j * self.grid.kr.unsqueeze(0) * uqh - 1j * self.grid.ky.unsqueeze(0) * vqh

            return S, ph
        def forward(self, t, q):
            """The forward function which returns the rhs of the QG model in the 
            physical space of the vorticity"""
            qh=self.to_spectral(q)
            bs = q.shape[0]
            Nlin, ph = self.non_linear(qh)
            Lin_ = self.Lin*qh
            S = Nlin + Lin_ + self.Fs(bs)         #self.Fs_t(t,bs)  # using time varying forcing
            return self.to_physical(S)


class ODE_Block(nn.Module):
    def __init__(self, odefunc: nn.Module, qg_solver_cfg):
        """
        Initializes the QG_Block with the given ODE function and solver configuration.
        :param odefunc: The ODE function to be used in the block."""
        super().__init__()
        self.odefunc = odefunc
        self.rtol = qg_solver_cfg.rtol
        self.atol = qg_solver_cfg.atol
        self.step_size = qg_solver_cfg.step_size
        self.solver = qg_solver_cfg.solver
        self.use_adjoint = qg_solver_cfg.adjoint
        self.num_steps=qg_solver_cfg.num_steps
        self.integration_time = self.step_size * torch.arange(0, self.num_steps, dtype=torch.float32)

    @property
    def ode_method(self):
        return odeint_adjoint if self.use_adjoint else odeint

    def forward(self, x: torch.Tensor, adjoint: bool = True, integration_time=None):
        integration_time = self.integration_time if integration_time is None else integration_time
        integration_time = integration_time.to(x.device)
        ode_method = odeint_adjoint if adjoint else odeint
        out = ode_method(
            self.odefunc, x, integration_time, rtol=self.rtol,
            atol=self.atol, method=self.solver, adjoint_params=()
        )
        return out[-1]

