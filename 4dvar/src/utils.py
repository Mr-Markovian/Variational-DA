import torch
import torch.nn.functional as F

def initialize_optimizer(X_torch, fourDvar_cfg):
    """Initialize the optimizer based on the configuration."""
    if fourDvar_cfg.optimizer == 'SGD':
        return torch.optim.SGD([X_torch], lr=fourDvar_cfg.delta, momentum=0.5, nesterov=True)
    if fourDvar_cfg.optimizer == 'ADAM':
        return torch.optim.Adam([X_torch], lr=fourDvar_cfg.delta, eps=1e-3, amsgrad=True)
    if fourDvar_cfg.optimizer == 'LBFGS':
        return torch.optim.LBFGS([X_torch], lr=1.0, history_size=20, line_search_fn='strong_wolfe')
    raise ValueError(f"Unknown optimizer type: {fourDvar_cfg.optimizer}")


def extract_states_from_batch(batch, fourDvar_cfg, params):
    """Extract and preprocess states from the batch based on the initial condition type."""
    X_torch, X_sf_torch, YObs_torch, Masks_torch = batch
    if fourDvar_cfg.ic_type == 'true field':
        return X_torch.squeeze(0), X_sf_torch.squeeze(0), YObs_torch.squeeze(0), Masks_torch.squeeze(0)

    if fourDvar_cfg.ic_type == 'blurred_ic':
        X_torch = apply_gaussian_smoothing(X_torch.squeeze(0), fourDvar_cfg.kernel_size, fourDvar_cfg.blur_std).unsqueeze(0)
        return X_torch.squeeze(0), X_sf_torch.squeeze(0), YObs_torch.squeeze(0), Masks_torch.squeeze(0)

    if fourDvar_cfg.ic_type == 'coherent-space-shifted':
        displacement_x = generate_correlated_fields(params.Nx, fourDvar_cfg.l_scale, fourDvar_cfg.t_scale, 
                                                    fourDvar_cfg.sigma_field, device=params.device, seed=fourDvar_cfg.seed1)
        displacement_y = generate_correlated_fields(params.Nx, fourDvar_cfg.l_scale, fourDvar_cfg.t_scale, 
                                                    fourDvar_cfg.sigma_field, device=params.device, seed=fourDvar_cfg.seed2)
        X_torch = warp_field(X_torch.squeeze(0).unsqueeze(1), displacement_x + fourDvar_cfg.mean_x, 
                             displacement_y + fourDvar_cfg.mean_y).squeeze(1)
        return X_torch.squeeze(0), X_sf_torch.squeeze(0), YObs_torch.squeeze(0), Masks_torch.squeeze(0)
    
    if fourDvar_cfg.ic_type == 'blurred-coherent-space-shifted':
        displacement_x = generate_correlated_fields(params.Nx, fourDvar_cfg.l_scale, fourDvar_cfg.t_scale, 
                                                    fourDvar_cfg.sigma_field, device=params.device, seed=fourDvar_cfg.seed1)
        displacement_y = generate_correlated_fields(params.Nx, fourDvar_cfg.l_scale, fourDvar_cfg.t_scale, 
                                                    fourDvar_cfg.sigma_field, device=params.device, seed=fourDvar_cfg.seed2)
        X_torch = warp_field(X_torch.squeeze(0).unsqueeze(1), displacement_x + fourDvar_cfg.mean_x, 
                             displacement_y + fourDvar_cfg.mean_y).squeeze(1)
        X_torch = apply_gaussian_smoothing(X_torch.squeeze(0), fourDvar_cfg.kernel_size, fourDvar_cfg.blur_std).unsqueeze(0)
        return X_torch.squeeze(0), X_sf_torch.squeeze(0), YObs_torch.squeeze(0), Masks_torch.squeeze(0)



    raise ValueError(f"Unknown ic_type: {fourDvar_cfg.ic_type}")




def generate_correlated_fields(N, L, T_corr, sigma, num_fields=10, device='cpu',seed=41):
    """
    Generate a series of 2D fields with both spatial and temporal correlations.

    Parameters:
        N (int): Grid size (assumed to be square, NxN).
        L (float): Spatial correlation length.
        T_corr (float): Temporal correlation length.
        sigma (float): Standard deviation of the field.
        num_fields (int): Number of fields to generate (default is 10).
        device (str): Device to run the computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Generated fields of shape (num_fields, N, N).
    """
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    # Define the time points for the fields
    time_points = torch.linspace(0, num_fields - 1, num_fields, device=device)

    # Compute the temporal covariance matrix
    C_temporal = torch.exp(-abs(time_points[:, None] - time_points[None, :]) / T_corr)

    # Perform Cholesky decomposition to get the temporal correlation factors
    L_chol = torch.linalg.cholesky(C_temporal)

    # Generate independent Gaussian white noise for each time point
    white_noises = torch.randn((num_fields, N, N), device=device)

    # Combine white noise using the Cholesky factors to induce temporal correlation
    temporal_correlated_noises = torch.matmul(L_chol, white_noises.view(num_fields, -1)).view(num_fields, N, N)

    # Generate 2D grid of wavenumbers for spatial correlation
    kx = torch.fft.fftfreq(N, device=device) * N
    ky = torch.fft.fftfreq(N, device=device) * N
    k = torch.sqrt(kx[:, None]**2 + ky[None, :]**2)
    cutoff_mask = (k < 20).float()  # High-frequency cutoff
    # apply - if we apply the same approach to vorticity, and then obtain 
    # stream function, 
    # Spatial covariance (Power spectrum) for Gaussian covariance
    P_k = torch.exp(-0.5 * (k * L)**3)
    P_k[0, 0] = 0.0
    P_k = P_k / torch.sum(P_k)

    # Generate fields using Fourier transform
    fields = []
    for i in range(num_fields):
        noise_ft = torch.fft.fft2(temporal_correlated_noises[i])
        field_ft = noise_ft * torch.sqrt(P_k) * cutoff_mask
        field = torch.fft.ifft2(field_ft).real
        field = sigma * (field - torch.mean(field) )/torch.std(field)
        fields.append(field)
    return torch.stack(fields)


def warp_field(field, dx, dy):
    """
    Warp a 2D field based on displacement fields dx and dy.
    field (torch.Tensor): Input field of shape (batch_size, channels, height, width)
    dx (torch.Tensor): X-displacement field of shape (batch_size, height, width)
    dy (torch.Tensor): Y-displacement field of shape (batch_size, height, width)
    """
    batch_size, _, height, width = field.shape
    dx = dx.to(field.dtype)
    dy = dy.to(field.dtype)
    # Create base grid
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    base_grid = torch.stack((x, y), dim=-1).float()
    
    # Add batch dimension and move to the same device as input field
    base_grid = base_grid.unsqueeze(0).repeat(batch_size,1,1,1).to(field.device)

    # Apply displacements
    sample_grid = base_grid + torch.stack((dx, dy), dim=-1)
    sample_grid[..., 0] = sample_grid[..., 0] % (width)
    sample_grid[..., 1] = sample_grid[..., 1] % (height)
    
    # Normalize grid to [-1, 1] range
    sample_grid[..., 0] = 2 * sample_grid[..., 0] / (width) - 1
    sample_grid[..., 1] = 2 * sample_grid[..., 1] / (height) - 1

    # Perform sampling
    warped_field = F.grid_sample(field, sample_grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    return warped_field


def get_corr_grf(device, T=10, T_corr=10.0, L=1.0, sigma=1.0, size=128):
    # Power spectrum for Gaussian covariance
    kx = torch.fft.fftfreq(size, device=device) * size
    ky = torch.fft.fftfreq(size, device=device) * size
    
    k = torch.sqrt(kx[:, None]**2 + ky[None, :]**2) 
    P_k = 10.0 * torch.exp(-0.5 *torch.pi * (k * L)**2)
    P_k[0,0]=0.0

    # Define the time points for 10 fields
    time_points = torch.linspace(0, 9, T, device=device)  # Time steps t1 to t10
    # Compute the temporal covariance matrix
    C_temporal = torch.zeros((T, T), device=device)
    for i in range(T):
        for j in range(T):
            C_temporal[i, j] = torch.exp(-abs(time_points[i] - time_points[j]) / T_corr)

    L_chol = torch.linalg.cholesky(C_temporal)
    white_noises = torch.randn((T, size, size), device=device)
    temporal_correlated_noises = torch.matmul(L_chol, white_noises.view(T, -1)).view(T, size, size)
    fields = []
    for i in range(T):
        noise_ft = torch.fft.fft2(temporal_correlated_noises[i])
        field_ft = noise_ft * torch.sqrt(P_k)
        field = 100* torch.fft.ifft2(field_ft).real
        print(field.shape)
        fields.append(field)
    
    return torch.stack(fields).unsqueeze(0)


# Function to create a 2D Gaussian kernel
def gaussian_kernel(kernel_size=5, sigma=1.0, dtype=torch.float32, device='cuda'):
    # Create a 2D grid of coordinates (x, y)
    x = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2
    y = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(x, y)

    # Compute the Gaussian function on the grid
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Normalize the kernel to ensure the sum of the weights is 1
    kernel = kernel / kernel.sum()

    return kernel


# Function to apply Gaussian smoothing to a batch of images
def apply_gaussian_smoothing(input_tensor, kernel_size=15, sigma=3.0):
    batch_size, height, width = input_tensor.shape
    
    # Ensure the input and kernel are on the same device and have the same type
    dtype = input_tensor.dtype
    device = input_tensor.device
    
    # Create the Gaussian kernel with the same dtype and device as the input tensor
    kernel = gaussian_kernel(kernel_size, sigma, dtype=dtype, device=device)
    
    # Reshape the kernel to be [1, 1, kernel_size, kernel_size] (for 2D convolution)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    # Expand the kernel to apply it across all channels and the batch
    kernel = kernel.expand(1, 1, kernel_size, kernel_size)

    # Add the channel dimension to the input tensor (assumed grayscale images, 1 channel)
    input_tensor = input_tensor.unsqueeze(1)  # Shape: [batch_size, 1, 128, 128]

    # Apply the convolution (use groups=batch_size to apply the kernel individually per batch element)
    smoothed_tensor = F.conv2d(input_tensor, kernel, padding='same', groups=1)

    # Remove the extra channel dimension
    smoothed_tensor = smoothed_tensor.squeeze(1)  # Shape: [batch_size, 128, 128]

    return smoothed_tensor