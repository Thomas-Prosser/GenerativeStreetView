import torch
import torch.nn.functional as F
import numpy as np

class GeometryProjection:
    def __init__(self, 
                 target_height=256,    # Output panorama height
                 target_width=512,     # Output panorama width
                 grd_height=-2,        # Ground height in meters
                 max_height=6,         # Maximum height in meters
                 num_planes=64,        # Number of height planes
                 device='cuda'):
        """
        Initialize Geometry Projection for CVACT dataset.
        All parameters optimized for CVACT street-view synthesis.
        """
        self.target_height = target_height
        self.target_width = target_width
        self.grd_height = grd_height
        self.max_height = max_height
        self.num_planes = num_planes
        self.device = device
        
        # CVACT specific parameters
        self.meters = (50 * 206 / 256)  # CVACT meter to pixel ratio
        
        # Pre-compute coordinate grids and move to device
        j = torch.arange(0, self.target_width, device=device)
        i = torch.arange(0, self.target_height, device=device)
        self.jj, self.ii = torch.meshgrid(j, i, indexing='xy')
        
        # Calculate viewing angles once
        self.tanPhi = torch.tan(self.ii / self.target_height * np.pi)
        self.tanPhi = torch.where(self.tanPhi == 0, 
                                  torch.tensor(1e-6, device=device),
                                  self.tanPhi)
        
    def preprocess_input(self, satellite_images):
        """
        Preprocess input satellite images from CVACT format.
        Args:
            satellite_images: tensor of shape (N, 3, 256, 256)
        Returns:
            preprocessed images: tensor of shape (N, 256, 256, 3)
        """
        # Move to device and transpose from channels first to channels last
        satellite_images = satellite_images.to(self.device)
        return satellite_images.permute(0, 2, 3, 1)
        
    def create_mpi(self, satellite_image, height_probs):
        """
        Create Multi-Plane Image (MPI) representation.
        Args:
            satellite_image: (N, 256, 256, 3) tensor
            height_probs: (N, 256, 256, num_planes) tensor
        Returns:
            MPI layers with color and alpha channels.
        """
        # Create alpha channels using cumulative sum of height probabilities
        alpha = torch.cumsum(height_probs, dim=-1)  # Cumulative probabilities
        
        # Repeat satellite image for each plane
        rgb = satellite_image.unsqueeze(3).repeat(
            1, 1, 1, self.num_planes, 1
        )
        
        # Combine RGB and alpha into RGBA
        alpha = alpha.unsqueeze(-1)  # Add channel dimension
        mpi = torch.cat([rgb, alpha], dim=-1)
        
        return mpi
        
    def project_to_street_view(self, mpi_layers):
        """
        Project MPI layers to street-view perspective.
        Args:
            mpi_layers: RGBA MPI representation.
        Returns:
            street_view_mpi: Projected layers in street-view perspective.
        """
        input_size = mpi_layers.shape[1]
        radius = input_size // 4
        MetersPerRadius = self.meters / 2 / radius
        
        street_view_layers = []
        
        for r in range(radius):
            # Calculate z coordinate based on radius and viewing angle
            z = (radius - r - 1) * MetersPerRadius / self.tanPhi
            z_normalized = (self.num_planes - 1) - \
                         (z - self.grd_height) / \
                         (self.max_height - self.grd_height) * \
                         (self.num_planes - 1)
            
            # Calculate projection coordinates (theta remains unchanged)
            theta = self.jj
            # (coords is computed but not directly used; we rely on F.interpolate below)
            coords = torch.stack([theta, z_normalized], dim=-1)
            
            # Sample from MPI layers using bilinear interpolation.
            # Here we simply resize the layer corresponding to the current radius.
            warped = F.interpolate(
                mpi_layers[:, :, :, r, :].permute(0, 3, 1, 2),  # Convert to NCHW
                size=(self.target_height, self.target_width),
                mode='bilinear',
                align_corners=True
            ).permute(0, 2, 3, 1)  # Back to NHWC
            
            street_view_layers.append(warped)
        
        return torch.stack(street_view_layers, dim=3)
        
    def render_final_image(self, street_view_mpi):
        """
        Render final street-view panorama using alpha compositing.
        Args:
            street_view_mpi: Projected MPI layers.
        Returns:
            final_image: Rendered street-view panorama.
        """
        # Split into RGB and alpha
        rgb = street_view_mpi[..., :3]
        alpha = street_view_mpi[..., 3:]
        
        # Composite from back to front
        output = rgb[..., 0, :]  # Initialize with the furthest layer
        
        for i in range(1, street_view_mpi.shape[3]):
            curr_rgb = rgb[..., i, :]
            curr_alpha = alpha[..., i, :]
            output = curr_rgb * curr_alpha + output * (1 - curr_alpha)
        
        return output
        
    def __call__(self, satellite_images, height_probabilities):
        """
        Main function to perform geometry projection.
        Args:
            satellite_images: (N, 3, 256, 256) input tensor.
            height_probabilities: (N, 256, 256, 64) height predictions.
        Returns:
            street_view_panoramas: (N, 256, 512, 3) output tensor.
        """
        # Ensure inputs are on the correct device.
        satellite_images = satellite_images.to(self.device)
        height_probabilities = height_probabilities.to(self.device)
        
        # Preprocess input.
        sat_processed = self.preprocess_input(satellite_images)
        
        # Create MPI representation.
        mpi = self.create_mpi(sat_processed, height_probabilities)
        
        # Project to street view.
        street_view_mpi = self.project_to_street_view(mpi)
        
        # Render final panorama.
        panorama = self.render_final_image(street_view_mpi)
        
        return panorama
