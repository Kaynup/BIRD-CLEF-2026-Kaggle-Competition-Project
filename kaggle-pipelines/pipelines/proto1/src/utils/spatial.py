import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from src.config import Config

def lat_lon_to_cartesian(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """
    Maps Latitude and Longitude to 3D Cartesian Coordinates on a unit sphere.
    Avoids boundaries and discontinuities.
    """
    # Convert degrees to radians
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    
    # Calculate Cartesian coordinates on a unit sphere
    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    
    return torch.stack([x, y, z], dim=-1)  # Shape [N, 3]

def get_sinusoidal_positional_encoding(coords: torch.Tensor, L: int = 6) -> torch.Tensor:
    """
    Applies sinusoidal positional encoding to a 3D coordinate vector (x, y, z).
    Output dimension: 3 * 2 * L = 36-dim for L=6.
    """
    encodings = []
    for i in range(L):
        freq = (2.0 ** i) * np.pi
        # Sin and Cos for each dimension
        encodings.append(torch.sin(coords * freq))
        encodings.append(torch.cos(coords * freq))
    
    return torch.cat(encodings, dim=-1)  # Shape [N, 3 * 2 * L]

def compute_pantanal_hotness_vector(
    train_csv_path: Path, 
    taxonomy_csv_path: Path, 
    sample_sub_path: Path, 
    sigma: float = 5.0, 
    epsilon: float = 0.01
) -> torch.Tensor:
    """
    Computes a static Pantanal regional species presence prior (Hotness Vector)
    for all 234 target species using a geographic Gaussian kernel centered on the Pantanal.
    """
    train_df = pd.read_csv(train_csv_path)
    taxonomy_df = pd.read_csv(taxonomy_csv_path)
    sub_df = pd.read_csv(sample_sub_path)
    
    # Target species list in correct submission order
    target_species = list(sub_df.columns)[1:]
    
    # Identify which species have training audio files
    train_audio_species = set(train_df['primary_label'].unique())
    soundscape_only_species = set(target_species) - train_audio_species
    
    # Calculate distance of train_audio recordings to Pantanal Center
    train_df['dist_to_pantanal'] = np.sqrt(
        (train_df['latitude'] - Config.PANTANAL_LAT_CENTER)**2 + 
        (train_df['longitude'] - Config.PANTANAL_LON_CENTER)**2
    )
    
    hotness_scores = []
    for species in target_species:
        if species in soundscape_only_species:
            # Soundscape-only species are recorded native inside the Pantanal
            hotness_scores.append(1.0)
        else:
            species_data = train_df[train_df['primary_label'] == species]
            distances = species_data['dist_to_pantanal'].values
            
            # Spatial Gaussian kernel
            kernel_vals = np.exp(-(distances ** 2) / (2 * (sigma ** 2)))
            mean_hotness = np.mean(kernel_vals)
            
            # Baseline probability limit
            hotness_scores.append(max(epsilon, mean_hotness))
            
    return torch.tensor(hotness_scores, dtype=torch.float32)
