import numpy as np
from scipy.special import erfc

# Background: The Ewald summation parameter alpha controls the balance between real-space and reciprocal-space contributions by determining the width of the Gaussian charge distribution. To balance convergence, alpha is typically chosen based on the smallest reciprocal lattice vector (in magnitude), as this vector contributes most significantly to the reciprocal-space sum. The formula used here scales the inverse of this smallest magnitude by a given factor (alpha_scaling) to determine alpha.


def get_alpha(recvec, alpha_scaling=5):
    '''Calculate the alpha value for the Ewald summation, scaled by a specified factor.
    Parameters:
        recvec (np.ndarray): A 3x3 array representing the reciprocal lattice vectors.
        alpha_scaling (float): A scaling factor applied to the alpha value. Default is 5.
    Returns:
        float: The calculated alpha value.
    '''
    # Compute the magnitudes of each reciprocal lattice vector (rows of recvec)
    g_magnitudes = np.linalg.norm(recvec, axis=1)
    # Find the smallest magnitude among the reciprocal vectors
    min_g_magnitude = np.min(g_magnitudes)
    # Calculate alpha using the scaling factor and smallest reciprocal vector magnitude
    alpha = alpha_scaling / (2 * min_g_magnitude)
    
    return alpha



# Background: To generate tiled lattice coordinates, we need to expand the unit cell in all three dimensions. The lattice coordinates are formed by taking integer linear combinations of the lattice vectors (rows of `latvec`). The integer indices range from `-nlatvec` to `nlatvec` (inclusive) in each direction, ensuring coverage of `(2 * nlatvec + 1)` points per direction. Combining these indices across all three directions gives a total of `(2 * nlatvec + 1)^3` lattice coordinates, each computed as a linear combination of the lattice vectors with the corresponding indices.


def get_lattice_coords(latvec, nlatvec=1):
    '''Generate lattice coordinates based on the provided lattice vectors.
    Parameters:
        latvec (np.ndarray): A 3x3 array representing the lattice vectors.
        nlatvec (int): The number of lattice coordinates to generate in each direction.
    Returns:
        np.ndarray: An array of shape ((2 * nlatvec + 1)^3, 3) containing the lattice coordinates.
    '''
    # Generate integer indices from -nlatvec to nlatvec (inclusive) for each direction
    indices = np.arange(-nlatvec, nlatvec + 1)
    # Create 3D meshgrid of indices to cover all combinations
    i, j, k = np.meshgrid(indices, indices, indices, indexing='ij')
    # Flatten the meshgrid to 1D arrays for each index component
    i_flat = i.flatten()
    j_flat = j.flatten()
    k_flat = k.flatten()
    # Compute each lattice coordinate as a linear combination of lattice vectors with indices
    lattice_coords = (i_flat[:, np.newaxis] * latvec[0] +
                      j_flat[:, np.newaxis] * latvec[1] +
                      k_flat[:, np.newaxis] * latvec[2])
    return lattice_coords
