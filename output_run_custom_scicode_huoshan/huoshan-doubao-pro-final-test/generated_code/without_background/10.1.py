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
