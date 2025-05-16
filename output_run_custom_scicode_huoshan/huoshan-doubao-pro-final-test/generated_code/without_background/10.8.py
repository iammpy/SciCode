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


# Background: To generate the distance matrix and pair indices, we need to compute the displacement vectors between each unique pair of particles. For a system with n particles, there are (n choose 2) unique pairs (i, j) where i < j. The distance vector for each pair is calculated as the difference between the coordinates of particle j and particle i. The pair indices list records these (i, j) pairs.



def distance_matrix(configs):
    '''Args:
        configs (np.array): (nparticles, 3)
    Returns:
        distances (np.array): distance vector for each particle pair. Shape (npairs, 3), where npairs = (nparticles choose 2)
        pair_idxs (list of tuples): list of pair indices
    '''
    nparticles = configs.shape[0]
    # Generate all unique (i, j) pairs with i < j
    pair_idxs = list(itertools.combinations(range(nparticles), 2))
    # Calculate distance vectors for each pair (j - i)
    distances = np.array([configs[j] - configs[i] for i, j in pair_idxs])
    return distances, pair_idxs


# Background: The real-space term in Ewald summation involves summing over all lattice cells to account for the periodic nature of the system. For each particle pair (i,j), the contribution from a lattice cell with vector nL is given by the complementary error function erfc of the product of alpha and the magnitude of (r_ij + nL), divided by that magnitude. The sum over all such lattice cells gives the total real-space contribution c_ij for the pair.



def real_cij(distances, lattice_coords, alpha):
    '''Calculate the real-space terms for the Ewald summation over particle pairs.
    Parameters:
        distances (np.ndarray): An array of shape (natoms, npairs, 1, 3) representing the distance vectors between pairs of particles where npairs = (nparticles choose 2).
        lattice_coords (np.ndarray): An array of shape (natoms, 1, ncells, 3) representing the lattice coordinates.
        alpha (float): The alpha value used for the Ewald summation.
    Returns:
        np.ndarray: An array of shape (npairs,) representing the real-space sum for each particle pair.
    '''
    # Sum distance vectors with lattice coordinates (broadcasting aligns dimensions)
    sum_vectors = distances + lattice_coords
    # Compute magnitude of each summed vector (axis=-1 for 3D components)
    magnitudes = np.linalg.norm(sum_vectors, axis=-1)  # Shape: (natoms, npairs, ncells)
    # Calculate erfc(alpha * magnitude) / magnitude for all terms
    erfc_terms = erfc(alpha * magnitudes) / magnitudes
    # Sum over lattice cells (axis=2) and atom types (axis=0) to get per-pair sums
    cij = np.sum(erfc_terms, axis=(0, 2))
    return cij


# Background: The real-space cross term in Ewald summation accounts for the electrostatic interaction between all pairs of charged particles (ions and electrons) considering periodic boundary conditions. This term is computed by summing over all unique particle pairs (i, j) with i < j, where each pair's contribution is the product of their charges and the real-space sum term c_ij. The term c_ij is calculated by summing over all lattice cells the complementary error function of the scaled distance divided by the distance itself.




def sum_real_cross(atom_charges, atom_coords, configs, lattice_coords, alpha):
    '''Calculate the sum of real-space cross terms for the Ewald summation.
    Parameters:
        atom_charges (np.ndarray): An array of shape (natoms,) representing the charges of the atoms.
        atom_coords (np.ndarray): An array of shape (natoms, 3) representing the coordinates of the atoms.
        configs (np.ndarray): An array of shape (nelectrons, 3) representing the configurations of the electrons.
        lattice_coords (np.ndarray): An array of shape (ncells, 3) representing the lattice coordinates.
        alpha (float): The alpha value used for the Ewald summation.
    Returns:
        float: The sum of ion-ion, electron-ion, and electron-electron cross terms.
    '''
    # Combine atom and electron coordinates into a single array
    all_coords = np.concatenate([atom_coords, configs], axis=0)
    nparticles = all_coords.shape[0]
    
    # Combine atom charges (ions) and electron charges (assumed -1 each)
    natoms = atom_charges.shape[0]
    nelectrons = configs.shape[0]
    all_charges = np.concatenate([atom_charges, -np.ones(nelectrons)])
    
    # Generate all unique (i, j) pairs with i < j
    pair_idxs = list(combinations(range(nparticles), 2))
    npairs = len(pair_idxs)
    
    # Calculate distance vectors for each pair (j - i)
    distance_vectors = np.array([all_coords[j] - all_coords[i] for i, j in pair_idxs])
    
    # Sum distance vectors with each lattice coordinate (broadcast to (npairs, ncells, 3))
    sum_vectors = distance_vectors[:, np.newaxis] + lattice_coords[np.newaxis, :]
    
    # Compute magnitude of each summed vector (shape: (npairs, ncells))
    magnitudes = np.linalg.norm(sum_vectors, axis=-1)
    
    # Avoid division by zero (though ideally magnitudes are positive)
    magnitudes = np.where(magnitudes == 0, np.finfo(float).eps, magnitudes)
    
    # Calculate erfc(alpha * magnitude) / magnitude for all terms
    erfc_terms = erfc(alpha * magnitudes) / magnitudes
    
    # Sum over lattice cells to get c_ij for each pair (shape: (npairs,))
    cij = np.sum(erfc_terms, axis=1)
    
    # Calculate product of charges for each pair
    q_pairs = np.array([all_charges[i] * all_charges[j] for i, j in pair_idxs])
    
    # Sum the product of charges and c_ij for all pairs
    val = np.sum(q_pairs * cij)
    
    return val


# Background: Reciprocal-space points (g-points) are generated to sample the reciprocal lattice for Ewald summation. The points are defined by integer indices (gx, gy, gz) where gx ranges from 0 to gmax, and gy, gz range from -gmax to gmax. Exclusions include the origin (0,0,0), points with x=0, y=0, z<0, and points with x=0, y<0 to avoid redundant or unwanted contributions. The final g-points are linear combinations of the reciprocal lattice vectors using these indices.


def generate_gpoints(recvec, gmax):
    '''Generate a grid of g-points for reciprocal space based on the provided lattice vectors.
    Parameters:
        recvec (np.ndarray): A 3x3 array representing the reciprocal lattice vectors.
        gmax (int): The maximum integer number of lattice points to include in one positive direction.
    Returns:
        np.ndarray: An array of shape (nk, 3) representing the grid of g-points.
    '''
    # Generate integer indices for gx, gy, gz
    gx_values = np.arange(0, gmax + 1)
    gy_values = np.arange(-gmax, gmax + 1)
    gz_values = np.arange(-gmax, gmax + 1)
    
    # Create 3D meshgrid of indices
    gx, gy, gz = np.meshgrid(gx_values, gy_values, gz_values, indexing='ij')
    
    # Flatten the meshgrid to 1D arrays
    gx_flat = gx.flatten()
    gy_flat = gy.flatten()
    gz_flat = gz.flatten()
    
    # Combine into an array of (gx, gy, gz) points
    gpoints = np.column_stack((gx_flat, gy_flat, gz_flat))
    
    # Exclude the origin (0,0,0)
    mask = ~np.all(gpoints == 0, axis=1)
    
    # Extract gx, gy, gz columns for easier condition checks
    gx_col = gpoints[:, 0]
    gy_col = gpoints[:, 1]
    gz_col = gpoints[:, 2]
    
    # Identify points where gx is 0
    gx_zero = (gx_col == 0)
    
    # For gx=0, gy must be >=0
    gy_ge_zero = (gy_col >= 0)
    
    # For gx=0 and gy=0, gz must be >0 (since z<0 is excluded and z=0 is origin)
    gy_zero = (gy_col == 0)
    gz_gt_zero = (gz_col > 0)
    
    # Combine conditions for gx=0 points: (gy >=0) and (if gy=0 then gz>0 else True)
    condition_gx_zero = gy_ge_zero & ( (gy_zero & gz_gt_zero) | (~gy_zero) )
    
    # Update mask: keep points where (gx !=0) OR (gx=0 and condition_gx_zero is True)
    mask = mask & ( (~gx_zero) | condition_gx_zero )
    
    # Apply mask to get valid (gx, gy, gz) integer points
    gpoints_filtered = gpoints[mask]
    
    # Convert integer points to actual reciprocal space vectors using recvec
    gpoints_all = gpoints_filtered @ recvec
    
    return gpoints_all


# Background: The weight at each reciprocal space point (g-point) in Ewald summation is given by the formula \( W(k) = \frac{4 \pi}{V} \frac{\mathrm{e}^{-\frac{k^2}{4 \alpha^2}}}{k^2} \), where \( k \) is the reciprocal vector, \( V \) is the unit cell volume, and \( \alpha \) is the Ewald parameter. This weight quantifies the contribution of each g-point to the reciprocal-space sum. Points with weights below a specified tolerance (tol) are considered negligible and are filtered out to optimize computation.


def select_big_weights(gpoints_all, cell_volume, alpha, tol=1e-10):
    '''Filter g-points based on weight in reciprocal space.
    Parameters:
        gpoints_all (np.ndarray): An array of shape (nk, 3) representing all g-points.
        cell_volume (float): The volume of the unit cell.
        alpha (float): The alpha value used for the Ewald summation.
        tol (float, optional): The tolerance for filtering weights. Default is 1e-10.
    Returns:
        tuple: 
            gpoints (np.array): An array of shape (nk, 3) containing g-points with significant weights.
            gweights (np.array): An array of shape (nk,) containing the weights of the selected g-points.       
    '''
    # Calculate the squared magnitude of each g-point vector (k²)
    k_squared = np.linalg.norm(gpoints_all, axis=1) ** 2
    
    # Compute the exponential term in the weight formula
    exponential_term = np.exp(-k_squared / (4 * alpha ** 2))
    
    # Calculate the weight for each g-point
    gweights = (4 * np.pi / cell_volume) * exponential_term / k_squared
    
    # Create a mask to select weights above the tolerance
    mask = gweights > tol
    
    # Apply the mask to filter g-points and their weights
    gpoints = gpoints_all[mask]
    gweights = gweights[mask]
    
    return gpoints, gweights



# Background: The reciprocal-space term in Ewald summation accounts for the electrostatic interaction between all pairs of charged particles (ions and electrons) in reciprocal space. This term is computed by summing over all non-zero reciprocal lattice points (g-points). For each g-point, the contribution is given by the weight W(k) multiplied by the squared magnitude of the sum of charges multiplied by the exponential of the dot product of the g-point vector with each particle's coordinates. The squared magnitude captures the constructive and destructive interference of these contributions, and the sum over all g-points gives the total reciprocal-space contribution.


def sum_recip(atom_charges, atom_coords, configs, gweights, gpoints):
    '''Calculate the reciprocal lattice sum for the Ewald summation.
    Parameters:
        atom_charges (np.ndarray): An array of shape (natoms,) representing the charges of the atoms.
        atom_coords (np.ndarray): An array of shape (natoms, 3) representing the coordinates of the atoms.
        configs (np.ndarray): An array of shape (nelectrons, 3) representing the configurations of the electrons.
        gweights (np.ndarray): An array of shape (nk,) representing the weights of the g-points.
        gpoints (np.ndarray): An array of shape (nk, 3) representing the g-points.
    Returns:
        float: The reciprocal lattice sum.
    '''
    # Combine atom and electron charges and coordinates
    nelectrons = configs.shape[0]
    all_charges = np.concatenate([atom_charges, -np.ones(nelectrons)])
    all_coords = np.concatenate([atom_coords, configs], axis=0)
    
    # Compute dot product of each g-point with each particle coordinate (k·r_i)
    k_dot_ri = np.dot(gpoints, all_coords.T)  # Shape: (nk, n_particles)
    
    # Compute exp(i k·r_i) for all g-points and particles
    exp_terms = np.exp(1j * k_dot_ri)  # Shape: (nk, n_particles)
    
    # Multiply by charges and sum over particles to get sum(q_i exp(i k·r_i))
    sum_q_exp = np.sum(all_charges * exp_terms, axis=1)  # Shape: (nk,)
    
    # Calculate the squared magnitude of the sum
    abs_squared = np.abs(sum_q_exp) ** 2  # Shape: (nk,)
    
    # Multiply by weights and sum over all g-points
    val = np.sum(gweights * abs_squared)
    
    return val
