import numpy as np



# Background: The conjugate gradient (CG) method is an iterative algorithm for solving symmetric positive-definite linear systems Ax = b. It iteratively improves the solution using conjugate search directions. Starting with an initial guess x, the residual r = b - Ax is computed. The search direction p is initialized to r. In each iteration, the step size alpha is determined by the residual norm and the product of the search direction with A. The solution x is updated, the residual is adjusted, and a new search direction is computed using the beta coefficient. This repeats until the residual norm falls below the specified tolerance.


def cg(A, b, x, tol):
    '''Inputs:
    A : Matrix, 2d array size M * M
    b : Vector, 1d array size M
    x : Initial guess vector, 1d array size M
    tol : tolerance, float
    Outputs:
    x : solution vector, 1d array size M
    '''
    r = b - np.dot(A, x)
    p = r.copy()
    rsold = np.dot(r, r)
    
    # Check initial guess
    if np.sqrt(rsold) < tol:
        return x
    
    while True:
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        
        if np.sqrt(rsnew) < tol:
            break
        
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x
