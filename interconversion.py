"""
Name: anisotropic-interconversion
Description: Python library to interconvert anisotropic
             relaxation and creep prony series
Author: Christopher Rehberg
Email: christopher.rehberg@knights.ucf.edu
"""

# External libaries to run the viscoelastic interconversion library
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.linalg as linalg
from numpy import linalg as la


def import_properties_excel(excel_file, coeff_size=None, invert=False):
    """Import matrices and creep/relaxation times from an excel
       format with a specific format for this library


    Parameters
    ----------
    excel_file : xlsx
        Excel file
    coeff_size : None or int
        sets the size of the matrix to be read, default to none
    invert : Bool
        flag if time consts need to be inverted

    Returns
    -------
    mat0 : numpy.array
        Instantaneous modulus
    matrix_coeff : numpy.array
        Modulus coefficients
    time_consts : numpy.array
        inverted time constants
    coeff_size : int
        Number of coefficent matrices
    """

    # Open the excel file with viscoelastic material properties
    with open(excel_file, 'br') as excel_loc:

        # Read in the number of matrix coefficients, and set as an integer
        num_coeff = pd.read_excel(excel_loc, header=None, usecols=[1], nrows=1)
        num_coeff = int(num_coeff.values)

        # Check to see if a matrix size has been set, default to none
        if coeff_size is None:
            # Read in the size of the matrix coefficients, and set as an int
            coeff_size = pd.read_excel(
                excel_loc, header=None, usecols=[3], nrows=1)
            coeff_size = int(coeff_size.values)

        # Read in the instantious coefficient as a size of 6x6
        mat0 = pd.read_excel(excel_loc, header=None,
                             usecols=[0, 1, 2, 3, 4, 5], nrows=6, skiprows=5)
        # Change from dataframe to a numpy array
        mat0 = mat0.to_numpy()
        # Select only the required coefficients
        mat0 = mat0[:coeff_size, 0:coeff_size]

        # Create an empy array to store coefficients
        matrix_coeff = np.empty((num_coeff, coeff_size, coeff_size))

        for i in range(num_coeff):

            # Read a coefficient matrix from excel
            temp_coeff = pd.read_excel(excel_loc, header=None,
                                       usecols=[j+(6*i) for j in range(6)],
                                       nrows=6, skiprows=13)

            # Convert to a numpy array and select required coefficients
            temp_coeff = temp_coeff.to_numpy()

            matrix_coeff[i, :, :] = temp_coeff[0:coeff_size, 0:coeff_size]

        # Read in the time constants
        time_consts = pd.read_excel(excel_loc, header=None,
                                    usecols=[1+i for i in range(num_coeff)],
                                    nrows=1, skiprows=2)

        # Convert to a numpy array and reshape to a 1D array
        time_consts = time_consts.to_numpy()
        time_consts = np.reshape(time_consts, num_coeff)

        # Invert time consts if required
        if(invert):
            time_consts = 1 / time_consts

    return mat0, matrix_coeff, time_consts, coeff_size


def StoC(S0, S_mats, lambdas, coeff_size):
    """Converts prony series creep complance to relaxtion modulus
       using Cholesky decomposition. Using algorithm 4 of "Interconversion of
       linearly viscoelastic material functions expressed as Prony series"


    Parameters
    ----------
    S0 : numpy.array
        Instantaneous creep modulus
    S_mats : numpy.array
        Creep modulus coefficient
    lambdas : numpy.array
        Creep time constants
    coeff_size : int
        The size dim of the the matrix i.e. 6 if 6x6

    Returns
    -------
    C0 : numpy.array
        Equilibrium relaxation
    C_mats : numpy.array
        Relaxtion modulus coefficient
    rhos : numpy.array
        Relaxation time constants
    """

    # Number of coefficents
    num_coeff = len(S_mats)

    # Final amount of coefficents returned, used to size different variables
    final_num_coeff = coeff_size * num_coeff

    # Following the step by step formula given by the paper
    A1 = S0

    A2 = np.linalg.cholesky(lambdas[0] * S_mats[0])

    for i in range(1, num_coeff):
        temp_mat = lambdas[i] * S_mats[i]
        temp_mat = np.linalg.cholesky(temp_mat)
        A2 = np.concatenate((A2, temp_mat), 1)

    A3 = np.identity(coeff_size, dtype=np.float64)
    A3 = lambdas[0]*A3

    if num_coeff >= 2:
        for i in range(1, num_coeff):
            A3 = linalg.block_diag(A3, (lambdas[i]*np.identity(coeff_size)))

    B_idnet = np.identity(final_num_coeff)

    L1 = np.linalg.inv(A1)
    L2 = A2.T @ L1
    L2 = L2.T
    L3 = A3 + A2.T @ L1 @ A2

    L3_star, PT = np.linalg.svd(L3)[1:]
    L3_star = np.diag(L3_star)

    L2_star = PT @ L2.T
    L2_star = L2_star.T

    # Preallocate a 3d numpy array for the coefficent matrices
    C_mats = np.empty((final_num_coeff, coeff_size, coeff_size))

    # Preallocate a numpy array for the time consts
    rhos = np.zeros(final_num_coeff)

    for m in range(0, final_num_coeff):
        for i in range(0, coeff_size):
            for j in range(0, coeff_size):
                C_mats[m, i, j] = (
                    (L2_star[i, m]*L2_star[j, m]) / L3_star[m, m])

        rhos[m] = L3_star[m, m] / B_idnet[m, m]

    C0 = L1

    for i in range(len(C_mats)):
        C0 = C0 - C_mats[i]

    return (C0, C_mats, rhos)


def CtoS(C0, C_mats, rhos, coeff_size):
    """Converts prony series relaxtion modulus to creep complance
       using Cholesky decomposition. Using algorithm 3 of "Interconversion of
       linearly viscoelastic material functions expressed as Prony series"


    Parameters
    ----------
    C0 : numpy.array
        Equilibrium relaxation
    C_mats : numpy.array
        Relaxtion modulus coefficient
    rhos : numpy.array
        Relaxtion time constants
    coeff_size : int
        The size dim of the the matrix i.e. 6 if 6x6

    Returns
    -------
    S0 : numpy.array
        Instantaneous creep modulus
    S_mats : numpy.array
        Creep modulus coefficient
    lambdas : numpy.array
        Creep time constants
    """

    # Number of coefficents
    num_coeff = len(C_mats)

    # Final amount of coefficents returned, used to size different variables
    final_num_coeff = coeff_size * num_coeff

    # Following the step by step formula given by the paper
    L1 = C0

    for i in range(num_coeff):
        L1 = L1 + C_mats[i]

    L2 = np.linalg.cholesky(rhos[0] * C_mats[0])

    for i in range(1, num_coeff):
        temp_mat = np.linalg.cholesky(rhos[i] * C_mats[i])
        L2 = np.concatenate((L2, temp_mat), 1)

    L3 = np.identity(coeff_size, dtype=np.float64)
    L3 = rhos[0]*L3

    if num_coeff >= 2:
        for i in range(1, num_coeff):
            L3 = linalg.block_diag(L3, (rhos[i]*np.identity(coeff_size)))

    B_idnet = np.identity(final_num_coeff)

    A1 = np.linalg.inv(L1)
    A2 = L2.T @ A1
    A2 = A2.T
    A3 = L3 - L2.T @ A1 @ L2

    A3_star, PT = np.linalg.svd(A3)[1:]
    A3_star = np.diag(A3_star)

    A2_star = PT @ A2.T
    A2_star = A2_star.T

    S0 = A1

    # Preallocate a 3d numpy array for the coefficent matrices
    S_mats = np.empty((final_num_coeff, coeff_size, coeff_size))

    # Preallocate a numpy array for the time consts
    lambdas = np.zeros(final_num_coeff)

    for m in range(0, final_num_coeff):
        for i in range(0, coeff_size):
            for j in range(0, coeff_size):
                S_mats[m, i, j] = (
                    (A2_star[i, m] * A2_star[j, m]) / A3_star[m, m])

        lambdas[m] = A3_star[m, m] / B_idnet[m, m]

    lambdas = np.flip(lambdas)

    return (S0, S_mats, lambdas)


def modulus_at_time(M0, M_mats, time_const, time, property):
    """Gives the matrix of creep or relaxation at a given time

    Parameters
    ----------
    M0 : numpy.array
        Instantaneous/Equilibrium modulus
    M_mats : numpy.array
        Coefficient moduli
    time_const : numpy.array
        Time constants
    time : float
        Time at which to calculate property
    property : string
        Switch for creep or relaxation calculations (relax or creep)

    Returns
    -------
    mod_time : numpy.array
        Modulus at a given time
    """

    # Number of coefficient matrices
    num_coeff = len(M_mats)

    # Funtion for matrix relaxation modulus at given time
    def relax_time(M_mats, rhos, time): return M_mats * \
        (np.exp(-1 * time * rhos.reshape(num_coeff, 1, 1)))
    # Funtion for matrix creep modulus at given time
    def creep_time(M_mats, lambdas, time): return M_mats * \
        (1 - np.exp(-1 * time * lambdas.reshape(num_coeff, 1, 1)))

    # Sets the proper relax or creep function to the variable time_func
    if property == "relax":
        time_func = relax_time
    elif property == "creep":
        time_func = creep_time
    else:
        raise Exception('Expected "relax" or "creep" proptery')

    # Caluclates the modulus at the given time for each matrix
    mod_time = time_func(M_mats, time_const, time)
    # Sums each matrix together
    mod_time = np.sum(mod_time, axis=0)
    # Adds the Instantaneous/Equilibrium modulus
    mod_time = mod_time + M0

    return mod_time


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    Parameters
    ----------
    A : numpy.array
        Matrix

    Returns
    -------
    A3: numpy.array
        Pos def matrix
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))

    Ident = np.eye(A.shape[0], dtype=np.float64)
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += Ident * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky

    Parameters
    ----------
    B : numpy.array
        Matrix

    Returns
    -------
    Bool : Bool
        Returns True or False
    """

    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def pos_def_update(M0, M_mats):
    """Checks a matrices for positive-definitness. If matrix is not
    positive-definite, finds nearest positive-definite matrix and replaces

    Parameters
    ----------
    M0 : numpy.array
        Instantaneous/Equilibrium modulus
    M_mats : numpy.array
        Coefficient moduli

    Returns
    -------
    M0 : numpy.array
        Positive-definite Instantaneous/Equilibrium modulus
    M_mats : numpy.array
        Positive-definite Coefficient moduli
    """

    def is_pos_def(matrix):
        if isPD(matrix):
            return matrix
        else:
            return nearestPD(matrix)

    M0 = is_pos_def(M0)

    for i in range(len(M_mats)):
        M_mats[i] = is_pos_def(M_mats[i])

    return (M0, M_mats)


def convolution_check_mat(C0, C_mats, rhos, S0, S_mats, lambdas, t):
    """Checks the convolution intergral of the C(t) and S(t) matrices.
       Should be the identiy matrix of t*I
       Also returns the errors from scipy.quad intergration

    Parameters
    ----------
    C0 : numpy.array
        Equilibrium relaxation
    C_mats : numpy.array
        Relaxtion modulus coefficient
    rhos : numpy.array
        Relaxation time constants
    S0 : numpy.array
        Instantaneous creep modulus
    S_mats : numpy.array
        Creep modulus coefficient
    lambdas : numpy.array
        Creep time constants
    t : float
        time

    Returns
    -------
    convolution : numpy.array
        Numpy array of the convolution matrix for C(t) and S(t)
    error : numpy.array
        Numpy array of the errors for each element of the convolution matrix
    """

    dim = max(C0.shape)

    convolution = np.empty(C0.shape)
    error = np.empty(C0.shape)

    def f(C0, C_mats, S0, S_mats, lambdas, rhos, t):
        def g(tau):

            C_converted = modulus_at_time(
                C0, C_mats, rhos, t - tau, "relax")

            S_converted = modulus_at_time(
                S0, S_mats, lambdas, tau, "creep")

            return np.dot(C_converted, S_converted)[i, j]
        return g

    u = f(C0, C_mats, S0, S_mats, lambdas, rhos, t)

    for i in range(dim):
        for j in range(dim):

            convolution[i, j], error[i, j] = integrate.quad(
                u, 0, t, epsabs=1e-12, epsrel=1e-12, limit=1000)

    return convolution, error
