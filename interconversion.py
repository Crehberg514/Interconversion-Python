"""
Summary: Python library to interconvert anisotropic material functions.

Name: anisotropic-interconversion

Author: Christopher Rehberg
Email: christopher.rehberg@knights.ucf.edu
"""

# External libraries to run the viscoelastic interconversion library
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.linalg as linalg
from numpy import linalg as la


def import_properties_excel(excel_file, coeff_size=6, invert=False):
    """Import prony series coefficient matrices and time constants from excel.

    Uses a specifically formatted excel document to extract the material
    function matrix coefficients along with time series constants. Potentially
    the dimension of the square matrix can be extracted as well.

    Time constants should be in an inverted form, if they are not, then setting
    the invert flag to True will invert the time constants for you.

    If `coeff_size` is set to None, the function will read the dimension size
    from the excel file.
    If an integer value is given, this value will be used instead, with default
    set to 6.

    Parameters
    ----------
    excel_file : str
        Location of a Microsoft excel file in the proper format
    coeff_size : None or int
        If given, sets the size of the matrix to be read, defaults to 6.
        If None is given, then this function will try to read from the size of
        the matrix from the excel file.
    invert : Bool
        Boolean flag to trigger then inverting of the time constants

    Returns
    -------
    mat0 : numpy.array
        The instantaneous modulus returned in a square numpy array
    matrix_coeff : numpy.array
        The modulus coefficients returned in a 3D numpy array. With the first
        dimension being each coefficient matrix. The second and third dimension
        are the row and columns of the individual matrix.
    time_consts : numpy.array
        A 1D numpy array of the inverted time constants
    """
    # Open the excel file with viscoelastic material properties
    with open(excel_file, 'br') as excel_loc:

        # Read in the number of matrix coefficients, and set as an integer
        num_coeff = pd.read_excel(excel_loc, header=None, usecols=[1], nrows=1)
        num_coeff = int(num_coeff.values)

        # Check to see if a matrix size has been set, default to none
        if coeff_size is None:
            # Read in the size of the matrix coefficients as an integer
            coeff_size = pd.read_excel(
                excel_loc, header=None, usecols=[3], nrows=1)
            coeff_size = int(coeff_size.values)

        # Read in the instantaneous coefficient as a size of 6x6
        mat0 = pd.read_excel(excel_loc, header=None,
                             usecols=[0, 1, 2, 3, 4, 5], nrows=6, skiprows=5)
        # Change from dataframe to a numpy array
        mat0 = mat0.to_numpy()
        # Select only the required coefficients
        mat0 = mat0[:coeff_size, 0:coeff_size]

        # Create an empty array to store coefficients
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

        # Invert time constants if required
        if(invert):
            time_consts = 1 / time_consts

    return mat0, matrix_coeff, time_consts


def StoC(S0, S_mats, lambdas):
    """Convert a prony series creep compliance to relaxation modulus.

    Python numerical implementation of algorithm 4 from "Interconversion of
    linearly viscoelastic material functions expressed as Prony series"[1].
    Requiring the instantaneous creep modulus, the matrix creep coefficient
    modulus, the inverted time series, and finally the size of the square
    matrices.

    This version is of the algorithm is not limited to 6 by 6 matrices, but has
    been minimally adjusted to accept any square matrices.

    Parameters
    ----------
    S0 : numpy.array
        The instantaneous creep modulus as a 2D square numpy array.
    S_mats : numpy.array
        The creep modulus coefficient as a 3D numpy array, with the first
        dimension being the coefficient matrices and the second and third being
        the row and columns respectfully.
    lambdas : numpy.array
        The inverted creep time constants in a 1D numpy array, in descending
        order.

    Returns
    -------
    C0 : numpy.array
        The equilibrium relaxation modulus returned as a 2D square numpy array.
    C_mats : numpy.array
        The relaxation modulus coefficients returned as a 3D numpy array, with
        the first dimension being the coefficient matrices and the second and
        third being the row and columns respectfully.
    rhos : numpy.array
        The inverted relaxation time constants in a 1D numpy array.

    References
    ----------
    [1] Luk-Cyr, J., Crochon, T., Li, C. et al. Interconversion of linearly
    viscoelastic material functions expressed as Prony series: a closure. Mech
    Time-Depend Mater 17, 53–82 (2013).
    https://doi.org/10.1007/s11043-012-9176-y
    """
    # Size of square dimension
    coeff_size = S0.shape[0]

    # Number of coefficients
    num_coeff = len(S_mats)

    # Final amount of coefficients returned, used to size different variables
    final_num_coeff = coeff_size * num_coeff

    # Following the step by step formula given by the paper
    A1 = S0

    A2 = np.linalg.cholesky(lambdas[0] * S_mats[0])

    for i in range(1, num_coeff):
        temp_mat = lambdas[i] * S_mats[i]
        temp_mat = np.linalg.cholesky(temp_mat)
        A2 = np.concatenate((A2, temp_mat), 1)

    A3 = np.identity(coeff_size)
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

    # Pre-allocate a 3d numpy array for the coefficient matrices
    C_mats = np.empty((final_num_coeff, coeff_size, coeff_size))

    # Pre-allocate a numpy array for the time constants
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


def CtoS(C0, C_mats, rhos):
    """Convert a prony series relaxation modulus to creep compliance.

    Python numerical implementation of algorithm 3 from "Interconversion of
    linearly viscoelastic material functions expressed as Prony series"[1].
    Requiring the equilibrium relaxation modulus, the relaxation coefficient
    modulus matrices, the inverted time series, and finally the size of the
    square matrices.

    This version is of the algorithm is not limited to 6 by 6 matrices, but has
    been minimally adjusted to accept any square matrices.

    Parameters
    ----------
    C0 : numpy.array
        The equilibrium relaxation modulus as a 2D square numpy array.
    C_mats : numpy.array
        The relaxation modulus coefficients as a 3D numpy array, with the first
        dimension being the coefficient matrices and the second and third being
        the row and columns respectfully.
    rhos : numpy.array
        The inverted relaxation time constants in a 1D numpy array, in
        descending order.

    Returns
    -------
    S0 : numpy.array
        The instantaneous creep modulus returned as a 2D square numpy array.
    S_mats : numpy.array
        The creep modulus coefficient returned as a 3D numpy array, with
        the first dimension being the coefficient matrices and the second and
        third being the row and columns respectfully.
    lambdas : numpy.array
        The inverted creep time constants in a 1D numpy array.

    References
    ----------
    [1] Luk-Cyr, J., Crochon, T., Li, C. et al. Interconversion of linearly
    viscoelastic material functions expressed as Prony series: a closure. Mech
    Time-Depend Mater 17, 53–82 (2013).
    https://doi.org/10.1007/s11043-012-9176-y
    """
    # Size of square dimension
    coeff_size = C0.shape[0]

    # Number of coefficients
    num_coeff = len(C_mats)

    # Final amount of coefficients returned, used to size different variables
    final_num_coeff = coeff_size * num_coeff

    # Following the step by step formula given by the paper
    L1 = C0

    for i in range(num_coeff):
        L1 = L1 + C_mats[i]

    L2 = np.linalg.cholesky(rhos[0] * C_mats[0])

    for i in range(1, num_coeff):
        temp_mat = np.linalg.cholesky(rhos[i] * C_mats[i])
        L2 = np.concatenate((L2, temp_mat), 1)

    L3 = np.identity(coeff_size)
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

    # Pre-allocate a 3d numpy array for the coefficient matrices
    S_mats = np.empty((final_num_coeff, coeff_size, coeff_size))

    # Pre-allocate a numpy array for the time constants
    lambdas = np.zeros(final_num_coeff)

    for m in range(0, final_num_coeff):
        for i in range(0, coeff_size):
            for j in range(0, coeff_size):
                S_mats[m, i, j] = (
                    (A2_star[i, m] * A2_star[j, m]) / A3_star[m, m])

        lambdas[m] = A3_star[m, m] / B_idnet[m, m]

    return (S0, S_mats, lambdas)


def modulus_at_time(M0, M_mats, time_const, time, property):
    """Computes the anisotropic modulus, at a given time.

    A single function that calculates the modulus of an anisotropic material
    function at a given time. Requires the instantaneous/equilibrium matrix,
    coefficient matrices, the inverted time constants, and the time at which
    the modulus will be calculated. A property flag is also passed to the
    function to switch between creep or relaxation calculations.

    Parameters
    ----------
    M0 : numpy.array
        An instantaneous or equilibrium modulus in a 2D numpy array.
    M_mats : numpy.array
        The coefficient moduli in a 3D numpy array, with the first dimension
        consisting of the matrices and the second and third being the row and
        columns of the matrix.
    time_const : numpy.array
        The corresponding inverted time constants in a 1D numpy array, in
        descending order.
    time : float
        The time at which to calculate the modulus.
    property : string
        Flag to set the function to calculate the modulus of a creep material
        function or of a relaxation material function.
        Valid inputs are: 'creep' or 'relax'

    Returns
    -------
    mod_time : numpy.array
        The modulus, at the given time, for the prescribed material function.
        Returned in a 2D numpy array.
    """

    # Number of coefficient matrices
    num_coeff = len(M_mats)

    # Function for matrix relaxation modulus at given time
    def relax_time(M_mats, rhos, time): return M_mats * \
        (np.exp(-1 * time * rhos.reshape(num_coeff, 1, 1)))
    # Function for matrix creep modulus at given time
    def creep_time(M_mats, lambdas, time): return M_mats * \
        (1 - np.exp(-1 * time * lambdas.reshape(num_coeff, 1, 1)))

    # Sets the proper relax or creep function to the variable time_func
    if property == "relax":
        time_func = relax_time
    elif property == "creep":
        time_func = creep_time
    else:
        raise Exception('Expected "relax" or "creep" property')

    # Calculates the modulus at the given time for each matrix
    mod_time = time_func(M_mats, time_const, time)
    # Sums each matrix together
    mod_time = np.sum(mod_time, axis=0)
    # Adds the Instantaneous/Equilibrium modulus
    mod_time = mod_time + M0

    return mod_time


def creep_modulus(M0, M_mats, time_const, time):
    """Shortcut to compute the anisotropic creep modulus, at a given time.

    A shortcut function that presets the creep flag in the modulus_at_time
    function. Calculates the creep modulus of an anisotropic material function
    at a given time . Requires the instantaneous matrix, coefficient matrices,
    the inverted time constants, and the time at which the modulus will be
    calculated.

    Parameters
    ----------
    M0 : numpy.array
        An instantaneous modulus in a 2D numpy array.
    M_mats : numpy.array
        The coefficient moduli in a 3D numpy array, with the first dimension
        consisting of the matrices and the second and third being the row and
        columns of the matrix.
    time_const : numpy.array
        The corresponding inverted time constants in a 1D numpy array, in
        descending order.
    time : float
        The time at which to calculate the modulus.

    Returns
    -------
    mod_time : numpy.array
        The modulus, at the given time, for the prescribed material function.
        Returned in a 2D numpy array.
        Uses the modulus_at_time function with the proper flag preset.
    """
    return modulus_at_time(M0, M_mats, time_const, time, property='creep')


def relax_modulus(M0, M_mats, time_const, time):
    """Shortcut to compute the anisotropic relaxation modulus, at a given time.

    A shortcut function that presets the relaxation flag in the modulus_at_time
    function. Calculates the creep modulus of an anisotropic material function
    at a given time . Requires the equilibrium matrix, coefficient matrices,
    the inverted time constants, and the time at which the modulus will be
    calculated.

    Parameters
    ----------
    M0 : numpy.array
        An equilibrium modulus in a 2D numpy array.
    M_mats : numpy.array
        The coefficient moduli in a 3D numpy array, with the first dimension
        consisting of the matrices and the second and third being the row and
        columns of the matrix.
    time_const : numpy.array
        The corresponding inverted time constants in a 1D numpy array, in
        descending order.
    time : float
        The time at which to calculate the modulus.

    Returns
    -------
    mod_time : numpy.array
        The modulus, at the given time, for the prescribed material function.
        Returned in a 2D numpy array.
        Uses the modulus_at_time function with the proper flag preset.
    """

    return modulus_at_time(M0, M_mats, time_const, time, property='relax')


def nearestPD(A):
    """Find the nearest positive-definite matrix to the input matrix.

    Finds the nearest semi-positive definite matrix from the inputted matrix.
    This is a a Python port of John D'Errico's `nearestSPD` MATLAB code [1],
    which is an implementation of [2].

    Parameters
    ----------
    A : numpy.array
        A 2D numpy array.

    Returns
    -------
    SPD : numpy.array
        Semi-positive-definite 2D numpy array.

    References
    ----------
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    # Symmetrize matrix A into matrix B
    B = (A + A.T) / 2

    # Compute the symmetric polar factor of matrix B
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))

    # Calculate the nearest semi-positive-definite matrix
    SPD = (B + H) / 2

    # Ensure the matrix is symmetric
    SPD = (SPD + SPD.T) / 2

    # Test if the matrix is positive-definite
    if isPD(SPD):
        return SPD

    # Calculate minor adjustments to the matrix if it is not positive-definite
    # This is because of the how computers handle and calculate numbers
    spacing = np.spacing(la.norm(A))

    Ident = np.eye(A.shape[0])
    k = 1

    while not isPD(SPD):
        mineig = np.min(np.real(la.eigvals(SPD)))
        SPD += Ident * (-mineig * k**2 + spacing)
        k += 1

    return SPD


def isPD(B):
    """Checks if the matrix is positive-definite.

    Checks via Cholesky decomposition if the given matrix is positive-definite.
    If the matrix is positive-definite then the function returns True,
    otherwise, catches the thrown exception and returns False.

    Parameters
    ----------
    B : numpy.array
        Square matrix represented in a numpy array

    Returns
    -------
    Bool : Bool
        Returns True if the matrix is positive-definite and false otherwise.
    """

    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def pos_def_update(M0, M_mats):
    """Checks and corrects a matrix for positive-definiteness.

    First, checks to see if a matrix has the property of positive-definiteness.
    If the matrix has the property, then it returns the matrix. If it does not,
    then the nearest semi-positive-definite form of the matrix is found and
    returned.

    Parameters
    ----------
    M0 : numpy.array
        The instantaneous or equilibrium matrix given in a 2D numpy array
    M_mats : numpy.array
        The coefficient moduli given in a 3D numpy array, with the first
        dimension addressing the matrix, and the second and third, the rows and
        columns.

    Returns
    -------
    M0 : numpy.array
        Returns the original matrix if it was already positive-definite.
        Or returns the nearest semi-positive-definite form of the instantaneous
        or equilibrium matrix in a 2D numpy array.
    M_mats : numpy.array
        Returns a 3D numpy array, with the same constraints as given to the
        function.
        Each array returned is the original matrix if it was already
        positive-definite.
        All other matrices in the array are the nearest semi-positive-definite
        form of the coefficient moduli
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


def convolution_check_mat(C0, C_mats, rhos, S0, S_mats, lambdas, t, **kwarg):
    """Checks the convolution integral of the C(t) and S(t) matrices.

       Uses the scipy.quad numerical integration function to perform the
       numerical convolution integration of the creep and relaxation material
       functions. The returned matrix of the time convolution should be an
       identity matrix multiplied by the given time.

       This function accepts any number of keyword arguments, which are fed to
       the scipy.quad function. If no keyword arguments are given, then the
       integration is carried out using the default parameters of the
       scipy.quad function. This function also returns the errors given by the
       scipy.quad integration function.

       The minimum suggested keyword arguments and parameters are given below:
       `epsabs=1e-12, epsrel=1e-12, limit=1000`

    Parameters
    ----------
    C0 : numpy.array
        The equilibrium relaxation in a 2D numpy array.
    C_mats : numpy.array
        The relaxation modulus coefficient matrices in a 3D numpy array.
        The first dimension is to access the matrix, while the second and third
        are the rows and columns.
    rhos : numpy.array
        The inverted relaxation time constants in a 1D numpy array, in
        descending order.
    S0 : numpy.array
        The instantaneous creep modulus in a 2D numpy array.
    S_mats : numpy.array
        The creep modulus coefficient matrices in a 3D numpy array.
        The first dimension is to access the matrix, while the second and third
        are the rows and columns.
    lambdas : numpy.array
        The inverted creep time constants in a 1D numpy array, in descending
        order.
    t : float
        The time at which to calculate the convolution.
    **kwarg : dict
        Keyword arguments to be passed to the scipy.quad integration

    Returns
    -------
    convolution : numpy.array
        Numpy array of the convolution matrix for C(t) and S(t)
    error : numpy.array
        Numpy array of the errors for each element of the convolution matrix
    """

    dim = C0.shape[0]

    convolution = np.zeros_like(C0)
    error = np.zeros_like(C0)

    def f(C0, C_mats, S0, S_mats, lambdas, rhos, t):
        def g(tau):

            C_converted = relax_modulus(C0, C_mats, rhos, t - tau)

            S_converted = creep_modulus(S0, S_mats, lambdas, tau)

            return np.dot(C_converted, S_converted)[i, j]
        return g

    u = f(C0, C_mats, S0, S_mats, lambdas, rhos, t)

    for i in range(dim):
        for j in range(dim):

            convolution[i, j], error[i, j] = integrate.quad(u, 0, t, **kwarg)

    return convolution, error
