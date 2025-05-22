import numpy as np
import qutip as qt


# ==============================================
# =============== Linear algebra ===============
# ==============================================

# To test the performance of weighted variational method,
# we need to evaluate the ground state of the time-dependent Hamiltonian.
# We get a few extremum eigenenergies and the corresponding eigenstates
# efficiently by using the power method in the following code.


def gram_schmidt(vectors):
    """
    Perform the Gram-Schmidt process on a list of vectors.
    vectors: a list of real or complex vectors
    return: a list of orthogonal vectors
    """ 
    q, r = np.linalg.qr(vectors.T)
    return q.T

def power_method(H, num, maxitr, tol, verbose = True):
    """
    Perform the power method to get  eigenstates [num]with the largest absolute values.
    
    The convergence is guaranteed for most of the case including the case of degenerate eigenvalues.
    The exception is when there is a pair of eigenvalues with the same absolute value with opposite signs,
    and at least one of them is within the required eigenvalues.
    """
    H_sparse = H.data_as("scipy_csr")

    # Get dimensions
    D = np.prod(H.shape[0]) # Dimension of the Hilbert space

    # Initial guess
    v_all = (np.random.rand(num,D) - 0.5) + (np.random.rand(num,D) - 0.5) * 1j
    v_all = gram_schmidt(v_all)
    e_all = np.zeros(num, dtype="complex")

    # Iterate
    for itr in range(maxitr):
        # Multiply the Hamiltonian to the vectors
        v_all_prev = v_all.copy()
        e_all_prev = e_all.copy()
        for i in range(num):
            # Check the normalization
            # if np.abs(np.linalg.norm(v_all_prev[i]) - 1) > 1e-10:
                # print("Warning: The vector is not normalized. Norm:", np.linalg.norm(v))
            
            mult = H_sparse.dot(v_all_prev[i])
            v_all[i] = mult
            e_all[i] = np.dot(v_all_prev[i].conj(), mult)
    
        # Orthogonalize the vectors
        v_all = gram_schmidt(v_all)
        
        # Check convergence
        is_converged = True
        for i in range(num):
            if np.abs(np.dot(v_all[i].conj(), v_all_prev[i])) < 1 - tol:
                is_converged = False
                break
            if np.abs(e_all[i] - e_all_prev[i]) > tol:
                is_converged = False
                break
        if is_converged:
            break

    if not is_converged and verbose:
        print("Warning: Power method did not converge after", maxitr, "iterations")
        for i in range(num):
            print("* Overlap:", np.abs(np.dot(v_all[i].conj(), v_all_prev[i])))
            print("* Eigenvalue difference:", np.abs(e_all[i] - e_all_prev[i]))

    # Check that the eigenvalues are close to real
    if verbose:
        for i in range(num):
            if np.abs(e_all[i].imag) > np.abs(e_all[i].real) * 1e-10:
                print("Warning: The eigenvalues are not close to real. The eigenstates may not be correct.")

    return e_all.real, v_all

    

def extremal_eigs(H, num = 3, maxitr = None, tol = 1e-8, cautious = True):
    """ 
    Perform the power method to get [num] highest and [num] lowest eigenstates/eigenvalues.
    
    H: Hamiltonian (a qutip.Qobj object)
    num: 2 * [num] is the number of extremal eigenstates to be calculated
    maxitr: maximum number of iterations
    tol: tolerance for convergence
    cautious: whether to repeat the power method with different initial guesses
    """

    # Set parameters
    if maxitr is None:
        maxitr = 10000 * num + 10000
    if cautious:
        rep = 4
    else:
        rep = 1
        
    # Get dimensions
    D = np.prod(H.dims[0]) # Dimension of the Hilbert space
    assert(D / 2 > num)

    # Check Hermicity
    if not H.isherm:
        print("Warning: Hamiltonian is not Hermitian. The eigenvalues may not be correct.")

    # Get mean and standard deviation of the eigenvalues of the Hamiltonian
    mean = H.tr() / D
    std = np.sqrt((H * H).tr() / D - mean ** 2)

    # First step. Get the eigenvalue with the largest absolute value
    shift = mean + 4 * std
    e1_all, v1_all = power_method(H - shift, 1, maxitr = maxitr, tol = tol)
    e1_all += shift

    # Second step. Get the largest or smallest eigenvalue
    shift = e1_all[0]
    e2_all_list = []
    for i in range(rep):
        e2_all, v2_all = power_method(H - shift, num, maxitr = maxitr, tol = tol)
        e2_all_list.append(e2_all)
    if cautious:
        relative_error = np.max(np.std(e2_all_list, axis = 0) / np.abs(np.mean(e2_all_list, axis = 0)))
        if relative_error > 2 * tol:
            print("Warning: The eigenvalues are not consistent. Relative error:", relative_error)
    e2_all = np.mean(e2_all_list, axis = 0)
    e2_all += shift

    # Third step. Get the smallest or largest eigenvalue
    shift = e2_all[0]
    e3_all_list = []
    for i in range(rep):
        e3_all, v3_all = power_method(H - shift, num, maxitr = maxitr, tol = tol)
        e3_all_list.append(e3_all)
    if cautious:
        relative_error = np.max(np.std(e3_all_list, axis = 0) / np.abs(np.mean(e3_all_list, axis = 0)))
        if relative_error > 2 * tol:
            print("Warning: The eigenvalues are not consistent. Relative error:", relative_error)
    e3_all = np.mean(e3_all_list, axis = 0)
    e3_all += shift

    if e2_all[0] > e3_all[0]:
        e_return = np.concatenate([e3_all[:], e2_all[::-1]])
        v_return = np.concatenate([v3_all[:,:], v2_all[::-1,:]])
    else:
        e_return = np.concatenate([e2_all[:], e3_all[::-1]])
        v_return = np.concatenate([v2_all[:,:], v3_all[::-1,:]])

    return e_return, v_return



# ===========================================================
# =============== Read files generated by C++ ===============
# ===========================================================


def read_PauliAlg(filename):    
    """
    Reads an array of C++ PauliAlg objects from a file generated by the C++ `PauliAlg::write_to_file` function. 
    Returns a list of QuTiP operators. Beware that it returns a list even if the file contains only one operator.
    
    The file consists of blocks separated by an empty line.
    
    Each block represents a PauliAlg object and contains the following lines:
    - The first line contains the number of sites in the form `# N = [N]`
    - The second and later lines contain the terms in the format: 
    `[coefficient.real()] [coefficient.imag()] [Pauli] [Pauli] [Pauli] ...`
    - Example of a line: `1.123456789012345e+01 -9.678901234567890e+00 X3 Y5 Z8`

    As a special case, a block represents the zero operator if it contains only the number of sites.
    """
    
    f = open(filename, 'r')
    A_list = []

    while True:
        # ===== Skip empty lines (if any) =====
        while True:
            line = f.readline()
            if line != '\n': # Non-empty line (the beginning of a block or the end of file)
                break
        
        # At the end of this loop, line contains a non-empty line.

        # ===== Check if the file has ended =====
        if line == '': # End of file
            break

        # ===== Determine the number of sites ===== 
        # The first non-empty line of a block should contain the number of sites
        if line[:6] != '$ N = ':
            raise ValueError('Invalid input file format: the first line of a block should start with "$ N = "')
        N = int(line[6:].strip())
        if N < 1:
            raise ValueError('Invalid number of sites:', N)

        # Prepare X, Y, and Z operators
        X = [qt.tensor([qt.qeye(2)]*i + [qt.sigmax()] + [qt.qeye(2)]*(N-i-1)) for i in range(N)]
        Y = [qt.tensor([qt.qeye(2)]*i + [qt.sigmay()] + [qt.qeye(2)]*(N-i-1)) for i in range(N)]
        Z = [qt.tensor([qt.qeye(2)]*i + [qt.sigmaz()] + [qt.qeye(2)]*(N-i-1)) for i in range(N)]
        
        # Prepare the identity operator
        Id = qt.tensor([qt.qeye(2)]*N)

        # Prepare the zero operator
        Zero = qt.tensor([qt.qzero(2)]*N)
        
        # ===== Reconstruct the operator =====
        A = Zero.copy()

        # If the block contains no terms, the following loop ends at the very beggining, remaining A as Zero.
        while(True):
            line = f.readline()
            if line == '': # End of file
                break
            elif line == '\n': # End of block
                break

            # Split the line
            line_split = line.split()
            if len(line_split) <= 2:
                raise ValueError('Invalid input file format: a line contains less than 3 elements')
            
            # Read the first two elements
            coeff = float(line_split[0]) + 1j*float(line_split[1])
            term = coeff * Id

            # Read the remaining elements
            if line_split[2] == 'I': # Identity operator
                pass 
            else:
                for op in line_split[2:]:
                    # op is a string of the form 'X0', 'Y13', 'Z25', etc.
                    kind = op[0]
                    site = int(op[1:])

                    if site >= N:
                        raise ValueError(f'Invalid input file format: site index ({site}) exceeds the total number of sites ({N})')
                    
                    if kind == 'X':
                        term *= X[site]
                    elif kind == 'Y':
                        term *= Y[site]
                    elif kind == 'Z':
                        term *= Z[site]
                    else:
                        raise ValueError(f"Invalid input file format: an operator should be 'X', 'Y', or 'Z', but got '{kind}'")
        
            A += term

        A_list.append(A)
    
    f.close()

    return A_list



def read_driving_coefficient(filename):
    """
    Reads the output of the C++ code that calculates the CDdriving coefficients.

    The file should have the following format:
    ```    
    # Parameters
    $ K = [K]
    $ N = [N]
    $ M = [M]
    $ Gamma = [Gamma]

    # lambda f1 f2 ... a1 a2 ... E 
    [lambda_series[0]] [H_coeff_series[0][0]] [H_coeff_series[0][1]] ... [CD_coeff_series[0][0]] [CD_coeff_series[0][1]] ... [E_series[0]]
    [lambda_series[1]] [H_coeff_series[1][0]] [H_coeff_series[1][1]] ... [CD_coeff_series[1][0]] [CD_coeff_series[1][1]] ... [E_series[1]]
    [lambda_series[2]] [H_coeff_series[2][0]] [H_coeff_series[2][1]] ... [CD_coeff_series[2][0]] [CD_coeff_series[2][1]] ... [E_series[2]]
    ...
    ```
    where `[x]` denotes the value of the variable `x`.

    Returns a 1d numpy array `lambda_series`, a 2d numpy array `H_coeff_series`, 
    a 2d numpy array `CD_coeff_series`, and a 1d numpy array `E_series`.
    `H_coeff_series[i,gamma]`, `CD_coeff_series[i,mu]`, and `E_series[i]` are 
    the corresponding coefficients at `lambda_series[i]`.
    """

    f = open(filename, 'r')

    # ===== Skip empty lines (if any) =====
    while True:
        line = f.readline()
        if line != '\n': # Non-empty line (the beginning of a block or the end of file)
            break
    
    # ===== Read the parameters =====
    if line.strip() != '# Parameters':
        raise ValueError('The first line of the file should be "# Parameters"')

    line = f.readline()
    if line.strip()[:6] != '$ K = ':
        raise ValueError('The second line of the file should start with "$ K ="')
    K = int(line.split()[3])

    line = f.readline()
    if line.strip()[:6] != '$ N = ':
        raise ValueError('The third line of the file should start with "$ N ="')
    N = int(line.split()[3])

    line = f.readline()
    if line.strip()[:6] != '$ M = ':
        raise ValueError('The fourth line of the file should start with "$ M = "')
    M = int(line.split()[3])

    line = f.readline()
    if line.strip()[:10] != '$ Gamma = ':
        raise ValueError('The fifth line of the file should start with "$ Gamma ="')
    Gamma = int(line.split()[3])

    params = {'K': K, 'N': N, 'M': M, 'Gamma': Gamma}

    # ===== Skip empty lines (if any) =====
    while True:
        line = f.readline()
        if line != '\n': # Non-empty line (the beginning of a block or the end of file)
            break

    # ===== Check the header of the table =====
    header_guess = 'lambda\t'
    for gamma in range(Gamma):
        header_guess += f'f{gamma+1}\t'
    for mu in range(M):
        header_guess += f'a{mu+1}\t'
    header_guess += 'E'

    if line.strip() != header_guess:
        print("Warning: the header in the file does not match the expected format:")
        print("Expected: " + header_guess)
        print("Got:      " + line.strip())
        print("This may cause a wrong result.")

    length_guess = 1 + Gamma + M + 1

    # ===== Read the remaining lines =====
    lines = f.readlines()
    f.close()

    lambda_series = []
    H_coeff_series = []
    CD_coeff_series = []
    E_series = []
    
    for lineNo in range(len(lines)):
        line_split = list(map(float, lines[lineNo].split()))
        
        # Chcek number of elements
        if(len(line_split) != length_guess):
            raise ValueError(f'Line {lineNo+1} of the numerical table contains invalid number of elements. Got: {len(line_split)}, Expected: {length_guess}')
        
        # Read
        lambda_series.append(line_split[0])
        H_coeff_series.append(line_split[1:Gamma+1])
        CD_coeff_series.append(line_split[Gamma+1:Gamma+M+1])
        E_series.append(line_split[-1])
    
    return params, np.array(lambda_series), np.array(H_coeff_series), np.array(CD_coeff_series), np.array(E_series)


