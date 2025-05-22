import numpy as np
import qutip as qt
import weighted_variational_utility



def extremal_eigs_test(H):
    # H should be given by some other procedure.

    num_eig = 3
    # Power method
    e_power, v_power = weighted_variational_utility.extremal_eigs(H, num_eig)

    # Qutip
    e1, v1 = H.eigenstates(sort='low', eigvals=num_eig)
    e2, v2 = H.eigenstates(sort='high', eigvals=num_eig)
    e_qutip = np.concatenate([e1, e2[::-1]])
    v_qutip = np.zeros_like(v_power)
    for i in range(num_eig):
        v_qutip[i,:] = v1[i].full().flatten()
    for i in range(num_eig):
        v_qutip[2 * num_eig - i - 1,:] = v2[i].full().flatten()

    # Compare
    e_error = np.zeros(num_eig * 2)
    v_overlap = np.zeros(num_eig * 2)

    for i in range(num_eig * 2):
        e_error[i] = np.abs(e_power[i] - e_qutip[i]) / np.abs(e_qutip[i])
        v_overlap[i] = np.abs(np.dot(v_power[i], v_qutip[i].conj()))

    print("Eigenvalues (Power method):\n", e_power)
    print("Eigenvalues (Qutip):\n", e_qutip)
    print("Relative error in eigenvalues:\n", e_error)
    print("Overlap of eigenstates:\n", v_overlap)
    



def read_PauliAlg_test():
    # Read the output file of the pauli_alg_test.cpp.
    # The file contains the following two Pauli algebras:
    # PauliAlg D = 0.1 * X[0] - 0.2 * Y[0] * Y[1] * Z[2];
    # PauliAlg F = complex<double>(0.0, 3.0) * Id + 1e-40 * X[0] + complex<double>(1.4e5, 2.8e10) * Y[1] - 0.01 * Z[2] * Y[3] * X[12];

    A_list = weighted_variational_utility.read_PauliAlg('pauli_alg_test_output.txt')
    assert(len(A_list) == 2)
    D, F = A_list

    # Build the expected data
    I = qt.qeye(2)
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()

    D_expected = 0.1 * qt.tensor([X] + [I] * 14)
    D_expected -= 0.2 * qt.tensor([Y, Y, Z] + [I] * 12)

    F_expected = 3.0j * qt.tensor([I] * 15) 
    F_expected += 1e-40 * qt.tensor([X] + [I] * 14) 
    F_expected += (1.4e5 + 2.8e10j) * qt.tensor([I, Y] + [I] * 13) 
    F_expected -= 0.01 * qt.tensor([I, I, Z, Y, I, I, I, I, I, I, I, I, X, I, I])

    # Compare the expected data with the data from the file
    # This takes a bit of time. (~ 1 minite)
    print("Comparing the expected data with the data from the file...")
    D_difference = D - D_expected
    F_difference = F - F_expected
    D_norm = np.linalg.norm(D_difference.full())
    F_norm = np.linalg.norm(F_difference.full())
    print(D_norm)
    print(F_norm)
    assert(D_norm == 0)
    assert(F_norm == 0)

    print("All tests passed!")



if __name__ == '__main__':
    read_PauliAlg_test()
