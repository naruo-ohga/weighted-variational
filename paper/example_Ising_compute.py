 
import numpy as np
import qutip as qt
import weighted_variational_utility
import sys

 
# =========================================
# =============== Load data ===============
# =========================================

# Parameters from command line
args = sys.argv

Hamiltonian_name = args[1]
driving_name = args[2]
seed_idx = int(args[3])
N = int(args[4])
is_eigendecomposition = ('eigen' in args)
is_resenergy = ('resenergy' in args)

print("Eigen-decomposition:", is_eigendecomposition)
print("Residual energy:    ", is_resenergy)

# Other parameters
K_list = [0, 1, 2, 3, 4, 5]
data_list = [] # data_list[i] stores the data for K = K_list[i]
system_name = f'{Hamiltonian_name}-seed{seed_idx}{driving_name}' 
folder_name = 'example_Ising_data_' + system_name

if seed_idx < 10:
    tmax_list = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
else:
    tmax_list = [0.1, 0.05, 0.02, 0.01]


# Load Hamiltonian 
H_basis = weighted_variational_utility.read_PauliAlg(f'{folder_name}/example_Ising_H_basis_N={N}.txt')
CD_basis = weighted_variational_utility.read_PauliAlg(f'{folder_name}/example_Ising_CD_basis_N={N}.txt')

# Load coefficients
for K in K_list:
    params, lambda_series, H_coeff_series, CD_coeff_series, E_series = weighted_variational_utility.read_driving_coefficient(f'{folder_name}/example_Ising_coefficients_N={N}_K={K}.txt')
    print(params)
    assert(params['K'] == K)

    data = {'K': K, 
            'params': params, 
            'lambda_series': lambda_series, 
            'H_coeff_series': H_coeff_series, 
            'CD_coeff_series': CD_coeff_series, 
            'E_series': E_series}
    data_list.append(data)

# Check if lambda_series and H_coeff_series are the same for all K
for data in data_list[1:]:
    assert(np.array_equal(data['lambda_series'], data_list[0]['lambda_series']))
    assert(np.array_equal(data['H_coeff_series'], data_list[0]['H_coeff_series']))
lambda_series = data_list[0]['lambda_series']
assert(np.array_equal(lambda_series, np.sort(lambda_series))) # Ensure lambda_series is sorted

# Prepare qutip object of the Hamiltonian (as a function of lambda)
H_list_lambda = []
for gamma in range(len(H_basis)):
    H_list_lambda.append([H_basis[gamma], qt.coefficient(H_coeff_series[:, gamma], tlist = lambda_series)])
H_QobjEvo_lambda = qt.QobjEvo(H_list_lambda)

# The values of lambda for which the quantum state is evaluated
lambda_series_eval = np.linspace(lambda_series[0], lambda_series[-1], 41, endpoint=True)
 

# ===================================================
# =============== Eigen decomposition ===============
# ===================================================

# Number of eigenenergies to calculate
num_eig = 4 
assert(num_eig % 2 == 0)

# Prepare the header of the output file
header_str = " "
for j in range(int(num_eig/2)):
    header_str += str(j+1) + " "
for j in range(int(num_eig/2)-1, 0, -1):
    header_str += "D-" + str(j) + " "
header_str += "D"


if is_eigendecomposition:
    eigenenergy_series = []
    eigenstate_series = []

    for lam in lambda_series_eval:
        print("lambda =", lam)
        H_lambda = H_QobjEvo_lambda(lam)
        E_all, phi_all = weighted_variational_utility.extremal_eigs(H_lambda, int(num_eig/2), cautious=False)

        eigenenergy_series.append(E_all)
        eigenstate_series.append(phi_all)
        
    eigenenergy_series = np.array(eigenenergy_series)
    eigenstate_series = np.array(eigenstate_series)

    # Save to file
    np.save(f'{folder_name}/example_Ising_eigenenergy_N={N}.npy', eigenenergy_series, allow_pickle=False)
    np.save(f'{folder_name}/example_Ising_eigenstate_N={N}.npy', eigenstate_series, allow_pickle=False)
    np.savetxt(f'{folder_name}/example_Ising_eigenenergy_N={N}.txt', 
               np.concatenate([lambda_series_eval[:,None], eigenenergy_series], axis=1),
               header = "lambda" + header_str)

else:
    # Load from file
    eigenenergy_series = np.load(f'{folder_name}/example_Ising_eigenenergy_N={N}.npy')
    eigenstate_series = np.load(f'{folder_name}/example_Ising_eigenstate_N={N}.npy')


# Prepare the initial state (the ground state of the initial Hamiltonian)
H_init = H_QobjEvo_lambda(lambda_series_eval[0])
ket_dims = [H_init.dims[0], [1] * len(H_init.dims[0])] # Qutip-friendly representation of the demension of a ket
E0, psi0 = weighted_variational_utility.extremal_eigs(H_init, 1)
E0 = E0[0]
psi0 = qt.Qobj(psi0[0], dims=ket_dims)


 
# ==============================================
# =============== Time evolution ===============
# ==============================================

for tmax in tmax_list:
    # Schedule of the driving 
    def schedule(t): # Returns the value of lambda at time t
        return t/tmax
    def schedule_inv(lam): # The inverse function (currently only monotonic schedules are supported)
        return lam*tmax
    def schedule_derivative(t): # The time derivative of lambda
        return 1/tmax

    # Timepoints 
    time_series = np.array([schedule_inv(lam) for lam in lambda_series])
    time_series_eval = np.array([schedule_inv(lam) for lam in lambda_series_eval])

    # Derivative of the schedule
    lambda_deriv_series = np.array([schedule_derivative(t) for t in time_series])

    # Prepare qutip object of the Hamiltonian (as a function of time)
    H_list = []
    for gamma in range(len(H_basis)):
        H_list.append([H_basis[gamma], qt.coefficient(H_coeff_series[:, gamma], tlist = time_series)])
    H_QobjEvo = qt.QobjEvo(H_list)


    for data in data_list:
        # Extract data
        K = data['K']
        CD_coeff_series = data["CD_coeff_series"]

        # Prepare Qutip object of the CD Hamiltonian
        CD_list = []
        for mu in range(len(CD_basis)):
            CD_list.append([CD_basis[mu], qt.coefficient(lambda_deriv_series * CD_coeff_series[:, mu], tlist = time_series)])
        CD_QobjEvo = qt.QobjEvo(CD_list)
        Total_QobjEvo = H_QobjEvo + CD_QobjEvo
        
        # Calculate time evolution
        print(f"Time evolution: K = {K}, tmax = {tmax}")
        sesolve_result = qt.sesolve(Total_QobjEvo, psi0, time_series_eval, options={"progress_bar": False})
        psi_series = sesolve_result.states
    

        # Calculate overlap between the time-evolved state and the eigenstates 
        overlap_series = []
        for i in range(len(time_series_eval)):
            overlap = []
            psi = psi_series[i]

            for n in range(num_eig):
                phi_qutip = qt.Qobj(eigenstate_series[i,n,:], dims=ket_dims)
                overlap.append(np.abs(psi.overlap(phi_qutip)) ** 2)
                # overlap.append(np.abs(np.dot(psi.full()[:,0].conj(), eigenstate_series[ti,n,:])) ** 2)
            overlap_series.append(overlap)
        overlap_series = np.array(overlap_series)

        np.savetxt(f'{folder_name}/example_Ising_overlap_N={N}_K={K}_tmax={tmax}.txt', 
                   np.concatenate([time_series_eval[:,None], overlap_series], axis=1),
                   header = "time" + header_str)

        
        # Calculate residual energy
        if not is_resenergy:
            continue
        
        resenergy_series = []
        for i in range(len(time_series_eval)):
            t = time_series_eval[i]
            print("Time:", t)
            psi = psi_series[i]
            H_t = H_QobjEvo(t)

            normalization = psi.dag() * psi
            if np.abs(normalization - 1) > 1e-6:
                print("Normalization error:", normalization)
            
            resenergy = psi.dag() * H_t * psi
            if np.abs(resenergy.imag) > 1e-6:
                print("Residual energy imaginary:", resenergy)
            resenergy_series.append(resenergy.real)
        resenergy_series = np.array(resenergy_series)

        np.savetxt(f'{folder_name}/example_Ising_resenergy_N={N}_K={K}_tmax={tmax}.txt',
                    np.concatenate([time_series_eval[:,None], overlap_series[:,None]], axis=1),
                    header="time resenergy")