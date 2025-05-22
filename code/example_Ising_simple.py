import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import qutip as qt
import weighted_variational_utility


# ===== Load problem =====

# Degree of the method
K = 3

# Load Hamiltonian basis from the file and convert it to a list of qutip objects
H_basis = weighted_variational_utility.read_PauliAlg('example_Ising_simple_data/H_basis.txt')

# Load driving coefficients from the file and convert it to a list of qutip objects
CD_basis = weighted_variational_utility.read_PauliAlg('example_Ising_simple_data/CD_basis.txt')

# Load driving coefficients from the file
params, lambda_series, H_coeff_series, CD_coeff_series, E_series = weighted_variational_utility.read_driving_coefficient('example_Ising_simple_data/coefficients_K=' + str(K) + '.txt')
print("Parameters:\n", params)
assert(params['K'] == K)


# ===== Prepare qutip objects =====

# Initial and final time of the protocol
t_ini = 0.0
t_fin = 0.01 

# Schedule of the driving 
def schedule(t): # Returns the value of lambda at time t
    return t/t_fin
def schedule_inv(lam): # The inverse function
    return lam*t_fin
def schedule_derivative(t): # The time derivative of lambda
    return 1/t_fin

# Check that the range of lambda_series and the range [t_ini, t_fin] are consistent
assert(abs(schedule(t_ini) - lambda_series[0]) < 1e-10) 
assert(abs(schedule(t_fin) - lambda_series[-1]) < 1e-10)

# List of timepoints for which the driving coefficients have been calculated by the C++ program
timepoints = np.array([schedule_inv(lam) for lam in lambda_series])

# Prepare the Hamiltonian list object (qutip-friendly format)
# We specify the Hamiltonian by [[Hamiltonian1, coefficient1], [Hamiltonian2, coefficient2], ...].
H_list = []
for gamma in range(len(H_basis)):
    H_list.append([H_basis[gamma], qt.coefficient(H_coeff_series[:, gamma], tlist = timepoints)])

# Prepare the driving operator list object
lambda_deriv_series = np.array([schedule_derivative(t) for t in timepoints]) # time derivative of the schedule (lambda_t)
CD_list = []
for mu in range(len(CD_basis)):
    CD_list.append([CD_basis[mu], qt.coefficient(lambda_deriv_series * CD_coeff_series[:, mu], tlist = timepoints)])

# Convert the Hamiltonian list to the qutip.QobjEvo object (functions of time)
H_QobjEvo = qt.QobjEvo(H_list)         # Hamiltonian part
CD_QobjEvo = qt.QobjEvo(CD_list)       # CD driving part
Total_QobjEvo = H_QobjEvo + CD_QobjEvo # Total Hamiltonian



# ===== Quantum simulation =====

# Prepare the initial state
# Initial state is the ground state of the initial Hamiltonian
H_init = H_QobjEvo(t_ini)
ket_dims = [H_init.dims[0], [1] * len(H_init.dims[0])] # Qutip-friendly representation of the demension of a ket
E0, psi0 = weighted_variational_utility.extremal_eigs(H_init, 1) # Get the lowest/highest eigenenergies and the corresponding eigenstates
psi0 = qt.Qobj(psi0[0], dims=ket_dims) # Convert the ground state to qutip object

# timepoints for which we evaluate the quantum state
num_timepoints_eval = 21 # increase this number if qutip raises an error
timepoints_eval = np.linspace(t_ini, t_fin, num_timepoints_eval, endpoint=True) 

# Time evolution
print("===== Time evolution =====")
sesolve_result = qt.sesolve(Total_QobjEvo, psi0, timepoints_eval, options={"progress_bar": True})
psi_series = sesolve_result.states



# ===== Calculate fidelity =====

# Calculate eigenenergies and eigenstates.
# The (num_eig/2) highest and (num_eig/2) lowest eigenenergies and the corresponding eigenstates are calculated.
# The results are sorted in the ascending order of the eigenenergies.
num_eig = 6 
assert(num_eig % 2 == 0) 

print("===== Eigenenergies and Eigenstates =====")
eigenenergy_series = []
eigenstate_series = []
for t in timepoints_eval:
    print("t =", "{:.4f}".format(t))
    H_lambda = H_QobjEvo(t)   # Hamiltonian at the current value of lambda
    E_all, phi_all = weighted_variational_utility.extremal_eigs(H_lambda, int(num_eig/2), cautious=False) # Get the (num_eig/2) highest and (num_eig/2) lowest eigenenergies and eigenstates

    eigenenergy_series.append(E_all)
    eigenstate_series.append(phi_all)
    
eigenenergy_series = np.array(eigenenergy_series)
eigenstate_series = np.array(eigenstate_series)


# Calculate fidelity (overlap between the time-evolved state and the eigenstates)
overlap_series = []
for i in range(len(timepoints_eval)):
    overlap = []
    psi = psi_series[i] # get the time-evolved state at t = timepoints_eval[i]

    for n in range(num_eig):
        phi_qutip = qt.Qobj(eigenstate_series[i,n,:], dims=ket_dims) # convert the eigenstate to qutip object
        overlap.append(np.abs(psi.overlap(phi_qutip)) ** 2)          # calculate the overlap 
    overlap_series.append(overlap)

overlap_series = np.array(overlap_series)



# ========== Plot the results ==========

print("Final fidelity: ", overlap_series[-1,0])

fig = plt.figure(figsize=(9,3))

# Plot eigenenergies
ax = fig.add_subplot(131)
for j in range(num_eig):
    ax.plot(timepoints_eval, eigenenergy_series[:,j], c='mediumseagreen', lw=1.0)
ax.fill_between(timepoints_eval, eigenenergy_series[:,int(num_eig/2)-1], eigenenergy_series[:,int(num_eig/2)], facecolor='mediumseagreen', edgecolor="None", alpha=.3, zorder=-2) # to indicate that the intermediate eigenenergies are not calculated
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'Energy $\epsilon_n(\lambda_t)$')

# Plot overlap with the groundstate
ax = fig.add_subplot(132)
ax.plot(timepoints_eval, overlap_series[:,0])
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'Fidelity $\vert \langle \phi_1(\lambda_t) \vert \psi^{(K)}(t) \rangle \vert^2$')

# Plot coefficients
ax = fig.add_subplot(133)
cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=np.min(0), vmax=np.max(len(CD_basis))), cmap=mpl.cm.turbo).to_rgba # Prepare the colormap for distinguishing the coefficients
for mu in range(len(CD_basis)):
    ax.plot(timepoints, CD_coeff_series[:,mu], c=cmap(mu), label=r'$\alpha_{' + str(mu+1) + r'}$', lw=1.0, alpha=0.7)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'Driving coefficient $\alpha_{\mu}(\lambda_t)$')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()



# ========== Save the results ==========
# header string for the eigenenergies and overlap
# For example, it will be "time 1 2 3 D-2 D-1 D" when num_eig = 6
header_str = "time "
for j in range(int(num_eig/2)):
    header_str += str(j+1) + "\t"
for j in range(int(num_eig/2)-1, 0, -1):
    header_str += "D-" + str(j) + "\t"
header_str += "D"

# Save the eigenenergies
eigenenergy_data = np.concatenate([timepoints_eval[:,None], eigenenergy_series], axis=1)
np.savetxt('example_Ising_simple_data/eigenenergy_K=' + str(K) + '.txt', eigenenergy_data, header=header_str, delimiter="\t", comments="")

# Save the overlap
overlap_data = np.concatenate([timepoints_eval[:,None], overlap_series], axis=1)
np.savetxt(f'example_Ising_simple_data/overlap_K=' + str(K) + '.txt', overlap_data, header=header_str, delimiter="\t", comments="")

# Save the plot
plt.savefig(f'example_Ising_simple_data/result_K=' + str(K) + '.pdf')

print("\nResults saved in the folder example_Ising_simple_data/")
