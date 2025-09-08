import example_Ising_loader as loader
import weighted_variational_utility
import numpy as np
import pandas as pd


# ===== Parameters =====
parent_folder = 'data-main/'
K_list = loader.K_list
driving_name = ''
N = 12
D = 2**N
i = 100 # index of lambda

# We choose representative setups with the fidelity and gain close to median
setup_list = [('ferromagnetic', 0), ('Gaussian', 9)]


for Hamiltonian_name, seed_idx in setup_list:
    print(f"Processing {Hamiltonian_name} with seed_idx = {seed_idx}")

    # ===== Load data =====
    folder_name = loader.get_folder_name(Hamiltonian_name, driving_name, seed_idx, parent_folder)

    lambda_series, H_QobjEvo_lambda = loader.load_single_Hamiltonian(Hamiltonian_name=Hamiltonian_name,
                                                                    driving_name=driving_name,
                                                                    seed_idx=seed_idx,
                                                                    N=N,
                                                                    parent_folder=parent_folder)

    lambda_series2, CD_coeff_series_all, E_series_all = loader.load_driving_coefficients(Hamiltonian_name=Hamiltonian_name,
                                                                                        driving_name=driving_name,
                                                                                        seed_idx=seed_idx,
                                                                                        N=N,
                                                                                        parent_folder=parent_folder)

    assert(np.array_equal(lambda_series, lambda_series2))

    CD_basis = weighted_variational_utility.read_PauliAlg(f'{folder_name}/example_Ising_CD_basis_N={N}.txt')
    M = len(CD_basis)


    # ===== Extract quantities at i =====
    energy, evecs = H_QobjEvo_lambda(lambda_series[i]).eigenstates()
    assert(np.array_equal(energy, np.sort(energy))) # Ensure energy is sorted

    lam_after = lambda_series[i+1]
    lam_before = lambda_series[i-1]
    dlambda_H = (H_QobjEvo_lambda(lam_after) - H_QobjEvo_lambda(lam_before)) / (lam_after - lam_before)

    CD_coeff = CD_coeff_series_all[:,i]
    E_shift = E_series_all[:,i]


    # ===== Compute matrix elements of A and Phi =====
    # In the following, matrices are in the basis of the eigenstates of the Hamiltonian

    # Elements of the basis of CD driving
    # A[mu, m, n] = <phi_m | A_mu | phi_n>
    print("A...", end = " ", flush = True)
    A_elem = np.zeros((M, D, D), dtype = complex)

    for n in range(D):
        if n % 20 == 0: print(n, end = " ", flush = True)
        for mu in range(M):
            CD_times_evec = CD_basis[mu] * evecs[n]
        
            for m in range(D):
                A_elem[mu, m, n] = evecs[m].dag() * CD_times_evec
    print("")

    # Elements of the adiabatic gauge potential (m != n)
    # Phi[m,n] = i <phi_m | partial_lambda phi_n> 
    #          = i <phi_m | (partial_lambda H) |phi_n> / (E_n - E_m) 
    print("Phi...", end = " ", flush = True)
    Phi = np.zeros((D, D), dtype = complex)
    
    for n in range(D):
        if n % 20 == 0: print(n, end = " ", flush = True)
        dlambda_H_times_vec = dlambda_H * evecs[n]

        for m in range(D):
            if m != n:
                if np.abs(energy[n] - energy[m]) < 1e-10:
                        print("Eigenvalue very close to degeneracy: Diff =", np.abs(energy[n] - energy[m]))
                Phi[m, n] = 1.0j * evecs[m].dag() * dlambda_H_times_vec / (energy[n] - energy[m])
    print("")


    # ===== Compute partial action =====
    print("partial action...", end = " ", flush = True)
    df = pd.DataFrame()

    # To recover the total action for checking
    Q_partial_sum = np.zeros((len(K_list), M, M))
    r_partial_sum = np.zeros((len(K_list), M))

    for n in range(D):
        if n % 20 == 0: print(n, end = " ", flush = True)

        # ===== Elements of the Q and r =====
        # Expanding the elements of the action:
        # | [V - Phi]_{mn} |^2 = \sum_{mu,nu} Q_elem[mu,nu,m,n] a_mu a_nu - 2 \sum_mu r_elem[mu,m,n] a_mu + |Phi[m,n]|^2

        Q_elem = np.zeros((M, M, D), dtype = float) # Q_elem[mu, nu, m, n=n] 
        r_elem = np.zeros((M, D), dtype = float)    # r_elem[mu, m, n=n]

        for mu in range(M):
            r_elem[mu,:] = (Phi[:,n] * A_elem[mu,n,:]).real

        for mu in range(M):
            for nu in range(M):
                Q_elem[mu,nu,:] = (A_elem[mu,:,n] * A_elem[nu,n,:]).real


        # ===== Minimize the ideal action if n==0 =====
        if n == 0:
            Q_ideal = np.sum(Q_elem[:,:,1:], axis=2)
            r_ideal = np.sum(r_elem[:,1:], axis=1)
            a_ideal = np.linalg.solve(Q_ideal, r_ideal)

            df = loader.append_row(df, {'kind': 'ideal', 'K': np.nan, 'n': np.nan, 'energy': np.nan} | {f'a_{mu}': a_ideal[mu] for mu in range(M)})


        # ===== Minimize the partial action =====
        for K_idx in range(1, len(K_list)):
            K = K_list[K_idx]
            e_shifted = energy - E_shift[K_idx]
            weight_state = e_shifted ** (2*K - 2)
            
            # We omit the denominator w_n, since it is constant for given n.
            # weight_elem[m], chi[m], mask[m] are of shape = (D,) for fixed n.
            weight_elem = ((e_shifted**K) - (e_shifted[n]**K)) ** 2
            chi = np.where(weight_state == weight_state[n], 0.5, 1.0)
            mask = np.where(weight_state <= weight_state[n], 1.0, 0.0)

            coeff = weight_elem * chi * mask      

            # compute partial action
            Q_partial_n = np.sum(Q_elem[:,:,:] * coeff[None,None,:], axis = 2) # sum over m
            r_partial_n = np.sum(r_elem[:,:] * coeff[None,:], axis = 1) # sum over m

            singular_n = []
            try:
                a_partial_n = np.linalg.solve(Q_partial_n, r_partial_n)
                df = loader.append_row(df, {'kind': 'partial', 'K': K, 'n': n, 'energy': energy[n]} | {f'a_{mu}': a_partial_n[mu] for mu in range(M)})
            except np.linalg.LinAlgError:
                singular_n.append(n)
                df = loader.append_row(df, {'kind': 'partial', 'K': K, 'n': n, 'energy': energy[n]} | {f'a_{mu}': np.nan for mu in range(M)})

            if len(singular_n) >= 2:
                print(f"Warning: Singular at n = {singular_n} at seed_idx = {seed_idx}, K = {K}, i = {i}")

            Q_partial_sum[K_idx,:,:] += Q_partial_n
            r_partial_sum[K_idx,:] += r_partial_n

    print("")


    # ===== Check consistency =====
    for K_idx in range(1, len(K_list)):
        K = K_list[K_idx]
        CD_coeff_K = CD_coeff[K_idx]

        a_recover = np.linalg.solve(Q_partial_sum[K_idx,:,:], r_partial_sum[K_idx,:])
        if not np.isclose(a_recover, CD_coeff_K).all():
            print(f"Warning: a and CD_coeff do not match at seed_idx: {seed_idx}, K: {K}, i: {i}")
            print(a_recover)
            print(CD_coeff_K)

        # Save CD_coeff_K for convenience
        df = loader.append_row(df, {'kind': 'total', 'K': K, 'n': np.nan, 'energy': np.nan} | {f'a_{mu}': CD_coeff_K[mu] for mu in range(M)})


    # ===== Compute deviation from ideal action =====
    df['deviation'] = df.apply(lambda row: np.sum((row[[f'a_{mu}' for mu in range(M)]].values - a_ideal)**2), axis='columns')

    # ===== Save the results =====
    df = df.sort_values(by = ['kind', 'K', 'n'])
    df.to_csv(f'{folder_name}/example_Ising_partial-action_N={N}_i={i}.txt', sep = "\t", index = False, na_rep='NaN')