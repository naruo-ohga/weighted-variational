import numpy as np
import pandas as pd
import qutip as qt
import weighted_variational_utility


# ===== Helper for pandas =====
def append_row(df, dictionary): 
    names = list(dictionary.keys())
    values = list(dictionary.values())
    sf = pd.DataFrame([values], columns=names)
    df_new = pd.concat([df, sf], ignore_index=True)

    return df_new


# ===== Helper for loading strings =====
def get_folder_name(Hamiltonian_name, driving_name, seed_idx, parent_folder = ''):
    system_name = f'{Hamiltonian_name}-seed{seed_idx}{driving_name}' 
    folder_name = parent_folder + 'example_Ising_data_' + system_name

    return folder_name


def read_header(filename):
    with open(filename, 'r') as f:
        header = f.readline().strip()
    return header



# ===== Data loaders =====

# Hardcoded parameter
K_list = [0, 1, 2, 3, 4, 5]


def load_energygap(Hamiltonian_name_list, driving_name_list, seed_list, N_list, parent_folder = ''):
    df = pd.DataFrame([])

    for Hamiltonian_name in Hamiltonian_name_list:
        for driving_name in driving_name_list:
            for N in N_list:
                for seed_idx in seed_list:    
                    folder_name = get_folder_name(Hamiltonian_name, driving_name, seed_idx, parent_folder)
                    
                    header = read_header(f'{folder_name}/example_Ising_eigenenergy_N={N}.txt')
                    if header != '# lambda 1 2 D-1 D':
                        print(f'Invalid header: "{header}" for {folder_name}/example_Ising_eigenenergy_N={N}.txt')
                    eigenenergy_series = np.loadtxt(f'{folder_name}/example_Ising_eigenenergy_N={N}.txt')

                    # Calculate energy gap
                    energy_gap = eigenenergy_series[:,2] - eigenenergy_series[:,1]
                    df = append_row(df, {'Hamiltonian': Hamiltonian_name, 
                                         'driving': driving_name, 
                                         'N': N, 
                                         'seed': seed_idx, 
                                         'gap_min': np.min(energy_gap), 
                                         'gap_absmin': np.min(np.abs(energy_gap)), 
                                         'gap_final': energy_gap[-1], 
                                         'gap_max': np.max(energy_gap)})

    return df



def load_final_overlap(Hamiltonian_name_list, driving_name_list, seed_list, N_list, tmax_list, parent_folder = ''):
    df = pd.DataFrame([])

    for Hamiltonian_name in Hamiltonian_name_list:
        for driving_name in driving_name_list:
            for N in N_list:
                for seed_idx in seed_list:    
                    for tmax in tmax_list:
                        folder_name = get_folder_name(Hamiltonian_name, driving_name, seed_idx, parent_folder)
                        
                        final_overlap = []
                        for K in K_list:
                            header = read_header(f'{folder_name}/example_Ising_overlap_N={N}_K={K}_tmax={tmax}.txt')
                            if header != '# time 1 2 D-1 D':
                                print(f'Invalid header: "{header}" for {folder_name}/example_Ising_overlap_N={N}_K={K}_tmax={tmax}.txt')
                            overlap_series = np.loadtxt(f'{folder_name}/example_Ising_overlap_N={N}_K={K}_tmax={tmax}.txt')
                            final_overlap.append(overlap_series[-1,1])

                        df = append_row(df, {'Hamiltonian': Hamiltonian_name, 'driving': driving_name, 'N': N, 'seed': seed_idx, 'tmax': tmax} | dict(zip([f"K={K}" for K in K_list], final_overlap)))
    
    return df



def load_single_data(Hamiltonian_name, driving_name, seed_idx, N, tmax, parent_folder = ''):
    folder_name = get_folder_name(Hamiltonian_name, driving_name, seed_idx, parent_folder)

    time_series_eval_all = []
    overlap_series_all = []

    # Load overlap
    for K in K_list:
        header = read_header(f'{folder_name}/example_Ising_overlap_N={N}_K={K}_tmax={tmax}.txt')
        if header != '# time 1 2 D-1 D':
            print(f'Invalid header: "{header}" for {folder_name}/example_Ising_overlap_N={N}_K={K}_tmax={tmax}.txt')
        overlap_series = np.loadtxt(f'{folder_name}/example_Ising_overlap_N={N}_K={K}_tmax={tmax}.txt')
        time_series_eval = overlap_series[:,0]
        overlap_series = overlap_series[:,1:]

        time_series_eval_all.append(time_series_eval)
        overlap_series_all.append(overlap_series)

    # Check if time_series_eval is the same for all K
    for idx in range(1, len(K_list)):
        assert(np.array_equal(time_series_eval_all[0], time_series_eval_all[idx]))
    time_series_eval = time_series_eval_all[0]

    # Load eigenenergy
    header = read_header(f'{folder_name}/example_Ising_eigenenergy_N={N}.txt')
    if header != '# lambda 1 2 D-1 D':
        print(f'Invalid header: "{header}" for {folder_name}/example_Ising_overlap_N={N}_K={K}_tmax={tmax}.txt')    
    eigenenergy_series = np.loadtxt(f'{folder_name}/example_Ising_eigenenergy_N={N}.txt')
    eigenenergy_series = eigenenergy_series[:,1:]
    assert(len(time_series_eval) == len(eigenenergy_series))
    
    return eigenenergy_series, time_series_eval, np.array(overlap_series_all)



def load_single_Hamiltonian(Hamiltonian_name, driving_name, seed_idx, N, parent_folder = ''):
    folder_name = get_folder_name(Hamiltonian_name, driving_name, seed_idx, parent_folder)

    data_list = [] # data_list[i] stores the data for K = K_list[i]
    
    # Load Hamiltonian 
    H_basis = weighted_variational_utility.read_PauliAlg(f'{folder_name}/example_Ising_H_basis_N={N}.txt')

    # Load coefficients
    for K in K_list:
        params, lambda_series, H_coeff_series, CD_coeff_series, E_series = weighted_variational_utility.read_driving_coefficient(f'{folder_name}/example_Ising_coefficients_N={N}_K={K}.txt')
        assert(params['K'] == K)

        data = {'K': K, 
                'lambda_series': lambda_series, 
                'H_coeff_series': H_coeff_series}
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
    
    return lambda_series, H_QobjEvo_lambda



def load_driving_coefficients(Hamiltonian_name, driving_name, seed_idx, N, parent_folder = ''):
    folder_name = get_folder_name(Hamiltonian_name, driving_name, seed_idx, parent_folder)

    # Load driving coefficients
    lambda_series_all = []
    CD_coeff_series_all = []
    E_series_all = []
    for idx in range(len(K_list)):
        K = K_list[idx]
        params, lambda_series, H_coeff_series, CD_coeff_series, E_series = weighted_variational_utility.read_driving_coefficient(f'{folder_name}/example_Ising_coefficients_N={N}_K={K}.txt')
        assert(params['K'] == K)
        
        lambda_series_all.append(lambda_series)
        CD_coeff_series_all.append(CD_coeff_series)
        E_series_all.append(E_series)

    # Check if lambda_series is the same for all K
    for idx in range(1, len(K_list)):
        assert(np.array_equal(lambda_series_all[0], lambda_series_all[idx]))
    lambda_series = lambda_series_all[0]
    
    return lambda_series, np.array(CD_coeff_series_all), np.array(E_series_all)



def load_Hamiltonian_coefficients(Hamiltonian_name, driving_name, seed_list, N_list, parent_folder = ''):
    coeff_X = []
    coeff_Z = []
    coeff_ZZ = []

    for seed_idx in seed_list:
        for N in N_list:
            folder_name = get_folder_name(Hamiltonian_name, driving_name, seed_idx, parent_folder)

            with open(f"{folder_name}/example_Ising_H_basis_N={N}.txt", 'r') as f:
                for line in f:
                    if line.count("Z") == 2:
                        coeff_ZZ.append(float(line.split()[0]))
                    elif line.count("Z") == 1:
                        coeff_Z.append(float(line.split()[0]))
                    elif line.count("X") == 1:
                        coeff_X.append(float(line.split()[0]))       

    return np.array(coeff_X), np.array(coeff_Z), np.array(coeff_ZZ) 