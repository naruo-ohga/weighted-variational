import numpy as np
import pandas as pd
import example_Ising_loader as loader


# ===== parameters =====
N = 15 # the following code assumes N=15
i = 100 # index of lambda
Hamiltonian_name_list = ["ferromagnetic", "antiferromagnetic", "Gaussian"]
driving_name_list = ["", "-YZ"]
K_list = loader.K_list

perturb_ratio = 1.02
parent_folder1 = 'data-main' # without perturbation
parent_folder2 = 'data-main_delta=1.02' # with perturbation

columns = ['mu=' + str(mu) for mu in range(100)] # Too long. Automatically shortend


# ===== Reorder basis =====
# labeling in this code:
#   0  1  2  3  4
#   5  6  7  8  9
#  10 11 12 13 14

# ordering used in the C++ code
basis_order_cpp = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), 
               (0, 5), (0, 1), (5, 10), (5, 6), (10, 11), (1, 6), (1, 2), (6, 11), (6, 7), (11, 12), 
               (2, 7), (2, 3), (7, 12), (7, 8), (12, 13), (3, 8), (3, 4), (8, 13), (8, 9), (13, 14), 
               (4, 9), (9, 14)]

# ordering used in the paper
basis_order_paper = [(0,), (5,), (1,), (10,), (6,), (2,), (11,), (7,), (3,), 
                 (12,), (8,), (4,), (13,), (9,), (14,),
                 (0, 5), (0, 1), (5, 10), (5, 6), (1, 6), (1, 2), 
                 (10, 11), (6, 11), (6, 7), (2, 7), (2, 3), 
                 (11, 12), (7, 12), (7, 8), (3, 8), (3, 4), 
                 (12, 13), (8, 13), (8, 9), (4, 9), (13, 14), (9, 14)]

assert(set(basis_order_cpp) == set(basis_order_paper))

idx_order = [basis_order_cpp.index(b) for b in basis_order_paper]
print(np.array(idx_order))


# ===== Compute relative difference =====
for Hamiltonian_name in Hamiltonian_name_list:
    for driving_name in driving_name_list:
        print(f"Processing {Hamiltonian_name}{driving_name}...")
        df = pd.DataFrame()

        for seed_idx in range(0, 100):
            # Load data
            lambda_series1, CD_coeff_series_all1, _ = loader.load_driving_coefficients(Hamiltonian_name=Hamiltonian_name,
                                                                                        driving_name=driving_name,
                                                                                        seed_idx=seed_idx,
                                                                                        N=N,
                                                                                        parent_folder=parent_folder1)

            lambda_series2, CD_coeff_series_all2, _ = loader.load_driving_coefficients(Hamiltonian_name=Hamiltonian_name,
                                                                                        driving_name=driving_name,
                                                                                        seed_idx=seed_idx,
                                                                                        N=N,
                                                                                        parent_folder=parent_folder2)

            assert(np.array_equal(lambda_series1, lambda_series2))
            assert(CD_coeff_series_all1.shape == CD_coeff_series_all2.shape)

            M = CD_coeff_series_all1.shape[2] # number of driving basis 

            # Compute rel_diff
            for K_idx in range(1, len(K_list)):
                K = K_list[K_idx]
                CD_coeff1 = CD_coeff_series_all1[K_idx,i,:]
                CD_coeff2 = CD_coeff_series_all2[K_idx,i,:]

                if (CD_coeff1 * CD_coeff2 < 0).any():
                    print(f"Warning: sign change detected at {np.count_nonzero(CD_coeff1 * CD_coeff2 < 0)} element(s). seed_idx={seed_idx}, K={K}")

                response = np.log(np.abs(CD_coeff1)) - np.log(np.abs(CD_coeff2))
                response /= np.log(perturb_ratio)

                # collect data into a dataframe
                df_K = pd.DataFrame(np.array([CD_coeff1, CD_coeff2, response]).T, columns=['coeff_without_perturb', 'coeff_with_perturb', 'response'])
                df_K["mu_cpp"] = np.arange(M)
                df_K = df_K.reindex(index = idx_order[:M]) # reorder basis
                df_K["mu_paper"] = np.arange(M)
                df_K["seed_idx"] = seed_idx
                df_K["K"] = K
                df_K = df_K[['seed_idx', 'K', 'mu_cpp', 'mu_paper', 'coeff_without_perturb', 'coeff_with_perturb', 'response']]

                df = pd.concat([df, df_K], axis=0, ignore_index=True)

        # Save data
        df.to_csv(f'{parent_folder2}/example_Ising_response_{Hamiltonian_name}{driving_name}_N={N}_i={i}_delta={perturb_ratio}.txt', sep='\t', index=False)

