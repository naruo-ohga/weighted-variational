/** 
 * weighted_variational.h: Implementation of the weighted variational method.
 *
 * Please consult README.md for the usage.
 * 
 * This file contains three classes:
 * - SingleProblem: Computes the driving coefficients for a single value of lambda.
 * - SerialProblem: Computes the traces of operators for treating a lambda-dependent Hamiltonian.
 * - SerialProblemTrace: Loads the output of SerialProblem and computes the driving coefficients
 *   for the entire series of a lambda-dependent Hamiltonian.
 */ 


#ifndef WEIGHTED_VARIATIONAL_H
#define WEIGHTED_VARIATIONAL_H

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <vector>
#include <map>
#include <cmath>
#include <functional>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include "stopwatch.h"
#include "pauli_alg.h"
#include "weighted_variational_utility.h" // includes weighted_variational::utility functions
using paulialg::PauliAlg;


namespace weighted_variational{

    // ========== Global option ==========
    // The size of the chunk for calculating the traces of operators.
    // Smaller chunks reduces the memory usage while larger chunks reduce the computational time.
    // If the program slows down due to memory shortage, try reducing this value.
    constexpr const std::size_t chunk_size = 1 << 20; // 2^20 = 1048576



    // =============================================================================
    // =============== Weighted variational method for single lambda ===============
    // =============================================================================


    /// @brief Weighted variational method for a single lambda.
    /// An instance of this class will hold all the parameters of the problem.
    /// Then the method `compute` will perform the weighted variational method and return the driving coefficients.
    class SingleProblem{
        private:
            int K_; // Order of the method
            int N_; // Number of sites
            int M_; // Number of CD_basis operators
            
            std::vector<PauliAlg> CD_basis_; // basis of the CD driving
            PauliAlg H_;                     // Hamiltonian
            PauliAlg Hderiv_;                // lambda derivative of the Hamiltonian

            std::string E_mode_; // "manual" or "auto"
            double E_manual_;    // Energy shift 

        public:
            // ========== Constructor ==========
            SingleProblem(int K, int N) : K_(K), N_(N), M_(0), CD_basis_(std::vector<PauliAlg>()), H_(PauliAlg::zero(N)), Hderiv_(PauliAlg::zero(N)) {
                if(K < 0) WEIGHTED_VARIATIONAL_ERROR("The order K must be a nonnegative integer.");
                if(N < 1) WEIGHTED_VARIATIONAL_ERROR("The number of sites must be a positive integer.");
            }

            // ========== Setters ==========
            void set_CD_basis(const std::vector<PauliAlg> &CD_basis){
                if(CD_basis.size() < 1) WEIGHTED_VARIATIONAL_ERROR("The CD_basis operators must not be empty.");
                for(auto L : CD_basis){
                    if(L.get_num_sites() != N_) WEIGHTED_VARIATIONAL_ERROR("The number of sites of the CD_basis operators must be N.");
                }
                CD_basis_ = CD_basis;
                M_ = CD_basis.size();
            }

            void set_H(const PauliAlg &H){
                if(H.get_num_sites() != N_) WEIGHTED_VARIATIONAL_ERROR("The number of sites of H must be N.");
                H_ = H;
            }

            void set_Hderiv(const PauliAlg &Hderiv){
                if(Hderiv.get_num_sites() != N_) WEIGHTED_VARIATIONAL_ERROR("The number of sites of Hderiv must be N.");
                Hderiv_ = Hderiv;
            }

            void set_E_auto(){
                E_mode_ = "auto";
            }

            void set_E_manual(double E){
                E_mode_ = "manual";
                E_manual_ = E;
            }

            
            // ========== Solvers ==========
            // We implement two algorithms for the weighted variational method. They should return the same result.

            // 1: The fast-commutator algorithm. 
            // Faster and recommended. This is written in the paper.
            std::pair<std::vector<double>, double> compute(const std::string& filename = "") const &;

            // 2: The K-nested-commutator algorithm.
            // This is not written in the paper. It is slower. 
            // It is used for validation.
            // The implementation is in the weighted_variational_test_algorithm2.h file.
            std::vector<double> compute_algorithm2() const &;
    };




    /// @brief Performs weighted variational method for a single lambda.
    /// @param filename The name of the output file. (optional)
    /// @return The vector of driving coefficients `a`, and the energy shift `E`. 
    /// `a[mu]` corresponds to the operator `CD_basis[mu]`.
    /// If `K = 0`, the vector is all zero (no CD driving), and E = 0.
    /// If the filename is not empty, writes the vector `a` to the file.
    std::pair<std::vector<double>, double> SingleProblem::compute(const std::string& filename) const & {
        // ========== Check ==========
        if(E_mode_ != "auto" && E_mode_ != "manual") WEIGHTED_VARIATIONAL_ERROR("Either set_E_auto or set_E_manual should be called before computation.");
        if(M_ < 1) WEIGHTED_VARIATIONAL_ERROR("The CD_basis operators must not be empty.");

        // ========== Open file ==========
        std::ofstream fout;
        if(!filename.empty()){
            fout.open(filename);
            if(!fout) WEIGHTED_VARIATIONAL_ERROR("Unable to open output file \"" + filename + "\"");
        }


        // ========== Early return if K = 0 (no CD driving) ==========
        if(K_ == 0){
            if(!filename.empty()){
                std::cout << "Saving the driving coefficients to file" << std::endl;
                for(int mu = 0; mu < M_; mu++){
                    fout << "0 ";
                }
                fout.close();
            }
            return {std::vector<double>(M_, 0.0), 0.0};
        }


        // ========== Prepare the table of binomial coefficients ==========
        std::vector<std::vector<int> > binomial = utility::make_binomial_table(std::max(K_, 2*K_ - 2));


        // ========== Compute the powers of H (up to Kth) ==========
        std::cout << "Computing the powers of H" << std::endl;
        StopWatch stopwatch;
        stopwatch.start();
        
        /// H_powers[k] = H^k
        std::vector<PauliAlg> H_powers;

        H_powers.push_back(PauliAlg::identity(N_));
        for(int k = 1; k <= K_; k++){
            H_powers.emplace_back(H_powers[k-1] * H_);
        }
        std::cout << "Time = " << stopwatch.stop() << " sec" << std::endl;


        // ========== Compute the optimal E ==========
        double E;

        if(E_mode_ == "manual"){
            E = E_manual_;
        }
        else if(E_mode_ == "auto"){
            std::cout << "Computing the optimal E" << std::endl;
            stopwatch.start();

            // Compute the trace of the powers of H (up to (2K-1)th) 
            /// omega[k] = trace(H^k)
            std::vector<double> omega(2*K_);

            omega[0] = 1.0;
            for(int k = 1; k <= K_; k++){
                omega[k] = PauliAlg::trace_normalized(H_powers[k]).real();
            }
            for(int k = K_ + 1; k <= 2*K_ - 1; k++){
                omega[k] = PauliAlg::trace_normalized_multiply(H_powers[k-K_], H_powers[K_]).real();
            }

            // Compute the optimal energy shift E

            // If K = 1, the optimal E is undefined.
            if(K_ == 1){
                E = 0.0;
            }
            
            // Solve the optimization problem for K = 2, which is analytically solvable.
            // If K_ >= 3, this result will be used for the initial value for the Newton method.
            if(K_ >= 2){
                std::vector<double> Omega_numerator_2(3, 0.0);
                std::vector<double> Omega_denominator_2(3, 0.0);

                int sign = 1; // To generate (-1)^k
                for(int k = 0; k <= 2; k++){
                    Omega_numerator_2[k] = sign * binomial[2][k] * omega[3-k];
                    Omega_denominator_2[k] = sign * binomial[2][k] * omega[2-k];
                    sign *= -1;
                } 

                auto [E_optimal_2, Omega_optimal_2] = utility::find_extremum(Omega_numerator_2, Omega_denominator_2, 0.0, true);
                E = E_optimal_2;
            }

            // Solve the optimization problem for the specified K
            // using the result of the optimization for K = 2 as the initial guess.
            if(K_ >= 3){
                std::vector<double> Omega_numerator_K(2*K_ - 1, 0.0);
                std::vector<double> Omega_denominator_K(2*K_ - 1, 0.0);

                int sign = 1; // To generate (-1)^k
                for(int k = 0; k <= 2*K_ - 2; k++){
                    Omega_numerator_K[k] = sign * binomial[2*K_ - 2][k] * omega[2*K_ - k - 1];
                    Omega_denominator_K[k] = sign * binomial[2*K_ - 2][k] * omega[2*K_ - k - 2];
                    sign *= -1;
                }

                // Solve the optimization problem
                auto [E_optimal_K, Omega_optimal_K] = utility::find_extremum(Omega_numerator_K, Omega_denominator_K, E, false);
                
                E = E_optimal_K;
            }

            std::cout << "Time = " << stopwatch.stop() << " sec" << std::endl;
        }


        // ========== Compute the powers of (H - E) (up to Kth) ==========
        std::cout << "Computing the powers of (H - E)" << std::endl;
        stopwatch.start();
        
        /// HE_powers[k] = (H - E)^k
        std::vector<PauliAlg> HE_powers = std::move(H_powers); // destroy H_powers and reuse the data to reduce copy cost
        
        for(int k = K_; k >= 0; k--){
            for(int s = 0; s < k; s++){
                HE_powers[k] += binomial[k][s] * HE_powers[s] * std::pow(-E, k - s);
            }
        }
        std::cout << "Time = " << stopwatch.stop() << " sec" << std::endl;
            

        // ========== Compute the lambda derivative of (H - E)^K ==========
        std::cout << "Computing the lambda derivative of (H - E)^K" << std::endl;
        stopwatch.start();

        PauliAlg Hderiv_power = PauliAlg::zero(N_);
        // The derivative of (H - E)^K is given by
        // \sum_{s=0}^{K-1} [(H - E)^s * Hderiv * (H - E)^(K-1-s)]
        for(int s = 0; s <= K_ - 1; s++){
            Hderiv_power += HE_powers[s] * Hderiv_ * HE_powers[K_-1-s];
        }

        std::cout << "Time = " << stopwatch.stop() << " sec" << std::endl;


        // ========== Compute the commutators [(H - E)^K, L] ==========
        std::cout << "Computing the commutators" << std::endl;
        stopwatch.start();

        // commutator[mu] = [(H - E)^K, CD_basis[mu]]
        std::vector<PauliAlg> commutator = PauliAlg::commutator_for_each(HE_powers[K_], CD_basis_);
        std::cout << "Time = " << stopwatch.stop() << " sec" << std::endl;


        // ========== Compute the double commutators [[(H - E)^K, L], L] ==========
        std::cout << "Computing the double commutators" << std::endl;
        stopwatch.start();

        // double_commutator[mu][nu] = [[(H - E)^K, CD_basis[mu]], CD_basis[nu]]
        // To reduce computational time, nu is limited to nu <= mu.
        std::vector<std::vector<PauliAlg> > double_commutator;
        for(int mu = 0; mu < M_; mu++){
            std::vector<PauliAlg> CD_basis_partial(CD_basis_.begin(), CD_basis_.begin() + mu + 1);
            assert(CD_basis_partial.size() == static_cast<size_t>(mu + 1)); 
            double_commutator.push_back(PauliAlg::commutator_for_each(commutator[mu], CD_basis_partial));
        }
        std::cout << "Time = " << stopwatch.stop() << " sec" << std::endl;


        // ========== Compute the Q matrix ==========
        std::cout << "Computing the Q matrix" << std::endl;
        stopwatch.start();

        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(M_, M_);
        for(int mu = 0; mu < M_; mu++){
            for(int nu = 0; nu <= mu; nu++){
                Q(mu, nu) = PauliAlg::trace_normalized_multiply(HE_powers[K_], double_commutator[mu][nu]).real();
                if(mu != nu){
                    Q(nu, mu) = Q(mu, nu);
                }
            }
        }
        std::cout << "Time = " << stopwatch.stop() << " sec" << std::endl;


        // ========== Compute the r vector ==========
        std::cout << "Computing the r vector" << std::endl;
        stopwatch.start();

        Eigen::VectorXd r = Eigen::VectorXd::Zero(M_);
        for(int mu = 0; mu < M_; mu++){
            r(mu) = - PauliAlg::trace_normalized_multiply(Hderiv_power, commutator[mu]).imag();
        }
        std::cout << "Time = " << stopwatch.stop() << " sec" << std::endl;


        // ========== Solve the linear equation Qa = r ==========
        std::cout << "Solving the linear equation Qa = r" << std::endl;
        Eigen::VectorXd a = Q.colPivHouseholderQr().solve(r);

        // Copy the solution to a std::vector object
        std::vector<double> a_vec(M_, 0.0);
        for(int mu = 0; mu < M_; mu++){
            a_vec[mu] = a(mu);
        }
        

        // ========== Save the solution to file ==========
        if(!filename.empty()){
            std::cout << "Saving the driving coefficients to file" << std::endl;
            for(int mu = 0; mu < M_; mu++){
                fout << a_vec[mu] << " ";
            }
            fout.close();
        }


        // ========== End ==========
        std::cout << std::endl;
        return {a_vec, E};        
    }





    // ==============================================================================================
    // =============== Weighted variational method for a lambda-dependent Hamiltonian ===============
    // ==============================================================================================



    /// @brief Weighted variational method for a lambda-dependent Hamiltonian.
    /// An instance of this class will hold all the parameters of the problem.
    /// The method `compute` will evaluate the traces of operators and save the results to a file,
    /// which will later be used to compute the driving coefficients for the entire series of lambda.
    class SerialProblem{
        private:
            // ========== Problem Parameters ==========
            int K_;     // Order of the method
            int N_;     // Number of sites
            int M_;     // Number of CD_basis operators
            int Gamma_; // Number of H_basis operators

            std::vector<PauliAlg> CD_basis_; // basis of the CD driving
            std::vector<PauliAlg> H_basis_;  // basis of Hamiltonian
            
        public:
            // ========== Constructor ==========
            SerialProblem(int K, int N) : K_(K), N_(N), M_(0), Gamma_(0) { // All other variables are initialized with empty vectors
                if(K < 0) WEIGHTED_VARIATIONAL_ERROR("The order K must be a nonnegative integer.");
                if(N < 1) WEIGHTED_VARIATIONAL_ERROR("The number of sites must be a positive integer.");
            }

            // ========== Setters ==========
            void set_CD_basis(const std::vector<PauliAlg> &CD_basis){
                if(CD_basis.size() < 1) WEIGHTED_VARIATIONAL_ERROR("The CD_basis operators must not be empty.");
                for(auto L : CD_basis){
                    if(L.get_num_sites() != N_) WEIGHTED_VARIATIONAL_ERROR("The number of sites of the CD_basis operators must be N.");
                }
                CD_basis_ = CD_basis;
                M_ = CD_basis.size();
            }

            void set_H_basis(const std::vector<PauliAlg> &H_basis){
                if(H_basis.size() < 1) WEIGHTED_VARIATIONAL_ERROR("The basis Hamiltonians must not be empty.");
                for(auto H : H_basis){
                    if(H.get_num_sites() != N_) WEIGHTED_VARIATIONAL_ERROR("The number of sites of the basis Hamiltonians must be N.");
                }
                H_basis_ = H_basis;
                Gamma_ = H_basis.size();
            }

            // ========== Solvers ==========
            void compute(const std::string& filename);
    };


    /// @brief Computes the basis of the Q matrix, the r vector, and omega for the weighted variational method.
    /// The results are saved to the specified file.
    /// @param filename The name of the output file. Mandatory.
    void SerialProblem::compute(const std::string& filename){
        StopWatch stopwatch1, stopwatch2;

        // ========== Check ==========
        if(M_ < 1) WEIGHTED_VARIATIONAL_ERROR("The CD_basis operators must be specified.");
        if(Gamma_ < 1) WEIGHTED_VARIATIONAL_ERROR("The basis Hamiltonians must be specified.");

        // ========== Open file ==========
        std::ofstream fout(filename);
        if(!fout) WEIGHTED_VARIATIONAL_ERROR("Unable to open output file \"" + filename + "\"");
            
        // ========== Early return if K = 0 (no CD driving) ==========
        if(K_ == 0){
            // If K == 0, only the paramters are written to the file.
            fout << "# Parameters" << std::endl;
            fout << "$ K = " << K_ << std::endl;
            fout << "$ N = " << N_ << std::endl;
            fout << "$ M = " << M_ << std::endl;
            fout << "$ Gamma = " << Gamma_ << std::endl;
            fout << std::endl;
            fout.close();

            return;
        }

        // ========== Fill the power table ==========
        // Power table contains the combinatoric information for expanding the powers of H 
        // into the terms with different lambda-dependent coefficients.
        // See the documentation in weighted_variational_utility.h for the details.
        std::vector<utility::power_table_row> power_table = utility::make_power_table(K_, Gamma_);
        int power_table_size = static_cast<int>(power_table.size()); // Cast to avoid warnings of comparison between signed and unsigned integers


        // ========== Prepare for parallelization ==========
        // Prepare the indices for two-level nested loops
        std::vector<std::pair<int, int> > loop_indices;
        for(int g = 0; g < power_table_size; g++){
            for(int mu = 0; mu < M_; mu++){
                loop_indices.emplace_back(g, mu);
            }
        }

        // Shuffle the indices for a better load balancing
        std::mt19937 rand_engine(12345);
        std::shuffle(loop_indices.begin(), loop_indices.end(), rand_engine);
        

        // ========== Compute the powers of H ==========    
        std::cout << "Computing powers of H" << std::endl;
        stopwatch1.start();

        // H_power_basis[g] contains the sum of products of H_basis specified by power_table_[g]. For example, 
        // - if power_table_[g].combination = (2, 1), then H_power_basis[g] = H1 * H1 * H2 + H1 * H2 * H1 + H2 * H1 * H1.
        // - if power_table_[g].combination = (2, 0), then H_power_basis[g] = H1 * H1
        std::vector<PauliAlg> H_power_basis(power_table_size, PauliAlg::zero(N_));
        for(int g = 0; g < power_table_size; g++){
            std::cout << "g = " << g << ": " << "order N^" << power_table[g].total_power << ", " << std::flush;
            stopwatch2.start();

            for(const auto &ordering : power_table[g].orderings){
                if(ordering.size() == 0){
                    H_power_basis[g] += PauliAlg::identity(N_);
                }
                else{
                    // Multiply the basis Hamiltonians according to the ordering
                    // Use normal multiplication except for the last term
                    PauliAlg product = PauliAlg::identity(N_);
                    for(std::size_t i = 0; i < ordering.size() - 1; i++){
                        product = product * H_basis_[ordering[i] - 1];
                    }

                    // Add the final multiplication directly to H_power_basis[g]
                    H_power_basis[g].add_multiply(product, H_basis_[ordering.back() - 1]);
                }
            }
            std::cout << H_power_basis[g].get_num_terms() << " terms, time = " << stopwatch2.stop() << " sec" << std::endl;
        }
        std::cout << "Time = " << stopwatch1.stop() << " sec" << std::endl;

        
        // ========== Compute commutators and traces ==========
        std::cout << "Computing commutators and traces" << std::endl;
        stopwatch1.start();

        // Traces appearing in Q matrix and r vector. Initialized with zeros.
        std::vector<std::vector<std::vector<std::vector<double> > > > Q_basis(power_table_size, std::vector<std::vector<std::vector<double> > >(power_table_size, std::vector<std::vector<double> >(M_, std::vector<double>(M_, 0)))); 
        std::vector<std::vector<std::vector<double> > > r_basis(power_table_size, std::vector<std::vector<double> >(power_table_size, std::vector<double>(M_, 0))); 

        // omega_basis_former[g2] will hold the trace of H_power_basis[g2].
        // omega_basis_latter[g1][g2] will hold the trace of H_power_basis[g1] * H_power_basis[g2] if
        // - power_table[g1] corresponds to a term of H^K, and
        // - power_table[g2] corresponds to a term of H^k for 1 <= k <= K-1.
        // Otherwise omega_basis_latter[g1][g2] = 0 to avoid unnecessary computation.
        std::vector<double> omega_basis_former(power_table_size, 0.0);
        std::vector<std::vector<double> > omega_basis_latter(power_table_size, std::vector<double>(power_table_size, 0.0));

        for(int g2 = 0; g2 < power_table_size; g2++){
            std::cout << "g = " << g2 << ": " << "order N^" << power_table[g2].total_power << ", " << std::flush;
            stopwatch2.start();
            
            // Since the following calculation is linear in H_power_basis[g2], 
            // we can split H_power_basis[g2] into chunks and sum up the results to save memory.
            // The chunk size is determined by the parameter chunk_size defined above.

            for(std::size_t i_chunk = 0; i_chunk < H_power_basis[g2].get_num_terms(); i_chunk += chunk_size){
                // ===== Construct the chunk =====
                PauliAlg H_power_basis_g2_chunk(N_);
                
                for(std::size_t i = i_chunk; i < i_chunk + chunk_size && i < H_power_basis[g2].get_num_terms(); i++){
                    const auto& [prod2, coeff2] = H_power_basis[g2].term_at(i);
                    H_power_basis_g2_chunk.append(prod2, coeff2);
                }    

                // ===== Compute omega =====
                omega_basis_former[g2] += PauliAlg::trace_normalized(H_power_basis_g2_chunk).real();

                if(power_table[g2].total_power <= K_ - 1 && power_table[g2].total_power >= 1){
                    for(int g1 = 0; g1 < power_table_size; g1++){
                        if(power_table[g1].total_power == K_){
                            omega_basis_latter[g1][g2] += PauliAlg::trace_normalized_multiply(H_power_basis[g1], H_power_basis_g2_chunk).real();
                        }
                    }
                }

                // ===== Compute commutators and r vector =====
                // commutator[mu] = [H_power_basis_g2_chunk, CD_basis[mu]]
                std::vector<PauliAlg> commutator = PauliAlg::commutator_for_each(H_power_basis_g2_chunk, CD_basis_);
                
                // compute the traces appearing in r vector
                #pragma omp parallel for
                for(std::size_t l = 0; l < loop_indices.size(); l++){
                    auto [g1, mu] = loop_indices[l];
                    r_basis[g1][g2][mu] += - PauliAlg::trace_normalized_multiply(H_power_basis[g1], commutator[mu]).imag();
                }

                // ===== Compute double commutators and Q matrix =====
                for(int mu = 0; mu < M_; mu++){
                    // double_commutator[nu] = [[H_power_basis_g2_chunk, CD_basis[mu]], CD_basis[nu]]
                    // To reduce computational time, we restrict nu to 0 <= nu <= mu.
                    std::vector<PauliAlg> CD_basis_partial(CD_basis_.begin(), CD_basis_.begin() + mu + 1); 
                    std::vector<PauliAlg> double_commutator = PauliAlg::commutator_for_each(commutator[mu], CD_basis_partial);

                    // Compute the traces appearing in Q matrix
                    // We only compute for nu <= mu because Q is symmetric
                    #pragma omp parallel for
                    for(std::size_t l = 0; l < loop_indices.size(); l++){
                        auto [g1, nu] = loop_indices[l];
                        if(nu <= mu){
                            Q_basis[g1][g2][mu][nu] += PauliAlg::trace_normalized_multiply(H_power_basis[g1], double_commutator[nu]).real();
                        }
                    }
                }
            }

            std::cout << "time = " << stopwatch2.stop() << " sec" << std::endl;
        }
        std::cout << "Time = " << stopwatch1.stop() << " sec" << std::endl;


        // ========== Save the results to file ==========
        // Write parameters
        fout << "# Parameters" << std::endl;
        fout << "$ K = " << K_ << std::endl;
        fout << "$ N = " << N_ << std::endl;
        fout << "$ M = " << M_ << std::endl;
        fout << "$ Gamma = " << Gamma_ << std::endl;
        fout << std::endl;

        // Write the Q matrix
        for(int g1 = 0; g1 < power_table_size; g1++){
            for(int g2 = 0; g2 < power_table_size; g2++){
                fout << "# Q_basis" << std::endl;
                fout << "$ g1 = " << g1 << std::endl;
                fout << "$ g2 = " << g2 << std::endl;
                for(int mu = 0; mu < M_; mu++){
                    for(int nu = 0; nu < M_; nu++){
                        if(nu <= mu){
                            fout << paulialg::utility::double_format_full(Q_basis[g1][g2][mu][nu]) << " ";
                        }
                        else{
                            fout << paulialg::utility::double_format_full(Q_basis[g1][g2][nu][mu]) << " ";
                        }
                    }
                    fout << std::endl;
                }
                fout << std::endl;
            }
        }

        // Write the r vector
        for(int g1 = 0; g1 < power_table_size; g1++){
            for(int g2 = 0; g2 < power_table_size; g2++){
                fout << "# r_basis" << std::endl;
                fout << "$ g1 = " << g1 << std::endl;
                fout << "$ g2 = " << g2 << std::endl;
                for(int mu = 0; mu < M_; mu++){
                    fout << paulialg::utility::double_format_full(r_basis[g1][g2][mu]) << " ";
                }
                fout << std::endl << std::endl;
            }
        }

        // Write omega_basis
        fout << "# omega_basis_former" << std::endl;
        for(int g = 0; g < power_table_size; g++){
            fout << paulialg::utility::double_format_full(omega_basis_former[g]) << " ";
        }
        fout << std::endl << std::endl;

        fout << "# omega_basis_latter" << std::endl;
        for(int g1 = 0; g1 < power_table_size; g1++){
            for(int g2 = 0; g2 < power_table_size; g2++){
                fout << paulialg::utility::double_format_full(omega_basis_latter[g1][g2]) << " ";
            }
            fout << std::endl;
        }
        fout << std::endl;

        // Close the file
        fout.close();
        std::cout << "Results saved to " << filename << std::endl << std::endl;
    }






    // ==============================================================================================
    // =============== Weighted variational method for a lambda-dependent Hamiltonian ===============
    // ==============================================================================================


    /// @brief Loads the output of SerialProblem and computes the driving coefficients 
    /// for the entire series of a lambda-dependent Hamiltonian.
    /// An instance of this class will hold all the parameters and the results of the SerialProblem, namely the traces of operators.
    /// The method `compute_driving_coeff` will compute the driving coefficients for the entire series of lambda.
    class SerialProblemTrace{
        private:
            // ========== Problem Parameters ==========
            int K_;     // Order of the method
            int N_;     // Number of sites
            int M_;     // Number of CD_basis operators
            int Gamma_; // Number of H_basis operators
            
            std::vector<double> lambda_series_;                       // The series of lambda for which the driving coefficients are computed
            std::function<std::vector<double>(double)> H_coeff_;      // lambda-dependent coefficient of the Hamiltonian
            std::function<std::vector<double>(double)> Hderiv_coeff_; // lambda derivatives of the coefficient of the Hamiltonian
            
            std::string E_mode_;           // "auto" if the energy shift E is automatically computed, otherwise "manual"
            std::vector<double> E_series_; // Series of the energy shift E for E_mode_ == "manual"

            // ========== Trace of operators calculated by SerialProblem::compute() ==========
            std::vector<std::vector<Eigen::MatrixXd> > Q_basis_; // Traces appearing in Q matrix
            std::vector<std::vector<Eigen::VectorXd> > r_basis_; // Traces appearing in r vector
            Eigen::VectorXd omega_basis_former_; // Tr(H^k)
            Eigen::MatrixXd omega_basis_latter_; // Tr(H^K * H^k)
            
            // ========== Power table ==========
            std::vector<utility::power_table_row> power_table_; // Table of the combination of powers. Filled in the constructor SerialProblemTrace()
            int power_table_size_; // Size of the power_table_

        public:
            // ========== Constructor ==========
            SerialProblemTrace(const std::string &filename);
            
            // ========== Setters ==========
            void set_lambda_series(const std::vector<double> &lambda_series){
                if(lambda_series.size() < 1) WEIGHTED_VARIATIONAL_ERROR("lambda_series must not be empty.");
                lambda_series_ = lambda_series;
            }

            void set_H_coeff(const std::function<std::vector<double>(double)> &H_coeff){
                // We do not check the size of the return value of H_coeff here.
                H_coeff_ = H_coeff;
            }

            void set_Hderiv_coeff(const std::function<std::vector<double>(double)> &Hderiv_coeff){
                // We do not check the size of the return value of Hderiv_coeff here.
                Hderiv_coeff_ = Hderiv_coeff;
            }
            
            void set_E_auto(){
                E_mode_ = "auto";
            }

            void set_E_manual(const std::vector<double> &E_series){
                E_mode_ = "manual";
                if(E_series.size() < 1) WEIGHTED_VARIATIONAL_ERROR("E_series must not be empty.");
                E_series_ = E_series;
            }


            // ========== Compute the lambda-dependent driving coefficients ==========
            std::tuple<std::vector<std::vector<double> >, 
                    std::vector<std::vector<double> >,
                    std::vector<double> > 
            compute_driving_coeff(const std::string &filename = "") const & ;
    };


    /// @brief Constructor of SerialProblemTrace from an output file of SerialProblem.
    /// @param filename The name of the file.
    SerialProblemTrace::SerialProblemTrace(const std::string &filename) : K_(-1), N_(-1), M_(-1), Gamma_(-1), E_mode_("") {
        std::cout << "Loading trace from file" << std::endl;
        StopWatch stopwatch;
        stopwatch.start();

        // Open and read the entire file
        std::ifstream fin(filename);
        if(!fin) WEIGHTED_VARIATIONAL_ERROR("Unable to open input file \"" + filename + "\"");
        std::vector<utility::file_paragraph> data = utility::split_file_structure(fin);
        fin.close();

        // Check if the first paragraph is "Parameters".
        if(data.empty() || data[0].header != "Parameters"){
            WEIGHTED_VARIATIONAL_ERROR("The file must start with '# Parameters'.");
        }

        // Read each paragraph
        for(const auto& paragraph : data){
            if(paragraph.header == "Parameters"){
                // Read parameter values
                for (auto const& [key, value] : paragraph.parameters){
                    if(key == "K"){
                        K_ = value;
                    }
                    else if(key == "N"){
                        N_ = value;
                    }
                    else if(key == "M"){
                        M_ = value;
                    }
                    else if(key == "Gamma"){
                        Gamma_ = value;
                    }
                }

                // Check the values
                if(K_ < 0) WEIGHTED_VARIATIONAL_ERROR("The order K must be a nonnegative integer.");
                if(N_ < 1) WEIGHTED_VARIATIONAL_ERROR("The number of sites (N) must be a positive integer.");
                if(M_ < 1) WEIGHTED_VARIATIONAL_ERROR("The number of CD_basis operators (M) must be a positive integer.");
                if(Gamma_ < 1) WEIGHTED_VARIATIONAL_ERROR("The number of basis Hamiltonians (Gamma) must be a positive integer.");
                
                if(K_ != 0){
                    // Reconstruct power_table
                    power_table_ = utility::make_power_table(K_, Gamma_);
                    power_table_size_ = static_cast<int>(power_table_.size());

                    // Initialize the results
                    Q_basis_ = std::vector<std::vector<Eigen::MatrixXd> >(power_table_size_, std::vector<Eigen::MatrixXd>(power_table_size_, Eigen::MatrixXd::Zero(M_, M_)));
                    r_basis_ = std::vector<std::vector<Eigen::VectorXd> >(power_table_size_, std::vector<Eigen::VectorXd>(power_table_size_, Eigen::VectorXd::Zero(M_)));
                    omega_basis_former_ = Eigen::VectorXd::Zero(power_table_size_);
                    omega_basis_latter_ = Eigen::MatrixXd::Zero(power_table_size_, power_table_size_);
                }
            }
            else if(paragraph.header == "Q_basis"){                
                // Read parameter values
                int g1 = -1;
                int g2 = -1;
                for (auto const& [key, value] : paragraph.parameters){
                    if(key == "g1"){
                        g1 = value;
                    }
                    else if(key == "g2"){
                        g2 = value;
                    }    
                }      

                // Check g1 and g2
                if(g1 < 0 || g1 >= power_table_size_ || g2 < 0 || g2 >= power_table_size_){
                    WEIGHTED_VARIATIONAL_ERROR("The indices g1 and g2 are out of range.");
                }
                
                // Read the matrix
                Q_basis_[g1][g2] = utility::read_dmatrix(paragraph.contents, M_, M_, "Q_basis");
            }
            else if(paragraph.header == "r_basis"){
                // Read parameter values
                int g1 = -1;
                int g2 = -1;
                for (auto const& [key, value] : paragraph.parameters){
                    if(key == "g1"){
                        g1 = value;
                    }
                    else if(key == "g2"){
                        g2 = value;
                    }    
                }      

                // Check g1 and g2
                if(g1 < 0 || g1 >= power_table_size_ || g2 < 0 || g2 >= power_table_size_){
                    WEIGHTED_VARIATIONAL_ERROR("The indices g1 and g2 are out of range.");
                }
                
                // Read the vector
                r_basis_[g1][g2] = utility::read_dmatrix(paragraph.contents, 1, M_, "r_basis").row(0);
            }    
            else if(paragraph.header == "omega_basis_former"){
                // There is no parameter to read. Directly read the vector
                omega_basis_former_ = utility::read_dmatrix(paragraph.contents, 1, power_table_size_, "omega_basis_former").row(0);
            }    
            else if(paragraph.header == "omega_basis_latter"){
                // There is no parameter to read. Directly read the matrix
                omega_basis_latter_ = utility::read_dmatrix(paragraph.contents, power_table_size_, power_table_size_, "omega_basis_latter"); 
            }
            else{
                std::cout << "Warning: Paragraph header must be one of 'Parameters', 'Q_basis', 'r_basis', 'omega_basis_former', 'omega_basis_latter'." << std::endl;
                std::cout << "Unknown header '" << paragraph.header << "' is ignored." << std::endl;
            }
        }

        std::cout << "Time = " << stopwatch.stop() << " sec" << std::endl;
    }



    /// @brief Computes the lambda-dependent driving coefficients.
    /// @param filename Optional. The filename to save the results. If not specified, the results will not be saved.
    /// @return A triplet {H_coeff_series, CD_coeff_series, E_series}.
    /// `H_coeff_series[li][gamma]` is the coefficient of the operator `H_basis[gamma]` at lambda `lambda_series_[li]`.
    /// `CD_coeff_series[li][mu]` is the coefficient of the operator `CD_basis[mu]` at lambda `lambda_series_[li]`.
    /// `E_series[li]` is the energy shift at the lambda `lambda_series_[li]`.
    /// @note The output file has the following format:
    /// ```
    /// # Parameters
    /// $ K = [K]
    /// $ N = [N]
    /// $ M = [M]
    /// $ Gamma = [Gamma]
    ///
    /// # lambda f1 f2 ... a1 a2 ... E 
    /// [lambda_series[0]] [H_coeff_series[0][0]] [H_coeff_series[0][1]] ... [CD_coeff_series[0][0]] [CD_coeff_series[0][1]] ... [E_series[0]]
    /// [lambda_series[1]] [H_coeff_series[1][0]] [H_coeff_series[1][1]] ... [CD_coeff_series[1][0]] [CD_coeff_series[1][1]] ... [E_series[1]]
    /// [lambda_series[2]] [H_coeff_series[2][0]] [H_coeff_series[2][1]] ... [CD_coeff_series[2][0]] [CD_coeff_series[2][1]] ... [E_series[2]]
    /// ...
    /// ```
    /// where `[x]` denotes the value of the variable `x`.
    std::tuple<std::vector<std::vector<double> >, 
               std::vector<std::vector<double> >,
               std::vector<double> > 
    SerialProblemTrace::compute_driving_coeff(const std::string &filename) const & {
        // ========== Checks ==========
        if(E_mode_ != "auto" && E_mode_ != "manual") WEIGHTED_VARIATIONAL_ERROR("Either set_E_auto or set_E_manual should be called before computation.");
        if(lambda_series_.size() < 1) WEIGHTED_VARIATIONAL_ERROR("lambda_series must be specified.");
        if(H_coeff_ == nullptr) WEIGHTED_VARIATIONAL_ERROR("H_coeff must be specified.");
        if(Hderiv_coeff_ == nullptr) WEIGHTED_VARIATIONAL_ERROR("Hderiv_coeff must be specified.");
        if(E_mode_ == "manual" && E_series_.size() != lambda_series_.size()) WEIGHTED_VARIATIONAL_ERROR("The size of E_series must be the same as that of lambda_series.");

        // ========== Open file ==========
        std::ofstream fout;
        if(!filename.empty()){
            fout.open(filename);
            if(!fout) WEIGHTED_VARIATIONAL_ERROR("Unable to open output file \"" + filename + "\"");
        }

        // ========== Prepare the table of binomial coefficients ==========
        std::vector<std::vector<int> > binomial = utility::make_binomial_table(std::max(K_, 2*K_ - 2));
        

        // ========== Compute the series ==========
        std::cout << "Generating the series" << std::endl;
        StopWatch stopwatch;
        stopwatch.start();

        // Variables to return
        std::vector<std::vector<double> > H_coeff_series(lambda_series_.size());
        std::vector<std::vector<double> > CD_coeff_series(lambda_series_.size());
        std::vector<double> E_series(lambda_series_.size());

        #pragma omp parallel for
        for(std::size_t li = 0; li < lambda_series_.size(); li++){
            // ===== Evaluate functions =====
            double lambda = lambda_series_[li];
            std::vector<double> H_coeff = H_coeff_(lambda);
            std::vector<double> Hderiv_coeff = Hderiv_coeff_(lambda);
            
            // Check the size of the return values
            if(static_cast<int>(H_coeff.size()) != Gamma_) WEIGHTED_VARIATIONAL_ERROR("The size of the return vector of H_coeff must be the same as the number of basis Hamiltonians.");
            if(static_cast<int>(Hderiv_coeff.size()) != Gamma_) WEIGHTED_VARIATIONAL_ERROR("The size of the return vector of Hderiv_coeff must be the same as the number of basis Hamiltonians.");
            
            H_coeff_series[li] = H_coeff; // Save the value to return it at the end of the function


            // ========== Early finish if K = 0 (no CD driving) ==========
            if(K_ == 0){
                CD_coeff_series[li] = std::vector<double>(M_, 0.0);
                E_series[li] = 0.0;
                continue;
            }


            // ===== Compute the powers of the coefficients =====
            std::vector<double> H_power_coeff(power_table_size_, 0.0);
            std::vector<double> Hderiv_power_coeff(power_table_size_, 0.0);
            
            for(int g = 0; g < power_table_size_; g++){
                utility::power_combination comb = power_table_[g].combination;
                
                // Compute the lambda-dependent coefficient specified by power_table_[g].
                // For example, if `power_table_[g].combination = (2, 1)`, the coefficient is `f1^2 * f2^1`. 
                H_power_coeff[g] = 1.0;
                for(int gamma = 0; gamma < Gamma_; gamma++){
                    H_power_coeff[g] *= std::pow(H_coeff[gamma], comb[gamma]);
                }
                
                // Compute the lambda-derivative of the lambda-dependent coefficient using the Leibniz rule.
                // For example, if `power_table_[g].combination = (2, 1)`, the coefficient is the derivative of `f1^2 * f2^1`, 
                // which is given by `(2 * f'1 * f1) * f2 + f1^2 * f'2`, where f'i is the derivative of fi.
                for(int gamma_deriv = 0; gamma_deriv < Gamma_; gamma_deriv++){ // determine which of the coefficients will be differentiated
                    if(comb[gamma_deriv] == 0){
                        continue;
                    }

                    double coeff_term = 1.0;
                    for(int gamma = 0; gamma < Gamma_; gamma++){
                        if(gamma == gamma_deriv){
                            coeff_term *= comb[gamma] * std::pow(H_coeff[gamma], comb[gamma] - 1);
                            coeff_term *= Hderiv_coeff[gamma];
                        }
                        else{
                            coeff_term *= std::pow(H_coeff[gamma], comb[gamma]);
                        }
                    }

                    Hderiv_power_coeff[g] += coeff_term;
                }
                
            }


            // ===== Construct omega =====    
            std::vector<double> omega(2*K_, 0.0);

            // omega_0
            omega[0] = 1.0;

            // omega_1, ..., omega_K
            for(int g = 0; g < power_table_size_; g++){
                int total_power = power_table_[g].total_power;
                if(total_power >= 1){
                    omega[total_power] += H_power_coeff[g] * omega_basis_former_(g);
                }
            }

            // omega_{K+1}, ..., omega_{2K-1}
            for(int g1 = 0; g1 < power_table_size_; g1++){
                if(power_table_[g1].total_power == K_){
                    for(int g2 = 0; g2 < power_table_size_; g2++){
                        int total_power = power_table_[g2].total_power;
                        if(total_power <= K_ - 1 && total_power >= 1){
                            omega[K_ + total_power] += H_power_coeff[g1] * H_power_coeff[g2] * omega_basis_latter_(g1, g2);
                        }
                    }
                }
            }

            
            // ===== Construct E =====
            double E = 0.0;

            if(E_mode_ == "auto"){
                // If K = 1, the optimal E is undefined.
                if(K_ == 1){
                    E = 0.0;
                }
                
                // Solve the optimization problem for K = 2, which is analytically solvable.
                // If K_ >= 3, this result will be used for the initial value for the Newton method.
                if(K_ >= 2){
                    std::vector<double> Omega_numerator_2(3, 0.0);
                    std::vector<double> Omega_denominator_2(3, 0.0);

                    int sign = 1; // To generate (-1)^k
                    for(int k = 0; k <= 2; k++){
                        Omega_numerator_2[k] = sign * binomial[2][k] * omega[3-k];
                        Omega_denominator_2[k] = sign * binomial[2][k] * omega[2-k];
                        sign *= -1;
                    } 

                    auto [E_optimal_2, Omega_optimal_2] = utility::find_extremum(Omega_numerator_2, Omega_denominator_2, 0.0, true);
                    E = E_optimal_2;
                }

                // Solve the optimization problem for the specified K
                // using the result of the optimization for K = 2 as the initial guess.
                if(K_ >= 3){
                    std::vector<double> Omega_numerator_K(2*K_ - 1, 0.0);
                    std::vector<double> Omega_denominator_K(2*K_ - 1, 0.0);

                    int sign = 1; // To generate (-1)^k
                    for(int k = 0; k <= 2*K_ - 2; k++){
                        Omega_numerator_K[k] = sign * binomial[2*K_ - 2][k] * omega[2*K_ - k - 1];
                        Omega_denominator_K[k] = sign * binomial[2*K_ - 2][k] * omega[2*K_ - k - 2];
                        sign *= -1;
                    }

                    // Solve the optimization problem
                    auto [E_optimal_K, Omega_optimal_K] = utility::find_extremum(Omega_numerator_K, Omega_denominator_K, E, false);
                    
                    E = E_optimal_K;
                }
            }
            else if(E_mode_ == "manual"){
                E = E_series_[li];
            }

            E_series[li] = E;  // Save the value to return it at the end of the function


            // ===== Compute factors in (H - E)^K =====
            // To construct (H - E)^K, we compute the factor associated with each term of the expansion of (H - E)^K.
            // The factor is the product of (-E)^(K - k) and the binomial coefficient binomial[K][k], which corresponds to 
            // the number of ways to choose k position for the Hamiltonians and (K - k) positions for E in the product of K things.
            std::vector<double> HE_power_factor(power_table_size_, 0.0);

            for(int g = 0; g < power_table_size_; g++){
                int total_power = power_table_[g].total_power;
                HE_power_factor[g] = std::pow(-E, K_ - total_power) * binomial[K_][total_power];
            }


            // ===== Compute Q matrix and r vector =====
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(M_, M_);
            Eigen::VectorXd r = Eigen::VectorXd::Zero(M_);
            
            for(int g1 = 0; g1 < power_table_size_; g1++){
                for(int g2 = 0; g2 < power_table_size_; g2++){
                    Q += (HE_power_factor[g1] * HE_power_factor[g2] * H_power_coeff[g1] * H_power_coeff[g2]) * Q_basis_[g1][g2];
                    r += (HE_power_factor[g1] * HE_power_factor[g2] * Hderiv_power_coeff[g1] * H_power_coeff[g2]) * r_basis_[g1][g2];
                }
            }   


            // ===== Solve the linear equation Qa = r =====
            Eigen::VectorXd a = Q.colPivHouseholderQr().solve(r);

            // copy the result to std::vector
            CD_coeff_series[li] = std::vector<double>(M_, 0.0);
            for(int mu = 0; mu < M_; mu++){
                CD_coeff_series[li][mu] = a(mu);
            }
        }

        std::cout << "Time = " << stopwatch.stop() << " sec" << std::endl;

        // ========== Save to file ==========
        if (!filename.empty()){
            // Write the parameters
            fout << "# Parameters" << std::endl;
            fout << "$ K = " << K_ << std::endl;
            fout << "$ N = " << N_ << std::endl;
            fout << "$ M = " << M_ << std::endl;
            fout << "$ Gamma = " << Gamma_ << std::endl;
            fout << std::endl;

            // Write the text header
            fout << "lambda\t";
            for(int gamma = 1; gamma <= Gamma_; gamma++){
                fout << "f" << gamma << "\t";
            }
            for(int mu = 1; mu <= M_; mu++){
                fout << "a" << mu << "\t";
            }
            fout << "E" << std::endl;
            
            // Write the data
            for(int li = 0; li < static_cast<int>(lambda_series_.size()); li++){
                fout << paulialg::utility::double_format_full(lambda_series_[li]) << "\t";
                for(int gamma = 0; gamma < Gamma_; gamma++){
                    fout << paulialg::utility::double_format_full(H_coeff_series[li][gamma]) << "\t";
                }
                for(int mu = 0; mu < M_; mu++){
                    fout << paulialg::utility::double_format_full(CD_coeff_series[li][mu]) << "\t";
                }
                fout << paulialg::utility::double_format_full(E_series[li]);
                fout << std::endl;
            }

            fout.close();

            std::cout << "Results saved to " << filename << std::endl << std::endl;
        }

        // ========== Return ==========
        std::cout << std::endl;
        return {H_coeff_series, CD_coeff_series, E_series};
    }


} // namespace weighted_variational


#endif // WEIGHTED_VARIATIONAL_H