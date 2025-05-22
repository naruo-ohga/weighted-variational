/**
 * This file implements the weighted variational method for a single lambda using the K-nested-commutator algorithm.
 * The algorithm is slower than the one implemented in weighted_variational.h, which is called the fast-commutator algorithm.
 *
 * This file is for testing the validity of weighted_variational.h.
 * Unless you want to perform the test, you do not need to include this file.
*/

#include "pauli_alg.h"
#include "weighted_variational.h"


/// @brief Performs weighted variational method for a single lambda using the K-nested-commutator algorithm (slower).
/// @return The vector of driving coefficients `a`. `a[mu]` corresponds to the operator `CD_basis[mu]`.
/// If `K = 0`, the vector is all zero (no CD driving).
std::vector<double> weighted_variational::SingleProblem::compute_algorithm2() const & {
    // ========== Check ==========
    if(M_ < 1) WEIGHTED_VARIATIONAL_ERROR("The CD_basis operators must not be empty.");
    if(E_mode_ != "manual") WEIGHTED_VARIATIONAL_ERROR("Algorithm2 only supports E_mode_ = \"manual\".");
    double E = E_manual_;

    // ========== Early return if K = 0 (no CD driving) ==========
    if(K_ == 0){
        return std::vector<double>(M_, 0.0);
    }
    
        
    // ========== Compute the integer-valued coefficients ==========
    std::vector<std::vector<int> > binomial = weighted_variational::utility::make_binomial_table(K_+1);

    /// The coefficients of the nested commutators
    std::vector<int> g_coeff(K_+1, 0);
    
    int sign = 1; // To generate (-1)^s
    for(int s = 0; s <= K_; s++){
        g_coeff[s] = 2 * binomial[K_+1][s+1] - binomial[K_][s];
        g_coeff[s] *= sign * s;
        sign *= -1;
    }

    // ========== Compute the nested commutators (up to Kth)==========
    StopWatch stopwatch;

    std::cout << "Computing the nested commutators" << std::endl;
    stopwatch.start();

    /// commutators[mu][0] = CD_basis[mu]
    /// commutators[mu][1] = [H, CD_basis[mu]]
    /// ...
    /// commutators[mu][k] = [H, [H, ...[H, CD_basis[mu]]...]] where there are k H's in the commutator.
    std::vector<std::vector<PauliAlg> > commutators;

    for(int mu = 0; mu < M_; mu++){
        commutators.emplace_back(std::vector<PauliAlg>({CD_basis_[mu]}));
        for(int k = 1; k <= K_; k++){
            commutators[mu].emplace_back(PauliAlg::commutator(H_, commutators[mu][k-1]));
        }
    }
    std::cout << "Time: " << stopwatch.stop() << " sec" << std::endl;

    // ========== Compute the powers of (H - E) (up to Kth) ==========
    std::cout << "Computing the powers of (H - E)" << std::endl;
    stopwatch.start();

    PauliAlg H_minus_E = H_ - E * PauliAlg::identity(N_);
    
    /// HE_powers[k] = (H - E)^k
    std::vector<PauliAlg> HE_powers;

    HE_powers.push_back(PauliAlg::identity(N_));
    for(int k = 1; k <= K_; k++){
        HE_powers.emplace_back(HE_powers[k-1] * H_minus_E);
    }
    std::cout << "Time: " << stopwatch.stop() << " sec" << std::endl;
        

    // ========== Compute the Q matrix ==========
    std::cout << "Computing the Q matrix" << std::endl;
    stopwatch.start();

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(M_, M_);
    for(int mu = 0; mu < M_; mu++){
        for(int nu = 0; nu <= mu; nu++){
            // The summand for s = 1 to K-1
            for(int s = 1; s <= K_-1; s++){
                PauliAlg rhs = HE_powers[K_-1-s] * (commutators[mu][s] * commutators[nu][1]);
                Q(mu, nu) += (double)g_coeff[s] * PauliAlg::trace_normalized_multiply(HE_powers[K_], rhs).real();
            }

            // The summand for s = K
            PauliAlg rhs = commutators[mu][K_] * commutators[nu][1];
            Q(mu, nu) += (double)g_coeff[K_] * PauliAlg::trace_normalized_multiply(HE_powers[K_-1], rhs).real();
            
            // Copy the value to the other half of the matrix. Q is symmetric.
            if(mu != nu){
                Q(nu, mu) = Q(mu, nu);
            }
        }
    }
    std::cout << "Time: " << stopwatch.stop() << " sec" << std::endl;

    // ========== Compute the r vector ==========
    std::cout << "Computing the r vector" << std::endl;
    stopwatch.start();

    Eigen::VectorXd r = Eigen::VectorXd::Zero(M_);
    for(int mu = 0; mu < M_; mu++){
        // s = 1 to K-1
        for(int s = 1; s <= K_-1; s++){
            PauliAlg rhs = HE_powers[K_-1-s] * (commutators[mu][s] * Hderiv_);
            r(mu) += (double)g_coeff[s] * PauliAlg::trace_normalized_multiply(HE_powers[K_], rhs).imag();
        }

        // s = K
        PauliAlg rhs = commutators[mu][K_] * Hderiv_;
        r(mu) += (double)g_coeff[K_] * PauliAlg::trace_normalized_multiply(HE_powers[K_-1], rhs).imag();
    }
    std::cout << "Time: " << stopwatch.stop() << " sec" << std::endl;

    // ========== Solve the linear equation Qa = r ==========
    std::cout << "Solving the linear equation Qa = r" << std::endl;
    Eigen::VectorXd a = Q.colPivHouseholderQr().solve(r);

    // Copy the solution to a std::vector object
    std::vector<double> a_vec(M_, 0.0);
    for(int mu = 0; mu < M_; mu++){
        a_vec[mu] = a(mu);
    }

    // ========== End ==========
    std::cout << std::endl;
    return a_vec;
}

