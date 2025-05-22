#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <cassert>
#include <functional>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <map>
#include "weighted_variational.h"
#include "weighted_variational_test_algorithm2.h"
using paulialg::PauliAlg;
using paulialg::PauliProd;

/// Return a vector x[] of n evenly spaced numbers between a and b
/// x[0] = a, x[n-1] = b.
std::vector<double> linspace(double a, double b, int n){
    std::vector<double> v;
    for(int i = 0; i < n; i++){
        v.push_back(a + ((double)i / (double)(n - 1)) * (b - a) );
    }
    return v;
}


/// Print a vector in the form of `(x1, x2, ..., xN)`
template<typename T>
void print_vector(std::vector<T> v){
    bool first = true;
    std::cout << "(";
    for(auto& elem : v){
        if(first){
            first = false;
        }
        else{
            std::cout << ", ";
        }
        std::cout << elem;
    }
    std::cout << ")";
}


void generate_orderings_test(int Gamma, int s){
    std::cout << "===== Test of generate_orderings =====" << std::endl;

    // Test of generate_orderings
    std::cout << "Verify that all " << s << " combinations of numbers 1 to " << Gamma << " are listed below." << std::endl;
    std::vector<weighted_variational::utility::power_ordering> orderings = weighted_variational::utility::generate_orderings(Gamma, s);
    for(auto& ordering: orderings){
        print_vector(ordering);
        std::cout << " ";
    }
    std::cout << std::endl;
}


void make_power_table_test(int K, int Gamma){
    std::cout << "===== Test of make_power_table =====" << std::endl;
    std::cout << "Verify that the second element contains all possible divisions of the first element into " << Gamma << " parts." << std::endl;
    std::cout << "Verify that the third element contains all possible orderings of numbers from 1 to " << Gamma << "." << std::endl;
    std::cout << "Verify that the numbers of each integers of the third element agrees with the second element." << std::endl;
    // Test of make_power_table
    std::vector<weighted_variational::utility::power_table_row> power_table = weighted_variational::utility::make_power_table(K, Gamma);
    for(auto& row : power_table){
        std::cout << row.total_power << " : ";
        print_vector(row.combination);
        std::cout << " : ";
        for(auto& ordering: row.orderings){
            print_vector(ordering);
            std::cout << " ";
        }
        std::cout << std::endl;
    }
}


void find_extremum_test(int N){
    std::cout << "===== Test of find_extremum =====" << std::endl;

    // ===== Preparation of the Hamiltonian =====
    const PauliAlg Id = PauliAlg::identity(N);
    const std::vector<PauliAlg> X = PauliAlg::X_array(N);
    const std::vector<PauliAlg> Y = PauliAlg::Y_array(N);
    const std::vector<PauliAlg> Z = PauliAlg::Z_array(N);

    // Prepare random number generator for the Hamiltonian
    const int mt_seed = 828495664;
    std::mt19937 rand_engine(mt_seed);
    std::normal_distribution<double> rand_normal(1.0, 0.5); // Specify mean and standard deviation
    
    // Generate a random Hamiltonian
    PauliAlg HX = PauliAlg::zero(N);
    PauliAlg HZ = PauliAlg::zero(N);
    PauliAlg HZZ = PauliAlg::zero(N);

    for(int i = 0; i < N; i++){
        HX += rand_normal(rand_engine) * X[i];
        HZ += rand_normal(rand_engine) * Z[i];
        if(i < N-1){    
            HZZ -= rand_normal(rand_engine) * Z[i] * Z[i+1];
        }
    }

    PauliAlg H = 1.3 * HX + 1.1 * HZ + 0.9 * HZZ; // + 12.5 * Id;
    PauliAlg Hderiv = 1.3 * HX - 1.1 * HZ + 1.8 * HZZ;
    

    // ===== Calculate the trace of the powers of the Hamiltonian =====
    const int K = 5;

    std::vector<PauliAlg> H_power(K+1, PauliAlg::zero(N));
    H_power[0] = Id;
    for(int k = 1; k <= K; k++){
        H_power[k] = H_power[k-1] * H;
    }

    std::vector<double> trace_H_powers(2*K);
    trace_H_powers[0] = 1.0;
    for(int k = 1; k <= K; k++){
        trace_H_powers[k] = PauliAlg::trace_normalized(H_power[k]).real();
    }
    for(int k = K+1; k <= 2*K-1; k++){
        trace_H_powers[k] = PauliAlg::trace_normalized_multiply(H_power[k-K], H_power[K]).real();
    }

    // ===== Set the optimization problem =====
    std::vector<std::vector<int> > binomial = weighted_variational::utility::make_binomial_table(2*K-2);
    int sign; // To generate (-1)^k

    // problem for K = 2. This will be analytically solvable and used to generate the initial value for the optimization.
    std::vector<double> Phi_numerator_2(3, 0.0);
    std::vector<double> Phi_denominator_2(3, 0.0);

    sign = 1;
    for(int k = 0; k <= 2; k++){
        Phi_numerator_2[k] = sign * binomial[2][k] * trace_H_powers[3-k];
        Phi_denominator_2[k] = sign * binomial[2][k] * trace_H_powers[2-k];
        sign *= -1;
    } 

    // problem for the specified K
    std::vector<double> Phi_numerator_K(2*K-1, 0.0);
    std::vector<double> Phi_denominator_K(2*K-1, 0.0);

    sign = 1;
    for(int k = 0; k <= 2*K-2; k++){
        Phi_numerator_K[k] = sign * binomial[2*K-2][k] * trace_H_powers[2*K-k-1];
        Phi_denominator_K[k] = sign * binomial[2*K-2][k] * trace_H_powers[2*K-k-2];
        // std::cout << Phi_numerator_K[k] << " " << Phi_denominator_K[k] << std::endl;
        sign *= -1;
    }
    // std::cout << std::endl; 

    // ===== Find the extremum =====
    std::string filename = "weighted_variational_test_extremum.txt";
    auto [E_extremum_2, Phi_extremum_2] = weighted_variational::utility::find_extremum(Phi_numerator_2, Phi_denominator_2, 0.0, true);
    auto [E_extremum_K, Phi_extremum_K] = weighted_variational::utility::find_extremum(Phi_numerator_K, Phi_denominator_K, E_extremum_2, false, 100, 1e-8, filename);

    std::cout << "The extremum for K = 2 is found at E = " << E_extremum_2 << " with Phi = " << Phi_extremum_2 << std::endl;
    std::cout << "The extremum for K = " << K << " is found at E = " << E_extremum_K << " with Phi = " << Phi_extremum_K << std::endl << std::endl;
    std::cout << "The result is saved in " << filename << std::endl;
    std::cout << "Please run weighted_variational_test_extremum.py to plot the result." << std::endl;
    std::cout << std::endl;
}


void check_hash_collision(PauliAlg A){
    std::cout << "Test of hash collision:" << std::endl;

    // Check hash collision
    
    // Key: hash value, Value: frequency
    std::unordered_map<size_t, unsigned int> hash_freq;
    for(const auto& [prod, coeff]: A.get_map()){
        size_t hashterm = ankerl::unordered_dense::hash<PauliProd>{}(prod);
        if(hash_freq.count(hashterm) != 0){
            hash_freq[hashterm]++;
        }
        else{
            hash_freq[hashterm] = 1;
        }
    }

    // Key: frequency, Value: count of the frequency
    std::map<unsigned int, unsigned int> freq_count;
    for(const auto& [hashval, freq]: hash_freq){
        if(freq_count.count(freq) != 0){
            freq_count[freq]++;
        }
        else{
            freq_count[freq] = 1;
        }
    }

    // Output
    // e.g. 1 : 103 means that there are 103 terms with unique hash value
    //      2 : 53  means that there are 53 pairs of terms with the same hash value
    //      3 : 2   means that there are 2 triplets of terms with the same hash value
    for(const auto& [freq, count]: freq_count){
        std::cout << freq << " : " << count << std::endl;
    }
}


void single_lambda_test(const int K, const int N, bool is_autoE, bool is_collision_test = true, bool is_save = false){
    std::cout << "===== Test of weighted_variational (single lambda) =====" << std::endl;

    // Number of sites must be a multiple of 4 in this test due to the way the Hamiltonian is constructed.
    // Of course, in general, the number of sites can be any number.
    assert(N % 4 == 0); 
    
    const PauliAlg Id = PauliAlg::identity(N);
    const std::vector<PauliAlg> X = PauliAlg::X_array(N);
    const std::vector<PauliAlg> Y = PauliAlg::Y_array(N);
    const std::vector<PauliAlg> Z = PauliAlg::Z_array(N);

    // ===== Preparation of the problem =====

    // Prepare random number generator for the Hamiltonian
    const int mt_seed = 828495664;
    std::mt19937 rand_engine(mt_seed);
    std::normal_distribution<double> rand_normal(2.0, 1.0); // Specify mean and standard deviation
    
    // Generate a random Hamiltonian
    PauliAlg HX = PauliAlg::zero(N);
    PauliAlg HZ = PauliAlg::zero(N);
    PauliAlg HZZ = PauliAlg::zero(N);
    PauliAlg HYYYY = PauliAlg::zero(N);

    for(int i = 0; i < N; i++){
        HX += rand_normal(rand_engine) * X[i];
        HZ += rand_normal(rand_engine) * Z[i];
        if(i < N-4){
            HZZ -= rand_normal(rand_engine) * Z[i] * Z[i+4];
        }
        if(i % 4 == 0){
            HYYYY -= rand_normal(rand_engine) * Y[i] * Y[i+1] * Y[i+2] * Y[i+3];
        }
    }

    PauliAlg H = 1.3 * HX + 1.1 * HZ + 0.9 * HZZ + 0.7 * HYYYY;
    PauliAlg Hderiv = 1.3 * HX - 1.1 * HZ + 1.8 * HZZ - 1.4 * HYYYY;

    // CD_basis
    std::vector<PauliAlg> CD_basis;
    for(int i = 0; i < N; i+=2){
        CD_basis.push_back(Y[i]);
        CD_basis.push_back(X[i] * Y[i+1]);
    }
    const int M = CD_basis.size();


    // ===== Check Hash Collision =====
    double E = 3.7;
    PauliAlg H_minus_E = H - E * Id;

    PauliAlg H_minus_E_Kth = Id;
    for(int k = 1; k <= K; k++){
        H_minus_E_Kth *= H_minus_E;
    }
    if(is_collision_test){
        check_hash_collision(H_minus_E_Kth);
    }


    // ===== Perform the weighted variational method (the principal algorithm) =====    
    weighted_variational::SingleProblem prob(K, N);
    prob.set_H(H);
    prob.set_Hderiv(Hderiv);
    prob.set_CD_basis(CD_basis);
    
    if(is_autoE){
        prob.set_E_auto();
    }
    else{
        prob.set_E_manual(E);
    }

    std::cout << "The fast-commutator algorithm." << std::endl;
    std::string filename;
    if(is_save){
        filename = "weighted_variational_test_single.txt";
    }
    else{
        filename = "";
    }
    auto [a_fast, E_fast] = prob.compute(filename);


    // ===== Perform the weighted variational method (the alternative algorithm) =====    
    std::cout << "The K-fold commutator algorithm." << std::endl;
    prob.set_E_manual(E_fast);
    std::vector<double> a_Kfold = prob.compute_algorithm2();


    // ===== Compare the result =====
    std::cout << "Comparison of the result:" << std::endl;
    for(int mu = 0; mu < M; mu++){
        std::cout << a_Kfold[mu] << " " << a_fast[mu] << " " << abs(a_Kfold[mu] - a_fast[mu]) << std::endl;
    }
    for(int mu = 0; mu < M; mu++){
        assert(abs(a_Kfold[mu] - a_fast[mu]) < 1e-10);
    }
    std::cout << std::endl;

    std::cout << "All of the driving coefficients are consistent!" << std::endl << std::endl;
}


std::vector<double> H_coeff(double t){
    double T = 2.0;
    return {1 - t/T, t/T, t/T, t/T * (1 - t/T)};
};
std::vector<double> Hderiv_coeff(double t){
    double T = 2.0;
    return {-1/T, 1/T, 1/T, (1 - 2*t/T)/T};
};


void lambda_series_test(const int K, const int N, bool is_autoE, bool is_save = true){
    std::cout << "===== Test of weighted_variational (series) =====" << std::endl;

    // Number of sites must be a multiple of 4 in this test due to the way the Hamiltonian is constructed.
    // Of course, in general, the number of sites can be any number.
    assert(N % 4 == 0); 
    
    const PauliAlg Id = PauliAlg::identity(N);
    const std::vector<PauliAlg> X = PauliAlg::X_array(N);
    const std::vector<PauliAlg> Y = PauliAlg::Y_array(N);
    const std::vector<PauliAlg> Z = PauliAlg::Z_array(N);

    // ===== Preparation of the problem =====

    // Prepare random number generator for the Hamiltonian
    const int mt_seed = 452956305;
    std::mt19937 rand_engine(mt_seed);
    std::normal_distribution<double> rand_normal(2.0, 1.0); // Specify mean and standard deviation
    
    // Generate a random Hamiltonian
    PauliAlg HX = PauliAlg::zero(N);
    PauliAlg HZ = PauliAlg::zero(N);
    PauliAlg HZZ = PauliAlg::zero(N);
    PauliAlg HYYYY = PauliAlg::zero(N);

    for(int i = 0; i < N; i++){
        HX += rand_normal(rand_engine) * X[i];
        HZ += rand_normal(rand_engine) * Z[i];
        if(i < N-4){
            HZZ -= rand_normal(rand_engine) * Z[i] * Z[i+4];
        }
        if(i % 4 == 0){
            HYYYY -= rand_normal(rand_engine) * Y[i] * Y[i+1] * Y[i+2] * Y[i+3];
        }
    }
    std::vector<PauliAlg> H_basis = {HX, HZ, HZZ, HYYYY};
    const int Gamma = H_basis.size();

    // CD_basis
    std::vector<PauliAlg> CD_basis;
    for(int i = 0; i < N; i++){
        CD_basis.push_back(Y[i]);
    }
    for(int i = 0; i < N-1; i++){
        CD_basis.push_back(X[i] * Y[i+1]);
    }
    const int M = CD_basis.size();

    // lambda_series
    const int len_series = 41;
    std::vector<double> lambda_series = linspace(0.0, 2.0, len_series);
    std::vector<double> E_series = linspace(2.0, 4.0, len_series);


    // ===== Perform the weighted variational method for lambda-dependent Hamiltonian =====

    // Compute the basis of matrix Q and vector r
    weighted_variational::SerialProblem prob(K, N);
    prob.set_CD_basis(CD_basis);
    prob.set_H_basis(H_basis);
    prob.compute("weighted_variational_test_trace.txt");

    // Compute the driving coefficients
    weighted_variational::SerialProblemTrace data("weighted_variational_test_trace.txt");
    data.set_H_coeff(H_coeff);
    data.set_Hderiv_coeff(Hderiv_coeff);
    data.set_lambda_series(lambda_series);
    if(is_autoE){
        data.set_E_auto();
    }
    else{
        data.set_E_manual(E_series);
    }

    std::string filename;
    if(is_save){
        filename = "weighted_variational_test_coefficient.txt";
    }
    else{
        filename = "";
    }
    auto [H_coeff_series, CD_coeff_series, E_series_return] = data.compute_driving_coeff(filename);


    // ===== Single-lambda calculation for comparison =====
    int t_index = (int)(len_series * 0.25);
    std::cout << "Comparison at ti = " << t_index << std::endl;
    double t = lambda_series[t_index];

    PauliAlg H_t = PauliAlg::zero(N);
    PauliAlg Hderiv_t = PauliAlg::zero(N);

    for(int gamma = 0; gamma < Gamma; gamma++){
        H_t += H_coeff(t)[gamma] * H_basis[gamma];
        Hderiv_t += Hderiv_coeff(t)[gamma] * H_basis[gamma];
    }

    weighted_variational::SingleProblem prob_single(K, N);
    prob_single.set_H(H_t);
    prob_single.set_CD_basis(CD_basis);
    prob_single.set_Hderiv(Hderiv_t);
    if(is_autoE){
        prob_single.set_E_auto();
    }
    else{
        prob_single.set_E_manual(E_series[t_index]);
    }
    
    auto [a_single, E_single] = prob_single.compute();


    // ===== Compare the result =====
    std::cout << "Comparison of the result:" << std::endl;

    std::vector<double> a_series = CD_coeff_series[t_index];
    for(int mu = 0; mu < M; mu++){
        std::cout << a_series[mu] << " " << a_single[mu] << " " << abs(a_series[mu] - a_single[mu]) << std::endl;
    }
    for(int mu = 0; mu < M; mu++){
        assert((a_series[mu] - a_single[mu]) < 1e-10);
    }
    std::cout << std::endl;

    std::cout << "All of the driving coefficients are consistent!" << std::endl << std::endl;


    // ===== Output the result =====
    // std::cout << std::endl; 
    // std::cout << "Series of the driving coefficients:" << std::endl;

    // for(int i = 0; i < len_series; i++){
    //     std::cout << "[t = " << lambda_series[i] << "] ";
    //     for(int gamma = 0; gamma < Gamma; gamma++){
    //         std::cout << H_coeff_series[i][gamma] << " ";
    //     }
    //     std::cout << "| ";
    //     for(int mu = 0; mu < M; mu++){
    //         std::cout << CD_coeff_series[i][mu] << " ";
    //     }
    //     std::cout << "| ";
    //     std::cout << E_series[i] << std::endl;
    // }
    // std::cout << std::endl;

}




int main(){   
    // ===== Test of combinatoric operations =====
    // The following two tests, `generate_orderings_test` and `make_power_table_test` 
    // are not automatically checked.
    // Verify the results by looking at the output.

    // generate_orderings_test(1, 3);
    // generate_orderings_test(2, 3);
    // generate_orderings_test(3, 4);
    // generate_orderings_test(3, 0);
    // generate_orderings_test(1, 0);

    make_power_table_test(3, 2);
    make_power_table_test(3, 3);
    // make_power_table_test(4, 2);
    // make_power_table_test(1, 2);
    // make_power_table_test(2, 1);
    // make_power_table_test(1, 1);


    // ===== Test of Newtonian optimization =====
    // Verify the results by running the Python script weighted_variational_test_extremum.py
    find_extremum_test(11);

    

    // ===== Test for all possible combinations of the implementation =====
    for (bool is_autoE : {false, true}){
        for(bool is_save : {false, true}){
            single_lambda_test(0, 16, is_autoE, true, is_save);
            single_lambda_test(1, 16, is_autoE, true, is_save);
            single_lambda_test(2, 16, is_autoE, true, is_save);
            single_lambda_test(3, 16, is_autoE, true, is_save);
        }
    }
            
    for(bool is_autoE : {false, true}){
        for(bool is_save : {false, true}){
            lambda_series_test(0, 16, is_autoE, is_save);
            lambda_series_test(1, 16, is_autoE, is_save);
            lambda_series_test(2, 16, is_autoE, is_save);
            lambda_series_test(3, 16, is_autoE, is_save);
        }
    }



    // ===== Test for larger inputs =====
    // single_lambda_test(4, 16, true, false);
    // single_lambda_test(4, 28, true);
    // single_lambda_test(4, 40, true);
    // single_lambda_test(3, 48, true);
    // single_lambda_test(3, 100, true);
    // single_lambda_test(4, 48, true);

    // lambda_series_test(3, 16, true);
    // lambda_series_test(3, 100, true);
    // lambda_series_test(4, 16, true);


    return 0;
}