#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include "pauli_alg.h"
#include "weighted_variational.h"
using namespace std;
using paulialg::PauliAlg;


/// Returns a vector x of n evenly spaced numbers between a and b such that
/// x[0] = a, x[n-1] = b.
vector<double> linspace(double a, double b, int n){
    vector<double> v;
    for(int i = 0; i < n; i++){
        v.push_back(a + ((double)i / (double)(n - 1)) * (b - a));
    }
    return v;
}



// Coefficients of the lambda-dependent Hamiltonian
vector<double> H_coeff(double lambda){
    return {1.0 - lambda, lambda};
}
vector<double> Hderiv_coeff(double lambda){
    return {-1.0, 1.0};
}



int main(int argc, char* argv[]){
    // ========== Read parameters ==========
    if(argc != 5){
        cout << "Invalid arguments." << endl;
        cout << "Usage: ./example_Ising <Hamiltonian_name> <driving_name> <seed_idx> <width>" << endl;
        exit(1);
    }
    
    // Hamiltonian
    const string Hamiltonian_name = argv[1];
    vector<string> Hamiltonian_list = {"ferromagnetic", "antiferromagnetic", "Gaussian"};
    if(find(Hamiltonian_list.begin(), Hamiltonian_list.end(), Hamiltonian_name) == Hamiltonian_list.end()){
        cout << "Invalid Hamiltonian name" << endl;
        exit(1);
    }

    // Driving scheme
    const string driving_name = argv[2];
    vector<string> driving_list = {"", "-YZ"};
    if(find(driving_list.begin(), driving_list.end(), driving_name) == driving_list.end()){
        cout << "Invalid driving name" << endl;
        exit(1);
    }

    // Seed for random number generator
    const int mt_seed_offset = std::atoi(argv[3]); 
    
    // The name of the system
    const string system_name = Hamiltonian_name + "-seed" + to_string(mt_seed_offset) + driving_name;
    const string folder_name = "example_Ising_data_" + system_name + "/";
    
    // Define the lattice
    const int width = std::atoi(argv[4]); // Change this among 3, 4, 5
    const int height = 3;
    const int N = width * height; // Number of sites 

    
    // ========== Preparation of input data ==========

    // Load elemntary Pauli operators
    const PauliAlg Id = PauliAlg::identity(N);
    const vector<PauliAlg> X = PauliAlg::X_array(N);
    const vector<PauliAlg> Y = PauliAlg::Y_array(N);
    const vector<PauliAlg> Z = PauliAlg::Z_array(N);

    // Prepare random number generator for the Hamiltonian
    const int mt_seed = 270385947 + 1000*N + mt_seed_offset;

    std::mt19937 rand_engine(mt_seed);
    const double rand_mean = 1.0;
    const double rand_std = 0.5;
    std::gamma_distribution<double> rand_gamma(rand_mean*rand_mean/rand_std/rand_std, rand_std*rand_std/rand_mean);
    std::normal_distribution<double> rand_normal(0.0, 1.0);

    // Basis of the Hamiltonian
    PauliAlg HX = PauliAlg::zero(N);
    PauliAlg HZ = PauliAlg::zero(N);
    PauliAlg HZZ = PauliAlg::zero(N);

    for(int x = 0; x < width; x++){
        for(int y = 0; y < height; y++){
            int i = x + y*width;
            
            HX += 1.0 * X[i];
            if(i == 0){
                HZ += (1.0 * rand_gamma(rand_engine)) * Z[i];
            }
            else{
                HZ += (1.0 * rand_gamma(rand_engine)) * Z[i];
            }

            if(y < height - 1){
                if(Hamiltonian_name == "ferromagnetic"){
                    HZZ -= rand_gamma(rand_engine) * Z[i] * Z[i+width];
                }
                else if(Hamiltonian_name == "antiferromagnetic"){
                    HZZ += rand_gamma(rand_engine) * Z[i] * Z[i+width];
                }   
                else if(Hamiltonian_name == "Gaussian"){
                    HZZ += rand_normal(rand_engine) * Z[i] * Z[i+width];
                }
            }

            if(x < width - 1){
                if(Hamiltonian_name == "ferromagnetic"){
                    HZZ -= rand_gamma(rand_engine) * Z[i] * Z[i+1];
                }
                else if(Hamiltonian_name == "antiferromagnetic"){
                    HZZ += rand_gamma(rand_engine) * Z[i] * Z[i+1];
                }
                else if(Hamiltonian_name == "Gaussian"){
                    HZZ += rand_normal(rand_engine) * Z[i] * Z[i+1];
                }
            }
        }
    }

    vector<PauliAlg> H_basis = {HX, HZ + HZZ};
    PauliAlg::write_to_file(H_basis, folder_name + "example_Ising_H_basis_N=" + to_string(N) + ".txt");
    
    // Basis of CD-driving
    vector<PauliAlg> CD_basis;
    for(int i = 0; i < N; i++){
        CD_basis.push_back(Y[i]);
    }

    for(int x = 0; x < width; x++){
        for(int y = 0; y < height; y++){
            int i = x + y*width;

            if(y < height - 1){
                if(driving_name == "-YZ"){
                    CD_basis.push_back(Y[i]*Z[i+width] + Z[i]*Y[i+width]);
                }
            }
            
            if(x < width - 1){
                if(driving_name == "-YZ"){
                    CD_basis.push_back(Y[i]*Z[i+1] + Z[i]*Y[i+1]);
                }
            }            
        }
    }

    PauliAlg::write_to_file(CD_basis, folder_name + "example_Ising_CD_basis_N=" + to_string(N) + ".txt");

    // Values of lambda for which coefficients are computed
    const int len_series = 401;
    vector<double> lambda_series = linspace(0.0, 1.0, len_series);


    // ========== Run the biased variational method ==========
    const int K_max = 5; // Order of the method

    for(int K = 0; K <= K_max; K++){
        // Prepare string for the output file
        string param_str = "_N=" + std::to_string(N) + "_K=" + std::to_string(K) + ".txt";

        // Compute the basis of matrix Q and vector r
        weighted_variational::SerialProblem prob(K, N);
        prob.set_CD_basis(CD_basis);
        prob.set_H_basis(H_basis);
        prob.compute(folder_name + "example_Ising_trace" + param_str);

        // Compute the time series
        weighted_variational::SerialProblemTrace data(folder_name + "example_Ising_trace" + param_str);
        data.set_H_coeff(H_coeff);
        data.set_Hderiv_coeff(Hderiv_coeff);
        data.set_lambda_series(lambda_series);
        data.set_E_auto();
        auto [H_coeff_series, CD_coeff_series, E_series] = data.compute_driving_coeff(folder_name + "example_Ising_coefficients" + param_str);
    }
    
    return 0;
}