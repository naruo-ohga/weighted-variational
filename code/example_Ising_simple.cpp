#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include <string>
#include "pauli_alg.h"
#include "weighted_variational.h"
using namespace std;
using paulialg::PauliAlg; // To abbreviate paulialg::PauliAlg to PauliAlg


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



int main(){
    // ========== Preparation of input data ==========
    // Constants
    const int width = 4;          // Width of the 2D lattice
    const int height = 3;         // Height of the 2D lattice
    const int N = width * height; // Number of sites 
    const int K = 3;              // Degree of the method
    
    // Load elemntary Pauli operators
    const PauliAlg Id = PauliAlg::identity(N);       // The identity operator
    const vector<PauliAlg> X = PauliAlg::X_array(N); // X[i] is the Pauli X operator acting on the ith site
    const vector<PauliAlg> Y = PauliAlg::Y_array(N); // Y[i] is the Pauli Y operator acting on the ith site
    const vector<PauliAlg> Z = PauliAlg::Z_array(N); // Z[i] is the Pauli Z operator acting on the ith site

    // Prepare generator of gamma-distributed random numbers
    const int mt_seed = 270397947; 
    std::mt19937 rand_engine(mt_seed);
    const double rand_mean = 1.0;
    const double rand_std = 0.5;
    std::gamma_distribution<double> rand_gamma(rand_mean*rand_mean/rand_std/rand_std, rand_std*rand_std/rand_mean);

    // Prepare the lambda-independent basis operators of the Hamiltonian
    PauliAlg HX = PauliAlg::zero(N);
    PauliAlg HZ = PauliAlg::zero(N);
    PauliAlg HZZ = PauliAlg::zero(N);

    for(int x = 0; x < width; x++){
        for(int y = 0; y < height; y++){
            int i = x + y*width; // The index of the site in the 2D lattice
            
            HX += 1.0 * X[i];
            HZ += (1.0 * rand_gamma(rand_engine)) * Z[i];

            if(y < height - 1){
                HZZ -= rand_gamma(rand_engine) * Z[i] * Z[i+width];
            }
            if(x < width - 1){
                HZZ -= rand_gamma(rand_engine) * Z[i] * Z[i+1];
            }
        }
    }

    vector<PauliAlg> H_basis = {HX, HZ + HZZ}; // Collect the Hamiltonian operators in a vector
    PauliAlg::write_to_file(H_basis, "example_Ising_simple_data/H_basis.txt"); // Save the Hamiltonian operators to a file
    
    // Prepare lambda-independent basis operators of the CD-driving
    vector<PauliAlg> CD_basis;
    for(int i = 0; i < N; i++){
        CD_basis.push_back(Y[i]); // Use the Pauli Y operators as the basis 
    }
    PauliAlg::write_to_file(CD_basis, "example_Ising_simple_data/CD_basis.txt"); // Save the driving operators to a file

    // Values of lambda for which coefficients are computed
    const int len_series = 401;
    vector<double> lambda_series = linspace(0.0, 1.0, len_series);


    // ========== Run the biased variational method ==========
    // parameter string for the output filename
    string param_str = "_K=" + std::to_string(K); 

    // Compute the traces of operators
    weighted_variational::SerialProblem prob(K, N); // Create an instance of the problem
    prob.set_H_basis(H_basis);                      // Set the basis of the Hamiltonian   
    prob.set_CD_basis(CD_basis);                    // Set the basis of the CD-driving
    prob.compute("example_Ising_simple_data/trace" + param_str + ".txt"); // Compute the traces of operators and save them to a file

    // Compute the lambda-dependent coefficients
    // This step loads the traces from the file without using the data of the trace on memory. It can be in a separate program.
    weighted_variational::SerialProblemTrace data("example_Ising_simple_data/trace" + param_str + ".txt"); // Load the traces from the file and store them in an instance of the class
    data.set_H_coeff(H_coeff);              // Set the coefficients of the Hamiltonian  
    data.set_Hderiv_coeff(Hderiv_coeff);    // Set the coefficients of the lambda-derivatives of the Hamiltonian
    data.set_lambda_series(lambda_series);  // Set the series of lambda for which the coefficients are computed
    data.set_E_auto();                      // Set the energy shift to be automatically computed
    auto [H_coeff_series, CD_coeff_series, E_series] = data.compute_driving_coeff("example_Ising_simple_data/coefficients" + param_str + ".txt"); // Compute the driving coefficients and save them to a file
    
    // The output variables contain the following data:
    // H_coeff_series[i][gamma] = (The coefficient of H_basis[gamma] at lambda = lambda_series[i])
    // CD_coeff_series[i][mu] = (The coefficient of CD_basis[mu] at lambda = lambda_series[i])
    // E_series[i] = (The energy shift at lambda = lambda_series[i])
    // These data are also automatically saved to a file in the previous step.

    return 0;
}