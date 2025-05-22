# Weighted variational method

This code implements the **weighted variational method** for constructing an approximate counterdiabatic (CD) driving protocol to enhance ground-state evolution of arbitrary spin-1/2 systems. See our paper at [arXiv:25xx.xxxxx](https://arxiv.org/abs/25xx.xxxxx).

The code consists of two parts:
* C++ code for calculating the driving coefficients by the weighted variational method.
* Python code that supports quantum simulation of the resulting CD driving.

The C++ code is the main part of this repository. The Python code is optional; one can use other ways to realize the CD driving, such as real quantum systems and other classical simulators.


## I. Quick start

### Prepare libraries

1. Download the `code/` directory (or the entire repository).
2. The C++ code requires the `eigen` linear algebra package. Please follow the [getting started guide](https://eigen.tuxfamily.org/dox/GettingStarted.html) to install `eigen` and make sure that "A simple first program" on the website runs in your environment.
3. The Python code requires the `QuTip` quantum simulation package. Use the command `pip install qutip` to install qutip in your Python environment. We also need three standard packages, `numpy`, `matplotlib`, and `scipy`.

### Run the example code 

The example codes `example_Ising_simple.cpp` and `example_Ising_simple.py` perform the weighted variational method for a ferromagnetic Ising system with the Hamiltonian 
```math
H(\lambda) = (1-\lambda)\sum_{i = 1}^{N}X_{i}+\lambda\left(\sum_{i = 1}^{N}h_{i}Z_{i} - \!\!\sum_{(i,j) \in \Lambda_{\mathrm{NN}}}\!\! J_{ij}Z_{i}Z_{j}\right)
```
for $`0 \leq \lambda \leq 1`$, where the coefficients $`h_{i}`$ and $`J_{ij}`$ are randomly sampled as described in Sec. IV A of the paper. The system size is $`N = 12`$, and the degree of the method is $`K = 3`$.

The example codes run in the following steps.

1. Compile the C++ code by the following command: <br>
   ```g++ -O2 example_Ising_simple.cpp -std=c++17 -I/usr/include/eigen3 -o example_Ising_simple``` <br>
   The path `usr/include/eigen3` should be replaced by the installation path of the eigen package in your environment. Our code requires C++17, as indicated by `-std=c++17` option. This command also enables compiler optimization through the `-O2` option to speed up the calculation.
2. Run the compiled C++ program using the command `./example_Ising_simple`. The C++ program will generate output files in the directory `example_Ising_simple_data/` (make sure that the directory exists; otherwise, the C++ program raises an error).
3. Run the Python program by `python3 example_Ising_simple.py` to perform the quantum simulation. The Python program loads the output of the C++ program and outputs results in the same directory.

The C++ program generates the following four files:
                                  
* `H_basis.txt` contains the $`\lambda`$-independent basis operators of the Hamiltonian ($`F_\gamma`$ in the paper).
* `CD_basis.txt` contains the basis operators of the CD driving ($`A_\mu`$ in the paper).
* `trace_K=3.txt` contains the traces of the operators ($`\tilde{r}_{\mu,gg'}`$, $`\tilde{Q}_{\mu\nu,gg'}`$, $`\tilde{\omega}_g`$, and $`\tilde{\omega}_{gg'}`$ in the paper). These are intermediate results used during the calculation of the driving coefficients. 
* `coefficients_K=3.txt` contains the final result, i.e., the $`\lambda`$-dependent driving coefficients of the driving terms ($`\alpha_\mu(\lambda)`$ in the paper), along with the $`\lambda`$-dependent coefficients of the Hamiltonian ($`f_\gamma(\lambda)`$ in the paper).
 
This Python program will automatically load `H_basis.txt`, `CD_basis.txt`, and `coefficients_K=3.txt` and perform a quantum simulation. The program will output the following three files:

* `eigenenergy_K=3.txt` contains time-dependent eigenenergies $`\epsilon_n(\lambda_t)`$ of the Hamiltonian.
* `overlap_K=3.txt` contains the time-dependent overlap between instantaneous eigenstates and the time-evolved state $`\vert \langle \phi_n(\lambda_t) \vert \psi^{(K)}(t) \rangle \vert^2`$.
* `result_K=3.pdf` contains simple plots of the results.

The Python program only calculates the three highest and three lowest eigenenergies and the corresponding eigenstates. The file `eigenenergy_K=3.txt` only contains the eigenenergies $`\epsilon_1(\lambda), \epsilon_2(\lambda), \epsilon_3(\lambda), \epsilon_{D-2}(\lambda), \epsilon_{D-1}(\lambda), \epsilon_D(\lambda)`$, where $`D`$ is the dimension of the Hilbert space, and similarly for the `overlap_K=3.txt` file.


### Modify the example

One can modify the example in many ways. The following are some examples:

* Change the system size by modifying `const int width = 4` and `const int height = 3` in `example_Ising_simple.cpp`.
* Change the degree of the method by modifying `const int K = 3` in `example_Ising_simple.cpp` and `K = 3` in `example_Ising_simple.py`. Any integer $`K\geq0`$ is supported. When $`K=0`$, the C++ code returns all zero driving coefficients, and the Python program generates the time evolution by the original Hamiltonian without CD driving (corresponding to $`K = \varnothing`$ in the paper).
* Change the schedule of $`\lambda`$ by modifying the three functions, `schedule(t)`, `schedule_inv(lam)`, and `schedule_derivative(t)`, in `example_Ising_simple.py`.
* Add other kinds of basis operators to CD driving by appending some elements to the vector of PauliAlg objects `CD_basis` in `example_Ising_simple.cpp`. For example, try `CD_basis.push_back(Y[0]*Z[1] + Y[1]*Z[0])`.
* Change the Hamiltonian by modifying l.56-76 of `example_Ising_simple.cpp`. Any Hamiltonian on the spin-1/2 system can be constructed by combining the Pauli operators `X[i]`, `Y[i]`, `Z[i]`, and `Id` of the `PauliAlg` class, as detailed in the next section.


### Use PauliAlg class

A `PauliAlg` object represents a linear combination of Pauli operators. To use the `PauliAlg` class, include the header file in your C++ program as follows:
```C++
#include "pauli_alg.h"
using paulialg::PauliAlg; // To abbreviate paulialg::PauliAlg to PauliAlg
```

To generalte a PauliAlg object, load elementary PauliAlg objects by specifying the number of sites and combine them. For example:
```C++
const int N = 6; // Number of sites

const PauliAlg Id = PauliAlg::identity(N);            // Load the identity operator
const std::vector<PauliAlg> X = PauliAlg::X_array(N); // Load the Pauli-X operator
const std::vector<PauliAlg> Y = PauliAlg::Y_array(N); // Load the Pauli-Y operator
const std::vector<PauliAlg> Z = PauliAlg::Z_array(N); // Load the Pauli-Z operator

PauliAlg H = PauliAlg::zero(N); // Initialize with the zero operator. PauliAlg H(N); is also allowed and equivalent.
H += 3.0 * Id + 1.5 * X[3] * Y[5] - std::complex<double>(0.0, 0.5) * Z[8]; // 3.0 I + 1.5 X3 Y5 - 0.5i Z8
```

The following operations are supported. 
`A`, `B`, `B1`, `B2` are PauliAlg objects with the number of sites `N`, and `c` is a scalar (`complex<double>`, `double`, or `int`):
```C++
+A; -A;                                      // Unary plus and minus
A.hermite_conj();                            // Hermitian conjugate
A + B; A += B;                               // Addition
A - B; A -= B;                               // Subtraction
A * B; A *= B;                               // Multiplication of two PauliAlg objects
A * c; c * A; A *= c;                        // Multiplication by a scalar
PauliAlg::commutator(A, B);                  // Commutator [A, B] = AB - BA
PauliAlg::commutator_for_each(A, {B1, B2});  // Commutator [A, B1] and [A, B2] (with arbitrary number of operators B1, B2, ...)
PauliAlg::trace_normalized(A);               // Trace Tr(A) / 2^N (returns a complex<double> number)
PauliAlg::trace_normalized_multiply(A, B);   // Trace Tr(AB) / 2^N (returns a complex<double> number)
```
The following expressions are not supported:
```C++
// A / c  ->  Use (1.0 / c) * A` instead
// A + c  ->  Use A + c * PauliAlg::identity(N) instead
// A - c  ->  Use A - c * PauliAlg::identity(N) instead
```

### Enable parallelization

The C++ code supports OpenMP parallelization. While the parallelization is disabled in the above procedure, it can be easily activated by installing OpenMP to your environment and adding a few options when compiling the C++ program without modifying any part of the code. For example, in my environment of Clang compiler on macOS with OpenMP installed via Homebrew, the command appears as <br>
```g++ -O2 -Xpreprocessor -fopenmp -L/opt/homebrew/opt/libomp/lib/ -I/opt/homebrew/opt/libomp/include -I/usr/include/eigen3 -lomp example_Ising_simple.cpp -std=c++17 -o example_Ising_simple```


## II. File contents

This repository contains the following directories.

### code/

This directory contains the C++ and Python packages along with example files.

The C++ package is made up of the following six files.
* `pauli_alg.h`: Implementation of basic algebraic operations (such as addition, multiplication, and trace) for arbitrary spin-1/2 systems. See below for usage.
* `weighted_variational.h`: Implementation of the weighted variational method for ground-state evolution for arbitrary spin-1/2 systems.
* `unordered_dense.h`, `stopwatch.h`, `vector_hash.h`: Helper functions loaded within `pauli_alg.h`. 
* `weighted_variational_utility.h`: Helper functions loaded within `weighted_variational.h`.

The Python package contains only one file.
* `weited_variational_utility.py`: Utility functions to support quantum simulation in Python.

The following files are for demonstration.
* `example_Ising_simple.cpp`: Example code for the C++ part.
* `example_Ising_simple.py`: Example code for the Python part.
* `example_Ising_simple_data/`: Containing the expected output of the example code. You should be able to reproduce this by running the example code.

### paper/

This directory contains codes we used to perform the numerical tests and generate figures in the paper. It also contains the data of the final fidelity (`example_Ising_final_fidelity.tsv`).

### test/

This directory contains test codes we used to check the validity of the C++ and Python codes.


## III. LICENSE

All code in this repository, except for `unordered_dense.h`, is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0) in accordance with the copyright policy of NTT Corporation. Please refer to the LICENSE file for more details. If you wish to use the code beyond the scope of this license, please feel free to contact us.

The file `unordered_dense.h` was copied from https://github.com/martinus/unordered_dense. It is copyrighted by Martin Leitner-Ankerl and is licensed under the MIT License. For details, please refer to the license notice included in the header of the file.
