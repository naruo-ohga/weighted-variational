#include <iostream>
#include <cassert>
#include <unordered_map>
#include <map>
#include <complex>
#include <vector>
#include <random>
#include "pauli_alg.h"
#include "stopwatch.h"
using namespace paulialg;

void test_PauliProd(){
    std::cout << "===== Test of PauliProd =====" << std::endl;

    // Test PauliProd::get_string and PauliProd::is_valid
    std::cout << "The following five output lines are the test of invalid PauliProd objects." << std::endl;
    PauliProd q0 = {};
    PauliProd q1 = {Pauli('X', 0), Pauli('Y', 5), Pauli('Z', 8)};
    PauliProd q2 = {Pauli('X', 3), Pauli('X', 5), Pauli('Z', 8)};
    PauliProd q3 = {Pauli('X', 0), Pauli('C', 5), Pauli('Z', 8)};
    PauliProd q4 = {Pauli('x', 0), Pauli('Y', 5), Pauli('Z', 8)};
    PauliProd q5 = {Pauli('X', 0), Pauli('Z', 8), Pauli('Y', 5)};
    PauliProd q6 = {Pauli('X', 0), Pauli('Y', 5), Pauli('X', 5)};
    PauliProd q7 = {Pauli('X', 0), Pauli('Y', 5), Pauli('Y', 8), Pauli('Z', 3)};

    assert(q0.get_string() == "I");
    assert(q1.get_string() == "X0 Y5 Z8");
    assert(q0.is_valid());
    assert(q1.is_valid());
    assert(q2.is_valid());
    assert(!q3.is_valid());
    assert(!q4.is_valid());
    assert(!q5.is_valid());
    assert(!q6.is_valid());
    assert(!q7.is_valid());


    // Test PauliProd::multiply
    PauliProd p = {Pauli('X', 2), Pauli('Y', 5), Pauli('Z', 8)};
    PauliProd p1 = {Pauli('Z', 1), Pauli('X', 4), Pauli('X', 5), Pauli('Y', 6), Pauli('Z', 7)};
    PauliProd p2 = {Pauli('Z', 1), Pauli('X', 4), Pauli('X', 5), Pauli('Y', 6), Pauli('Z', 8)};
    PauliProd p3 = {Pauli('Z', 1), Pauli('X', 4), Pauli('X', 5), Pauli('Y', 6), Pauli('Y', 8)};
    PauliProd p4 = {Pauli('Z', 1), Pauli('X', 4), Pauli('X', 5), Pauli('Y', 6), Pauli('Z', 10)};
    PauliProd p5 = {Pauli('Z', 1), Pauli('X', 4),                Pauli('Y', 6), Pauli('Z', 10)};
    PauliProd p6 = {Pauli('Z', 2), Pauli('X', 4), Pauli('Z', 5), Pauli('Y', 6), Pauli('Z', 10)};
    PauliProd p7 = {Pauli('Z', 3), Pauli('X', 4), Pauli('Z', 5), Pauli('Y', 6), Pauli('Z', 10)};
    PauliProd p0 = {};

    auto [prod, coeff] = PauliProd::multiply(p, p, false);
    assert(prod.get_string() == "I");
    assert(coeff == 0);
    
    auto [prod1, coeff1] = PauliProd::multiply(p, p1, false);
    assert(prod1.get_string() == "Z1 X2 X4 Z5 Y6 Z7 Z8");
    assert(coeff1 == 3);
    
    auto [prod2, coeff2] = PauliProd::multiply(p, p2, false);
    assert(prod2.get_string() == "Z1 X2 X4 Z5 Y6");
    assert(coeff2 == 3);

    auto [prod3, coeff3] = PauliProd::multiply(p, p3, false);
    assert(prod3.get_string() == "Z1 X2 X4 Z5 Y6 X8");
    assert(coeff3 == 2);

    auto [prod4, coeff4] = PauliProd::multiply(p, p4, false);
    assert(prod4.get_string() == "Z1 X2 X4 Z5 Y6 Z8 Z10");
    assert(coeff4 == 3);
    
    auto [prod5, coeff5] = PauliProd::multiply(p, p5, false);
    assert(prod5.get_string() == "Z1 X2 X4 Y5 Y6 Z8 Z10");
    assert(coeff5 == 0);

    auto [prod6, coeff6] = PauliProd::multiply(p, p6, false);
    assert(prod6.get_string() == "Y2 X4 X5 Y6 Z8 Z10");
    assert(coeff6 == 0);

    auto [prod7, coeff7] = PauliProd::multiply(p, p7, false);
    assert(prod7.get_string() == "X2 Z3 X4 X5 Y6 Z8 Z10");
    assert(coeff7 == 1);

    auto [prod0, coeff0] = PauliProd::multiply(p, p0, false);
    assert(prod0.get_string() == "X2 Y5 Z8");
    assert(coeff0 == 0);

    auto [prod00, coeff00] = PauliProd::multiply(p0, p, false);
    assert(prod00.get_string() == "X2 Y5 Z8");
    assert(coeff00 == 0);


    auto [prod6_f, coeff6_f] = PauliProd::multiply(p, p6, true);
    assert(prod6_f.get_string() == "Y2 X4 X5 Y6 Z8"); // This is incorrect but OK becauase is_commutator = true
    assert(coeff6_f == 0);

    auto [prod7_f, coeff7_f] = PauliProd::multiply(p, p7, true);
    assert(prod7_f.get_string() == "X2 Z3 X4 X5 Y6 Z8 Z10");
    assert(coeff7_f == 1);

    auto [prod3_f, coeff3_f] = PauliProd::multiply(p, p3, true);
    assert(prod3_f.get_string() == "Z1 X2 X4 Z5 Y6 X8");
    assert(coeff3_f == 0);
    
    std::cout << "All tests passed!" << std::endl << std::endl;
}


void test_PauliAlg(){
    std::cout << "===== Test of PauliAlg =====" << std::endl;

    std::cout << "Maximum number of sites: " << num_sites_max << std::endl;

    // ===== Test of initialized operators =====
    const int N = 15;
    PauliAlg Zero = PauliAlg::zero(N);
    PauliAlg Id = PauliAlg::identity(N);

    std::vector<PauliAlg> X = PauliAlg::X_array(N);
    std::vector<PauliAlg> Y = PauliAlg::Y_array(N);
    std::vector<PauliAlg> Z = PauliAlg::Z_array(N);

    assert(Zero.get_string() == "0");
    assert(Id.get_string() == "1.00 I");
    assert(X[0].get_string() == "1.00 X0");
    assert(Y[1].get_string() == "1.00 Y1");
    assert(Z[2].get_string() == "1.00 Z2");


    // ===== Test of changing output precision =====
    PauliAlg::get_string_precision = 3; // change precision
    assert(Z[3].get_string() == "1.000 Z3");
    assert(Y[4].get_string() == "1.000 Y4");
    PauliAlg::get_string_precision = 2; // change precision


    // ===== Test of get_num_sites =====
    assert(Zero.get_num_sites() == N);


    // ===== Test of get_num_terms =====
    assert(Zero.get_num_terms() == 0);
    assert(Id.get_num_terms() == 1);
    assert((X[0] + Y[3]).get_num_terms() == 2);


    // ===== Test of PauliAlg::is_close =====
    assert(PauliAlg::is_close(Zero, Zero));
    assert(PauliAlg::is_close(Id, Id));
    assert(PauliAlg::is_close(X[0], X[0]));

    assert(!PauliAlg::is_close(Zero, Id));
    assert(!PauliAlg::is_close(Id, Zero));
    assert(!PauliAlg::is_close(Zero, X[0]));
    assert(!PauliAlg::is_close(X[0], Zero));
    assert(!PauliAlg::is_close(X[1], Y[2]));

    assert(PauliAlg::is_close(Zero, Zero + Zero));
    assert(PauliAlg::is_close((X[0] + Y[1]) * 1e-15, Zero));
    assert(PauliAlg::is_close(0.1 * X[4] + 0.3 * X[4], 0.4 * X[4]));
    assert(PauliAlg::is_close(0.4 * X[4] + std::complex<double>(0.0,1e-13) * X[4], 0.4 * X[4]));


    // ===== Test of unary operators =====
    PauliAlg A = 0.11 * X[0] + std::complex<double>(0.44,0.88) * Y[1] + Z[2] * Y[3];

    // Unary plus
    assert((+A).get_string() == "0.11 X0 + (0.44+0.88i) Y1 + 1.00 Z2 Y3");
    assert((+(A + Id)).get_string() == "1.00 I + 0.11 X0 + (0.44+0.88i) Y1 + 1.00 Z2 Y3");
    assert((+Zero).get_string() == "0");

    // Unary minus
    assert((-A).get_string() == "- 0.11 X0 + (-0.44-0.88i) Y1 - 1.00 Z2 Y3");
    assert((-(A + Id)).get_string() == "- 1.00 I - 0.11 X0 + (-0.44-0.88i) Y1 - 1.00 Z2 Y3");
    assert((-Zero).get_string() == "0");

    // Unary hermite conjugate
    assert(A.hermite_conj().get_string() == "0.11 X0 + (0.44-0.88i) Y1 + 1.00 Z2 Y3");
    assert((A + Id).hermite_conj().get_string() == "1.00 I + 0.11 X0 + (0.44-0.88i) Y1 + 1.00 Z2 Y3");
    

    // ===== Test of addition and subtraction =====
    PauliAlg B = 0.11 * X[0] - 0.22 * (Y[1] * Z[2]) + 1.25 * Y[3] - 0.01 * Id;
    PauliAlg C = 0.11 * X[0] + 0.22 * (Y[1] * Z[2]) + 0.25 * Z[3];
    assert(B.get_string() == "- 0.01 I + 0.11 X0 - 0.22 Y1 Z2 + 1.25 Y3");
    assert(C.get_string() == "0.11 X0 + 0.22 Y1 Z2 + 0.25 Z3");
    
    // Addition
    assert((B + C).get_string() == "- 0.01 I + 0.22 X0 + 1.25 Y3 + 0.25 Z3");
    assert((C + B).get_string() == "- 0.01 I + 0.22 X0 + 1.25 Y3 + 0.25 Z3");
    assert(((B + Id) + C).get_string() == "0.99 I + 0.22 X0 + 1.25 Y3 + 0.25 Z3");
    assert((B + (C + Id)).get_string() == "0.99 I + 0.22 X0 + 1.25 Y3 + 0.25 Z3");
    assert(((B + Zero) + (C + Zero)).get_string() == "- 0.01 I + 0.22 X0 + 1.25 Y3 + 0.25 Z3");
    assert(((C + Zero) + (B + Zero)).get_string() == "- 0.01 I + 0.22 X0 + 1.25 Y3 + 0.25 Z3");

    assert((B + Zero).get_string() == "- 0.01 I + 0.11 X0 - 0.22 Y1 Z2 + 1.25 Y3");
    assert((Zero + B).get_string() == "- 0.01 I + 0.11 X0 - 0.22 Y1 Z2 + 1.25 Y3");

    assert(((-0.1) * X[0] + 0.100000000000009 * X[0]).get_string() == "0");
    assert(((std::complex<double>(-0.3,-0.1)) * X[0] + std::complex<double>(0.300000000000004,0.100000000000004) * X[0]).get_string() == "0");
    assert(((-0.1) * X[0] + 0.1000000000002 * X[0]).get_string() == "0.00 X0");

    // += operator
    PauliAlg B_copy1 = B;
    B_copy1 += C;
    assert(B_copy1.get_string() == "- 0.01 I + 0.22 X0 + 1.25 Y3 + 0.25 Z3");
    B_copy1 += Zero;
    assert(B_copy1.get_string() == "- 0.01 I + 0.22 X0 + 1.25 Y3 + 0.25 Z3");
    B_copy1 += (1.55 * X[3] + 0.2 * Y[4]);
    assert(B_copy1.get_string() == "- 0.01 I + 0.22 X0 + 1.55 X3 + 1.25 Y3 + 0.25 Z3 + 0.20 Y4");

    // append method
    PauliAlg B_copy2 = B;
    PauliProd pp_Y3({Pauli('Y', 3)});
    PauliProd pp_X1({Pauli('X', 1)});

    B_copy2.append(pp_Y3, -1.25);
    assert(B_copy2.get_string() == "- 0.01 I + 0.11 X0 - 0.22 Y1 Z2");
    B_copy2.append(pp_X1, 1.33);
    assert(B_copy2.get_string() == "- 0.01 I + 0.11 X0 + 1.33 X1 - 0.22 Y1 Z2");

    // Subtraction
    assert((B - C).get_string() == "- 0.01 I - 0.44 Y1 Z2 + 1.25 Y3 - 0.25 Z3");
    assert((C - B).get_string() == "0.01 I + 0.44 Y1 Z2 - 1.25 Y3 + 0.25 Z3");
    assert(((B + Id) - C).get_string() == "0.99 I - 0.44 Y1 Z2 + 1.25 Y3 - 0.25 Z3");
    assert((B - (C + Id)).get_string() == "- 1.01 I - 0.44 Y1 Z2 + 1.25 Y3 - 0.25 Z3");
    assert(((B + Zero) - (C + Zero)).get_string() == "- 0.01 I - 0.44 Y1 Z2 + 1.25 Y3 - 0.25 Z3");
    assert(((C + Zero) - (B + Zero)).get_string() == "0.01 I + 0.44 Y1 Z2 - 1.25 Y3 + 0.25 Z3");

    assert((B - Zero).get_string() == "- 0.01 I + 0.11 X0 - 0.22 Y1 Z2 + 1.25 Y3");
    assert((Zero - B).get_string() == "0.01 I - 0.11 X0 + 0.22 Y1 Z2 - 1.25 Y3");

    assert((0.1 * X[0] - 0.100000000000009 * X[0]).get_string() == "0");
    assert((std::complex<double>(0.3,0.1) * X[0] - std::complex<double>(0.300000000000004,0.100000000000004) * X[0]).get_string() == "0");
    assert((0.1 * X[0] - 0.1000000000002 * X[0]).get_string() == "- 0.00 X0");
    
    PauliAlg::addition_tolerance = 1e-15; // change tolerance
    assert((0.1 * X[0] - 0.100000000000009 * X[0]).get_string() == "- 0.00 X0");
    PauliAlg::addition_tolerance = 1e-13; // change tolerance

    // -= operator
    PauliAlg C_copy = C;
    C_copy -= B;
    assert(C_copy.get_string() == "0.01 I + 0.44 Y1 Z2 - 1.25 Y3 + 0.25 Z3");
    C_copy -= (0.25 * Z[3] + 0.44 * Y[1] * Z[2] + 0.03 * Z[5]);
    assert(C_copy.get_string() == "0.01 I - 1.25 Y3 - 0.03 Z5");


    // ===== Test of multiplication of PauliAlg objects =====

    // Multiplication of two PauliAlg objects
    PauliAlg D = 0.1 * X[0] - 0.2 * Y[0] * Y[1] * Z[2];
    PauliAlg E = 0.2 * Y[0] * Y[1] * Z[2] + 0.3 * X[0];

    assert((D * E).get_string() == "- 0.01 I + 0.08i Z0 Y1 Z2");
    assert(((D + Id) * E).get_string() == "- 0.01 I + 0.30 X0 + 0.20 Y0 Y1 Z2 + 0.08i Z0 Y1 Z2");
    assert((D * (E + Id)).get_string() == "- 0.01 I + 0.10 X0 - 0.20 Y0 Y1 Z2 + 0.08i Z0 Y1 Z2");
    assert(((D + Id) * (E + Id)).get_string() == "0.99 I + 0.40 X0 + 0.08i Z0 Y1 Z2");

    assert((D * Zero).get_string() == "0");
    assert((Zero * E).get_string() == "0");
    assert(((D + Id) * Zero).get_string() == "0");
    assert((Zero * (E + Id)).get_string() == "0");

    // *= operator
    PauliAlg D_copy = D;
    D_copy *= E;
    assert(D_copy.get_string() == "- 0.01 I + 0.08i Z0 Y1 Z2");

    // add_multiply method
    PauliAlg A_copy = A + 0.35 * Id;
    A_copy.add_multiply(D, E);
    assert(A_copy.get_string() == "0.34 I + 0.11 X0 + 0.08i Z0 Y1 Z2 + (0.44+0.88i) Y1 + 1.00 Z2 Y3");


    // ===== Test of scalar multiplication =====

    // Scalar multiplication with std::complex<double>
    assert((std::complex<double>(-0.22,0.33) * Y[5]).get_string() == "(-0.22+0.33i) Y5");
    assert((std::complex<double>(-0.22,0.33) * (Y[5] + Id)).get_string() == "(-0.22+0.33i) I + (-0.22+0.33i) Y5");
    assert((Y[5] * std::complex<double>(-0.22,0.33)).get_string() == "(-0.22+0.33i) Y5");
    assert(((Y[5] + Id) * std::complex<double>(-0.22,0.33)).get_string() == "(-0.22+0.33i) I + (-0.22+0.33i) Y5");

    PauliAlg Y5_copy = Y[5];
    Y5_copy *= std::complex<double>(0.22,-0.33);
    assert(Y5_copy.get_string() == "(0.22-0.33i) Y5");

    assert((std::complex<double>(-0.22,0.33) * Zero).get_string() == "0");
    assert((std::complex<double>(-0.22,0.33) * (Y[5] - Y[5])).get_string() == "0");
    assert((Zero * std::complex<double>(-0.22,0.33)).get_string() == "0");
    assert(((Y[5] - Y[5]) * std::complex<double>(-0.22,0.33)).get_string() == "0");

    assert((std::complex<double>(0.0, 0.0) * X[3]).get_string() == "0");
    assert((std::complex<double>(0.0, 0.0) * (X[3] + Id)).get_string() == "0");
    assert((X[3] * std::complex<double>(0.0, 0.0)).get_string() == "0");
    assert(((X[3] + Id) * std::complex<double>(0.0, 0.0)).get_string() == "0");


    // Scalar multiplication with double
    assert((0.11 * X[3]).get_string() == "0.11 X3");
    assert((0.11 * (X[3] + Id)).get_string() == "0.11 I + 0.11 X3");
    assert((X[3] * (-1.33)).get_string() == "- 1.33 X3");
    assert(((X[3] + Id) * (-1.33)).get_string() == "- 1.33 I - 1.33 X3");

    PauliAlg X3_copy = X[3];
    X3_copy *= 1.33;
    assert(X3_copy.get_string() == "1.33 X3");

    assert((0.11 * Zero).get_string() == "0");
    assert((0.11 * (X[3] - X[3])).get_string() == "0");
    assert((Zero * (-1.33)).get_string() == "0");
    assert(((X[3] - X[3]) * (-1.33)).get_string() == "0");

    assert((0.0 * X[3]).get_string() == "0");
    assert((0.0 * (X[3] + Id)).get_string() == "0");
    assert((X[3] * 0.0).get_string() == "0");
    assert(((X[3] + Id) * 0.0).get_string() == "0");

    assert((1e-18 * X[3]).get_string() == "0.00 X3");

    // Scalar multiplication with int
    assert((3 * Y[3]).get_string() == "3.00 Y3");
    assert((0 * Y[3]).get_string() == "0");


    // ===== Test of get_nontrivial_sites =====
    assert(Zero.get_nontrivial_sites().empty());
    assert(Id.get_nontrivial_sites().empty());
    assert(X[3].get_nontrivial_sites() == std::unordered_set<site_index_t>({3}));
    assert(A.get_nontrivial_sites() == std::unordered_set<site_index_t>({0, 1, 2, 3}));
    assert(D.get_nontrivial_sites() == std::unordered_set<site_index_t>({0, 1, 2}));


    // ===== Test of commutation =====
    assert(PauliAlg::is_close(A * B - B * A, PauliAlg::commutator(A, B)));
    assert(PauliAlg::is_close(A * C - C * A, PauliAlg::commutator(A, C)));
    assert(PauliAlg::is_close(A * D - D * A, PauliAlg::commutator(A, D)));
    assert(PauliAlg::is_close(A * E - E * A, PauliAlg::commutator(A, E)));
    assert(PauliAlg::is_close(B * C - C * B, PauliAlg::commutator(B, C)));
    assert(PauliAlg::is_close(C * D - D * C, PauliAlg::commutator(C, D)));
    assert(PauliAlg::is_close(D * E - E * D, PauliAlg::commutator(D, E)));
    assert(PauliAlg::is_close(PauliAlg::commutator(D, E), D * E - E * D));
    assert(PauliAlg::is_close(Zero, PauliAlg::commutator(D, Id)));
    assert(PauliAlg::is_close(Zero, PauliAlg::commutator(D, Zero)));

    PauliAlg G = X[0]*Y[1] + Y[1]*Z[2]*X[3];
    std::vector<PauliAlg> commutator_array = PauliAlg::commutator_for_each(G, {Id, Zero, Y[2], B, C, D, E, X[0]*Y[1], Y[0]*Z[1]});
    assert(PauliAlg::is_close(commutator_array[0], PauliAlg::commutator(G, Id)));
    assert(PauliAlg::is_close(commutator_array[1], PauliAlg::commutator(G, Zero)));
    assert(PauliAlg::is_close(commutator_array[2], PauliAlg::commutator(G, Y[2])));
    assert(PauliAlg::is_close(commutator_array[3], PauliAlg::commutator(G, B)));
    assert(PauliAlg::is_close(commutator_array[4], PauliAlg::commutator(G, C)));
    assert(PauliAlg::is_close(commutator_array[5], PauliAlg::commutator(G, D)));
    assert(PauliAlg::is_close(commutator_array[6], PauliAlg::commutator(G, E)));
    assert(PauliAlg::is_close(commutator_array[7], PauliAlg::commutator(G, X[0]*Y[1])));
    assert(PauliAlg::is_close(commutator_array[8], PauliAlg::commutator(G, Y[0]*Z[1])));


    // ===== Test of trace operations =====
    assert(PauliAlg::trace_normalized(Zero) == 0.0);
    assert(PauliAlg::trace_normalized(Id) == 1.0);
    assert(PauliAlg::trace_normalized(B) == -0.01);
    assert(PauliAlg::trace_normalized(B + Id) == 0.99);

    assert(abs(PauliAlg::trace_normalized_multiply(B, A) - PauliAlg::trace_normalized(B*A)) < 1e-13);
    assert(abs(PauliAlg::trace_normalized_multiply(A, B) - PauliAlg::trace_normalized(B*A)) < 1e-13);
    assert(abs(PauliAlg::trace_normalized_multiply(B, C) - PauliAlg::trace_normalized(B*C)) < 1e-13);
    assert(abs(PauliAlg::trace_normalized_multiply(C, B) - PauliAlg::trace_normalized(B*C)) < 1e-13);
    assert(abs(PauliAlg::trace_normalized_multiply(B, D) - PauliAlg::trace_normalized(B*D)) < 1e-13);
    assert(abs(PauliAlg::trace_normalized_multiply(D, B) - PauliAlg::trace_normalized(B*D)) < 1e-13);
    assert(abs(PauliAlg::trace_normalized_multiply(B, E) - PauliAlg::trace_normalized(B*E)) < 1e-13);
    assert(abs(PauliAlg::trace_normalized_multiply(E, B) - PauliAlg::trace_normalized(B*E)) < 1e-13);
    
    // ===== Test of write_file =====
    PauliAlg F = std::complex<double>(0.0, 3.0) * Id + 1e-40 * X[0] + std::complex<double>(1.4e5, 2.8e10) * Y[1] - 0.01 * Z[2] * Y[3] * X[12];
    PauliAlg::write_to_file({D, F}, "pauli_alg_test_output.txt");
    std::cout << "Please check pauli_alg_test_output.txt and run weighted_variational_test_utility.py" << std::endl;

    std::cout << "All tests passed!" << std::endl << std::endl;
}


void test_hash_collision(int N){
    std::cout << "===== Test of hash collision =====" << std::endl;

    const std::vector<PauliAlg> X = PauliAlg::X_array(N);
    const std::vector<PauliAlg> Z = PauliAlg::Z_array(N);

    PauliAlg H = PauliAlg::zero(N);

    for(int i = 0; i < N; i++){
        H += Z[i];
        H += X[i];
        H += Z[i] * Z[(i+1)%N];
        H += Z[i] * Z[(i+2)%N];
        H += Z[i] * Z[(i+3)%N];
    }
    std::cout << H.get_num_terms() << std::endl;
    
    StopWatch stopwatch;
    stopwatch.start();

    PauliAlg H2 = H * H;
    std::cout << H2.get_num_terms() << std::endl;
    
    PauliAlg H3 = H2 * H;
    std::cout << H3.get_num_terms() << std::endl;

    std::cout << "Time taken for multiplication: " << stopwatch.stop() << " sec" << std::endl;

    // ===== Check hash collision =====
    
    // Key: hash value, Value: frequency
    std::unordered_map<size_t, unsigned int> hash_freq;
    for(auto& [prod, coeff]: H3.get_map()){
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
    for(auto& [hashval, freq]: hash_freq){
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
    for(auto& [freq, count]: freq_count){
        std::cout << freq << " : " << count << std::endl;
    }
    std::cout << std::endl;
}


void test_large_system(const int N){
    std::cout << "===== Test for large systems =====" << std::endl;

    // ===== Elementary operators =====
    const std::vector<PauliAlg> X = PauliAlg::X_array(N);
    const std::vector<PauliAlg> Y = PauliAlg::Y_array(N);
    const std::vector<PauliAlg> Z = PauliAlg::Z_array(N);
    const PauliAlg Id = PauliAlg::identity(N);

    // ===== Example PauliAlg objects =====
    PauliAlg A(N), B(N), a(N);

    for(int i = 0; i < N; i++){
        A += X[i];
        B += 2.0 * Y[i];
    }
    for(int i = 0; i < N-2; i++){
        B += Z[i] * Z[i+2];
    }
    a = X[0] + X[1] + X[0]*X[1];

    std::cout<< "# of terms of A: " << A.get_num_terms() << std::endl;
    std::cout<< "# of terms of B: " << B.get_num_terms() << std::endl;

    // ===== Multiplication =====
    StopWatch stopwatch;
    stopwatch.start();
    PauliAlg AB = A * B;
    PauliAlg BA = B * A;
    PauliAlg aB = a * B;
    PauliAlg Ba = B * a;
    std::cout << "Time taken for multiplication: " << stopwatch.stop() << " sec" << std::endl;
    
    PauliAlg AB_ans(N);
    PauliAlg BA_ans(N);
    PauliAlg aB_ans(N);
    PauliAlg Ba_ans(N);

    // AB_ans
    stopwatch.start();
    for(int i = 0; i < N; i++){
        AB_ans += std::complex<double>(0,2.0) * Z[i];
    }
    for(int i = 0; i < N-2; i++){
        AB_ans -= std::complex<double>(0,1.0) * Y[i] * Z[i+2];
        AB_ans -= std::complex<double>(0,1.0) * Z[i] * Y[i+2];
    }
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(i != j){
                AB_ans += 2.0 * X[i] * Y[j];
            }
            if(i != j && i != j+2 && j < N-2){
                AB_ans += X[i] * Z[j] * Z[j+2];
            }
        }
    }

    BA_ans = AB_ans.hermite_conj();

    // aB_ans
    aB_ans += std::complex<double>(0,2.0) * (Z[0] + Z[1] + Z[0]*X[1] + X[0]*Z[1]);
    aB_ans += 2 * (X[0]*Y[1] + X[1]*Y[0]);
    aB_ans += std::complex<double>(0,-1.0) * (Y[0]*Z[2] + Y[0]*X[1]*Z[2] + Y[1]*Z[3] + X[0]*Y[1]*Z[3]);
    aB_ans += X[1]*Z[0]*Z[2] + X[0]*Z[1]*Z[3];
    for(int i = 2; i < N; i++){
        aB_ans += 2 * Y[i] * (X[0] + X[1] + X[0]*X[1]);
    }
    for(int i = 2; i < N-2; i++){
        aB_ans += Z[i] * Z[i+2] * (X[0] + X[1] + X[0]*X[1]);
    }

    Ba_ans = aB_ans.hermite_conj();

    std::cout << "Time taken for preparing answers: " << stopwatch.stop() << " sec" << std::endl;

    // Comparison
    stopwatch.start();
    PauliAlg AB_diff = AB - AB_ans;
    PauliAlg aB_diff = aB - aB_ans;
    std::cout << "AB_diff = " << AB_diff.get_string() << std::endl;
    std::cout << "aB_diff = " << aB_diff.get_string() << std::endl; 
    assert(PauliAlg::is_close(AB, AB_ans));
    assert(PauliAlg::is_close(aB, aB_ans));
    
    std::cout << "Time taken for comparison: " << stopwatch.stop() << " sec" << std::endl;
    
    // ===== Comutator =====
    stopwatch.start();
    PauliAlg comm_AB = PauliAlg::commutator(A, B);
    PauliAlg comm_aB = PauliAlg::commutator(a, B);
    std::cout << "Time taken for commutator: " << stopwatch.stop() << " sec" << std::endl;

    PauliAlg comm_AB_diff = comm_AB - (AB_ans - BA_ans);
    PauliAlg comm_aB_diff = comm_aB - (aB_ans - Ba_ans);

    std::cout << "comm_AB_diff = " << comm_AB_diff.get_string() << std::endl;
    std::cout << "comm_aB_diff = " << comm_aB_diff.get_string() << std::endl;
    assert(PauliAlg::is_close(comm_AB, AB_ans - BA_ans));
    assert(PauliAlg::is_close(comm_aB, aB_ans - Ba_ans));


    // ===== Trace operations =====
    std::complex<double> trace_BB1 = PauliAlg::trace_normalized_multiply(B, B + Id);
    std::complex<double> trace_BB2 = PauliAlg::trace_normalized_multiply(B + Id, B);
    assert(abs(trace_BB1 - std::complex<double>(4*N + N-2, 0)) < 1e-13);
    assert(abs(trace_BB2 - std::complex<double>(4*N + N-2, 0)) < 1e-13);

    std::cout << "All tests passed!" << std::endl << std::endl;
}


void test_very_large_system(const int N){
    std::cout << "===== Test for a very large system =====" << std::endl;

    // ===== Elementary operators =====
    const std::vector<PauliAlg> X = PauliAlg::X_array(N);
    const std::vector<PauliAlg> Y = PauliAlg::Y_array(N);
    const std::vector<PauliAlg> Z = PauliAlg::Z_array(N);
    const PauliAlg Id = PauliAlg::identity(N);

    // ===== Example PauliAlg objects =====
    std::mt19937 rand_engine(13257);
    std::uniform_real_distribution<double> rand_uniform(-1.0, 1.0);
    std::vector<double> coeff(N);
    for(int i = 0; i < N; i++){
        coeff[i] = rand_uniform(rand_engine);
    }

    PauliAlg A(N);
    for(int i = 0; i < N; i++){
        A += coeff[i] * X[i];
    }   
   
    // ===== Multiplication =====
    StopWatch stopwatch;
    stopwatch.start();
    PauliAlg AAA = A * A * A;
    std::cout << "Time taken for multiplication: " << stopwatch.stop() << " sec" << std::endl;

    // ===== Test =====
    stopwatch.start();
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                double coeff_ijk = coeff[i] * coeff[j] * coeff[k];
                if(i == j && j == k){
                    AAA -= coeff_ijk * X[i];
                }
                else if(i == j){
                    AAA -= coeff_ijk * X[k];
                }
                else if(j == k){
                    AAA -= coeff_ijk * X[i];
                }
                else if(k == i){
                    AAA -= coeff_ijk * X[j];
                }
                else if(i != j && j != k && k != i){
                    AAA -= coeff_ijk * X[i] * X[j] * X[k];
                }
            }
        }
    }
    std::cout << "Time taken for subtraction: " << stopwatch.stop() << " sec" << std::endl;

    PauliAlg::get_string_precision = 15;
    // std::cout << "AAA - AAA = " << AAA.get_string() << std::endl;
    std::cout << "Maximum error: " << PauliAlg::max_coeff_diff(AAA, PauliAlg::zero(N)) << std::endl;
    assert(PauliAlg::is_close(AAA, PauliAlg::zero(N), 1e-10));
    std::cout << "All tests passed!" << std::endl << std::endl;
}


int main(){
    test_PauliProd();
    test_PauliAlg();
    test_hash_collision(60);
    test_large_system(1000);
    test_very_large_system(300);
    return 0;
}
