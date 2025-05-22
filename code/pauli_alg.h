/**
 * pauli_alg.h: A fast library for manipulating a linear combination of tensor products of Pauli operators.
 * It supports computations such as `(3.0 I + 1.5 X3 Y5 - 0.5i Z8) * (3.0 X5) = 9.0 X5 - 4.5 X3 Z5 - 1.5i X5 Z8`.
 * `I` is the identity operator.
 * `Xi` is the Pauli operator X acting on site i, and similarly for `Yi` and `Zi`.
 * `Xi Yj` represents the tensor product of `Xi`, `Yj`, and implicit identity operators acting on other sites.
 * 
 * Please consult README.md for the usage.
 * 
 * The implementation uses three classes: `Pauli`, `PauliProd`, and `PauliAlg`.
 * `Pauli` represents a single Pauli operator at a signle site, such as `X3` or `Y5`.
 * `PauliProd` represents a tensor product of Pauli operators, such as `X3 Y5 Z8`.
 * `PauliAlg` represents a linear combination of PauliProd objects, such as `1.5 X3 Y5 - 0.5i Z8`.
 * `Pauli` and `PauliProd` are used as building blocks and are not intended to be used directly.
 */


#ifndef PAULI_ALG_H
#define PAULI_ALG_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <algorithm>
#include <complex>
#include <utility>
#include <cassert>
#include <cmath>
#include <limits>
#include <unordered_set>
#include "unordered_dense.h"
#include "vector_hash.h"
#include "stopwatch.h"

// OpenMP for parallelization
#ifdef _OPENMP
#include <omp.h>
#else
constexpr int omp_get_thread_num(){
    return 0;
}
constexpr int omp_get_num_threads(){
    return 1;
}
constexpr int omp_get_max_threads(){
    return 1;
}
#endif



// ========== Global options ==========
namespace paulialg{

// If the nuber of sites > 16383, change this to unsigned int
using site_index_t = unsigned short int; 
constexpr const site_index_t num_sites_max = (std::numeric_limits<site_index_t>::max() >> 2); // usually 16383


// If the number of terms exceeds this threshold, some operations are parallelized.
constexpr const int pauli_alg_parallelization_threshold = 500;

} // namespace paulialg


// The type of the map used in the implementation of PauliAlg class
#define PAULI_ALG_H_MAP_SIZE 1
// 1: normal size
// 2: big size
// We reocommend using option 1. It is faster.
// If the number of terms exceeds 2^32, it throws the error "eached max bucket size, cannot increase size". Then please try option 2.




// ==========================================================
// =============== Utility functions & Macros ===============
// ==========================================================


// Likely and unlikely macros to help optimization by the compiler
#define PAULI_ALG_LIKELY(x)   __builtin_expect(!!(x), 1)
#define PAULI_ALG_UNLIKELY(x) __builtin_expect(!!(x), 0)


namespace paulialg::utility{

    /// Returns x * i^power fast, where x is a complex number and power is in 0-3.
    std::complex<double> multiply_poweri(const std::complex<double>& x, unsigned int power){
        assert(power < 4);
        switch(power){
            case 0:
                return x;
            case 1:
                return std::complex<double>(-x.imag(), x.real());
            case 2:
                return -x;
            case 3:
                return std::complex<double>(x.imag(), -x.real());
            default:
                throw std::logic_error("Invalid power of the imaginary unit (should be 0-3)");
        }
    }

    /// Wrap printf formatter to return a C++ string object
    std::string double_format(const std::string format, const double x){
        int len = snprintf(nullptr, 0, format.c_str(), x);
        std::vector<char> x_formatted(len+1);

        std::snprintf(&x_formatted[0], len+1, format.c_str(), x);
        return std::string(x_formatted.begin(), --x_formatted.end());
    }

    /// Convert a double number to a string with full precision
    std::string double_format_full(double x){
        if(x == 0){
            return "0";
        }
        else{
            return double_format("%.16e", x);
        }
    }


    /// Convert a complex number to a string for expressing a coefficient of a term
    /// Returns one of the following forms depending on the signs:
    /// `+ 0.0`, 
    /// `+ 1.0`, `- 1.0`, `+ 1.0i`, `- 1.0i`, 
    /// `+ (1.0+1.0i)`, `+ (1.0-1.0i)`, `+ (-1.0+1.0i)`, `+ (-1.0-1.0i)` 
    std::string complex_format(const int precision, const std::complex<double> x){
        std::string format = "%." + std::to_string(precision) + "f";
        std::string format_alwayssigned = "%+." + std::to_string(precision) + "f";

        if(x.imag() == 0 && x.real() == 0){
            return "+ " + double_format(format, 0.0);
        }
        else if(x.imag() == 0){
            if(x.real() > 0){
                return "+ " + double_format(format, x.real());
            }
            else{
                return "- " + double_format(format, -x.real());
            }
        }
        else if(x.real() == 0){
            if(x.imag() > 0){
                return "+ " + double_format(format, x.imag()) + "i";
            }
            else{
                return "- " + double_format(format, -x.imag()) + "i";
            }
        }
        else{
            return "+ (" + double_format(format, x.real()) + double_format(format_alwayssigned, x.imag()) + "i)";
        }
    }

} // namespace paulialg::utility


// =====================================
// =============== Pauli ===============
// =====================================

namespace paulialg{

/// Represents a single Pauli operator of a spin system on a lattice.
/// A Pauli operator is specified by the kind ('X', 'Y', or 'Z') and the site index (0, 1, 2, ...).
/// Identity operators are always treated implicitly, and is not represented by this object.
class Pauli{
    private:
        site_index_t data_; // lower 2 bits: kind, upper bits: site

    public:
        // Constructor
        Pauli(char kind, site_index_t site) : data_((kind - 'X') | (site << 2)) {}

        // Getters
        char kind() const & {
            return (data_ & 3) + 'X';
        }
        site_index_t site() const & {
            return data_ >> 2;
        }
        site_index_t data() const & {
            return data_;
        }

        // Equality operators
        bool operator==(const Pauli& other) const & {
            return site() == other.site() && kind() == other.kind();
        }
        bool operator!=(const Pauli& other) const & {
            return !(*this == other);
        }

        // Comparison operator to sort Pauli operators by their site indices
        bool operator<(const Pauli& other) const & {
            if(site() < other.site()){
                return true;
            }
            else if(site() == other.site()){
                return kind() < other.kind();
            }
            else{
                return false;
            }
        }
};

} // namespace paulialg



// =========================================
// =============== PauliProd ===============
// =========================================

namespace paulialg{
    
/// Represents a tensor product of Pauli operators such as "X3 Y5 Z8".
/// The product is stored as a vector of Pauli objects, such as `{Pauli('X', 3), Pauli('Y', 5), Pauli('Z', 8)}`.
/// The Pauli operators should be stored in increasing order of their site indices, and the same site index should not appear twice. 
/// The empty vector represents the identity operator.
struct PauliProd{
    std::vector<Pauli> prod_;

    // =============== Constructor ===============
    PauliProd() {};
    PauliProd(std::initializer_list<Pauli> prod) : prod_(prod) {};

    // =============== Calculate the hash ===============
    uint64_t calc_hash() const & {
        VectorHash hasher(prod_.size());

        // Pass two elemnts of PauliProd at a time to the hasher
        for(size_t i = 1; i < prod_.size(); i += 2){
            uint64_t x = prod_[i-1].data();
            uint64_t y = prod_[i].data();
            hasher.update(x, y);
        }
        // If the size of the PauliProd is an odd number, pass the last element with 0
        if((prod_.size() & 1) == 1){
            uint64_t x = prod_.back().data();
            hasher.update(x, 0);
        }

        return hasher.get();
    }


    // =============== comparison operators ===============
    bool operator==(const PauliProd& other) const{
        return prod_ == other.prod_;
    }
    bool operator!=(const PauliProd& other) const{
        return !(prod_ == other.prod_);
    }
    bool operator<(const PauliProd& other) const{
        return prod_ < other.prod_;
    }


    // =============== Utility functions ===============
    /// Check the validity of the PauliProd object.
    bool is_valid() const &;

    /// Convert the product into a string
    std::string get_string() const &;


    // ========== Multiplication of two PauliProd objects ==========
    static int hash_pauli_pair(char op1, char op2);
    static std::pair<PauliProd, unsigned int> multiply(const PauliProd& prod1, const PauliProd& prod2, bool is_commutator);

    /// Tables holding the result of the multiplication of two Pauli operators.
    /// Indices are the hash of two Pauli operators generated by `hash_pauli_pair`.
    /// The first value is the kind of the resulting Pauli operator. 'I' means the identity operator.
    /// The second value is the power of the imaginary unit.
    /// `{'o', 99}` is a padding and should not be accessed.
    static constexpr const std::pair<char, unsigned int> multiply_table[12] = {
        {'I', 0}, {'Z', 1}, {'Y', 3}, {'o', 99},
        {'Z', 3}, {'I', 0}, {'X', 1}, {'o', 99},
        {'Y', 1}, {'X', 3}, {'I', 0}, {'o', 99}
    };
};


// =============== Utility functions ===============

/// Check the following validity conditions of a PauliProd object:
/// - The Pauli operators should be in increasing order of their site indices.
/// - The same site index should not appear twice.
/// - The kind of Pauli operator should be 'X', 'Y', or 'Z'.
bool PauliProd::is_valid() const & {
    int prev_site = -1;
    for(const auto& pauli : prod_){
        if(PAULI_ALG_UNLIKELY(pauli.kind() != 'X' && pauli.kind() != 'Y' && pauli.kind() != 'Z')){
            std::cerr << "Invalid Pauli operator kind: " << std::to_string(pauli.kind()) << std::endl;
            return false;
        }
        if(PAULI_ALG_UNLIKELY(static_cast<int>(pauli.site()) <= prev_site)){
            std::cerr << "Pauli operator sites are not in increasing order" << std::endl;
            return false;
        }
        prev_site = pauli.site();
    }
    return true;
}

/// Convert the product into a string, such as `"X3 Y5 Z8"`.
std::string PauliProd::get_string() const & {
    // Special case of no Pauli operators in the product
    if(prod_.empty()){
        return "I";
    }

    // Convert the product into a string
    std::string prod_str = "";
    bool is_first_op = true;
    for(auto& pauli : prod_){
        if(!is_first_op){
            prod_str += " ";
        }
        prod_str += pauli.kind() + std::to_string(pauli.site());
        is_first_op = false;
    }
    return prod_str;
}


// ========== Multiplication of two PauliProd objects ==========

/// hash two pauli operators into a single integer
/// op1 and op2 should be 'X', 'Y', or 'Z'
int PauliProd::hash_pauli_pair(char op1, char op2){
    return ((op1 - 'X') << 2) + (op2 - 'X');
}

/// Calculate the product of two PauliProd objects.
/// Returns the resulting PauliProd object and the coefficient.
/// The coefficient is either 1, i, -1, or -i, and is represented by the power of the imaginary unit (an integer from 0 to 3).
/// If is_commutator is true, the result will be used for a commutator. In this case, the resulting PauliProd object 
/// will be discarded if the coefficient is real, so we do a shortcut.
std::pair<PauliProd, unsigned int> PauliProd::multiply(const PauliProd& prod1, const PauliProd& prod2, bool is_commutator){
    PauliProd prod_result;
    prod_result.prod_.reserve(prod1.prod_.size() + prod2.prod_.size());
    unsigned int power_result = 0;

    std::size_t counter1 = 0;
    std::size_t counter2 = 0;

    // Iterates over the two PauliProd objects until one of them reaches the end.
    // This code assumes that the Pauli objects are sorted by their site indices in each PauliProd object.
    while(counter1 < prod1.prod_.size() && counter2 < prod2.prod_.size()){
        // If the same site appears in only one of the PauliProd objects, we simply append the Pauli operator.
        if(prod1.prod_[counter1].site() < prod2.prod_[counter2].site()){
            prod_result.prod_.push_back(prod1.prod_[counter1]);
            ++counter1;
        }
        else if(prod1.prod_[counter1].site() > prod2.prod_[counter2].site()){
            prod_result.prod_.push_back(prod2.prod_[counter2]);
            ++counter2;
        }
        // If the same site appears in both PauliProd objects, we multiply the Pauli operators.
        else{
            auto [kind, power] = PauliProd::multiply_table[hash_pauli_pair(prod1.prod_[counter1].kind(), prod2.prod_[counter2].kind())];
            if(kind != 'I'){
                prod_result.prod_.push_back(Pauli(kind, prod1.prod_[counter1].site()));
                power_result += power;
            }
            ++counter1;
            ++counter2;
        }
    }

    // If the result is used for a commutator and the coefficient is real, we do an early return
    // with an incomplete prod_result because the result will not be used.
    if(is_commutator && (power_result & 1) == 0){ // x & 1 is equivalent to x % 2
        return {prod_result, 0}; 
    }

    // Append the remaining Pauli operators
    // Only one of the following while loops will be executed
    while(counter1 < prod1.prod_.size()){
        prod_result.prod_.push_back(prod1.prod_[counter1]);
        ++counter1;
    }
    while(counter2 < prod2.prod_.size()){
        prod_result.prod_.push_back(prod2.prod_[counter2]);
        ++counter2;
    }
    
    // Shrink the vector and return
    prod_result.prod_.shrink_to_fit();
    return {prod_result, power_result & 3}; // x & 3 is equivalent to x % 4
}

} // namespace paulialg


/// Hash function to use PauliProd as a key in unordered_map
template <>
struct ankerl::unordered_dense::hash<paulialg::PauliProd> {
    using is_avalanching = void;

    uint64_t operator()(const paulialg::PauliProd& prod) const {
        return prod.calc_hash();
    }
}; 




// ========================================
// =============== PauliAlg ===============
// ========================================


namespace paulialg{

// Prepare custom map
#if PAULI_ALG_H_MAP_SIZE == 1
    using custom_map = ankerl::unordered_dense::segmented_map<PauliProd, std::complex<double> >;
#elif PAULI_ALG_H_MAP_SIZE == 2
    template <class Key,
              class T,
              class Hash = ankerl::unordered_dense::hash<Key>,
              class KeyEqual = std::equal_to<Key>,
              class AllocatorOrContainer = std::allocator<std::pair<Key, T>>,
              class Bucket = ankerl::unordered_dense::bucket_type::big>
    using unordered_dense_map_big = ankerl::unordered_dense::detail::table<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket, true>;
    using custom_map = unordered_dense_map_big<PauliProd, std::complex<double> >;
#endif


/// Represents a linear combination of Pauli products, such as a Hamiltonian.
/// The linear combination is stored as a map from a product of Pauli operators (a `PauliProd` object) to 
/// its coefficient (a `std::complex<double>` number).
/// The total number of sites in the spin system is fixed and stored, unlike the Pauli and PauliProd objects.
class PauliAlg{
    private:
        site_index_t num_sites_;
        custom_map terms_;

    public:
        // =============== Constructors ===============

        /// Constructor with no terms
        PauliAlg(site_index_t num_sites) : num_sites_(num_sites) {
            if(PAULI_ALG_UNLIKELY(num_sites > num_sites_max)) {
                throw std::invalid_argument("The number of sites exceeds the maximum value.\nEdit the type of site_index_t in pauli_alg.h to increase the maximum.");
            }
        }

        /// Constructor with one term.
        /// PauliAlg objects with more than one term should be created using append or + or +=.
        PauliAlg(site_index_t num_sites, const PauliProd& prod, std::complex<double> coeff) : num_sites_(num_sites), terms_({{prod, coeff}}) {
            if(PAULI_ALG_UNLIKELY(num_sites > num_sites_max)) {
                throw std::invalid_argument("The number of sites exceeds the maximum value.\nEdit the type of site_index_t in pauli_alg.h to increase the maximum.");
            }
        }

        // =============== Getters ===============

        /// Returns the number of sites
        site_index_t get_num_sites() const &{
            return num_sites_;
        }

        /// Returns the number of terms
        std::size_t get_num_terms() const &{
            return terms_.size();
        }

        /// Returns the term at position i
        /// The position may change when the map is updated.
        const std::pair<PauliProd, std::complex<double> >& term_at(const std::size_t i) const &{
            return terms_.values()[i];
        }

        /// Returns the map of all terms
        const custom_map& get_map() const &{
            return terms_;
        }
        
        /// Returns the set of sites on which the PauliAlg object nontrivially acts.
        std::unordered_set<site_index_t> get_nontrivial_sites() const &{
            std::unordered_set<site_index_t> nontrivial_sites;
            for(const auto& [prod, coeff] : terms_){
                for(const auto& pauli : prod.prod_){
                    nontrivial_sites.insert(pauli.site());
                }
            }
            return nontrivial_sites;
        }

        /// Precision of the coefficients when converted to string. Initialized at the end of this file.
        static int get_string_precision;
        /// Convert PauliAlg to string.
        std::string get_string() const &;
        

        // =============== Comparison ===============
        static double max_coeff_diff(const PauliAlg& A1, const PauliAlg& A2);
        static bool is_close(const PauliAlg& A1, const PauliAlg& A2, double atol = 1e-13);

        // =============== Arithmetic operators ===============
        // We overload some operators for lvalues and rvalues separately to avoid unnecessary copying.
        // This is not necessary if the speed is not critical.

        // ========== Unary operators ==========

        // Unary plus
        const PauliAlg& operator+() const & {
            return *this;
        }
        PauliAlg operator+() && {
            return std::move(*this);
        }

        // Unary minus
        PauliAlg operator-() const & {
            PauliAlg result = *this;
            for(auto& [prod, coeff] : result.terms_){
                coeff = -coeff;
            }
            return result;
        }
        PauliAlg operator-() && {
            PauliAlg result = std::move(*this);
            for(auto& [prod, coeff] : result.terms_){
                coeff = -coeff;
            }
            return result;
        }

        // Hermite conjugate
        PauliAlg hermite_conj() const & {
            PauliAlg result = *this;
            for(auto& [prod, coeff] : result.terms_){
                coeff = conj(coeff);
            }
            return result;
        }
        PauliAlg hermite_conj() && {
            PauliAlg result = std::move(*this);
            for(auto& [prod, coeff] : result.terms_){
                coeff = conj(coeff);
            }
            return result;
        }

        // ========== Addition and subtraction ==========

        /// Tolerance for the numerical error.
        /// When performing addition and subtraction, if the resulting coefficient is much smaller than the original ones,
        /// we construe it as a numerical error and remove the term.
        /// The maximum allowed ratio between the new and original coefficients is given by this tolerance.
        /// Default value is 1e-13, which is initialized at the end of this file.
        static double addition_tolerance;

        // Addition of a single term
        PauliAlg& append(const PauliProd& prod, const std::complex<double>& coeff){
            // Insert if the term does not exist
            const auto [itr, success] = terms_.try_emplace(prod, coeff);

            // If the same term already exists
            if(!success){ 
                (itr->second) += coeff;
                
                // If the result of addition is much smaller than the original values,
                // we construe it as a numerical error and remove the term.
                if(PAULI_ALG_UNLIKELY(std::abs(itr->second.real()) <= addition_tolerance * std::abs(coeff.real()) && 
                                      std::abs(itr->second.imag()) <= addition_tolerance * std::abs(coeff.imag()))){
                    terms_.erase(itr);
                }
            }
            
            return *this;
        } 

        // Addition of another PauliAlg object
        PauliAlg& operator+= (const PauliAlg& other){
            // Calculate the sum. If other has no term, nothing happens
            for(const auto& [prod, coeff] : other.terms_){
                append(prod, coeff);
            }
            return *this;
        }

        friend PauliAlg operator+ (const PauliAlg& A1, const PauliAlg& A2){
            if(A1.get_num_terms() < A2.get_num_terms()){
                PauliAlg result = A2;
                result += A1;
                return result;
            }
            else{
                PauliAlg result = A1;
                result += A2;
                return result;
            }
        }
        friend PauliAlg operator+ (PauliAlg&& A1, const PauliAlg& A2){
            PauliAlg result = std::move(A1);
            result += A2;
            return result;
        }
        friend PauliAlg operator+ (const PauliAlg& A1, PauliAlg&& A2){
            PauliAlg result = std::move(A2);
            result += A1;
            return result;
        }
        friend PauliAlg operator+ (PauliAlg&& A1, PauliAlg&& A2){
            if(A1.get_num_terms() < A2.get_num_terms()){
                PauliAlg result = std::move(A2);
                result += A1;
                return result;
            }
            else{
                PauliAlg result = std::move(A1);
                result += A2;
                return result;
            }
        }


        // Subtraction
        PauliAlg& operator-= (const PauliAlg& other){
            // Calculate the difference. If other has no term, nothing happens
            for(const auto& [prod, coeff] : other.terms_){
                append(prod, -coeff);
            }
            return *this;
        }

        friend PauliAlg operator- (const PauliAlg& A1, const PauliAlg& A2){
            if(A1.get_num_terms() < A2.get_num_terms()){
                PauliAlg result = -A2;
                result += A1;
                return result;
            }
            else{
                PauliAlg result = A1;
                result -= A2;
                return result;
            }
        }
        friend PauliAlg operator- (PauliAlg&& A1, const PauliAlg& A2){
            PauliAlg result = std::move(A1);
            result -= A2;
            return result;
        }
        friend PauliAlg operator- (const PauliAlg& A1, PauliAlg&& A2){
            PauliAlg result = -std::move(A2); 
            result += A1;
            return result;
        }
        friend PauliAlg operator- (PauliAlg&& A1, PauliAlg&& A2){
            if(A1.get_num_terms() < A2.get_num_terms()){
                PauliAlg result = -std::move(A2); 
                result += A1;
                return result;
            }
            else{
                PauliAlg result = std::move(A1);
                result -= A2;
                return result;
            }
        }


        // ========== Multiplication ==========

    private:
        // Add the result of A1 * A2 to *this
        PauliAlg& add_multiply_parallel(const PauliAlg& A1, const PauliAlg& A2); // Parallelized version
        PauliAlg& add_multiply_single(const PauliAlg& A1, const PauliAlg& A2); // Non-parallelized version
    
    public:
        PauliAlg& add_multiply(const PauliAlg& A1, const PauliAlg& A2, bool is_parallelized = true); // Switch between the two versions

        // Multiplication of two PauliAlg objects
        friend PauliAlg operator* (const PauliAlg& A1, const PauliAlg& A2){ 
            // Apply add_multiply to a zero object
            return PauliAlg(A1.num_sites_).add_multiply(A1, A2);
        }

        PauliAlg& operator*= (const PauliAlg& other){
            *this = *this * other;
            return *this;
        }
        

        // Multiplication by std::complex<double> scalar
        PauliAlg& operator*= (std::complex<double> scalar){
            if (PAULI_ALG_UNLIKELY(scalar == 0.0)){
                *this = PauliAlg(num_sites_); // empty object (represents zero)
            }
            else{
                for(auto& [prod, coeff] : terms_){
                    coeff *= scalar;
                }         
            }
            return *this;
        }

        friend PauliAlg operator* (const PauliAlg& A, std::complex<double> scalar){
            PauliAlg result = A;
            result *= scalar;
            return result;
        }
        friend PauliAlg operator* (std::complex<double> scalar, const PauliAlg& A){
            PauliAlg result = A;
            result *= scalar;
            return result;
        }
        friend PauliAlg operator* (PauliAlg&& A, std::complex<double> scalar){
            PauliAlg result = std::move(A);
            result *= scalar;
            return result;
        }
        friend PauliAlg operator* (std::complex<double> scalar, PauliAlg&& A){
            PauliAlg result = std::move(A);
            result *= scalar;
            return result;
        }



        // Multiplication by double scalar
        // Other arithmetic types, such as int, will be implicitly converted to double and come here.
        PauliAlg& operator*= (double scalar){
            if (PAULI_ALG_UNLIKELY(scalar == 0.0)){
                *this = PauliAlg(num_sites_); // empty object (represents zero)
            }
            else{
                for(auto& [prod, coeff] : terms_){
                    coeff *= scalar;
                }         
            }
            return *this;
        }

        friend PauliAlg operator* (const PauliAlg& A, double scalar){
            PauliAlg result = A;
            result *= scalar;
            return result;
        }
        friend PauliAlg operator* (double scalar, const PauliAlg& A){
            PauliAlg result = A;
            result *= scalar;
            return result;
        }
        friend PauliAlg operator* (PauliAlg&& A, double scalar){
            PauliAlg result = std::move(A);
            result *= scalar;
            return result;
        }
        friend PauliAlg operator* (double scalar, PauliAlg&& A){
            PauliAlg result = std::move(A);
            result *= scalar;
            return result;
        }

        
        // =============== Commutators ===============
        static std::vector<PauliAlg> commutator_for_each(const PauliAlg& A1, const std::vector<PauliAlg>& A2_array, bool is_parallelized = true);
        static PauliAlg commutator(const PauliAlg& A1, const PauliAlg& A2);


        // =============== Trace operations ===============
        static std::complex<double> trace_normalized_multiply(const PauliAlg& A1, const PauliAlg& A2);
        static std::complex<double> trace_normalized(const PauliAlg& A);


        // =============== Getting initialized operators ===============
    private:
        static std::vector<PauliAlg> initialize_array(char kind, site_index_t num_sites);
    public:
        static PauliAlg zero(site_index_t num_sites);
        static PauliAlg identity(site_index_t num_sites);
        static std::vector<PauliAlg> X_array(site_index_t num_sites);
        static std::vector<PauliAlg> Y_array(site_index_t num_sites);
        static std::vector<PauliAlg> Z_array(site_index_t num_sites);
        

        // =============== Save to file ===============
        static void write_to_file(const std::vector<PauliAlg>& A_list, const std::string& filename);
        static void write_to_file(const PauliAlg& A, const std::string& filename);
};


// =============== Getters ===============

/// Convert PauliAlg to a string, such as "0.9 X3 Y5 + (0.1+0.8i) Z8".
/// This function is slow and not recommended to use in large systems.
std::string PauliAlg::get_string() const &{
    // Special case of no terms
    if(terms_.size() == 0){
        return "0";
    }
    
    // Sort the terms in the increasing order of the PauliProd
    std::vector<PauliProd> terms_vector;
    terms_vector.reserve(terms_.size());

    for(const auto& [prod, coeff] : terms_){
        terms_vector.push_back(prod);
    }
    sort(terms_vector.begin(), terms_vector.end());

    // Create the string
    std::string str = "";
    for(const auto& prod : terms_vector){
        str += utility::complex_format(get_string_precision, terms_.at(prod)) + " ";
        str += prod.get_string() + " ";
    }

    // Remove the leading "+ " and trailing " " if they exist
    if (str.substr(0, 2) == "+ ") {
        str.erase(0, 2);
    }
    if (str.back() == ' ') {
        str.pop_back();
    }

    return str;
}


// =============== Comparison ===============

/// Compares two PauliAlg objects and returns the maximum error of the coefficients
double PauliAlg::max_coeff_diff(const PauliAlg& A1, const PauliAlg& A2){
    double max_diff = 0.0;

    // Check for all terms in A1
    for(const auto& [prod1, coeff1] : A1.terms_){
        // If the term does not exist in A2:
        if(A2.terms_.count(prod1) == 0){
            max_diff = std::max(max_diff, std::norm(coeff1));
        }
        // If the term exists in A2:
        else{
            max_diff = std::max(max_diff, std::norm(coeff1 - A2.terms_.at(prod1)));
        }
    }

    // Check for all terms in A2
    for(const auto& [prod2, coeff2] : A2.terms_){
        // If the term does not exist in A1:
        if(A1.terms_.count(prod2) == 0){
            max_diff = std::max(max_diff, std::norm(coeff2));
        }
    }

    return std::sqrt(max_diff);
}

/// Compares two PauliAlg objects and returns true if they are equal within the given absolute tolerance (default: `1e-13`).
/// The tolerance will be compared with the absolute value of the difference of the coefficients.
bool PauliAlg::is_close(const PauliAlg& A1, const PauliAlg& A2, double atol){
    assert(atol >= 0.0);
    return max_coeff_diff(A1, A2) <= atol;
}


// =============== Multiplications ===============
// Add the result of A1 * A2 to *this
// Parallelized version
PauliAlg& PauliAlg::add_multiply_parallel(const PauliAlg& A1, const PauliAlg& A2){
    // Vector of zero PauliAlg objects to store the result.
    std::vector<PauliAlg> results(omp_get_max_threads(), PauliAlg(A1.num_sites_));

    // Iterate over all pairs of terms. Outer loop is over the term with more terms.
    if(A1.get_num_terms() >= A2.get_num_terms()){
        #pragma omp parallel for
        for(std::size_t i1 = 0; i1 < A1.get_num_terms(); i1++){
            const auto& [prod1, coeff1] = A1.term_at(i1);
            for(std::size_t i2 = 0; i2 < A2.get_num_terms(); i2++){
                const auto& [prod2, coeff2] = A2.term_at(i2);
                const auto [prod_result, power_result] = PauliProd::multiply(prod1, prod2, false);
                std::complex<double> coeff_result = utility::multiply_poweri(coeff1 * coeff2, power_result);
                results.at(omp_get_thread_num()).append(prod_result, coeff_result);
            }
        }
    }
    else{
        #pragma omp parallel for
        for(std::size_t i2 = 0; i2 < A2.get_num_terms(); i2++){
            const auto& [prod2, coeff2] = A2.term_at(i2);
            for(std::size_t i1 = 0; i1 < A1.get_num_terms(); i1++){
                const auto& [prod1, coeff1] = A1.term_at(i1);
                const auto [prod_result, power_result] = PauliProd::multiply(prod1, prod2, false);
                std::complex<double> coeff_result = utility::multiply_poweri(coeff1 * coeff2, power_result);
                results.at(omp_get_thread_num()).append(prod_result, coeff_result);
            }
        }
    }

    // Sum the results to *this
    for(std::size_t i = 0; i < results.size(); i++){
        terms_.rehash(get_num_terms() + results[i].get_num_terms());
        *this += results[i];
        results[i] = PauliAlg(num_sites_); // clear the object to save memory
    }

    return *this;
}

// Add the result of A1 * A2 to *this
// Non-parallelized version
PauliAlg& PauliAlg::add_multiply_single(const PauliAlg& A1, const PauliAlg& A2){
    for(std::size_t i1 = 0; i1 < A1.get_num_terms(); i1++){
        const auto& [prod1, coeff1] = A1.term_at(i1);
        for(std::size_t i2 = 0; i2 < A2.get_num_terms(); i2++){
            const auto& [prod2, coeff2] = A2.term_at(i2);
            const auto [prod_result, power_result] = PauliProd::multiply(prod1, prod2, false);
            std::complex<double> coeff_result = utility::multiply_poweri(coeff1 * coeff2, power_result);
            append(prod_result, coeff_result);
        }
    }

    return *this;
}


// Add the result of A1 * A2 to *this
PauliAlg& PauliAlg::add_multiply(const PauliAlg& A1, const PauliAlg& A2, bool is_parallelized){
    // Disable parallelization if the number of terms is small
    if(omp_get_max_threads() <= 1 || (std::max)(A1.get_num_terms(), A2.get_num_terms()) < pauli_alg_parallelization_threshold){
        is_parallelized = false;
    }

    if(is_parallelized){
        add_multiply_parallel(A1, A2);
    }
    else{
        add_multiply_single(A1, A2);
    }

    return *this;
}



// =============== Commutators ===============


/// Given a PauliAlg object A1 and an array of PauliAlg objects A2_array, 
/// calculate the commutator [A1, A2] for each element A2 of A2_array.
/// Returns the same result as `{commutator(A1, A2_array[0]), commutator(A1, A2_array[1]), ...}`, 
/// but runs much faster if A2 are local operators.
std::vector<PauliAlg> PauliAlg::commutator_for_each(const PauliAlg& A1, const std::vector<PauliAlg>& A2_array, bool is_parallelized){
    // Disable parallelization if the number of terms is small
    if(omp_get_max_threads() <= 1 || A1.get_num_terms() < pauli_alg_parallelization_threshold){
        is_parallelized = false;
    }
    int num_boxes = 1;
    if(is_parallelized){
        num_boxes = omp_get_max_threads();
    }

    // Prepare the index. index[site] is the list of (mu, i2) such that A2_array[mu].term_at(i2) acts on the site. 
    std::vector<std::vector<std::pair<std::size_t, std::size_t> > > index(A1.num_sites_); // Initialize with empty vectors
    for(std::size_t mu = 0; mu < A2_array.size(); mu++){
        for(std::size_t i2 = 0; i2 < A2_array[mu].get_num_terms(); i2++){
            const auto& [prod, coeff] = A2_array[mu].term_at(i2);
            for(const auto& pauli : prod.prod_){
                index[pauli.site()].emplace_back(mu, i2);
            }
        }
    }

    // Calculate the commutators simultaneously
    // The sum of result_array[box][mu] over 0<=box<num_boxes will be the commutator of A1 and A2_array[mu].
    std::vector<std::vector<PauliAlg> > result_array(num_boxes, std::vector<PauliAlg>(A2_array.size(), PauliAlg(A1.num_sites_))); 
    
    // Used to avoid the calculation of the same commutator multiple times
    std::vector<std::vector<std::vector<std::size_t> > > finished_mark(num_boxes); 
    for(int box = 0; box < num_boxes; box++){
        for(std::size_t mu = 0; mu < A2_array.size(); mu++){
            finished_mark[box].emplace_back(A2_array[mu].get_num_terms(), 0);
        }
    }

    // Iterate through the terms of A1
    #pragma omp parallel for if(is_parallelized)
    for(std::size_t i1 = 0; i1 < A1.get_num_terms(); i1++){ 
        const auto& [prod1, coeff1] = A1.term_at(i1);

        // The following two for-loops pick up the indices (mu, i2) such that
        // A2_array[mu].term_at(i2) acts on the same site as prod1.
        for(const auto& pauli1 : prod1.prod_){
            for(const auto& [mu, i2] : index[pauli1.site()]){
                // Skip if A2_array[mu].term_at(i2) has already been calculated with the current i1
                if(finished_mark.at(omp_get_thread_num())[mu][i2] == i1 + 1){
                    continue;
                }

                // Calculate the commutator between prod1 and A2_array[mu].term_at(i2)
                const auto& [prod2, coeff2] = A2_array[mu].term_at(i2);
                const auto [prod_result, power_result] = PauliProd::multiply(prod1, prod2, true);

                // When A, B are tensor products of Pauli operators, the commutator is [A, B] = AB - BA = AB - conj(AB), 
                // where conj is the Hermitian conjugate.
                // Thus, [A, B] = 0 if the coefficient of AB is real, and [A, B] = 2AB if the coefficient is imaginary.
                if((power_result & 1) == 1){ // if power_result is odd, i.e., the coefficient is imaginary
                    std::complex<double> coeff_result = utility::multiply_poweri(2.0 * coeff1 * coeff2, power_result);
                    result_array.at(omp_get_thread_num())[mu].append(prod_result, coeff_result);
                }
                
                // Mark A2_array[mu].term_at(i2) as finished. We use (i1 + 1) because finished_mark is initialized with 0.
                finished_mark.at(omp_get_thread_num())[mu][i2] = i1 + 1; 
            }
        }
    }

    // Sum the results if parallelized
    if(is_parallelized){
        #pragma omp parallel for if(is_parallelized)
        for(std::size_t mu = 0; mu < A2_array.size(); mu++){
            for(int box = 1; box < num_boxes; box++){
                result_array[0][mu].terms_.rehash(result_array[0][mu].get_num_terms() + result_array[box][mu].get_num_terms());
                result_array[0][mu] += result_array[box][mu];
                result_array[box][mu] = PauliAlg(A1.num_sites_); // clear the object to save memory
            }
        }
    }

    return std::move(result_array[0]);
}


/// Calculate the commutator of two PauliAlg objects.
/// Returns the same results as `A1 * A2 - A2 * A1`, but runs faster.
PauliAlg PauliAlg::commutator(const PauliAlg& A1, const PauliAlg& A2){
    // This implementation is not very efficient if both A1 and A2 have a large number of terms.
    if(A1.get_num_terms() > A2.get_num_terms()){
        return std::move(PauliAlg::commutator_for_each(A1, {A2})[0]);
    }
    else{
        return -std::move(PauliAlg::commutator_for_each(A2, {A1})[0]);
    }
}


// =============== Trace operations ===============

/// Returns the normalized trace of the product of two PauliAlg objects, where "normalized" means that 
/// the trace is devided by 2^(number of sites).
/// Returns the same results as `trace_normlized(A1 * A2)`, but runs much faster.
/// Unlike the Hilbert-Schmidt inner product, the Hermitian conjugate is not taken.
std::complex<double> PauliAlg::trace_normalized_multiply(const PauliAlg& A1, const PauliAlg& A2){
    std::complex<double> trace = 0.0;

    // The normalized trace is equal to the sum over the products of coefficients of the terms with the same PauliProd object.
    // Iterate over the terms of the PauliAlg objects with fewer terms
    if(A1.get_num_terms() < A2.get_num_terms()){
        for(const auto& [prod1, coeff1] : A1.terms_){
            const auto itr2 = A2.terms_.find(prod1);
            if(itr2 != A2.terms_.end()){
                trace += coeff1 * (itr2->second);
            }
        }
    }
    else{
        for(const auto& [prod2, coeff2] : A2.terms_){
            const auto itr1 = A1.terms_.find(prod2);
            if(itr1 != A1.terms_.end()){
                trace += coeff2 * (itr1->second);
            }
        }
    }

    return trace;
}

/// Take the normalized trace of a PauliAlg object, where "normalized" means that the trace is devided by 2^num_sites.
std::complex<double> PauliAlg::trace_normalized(const PauliAlg& A){
    return PauliAlg::trace_normalized_multiply(PauliAlg::identity(A.num_sites_), A);
}



// =============== Getting initialized operators ===============

/// Returns a PauliAlg object with no terms
PauliAlg PauliAlg::zero(site_index_t num_sites){
    return PauliAlg(num_sites);
}

/// Returns a PauliAlg object with a single term of the identity operator
PauliAlg PauliAlg::identity(site_index_t  num_sites){
    PauliProd pauliprod; // empty PauliProd object represents the identity operator

    return PauliAlg(num_sites, pauliprod, 1.0);
}

/// Returns an array of PauliAlg objects, each of which has a single term of a Pauli operator
/// Should be used via the wrapper functions X_array, Y_array, and Z_array
std::vector<PauliAlg> PauliAlg::initialize_array(char kind, site_index_t num_sites){
    std::vector<PauliAlg> spins_array;
    spins_array.reserve(num_sites);

    for(site_index_t site = 0; site < num_sites; site++){
        Pauli pauli = Pauli(kind, site);

        PauliProd pauliprod;
        pauliprod.prod_.push_back(pauli);

        PauliAlg spins(num_sites, pauliprod, 1.0);
        spins_array.push_back(spins);
    }
    return spins_array;
}        

/// Returns an array of PauliAlg objects, whose ith element has a single X operator at site i.
std::vector<PauliAlg> PauliAlg::X_array(site_index_t num_sites){
    return PauliAlg::initialize_array('X', num_sites);
}

/// Returns an array of PauliAlg objects, whose ith element has a single Y operator at site i.
std::vector<PauliAlg> PauliAlg::Y_array(site_index_t num_sites){
    return PauliAlg::initialize_array('Y', num_sites);
}

/// Returns an array of PauliAlg objects, whose ith element has a single Z operator at site i.
std::vector<PauliAlg> PauliAlg::Z_array(site_index_t num_sites){
    return PauliAlg::initialize_array('Z', num_sites);
}




// =============== Save to file ===============

/// Writes a series of PauliAlg object to a file.
///
/// The file consists of blocks separated by an empty line.
/// Each block represents a PauliAlg object and contains the following lines:
///
/// - The first line contains the number of sites.
/// - The second and later lines contain the terms in the format: 
/// `[coefficient.real()] [coefficient.imag()] [Pauli] [Pauli] [Pauli] ...`
/// - Example of a line: `1.123456789012345e+01 -9.678901234567890e+00 X3 Y5 Z8`
///
/// As a special case, a block represents the zero operator if it contains only the number of sites.
void PauliAlg::write_to_file(const std::vector<PauliAlg>& A_list, const std::string& filename){
    std::ofstream fout(filename);
    if(!fout){
        throw std::runtime_error("Unable to open output file \"" + filename + "\"");
    }

    for(const auto& A : A_list){
        // Write the number of sites
        fout << "$ N = " << A.get_num_sites() << std::endl;
        
        // Write the terms
        for(const auto& [prod, coeff] : A.get_map()){
            fout << utility::double_format_full(coeff.real()) << " ";
            fout << utility::double_format_full(coeff.imag()) << " ";
            fout << prod.get_string() << std::endl;
        }

        // Write an empty line
        fout << std::endl;
    }

    fout.close();
}

/// Writes a PauliAlg object to a file.
/// The file consists of blocks separated by an empty line.
/// Each block represents a PauliAlg object and contains the following lines:
///
/// - The first line contains the number of sites.
/// - The second and later lines contain the terms in the format: 
/// `[coefficient.real()] [coefficient.imag()] [Pauli] [Pauli] [Pauli] ...`
/// - Example of a line: `1.123456789012345e+01 -9.678901234567890e+00 X3 Y5 Z8`
///
/// As a special case, a block represents the zero operator if it contains only the number of sites.
void PauliAlg::write_to_file(const PauliAlg& A, const std::string& filename){
    // Wrapper function for a single PauliAlg object
    std::vector<PauliAlg> A_list;
    A_list.push_back(A);
    
    write_to_file(A_list, filename);  
}



// =============== Initialize static variables ===============
double PauliAlg::addition_tolerance = 1e-13;
int PauliAlg::get_string_precision = 2;

} // namespace paulialg


#endif // PAULI_ALG_H