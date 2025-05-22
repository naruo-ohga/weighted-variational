#ifndef WEIGHTED_VARIATIONAL_UTILITY_H
#define WEIGHTED_VARIATIONAL_UTILITY_H


#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <map>
#include <Eigen/Dense>
#include "pauli_alg.h"


#define WEIGHTED_VARIATIONAL_ERROR(msg) {std::cerr << "Error: " << msg << std::endl; exit(1);}

namespace weighted_variational{
namespace utility{

    // ===========================================
    // =============== Power table ===============
    // ===========================================


    /// @brief Make a table of binomial coefficients using Pascal's triangle
    /// @param Nmax The maximum value of n in n choose k.
    /// @return An `(Nmax + 1) * (Nmax + 1)` 2d vector binomial where
    /// `binomial[n][k] = n choose k` for `k <= n` and
    /// `binomial[n][k] = 0` for `k > n`.
    std::vector<std::vector<int> > make_binomial_table(int Nmax){
        assert(Nmax >= 0);
        std::vector<std::vector<int> > binomial(Nmax+1, std::vector<int>(Nmax+1, 0));
        for(int n = 0; n <= Nmax; n++){
            binomial[n][0] = 1;
            binomial[n][n] = 1;
            for(int k = 1; k < n; k++){
                binomial[n][k] = binomial[n-1][k-1] + binomial[n-1][k];
            }
        }
    
        return binomial;
    }


    /// The following declarations and functions will be used to generate the "power table".
    /// When computing the weighted variational method for the entire series of lambda, 
    /// we need combinatorial treatments to expand the powers of Hamiltonians into terms with different lambda-dependent coefficients.
    /// The power table contains all required information for this expansion.
    ///
    /// For example, if the order is `K = 3` and the number of basis operators of Hamiltonian Gamma = 2, 
    /// the Hamiltonian is H = f1 * F1 + f2 * F2, where f1 and f2 are lambda-dependent coefficients and F1 and F2 are basis Hamiltonians.
    /// The table is as follows:
    /// ```
    /// table[0] = {0, (0, 0), [()]}
    /// table[1] = {1, (0, 1), [(2)]}
    /// table[2] = {1, (1, 0), [(1)]}
    /// table[3] = {2, (0, 2), [(2, 2)]}
    /// table[4] = {2, (1, 1), [(1, 2), (2, 1)]}
    /// table[5] = {2, (2, 0), [(1, 1)]}
    /// table[6] = {3, (0, 3), [(2, 2, 2)]}
    /// table[7] = {3, (1, 2), [(1, 2, 2), (2, 1, 2), (2, 2, 1)]}
    /// table[8] = {3, (2, 1), [(1, 1, 2), (1, 2, 1), (2, 1, 1)]}
    /// table[9] = {3, (3, 0), [(1, 1, 1)]}
    /// ```
    /// The second element is the power_combination `(n1, n2)`, which corresponds to the term with the coefficient `f1^n1 * f2^n2`.
    /// The first element is the total power of the combination `n1 + n2`.
    /// The third element is the corresponding list of possible orderings of the Hamiltonians.
    /// For example, `table[8]` corresponds to the term with the coefficient `f1^2 * f2`,
    /// and there are three possible orderings of the Hamiltonians, `F1 * F1 * F2`, `F1 * F2 * F1`, and `F2 * F1 * F1`, 
    /// represented by `(1, 1, 2)`, `(1, 2, 1)`, and `(2, 1, 1)`, respectively.


    using power_combination = std::vector<int>; // holds a combination of powers of the basis Hamiltonians
    using power_ordering = std::vector<int>;    // holds an ordering of the basis Hamiltonians

    // A row of the power table
    struct power_table_row{
        int total_power;
        power_combination combination;
        std::vector<power_ordering> orderings;

        // Constructer
        power_table_row(power_combination comb, std::vector<power_ordering> orderings) : combination(comb), orderings(orderings) {
            total_power = 0;
            for(int n : comb){
                total_power += n;
            }
        }

        // Comparison for sorting
        bool operator<(const power_table_row &other) const {
            if(total_power != other.total_power){
                return total_power < other.total_power;
            }
            return combination < other.combination;
        }
    };  


    /// @brief Generates all possible orderings of the indices of Hamiltonians.
    /// @param Gamma The number of basis Hamiltonians other than E (Gamma >= 1).
    /// @param s The total number of Hamiltonians in the ordering (s >= 0).
    /// @return All possible orderings of numbers from 1 to Gamma of length s.
    /// For example, if Gamma = 2 and s = 3, the return value is as follows:
    /// `{(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1), (1, 2, 2), (2, 1, 2), (2, 2, 1), (2, 2, 2)}`.
    std::vector<power_ordering> generate_orderings(int Gamma, int s){
        assert(Gamma >= 1);
        assert(s >= 0);

        std::vector<power_ordering> orderings;

        // Use the base Gamma representation to generate all possible orderings.
        for(int i = 0; i < std::pow(Gamma, s); i++){
            int x = i;
            power_ordering ordering(s, 0);
            for(int j = s - 1; j >= 0; j--){
                ordering[j] = x % Gamma + 1;
                x /= Gamma;
            }
            orderings.push_back(ordering);
        }
        
        return orderings;
    }


    /// @brief Generates the power table (explained above) up to the Kth power.
    /// @param K The order of the method (K >= 1). 
    /// @param Gamma The number of basis Hamiltonians other than E (Gamma >= 1). 
    /// @return The power table up to the Kth power. 
    std::vector<power_table_row> make_power_table(int K, int Gamma){
        assert(K >= 1);
        assert(Gamma >= 1);
        
        // We first generate a map from power_combination to power_ordering, then convert it to a vector.
        std::map<power_combination, std::vector<power_ordering> > power_table_map;

        // To generate the table, we iterate through all possible orderings of the Hamiltonians, 
        // and classify them into those having the same power_combination.

        // s denotes the number of Hamiltonians in the current ordering
        for(int s = 0; s <= K; s++){
            for(power_ordering ordering : generate_orderings(Gamma, s)){
                // Calculate the power_combination of the current ordering
                power_combination comb(Gamma, 0);
                for(int i = 0; i < s; i++){
                    ++comb[ordering[i]-1];
                }

                // Add the ordering to the table
                power_table_map[comb].push_back(ordering);
            }
        } 

        // Convert to a vector
        std::vector<power_table_row> power_table;
        for(auto [comb, orderings] : power_table_map){
            power_table.push_back(power_table_row(comb, orderings));
        }

        // Sort the vector
        std::sort(power_table.begin(), power_table.end());

        return power_table;
    }



    // ========================================
    // =============== File I/O ===============
    // ========================================

    // Helper functions used to read and write files.

    /// @brief Read a file and split it into a vector of strings
    /// @param fin The input file stream
    /// @return A triple-nested vector of strings. 
    /// The outermost vector is a vector of paragraphs (separated by empty lines).
    /// The next vector is a vector of lines.
    /// The innermost vector is a vector of words (separated by spaces).
    std::vector<std::vector<std::vector<std::string> > > split_file(std::ifstream& fin){
        std::vector<std::vector<std::vector<std::string> > > data;

        std::string line_str;
        while(true){
            // skip extra empty lines between paragraphs (if any)
            while(true){                
                // Read one line. Return if EOF.
                if(!std::getline(fin, line_str)){ // EOF
                    return data;
                }
                if(!line_str.empty()){
                    break;
                }
            }
            // After the end of the above loop, line_str contains the first line of the paragraph
    
            // Split the paragraph
            std::vector<std::vector<std::string> > paragraph;
            while(true){
                // Split the line
                std::vector<std::string> line;

                std::istringstream iss(line_str);
                std::string word;
                while(iss >> word){
                    line.push_back(word);
                }

                // add the line to the paragraph
                paragraph.push_back(line);

                // Read the next line. Break if EOF or empty line.
                if(!std::getline(fin, line_str)){ // EOF
                    break;
                }
                if(line_str.empty()){
                    break;
                }
            }
            // After the end of the above loop, either fin is EOF or line_str contains an empty line.

            data.push_back(paragraph);
        }  
    }


    /// @brief Represents a paragraph in a file. 
    /// A paragraph should have the following contents in order:
    /// - A header line of the form "# Header"
    /// - Parameter lines of the form "$ key = value" (value is an integer)
    /// - Contents lines (any form).
    /// The contents are stored in a vector of vectors of strings,
    /// where the outer vector is the lines (separated by line break),
    /// and the inner vector is the words (separated by a space).
    struct file_paragraph{
        std::string header;
        std::map<std::string, int> parameters;
        std::vector<std::vector<std::string> > contents;
    };


    /// @brief Read a file and split it into a structured format of paragraphs.
    /// @param fin The input file stream. 
    /// The file should consist of a series of paragraphs, separated by empty lines.
    /// Each paragraph should have the structure described in the documentation of struct file_paragraph.
    /// @return A vector of file_paragraph objects.
    std::vector<file_paragraph> split_file_structure(std::ifstream& fin){
        // Load the file and split it into paragraph/line/words.
        std::vector<std::vector<std::vector<std::string> > > paragraphs = split_file(fin);
        
        std::vector<file_paragraph> result;
        for(const auto& paragraph : paragraphs){
            file_paragraph p;

            // Read header
            if(paragraph.empty() || paragraph[0].size() != 2 || paragraph[0][0] != "#"){
                WEIGHTED_VARIATIONAL_ERROR("Error: Paragraph must start with a header line of the form '# Header'.");
            }
            p.header = paragraph[0][1];

            // Read parameter lines (if any)
            if(paragraph.size() <= 1){ // Reached the end of the paragraph
                continue;
            }
            size_t lineNo = 1;
            while(lineNo < paragraph.size() && paragraph[lineNo][0] == "$"){
                if(paragraph[lineNo].size() != 4 || paragraph[lineNo][0] != "$" || paragraph[lineNo][2] != "="){
                    WEIGHTED_VARIATIONAL_ERROR("Error: Lines for parameters must be of the form '$ key = value'.");
                }

                // parse into key and value
                std::string key = paragraph[lineNo][1];
                int value = std::stoi(paragraph[lineNo][3]);
                p.parameters[key] = value;

                lineNo++;
            }

            // Read the remainder of the paragraph (if exists)
            if(paragraph.size() <= lineNo){ // Already at the end of the paragraph
                p.contents = std::vector<std::vector<std::string> >(); // empty
            }
            else{
                p.contents = std::vector<std::vector<std::string> >(paragraph.begin() + lineNo, paragraph.end());
            }

            // Append the paragraph to the result
            result.push_back(p);        
        }

        return result;
    }


    /// @brief Load a 2-dimensional vector of strings and convert it to a Eigen matrix.
    /// @param data 2-dimensional vector of strings.
    /// @param rows The expected number of rows.
    /// @param cols The expected number of columns.
    /// @param matrix_name The name of the matrix. Used only for error messages.
    /// @return 2-dimensional double-valued matrix.
    Eigen::MatrixXd read_dmatrix(const std::vector<std::vector<std::string> >& data, int rows, int cols, const std::string& matrix_name){
        if(rows <= 0 || cols <= 0){
            WEIGHTED_VARIATIONAL_ERROR("Error: The number of rows and columns should be positive.");
        }
        Eigen::MatrixXd result(rows, cols);

        if(data.size() != static_cast<size_t>(rows)){
            WEIGHTED_VARIATIONAL_ERROR("Error: The number of rows in" + matrix_name + " should be " + std::to_string(rows) + ".");
        }
        for(int i = 0; i < rows; i++){
            if(data[i].size() != static_cast<size_t>(cols)){
                WEIGHTED_VARIATIONAL_ERROR("Error: The number of columns in" + matrix_name + " should be " + std::to_string(cols) + ".");
            }
            for(int j = 0; j < cols; j++){
                result(i, j) = std::stod(data[i][j]);
            }
        }

        return result;
    }



    // =======================================================
    // =============== Polynomial optimization ===============
    // =======================================================

    // Optimize rational functions using the Newton method.

    /// @brief Compute the value of a polynomial at a point.
    /// @param coeff The coefficients of the polynomial.
    /// [a0, a1, a2, ...] represents a0 + a1 * x + a2 * x^2 + ...
    /// @param x The point at which the polynomial is evaluated.
    /// @return The value of the polynomial at x.
    inline double evaluate_polynomial(const std::vector<double> &coeff, double x){
        double result = 0.0;
        double x_power = 1.0;
        for(double c : coeff){
            result += c * x_power;
            x_power *= x;
        }
        return result;
    }
    

    /// @brief Find an extremum point of a rational function using the Newton method.
    /// @param coeff_numer The coefficients of the numerator of the rational function. 
    /// [a0, a1, a2, ...] represents a0 + a1 * x + a2 * x^2 + ...
    /// @param coeff_denom The coefficients of the denominator of the rational function.
    /// [b0, b1, b2, ...] represents b0 + b1 * x + b2 * x^2 + ...
    /// @param x_init The initial guess of the extremum point.
    /// @param is_analytic If true, the extremum is computed analytically. x_init will be ignored.
    /// @param max_itr The maximum number of iterations in the Newton method.
    /// @param x_tol The tolerance of the variable of the Newton method.
    /// @param filename The name of the file to save the result for debug.
    /// @return x such that the derivative of the rational function is zero, and the value of the function at x. 
    /// If is_analytic = true and the degree of the derivative is 2, the larger root is returned.
    std::pair<double, double> find_extremum(const std::vector<double> &coeff_numer, 
                                            const std::vector<double> &coeff_denom, 
                                            double x_init, 
                                            bool is_analytic = false,
                                            int max_itr = 100, 
                                            double x_tol = 1e-8, 
                                            std::string filename = ""){
        const int deg_numer = coeff_numer.size() - 1; // degree of the numerator
        const int deg_denom = coeff_denom.size() - 1; // degree of the denominator
        
        // ===== Find the derivatives of the numerator and the denominator =====
        std::vector<double> coeff_numer_diff(deg_numer); 
        for(int r = 1; r <= deg_numer; r++){
            coeff_numer_diff[r-1] = r * coeff_numer[r];
        }
        std::vector<double> coeff_denom_diff(deg_denom);
        for(int r = 1; r <= deg_denom; r++){
            coeff_denom_diff[r-1] = r * coeff_denom[r];
        }

        // ===== Find the numerator of the derivative of (numer/denom) =====
        // The numerator is a polynomial, which we write as g(x)

        int deg_deriv = deg_numer + deg_denom - 1;

        std::vector<double> coeff_deriv(deg_deriv + 1, 0.0);
        for(int r = 0; r <= deg_numer; r++){
            for(int s = 0; s <= deg_denom; s++){
                if(r < deg_numer){
                    coeff_deriv[r+s] += coeff_numer_diff[r] * coeff_denom[s];
                }
                if(s < deg_denom){
                    coeff_deriv[r+s] -= coeff_numer[r] * coeff_denom_diff[s];
                }
            }
        }

        // If the numerator and the denominator has the same degree, the highest degree term of g(x) should be canceled.
        if(deg_numer == deg_denom){
            double criteria = 1e-10 * coeff_numer[deg_numer] * coeff_denom[deg_denom];
            if(abs(coeff_deriv[deg_deriv]) > abs(criteria)){
                std::cout << "Warning: In the optimization of rational function, the highest degree term of the derivative is not zero." << std::endl;
                std::cout << "This may be caused by a numerical error." << std::endl;
            } 
            coeff_deriv.pop_back();
            deg_deriv -= 1;
        }

        // ===== Find the derivative g'(x) =====
        std::vector<double> coeff_deriv_deriv(deg_deriv);    
        for(int r = 1; r <= deg_deriv; r++){
            coeff_deriv_deriv[r-1] = r * coeff_deriv[r];
        }

        // ===== Solve g(x) = 0 =====
        double x_result;

        if(is_analytic){
            // Compute the extremum analytically
            if(deg_deriv == 2){
                double a = coeff_deriv[2];
                double b = coeff_deriv[1];
                double c = coeff_deriv[0];

                if(a == 0){
                    x_result = - c / b;
                }
                else if(a < 0){
                    x_result = - b - std::sqrt(b * b - 4 * a * c);
                    x_result /= (2 * a);
                }
                else if(a > 0){
                    x_result = - b + std::sqrt(b * b - 4 * a * c);
                    x_result /= (2 * a);
                }
            }
            else if(deg_deriv == 1){
                x_result = - coeff_deriv[0] / coeff_deriv[1];
            }
            else{
                WEIGHTED_VARIATIONAL_ERROR("Analytic solution is not available although is_analytic = true.")
            }
        }
        else{
            // Perform Newton method
            double x = x_init;
            double g_val;
            double gprime_val;

            double x_change;
            double x_change_prev[3] = {x_tol * 1e10, x_tol * 1e10, x_tol * 1e10}; // To check convergence three times in a row
            
            int iter;
            bool is_converged = false;

            for(iter = 0; iter < max_itr; iter++){
                // Compute the function and its derivative
                g_val = evaluate_polynomial(coeff_deriv, x);
                gprime_val = evaluate_polynomial(coeff_deriv_deriv, x);

                // Update x
                x_change = - g_val / gprime_val;
                x += x_change;
                x_change_prev[iter % 3] = x_change;

                // Check convergence
                if(abs(x_change_prev[0]) < x_tol && abs(x_change_prev[1]) < x_tol && abs(x_change_prev[2]) < x_tol){
                    is_converged = true;
                    break;
                }
            }
            
            if(!is_converged){
                std::cout << "Warning: The Newton method did not converge in find_extremum." << std::endl;
                std::cout << "The size of the last update " << x_change << " is larger than the tolerance" << x_tol << "." << std::endl;
            }
            else{
                // std::cout << "Newton method converged in " << iter - 2 << " iterations" << std::endl;
            }

            x_result = x;
        }

        // Save the result to a file if filename is given.
        // This is for debugging.
        if(!filename.empty()){
            std::ofstream fout(filename);
            if(!fout) WEIGHTED_VARIATIONAL_ERROR("Unable to open output file");

            // Save the final result
            fout << "Optimization result" << std::endl;
            fout << x_result << std::endl << std::endl;

            // Save the value of x, f(x), the numerator of f'(x) for x in [x_min, x_max]
            double x_max = std::max(1.0, abs(x_result) * 10);
            double x_min = - x_max;
            double dx = (x_max - x_min) / 1000;

            fout << "x f(x) g(x)" << std::endl;

            for(double x = x_min; x <= x_max; x += dx){
                double numer_val = evaluate_polynomial(coeff_numer, x);
                double denom_val = evaluate_polynomial(coeff_denom, x);
                double deriv_val = evaluate_polynomial(coeff_deriv, x);
                
                fout << x << " " << numer_val / denom_val << " " << deriv_val << std::endl;
            }

            fout.close();
        }


        // Return the result with the value of the function at the extremum    
        double numer_val = evaluate_polynomial(coeff_numer, x_result);
        double denom_val = evaluate_polynomial(coeff_denom, x_result);
    
        return {x_result, numer_val / denom_val};
    }

} // namespace utility
} // namespace weighted_variational



#endif // WEIGHTED_VARIATIONAL_UTILITY_H