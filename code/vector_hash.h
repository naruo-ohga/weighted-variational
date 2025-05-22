/**
 * vector_hash.h
 * 
 * This file supports a hash function for a vector of integers based on wyhash (https://github.com/wangyi-fudan/wyhash).
 * I made this implementation by referencing l.219-264 of unordered_dense.h (https://github.com/martinus/unordered_dense),
 * which should be distributed along with this file.
 */


#ifndef VECTOR_HASH_H
#define VECTOR_HASH_H

#include "unordered_dense.h" 

struct VectorHash {
    int order; 
    uint64_t length;
    uint64_t seed[3];
    
    // magic numbers taken from l.220-223 of unordered_dense.h, which originally comes from wyhash
    static constexpr const uint64_t magic0 = 0xa0761d6478bd642fULL;
    static constexpr const uint64_t magic[3] = {0xe7037ed1a0b428dbULL, 0x8ebc6af09c88c6e3ULL, 0x589965cc75374cc3ULL};  

    // Constructor
    VectorHash(int length) : order(0), length(length), seed{magic0, magic0, magic0} {
    }
    
    // Update seed, given two values.
    // val1 and val2 should be within 8 bits
    // Should be called for ceil(length/2) times.
    // If the size of the vector is an odd number, val2 should be 0 for the last element.
    inline void update(u_int64_t val1, u_int64_t val2) {
        seed[order] = ankerl::unordered_dense::detail::wyhash::mix(val1 ^ magic[order], val2 ^ seed[order]);
        order = (order + 1) % 3;
    }

    // Get hash
    // Destroys the seed. Should be called only once.
    inline uint64_t get() {
        if(length > 2){
            seed[0] ^= seed[1];
        }
        if(length > 4){
            seed[0] ^= seed[2];
        }
        return ankerl::unordered_dense::detail::wyhash::mix(magic[0] ^ length, ankerl::unordered_dense::detail::wyhash::mix(magic[0], seed[0]));
    }
};

#endif // VECTOR_HASH_H