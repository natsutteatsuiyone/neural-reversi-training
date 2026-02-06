#pragma once

#include <cassert>
#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#define BSWAP64(x) _byteswap_uint64(x)
#define POPCOUNT64(x) __popcnt64(x)
#else
#define BSWAP64(x) __builtin_bswap64(x)
#define POPCOUNT64(x) __builtin_popcountll(x)
#endif

namespace dataset {

// Delta swap: swap bits selected by mask with bits shifted by delta positions
// Matches Rust: let tmp = mask & (bits ^ (bits << delta)); bits ^ tmp ^ (tmp >> delta)
inline uint64_t delta_swap(uint64_t x, uint64_t mask, int delta) {
    uint64_t t = mask & (x ^ (x << delta));
    return x ^ t ^ (t >> delta);
}

// Flip vertically (swap rows)
inline uint64_t flip_vertical(uint64_t x) { return BSWAP64(x); }

// Flip horizontally (mirror each row)
inline uint64_t flip_horizontal(uint64_t x) {
    constexpr uint64_t k1 = 0x5555555555555555ULL;
    constexpr uint64_t k2 = 0x3333333333333333ULL;
    constexpr uint64_t k4 = 0x0f0f0f0f0f0f0f0fULL;
    x = ((x >> 1) & k1) | ((x & k1) << 1);
    x = ((x >> 2) & k2) | ((x & k2) << 2);
    x = ((x >> 4) & k4) | ((x & k4) << 4);
    return x;
}

// Flip along A1-H8 diagonal (transpose)
// Matches Rust masks and order
inline uint64_t flip_diag_a1h8(uint64_t x) {
    x = delta_swap(x, 0x0f0f0f0f00000000ULL, 28);
    x = delta_swap(x, 0x3333000033330000ULL, 14);
    x = delta_swap(x, 0x5500550055005500ULL, 7);
    return x;
}

// Flip along A8-H1 diagonal (anti-transpose)
// Matches Rust masks and order
inline uint64_t flip_diag_a8h1(uint64_t x) {
    x = delta_swap(x, 0xf0f0f0f000000000ULL, 36);
    x = delta_swap(x, 0xcccc0000cccc0000ULL, 18);
    x = delta_swap(x, 0xaa00aa00aa00aa00ULL, 9);
    return x;
}

// Rotate 90 degrees clockwise
// Matches Rust: flip_diag_a8h1 then flip_vertical
inline uint64_t rotate_90_cw(uint64_t x) {
    return flip_vertical(flip_diag_a8h1(x));
}

// Rotate 180 degrees
inline uint64_t rotate_180(uint64_t x) {
    return flip_vertical(flip_horizontal(x));
}

// Rotate 270 degrees clockwise (= 90 degrees counter-clockwise)
// Matches Rust: flip_diag_a1h8 then flip_vertical
inline uint64_t rotate_270_cw(uint64_t x) {
    return flip_vertical(flip_diag_a1h8(x));
}

// Apply one of 8 symmetry transformations (0-7)
inline void apply_symmetry(uint64_t &player, uint64_t &opponent, int sym) {
    assert(sym >= 0 && sym < 8 && "symmetry index must be in range [0, 7]");
    switch (sym) {
    case 0: // Identity
        break;
    case 1: // Rotate 90 CW
        player = rotate_90_cw(player);
        opponent = rotate_90_cw(opponent);
        break;
    case 2: // Rotate 180
        player = rotate_180(player);
        opponent = rotate_180(opponent);
        break;
    case 3: // Rotate 270 CW
        player = rotate_270_cw(player);
        opponent = rotate_270_cw(opponent);
        break;
    case 4: // Flip vertical
        player = flip_vertical(player);
        opponent = flip_vertical(opponent);
        break;
    case 5: // Flip horizontal
        player = flip_horizontal(player);
        opponent = flip_horizontal(opponent);
        break;
    case 6: // Flip A1-H8 diagonal
        player = flip_diag_a1h8(player);
        opponent = flip_diag_a1h8(opponent);
        break;
    case 7: // Flip A8-H1 diagonal
        player = flip_diag_a8h1(player);
        opponent = flip_diag_a8h1(opponent);
        break;
    }
}

// Propagate flips in both directions along a ray.
// mask: opponent pieces with edge masking to prevent wrap-around.
// dir: shift amount in bits for the direction.
inline uint64_t get_some_moves(uint64_t player, uint64_t mask, int dir) {
    uint64_t flip = ((player << dir) | (player >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    return (flip << dir) | (flip >> dir);
}

// Get legal moves for player given opponent position
// Returns a bitboard of all legal move positions
inline uint64_t get_legal_moves(uint64_t player, uint64_t opponent) {
    uint64_t empty = ~(player | opponent);
    // Direction masks applied to opponent to prevent wrap-around:
    //   0x7E7E7E7E7E7E7E7E = excludes A-file and H-file (horizontal)
    //   0x00FFFFFFFFFFFF00 = excludes rank 1 and rank 8 (vertical)
    //   0x007E7E7E7E7E7E00 = excludes both edge files and edge ranks (diagonal)
    return (get_some_moves(player, opponent & 0x7E7E7E7E7E7E7E7EULL, 1) & empty)
         | (get_some_moves(player, opponent & 0x00FFFFFFFFFFFF00ULL, 8) & empty)
         | (get_some_moves(player, opponent & 0x007E7E7E7E7E7E00ULL, 7) & empty)
         | (get_some_moves(player, opponent & 0x007E7E7E7E7E7E00ULL, 9) & empty);
}

// Count the number of legal moves (mobility)
inline int count_mobility(uint64_t player, uint64_t opponent) {
    return static_cast<int>(POPCOUNT64(get_legal_moves(player, opponent)));
}

} // namespace dataset
