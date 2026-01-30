#pragma once

#include "dataset.h"

#include <cstdint>

namespace dataset {

// Maximum squares per pattern
constexpr size_t MAX_PATTERN_SIZE = 10;

// Sentinel value indicating end of pattern squares
constexpr uint8_t PATTERN_END = 0xFF;
constexpr uint8_t __ = PATTERN_END;  // Short alias for readability

// Square indices on an 8x8 board
// clang-format off
enum Square : uint8_t {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
};
// clang-format on

// Pattern definitions matching Rust EVAL_F2X
// __ marks end of pattern (for patterns with fewer than 10 squares)
constexpr uint8_t PATTERN_SQUARES[NUM_FEATURES][MAX_PATTERN_SIZE] = {
    // 8-square patterns (0-19), each has 3^8 = 6561 possible values
    {C2, D2, B3, C3, D3, B4, C4, D4, __, __},  // 0: inner top-left
    {F2, E2, G3, F3, E3, G4, F4, E4, __, __},  // 1: inner top-right
    {C7, D7, B6, C6, D6, B5, C5, D5, __, __},  // 2: inner bottom-left
    {F7, E7, G6, F6, E6, G5, F5, E5, __, __},  // 3: inner bottom-right

    {A1, B2, C3, D4, E5, F6, G7, H8, __, __},  // 4: diagonal A1-H8
    {H1, G2, F3, E4, D5, C6, B7, A8, __, __},  // 5: diagonal H1-A8

    {D3, E4, F5, D4, E5, C4, D5, E6, __, __},  // 6: center pattern 1
    {E3, D4, C5, E4, D5, F4, E5, D6, __, __},  // 7: center pattern 2

    {A1, B1, C1, D1, E1, F1, G1, H1, __, __},  // 8: row 1
    {A8, B8, C8, D8, E8, F8, G8, H8, __, __},  // 9: row 8
    {A1, A2, A3, A4, A5, A6, A7, A8, __, __},  // 10: column A
    {H1, H2, H3, H4, H5, H6, H7, H8, __, __},  // 11: column H

    {B1, C1, D1, E1, B2, C2, D2, E2, __, __},  // 12: top edge 2x4
    {G1, F1, E1, D1, G2, F2, E2, D2, __, __},  // 13: top edge 2x4 (mirrored)
    {B8, C8, D8, E8, B7, C7, D7, E7, __, __},  // 14: bottom edge 2x4
    {G8, F8, E8, D8, G7, F7, E7, D7, __, __},  // 15: bottom edge 2x4 (mirrored)
    {A2, A3, A4, A5, B2, B3, B4, B5, __, __},  // 16: left edge 2x4
    {A7, A6, A5, A4, B7, B6, B5, B4, __, __},  // 17: left edge 2x4 (mirrored)
    {H2, H3, H4, H5, G2, G3, G4, G5, __, __},  // 18: right edge 2x4
    {H7, H6, H5, H4, G7, G6, G5, G4, __, __},  // 19: right edge 2x4 (mirrored)

    // 9-square patterns (20-23), each has 3^9 = 19683 possible values
    {A1, B1, C1, A2, B2, C2, A3, B3, C3, __},  // 20: corner A1 3x3
    {H1, G1, F1, H2, G2, F2, H3, G3, F3, __},  // 21: corner H1 3x3
    {A8, B8, C8, A7, B7, C7, A6, B6, C6, __},  // 22: corner A8 3x3
    {H8, G8, F8, H7, G7, F7, H6, G6, F6, __},  // 23: corner H8 3x3
};

// Number of squares in each pattern
constexpr uint8_t PATTERN_SIZES[NUM_FEATURES] = {
    8, 8, 8, 8,
    8, 8, 8, 8,
    8, 8, 8, 8,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9
};

// Extract features from a board position
// player: bitboard of player's pieces
// opponent: bitboard of opponent's pieces
// out: array of NUM_FEATURES uint16_t values to store feature indices
inline void extract_features(uint64_t player, uint64_t opponent, uint16_t *out) {
    // Base-3 encoding matching Rust: player=0, opponent=1, empty=2
    // Calculation order: feature = feature * 3 + value (first square is MSB)
    for (size_t p = 0; p < NUM_FEATURES; ++p) {
        uint16_t feature = 0;
        for (size_t i = 0; i < PATTERN_SIZES[p]; ++i) {
            uint8_t sq = PATTERN_SQUARES[p][i];
            uint64_t mask = 1ULL << sq;

            uint8_t value;
            if (player & mask) {
                value = 0;  // player
            } else if (opponent & mask) {
                value = 1;  // opponent
            } else {
                value = 2;  // empty
            }

            feature = feature * 3 + value;
        }
        out[p] = feature;
    }
}

} // namespace dataset
