#include <gtest/gtest.h>
#include "huffman.h"

TEST(HuffmanTest, CodeGenerated) {
    int freq[256] = {0};
    freq['a'] = 3;
    freq['b'] = 1;

    char codes[256][256] = {{0}};
    huffmanEncoding2(freq, codes);

    EXPECT_STRNE(codes['a'], "");
    EXPECT_STRNE(codes['b'], "");

    const char* input = "ab";
    size_t input_len = 2;
    char output_bits[256] = {0};
    size_t bit_len = 0;

    encode_input_with_huffman(input, input_len, codes, output_bits, &bit_len);

    EXPECT_GT(bit_len, 0);
}
