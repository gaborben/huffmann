#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <stddef.h> // size_t miatt kell
#include <stdint.h> // ha uint8_t-t használnál

typedef struct Node {
    char charValue;
    int freq;
    struct Node* left;
    struct Node* right;
} Node;

void huffmanEncoding2(const int freq[256], char codes[256][256]);
void encode_input_with_huffman(const char* input, size_t input_len, char codes[256][256], char* output_bits, size_t* bit_len);

#endif
