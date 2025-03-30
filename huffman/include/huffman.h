#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <stddef.h>

typedef struct Node {
    char charValue;
    int freq;
    struct Node* left;
    struct Node* right;
} Node;

void huffmanEncoding(const char* word, size_t length, char codes[256][256]);
void huffmanEncoding2(const int freq[256], char codes[256][256]);




#endif