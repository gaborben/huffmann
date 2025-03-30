#ifndef HUFFMAN_H
#define HUFFMAN_H

typedef struct Node {
    char charValue;
    int freq;
    struct Node* left;
    struct Node* right;
} Node;

void huffmanEncoding(const char* word, char codes[256][256]);

#endif