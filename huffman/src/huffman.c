#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "huffman.h"

#define MAX_INPUT_SIZE 100000000

static Node* nodes[256];
static int nodeCount = 0;

static void calculateFrequencies(const char* word, size_t length) {
    int frequencies[256] = {0};

    for (size_t i = 0; i < length; i++) {
        frequencies[(unsigned char)word[i]]++;
    }

    for (int i = 0; i < 256; i++) {
        if (frequencies[i] > 0) {
            nodes[nodeCount] = (Node*)malloc(sizeof(Node));
            nodes[nodeCount]->charValue = (char)i;
            nodes[nodeCount]->freq = frequencies[i];
            nodes[nodeCount]->left = NULL;
            nodes[nodeCount]->right = NULL;
            nodeCount++;
        }
    }
}

static void insertionSort(Node* arr[], int n) {
    for (int i = 1; i < n; i++) {
        Node* key = arr[i];
        int j = i - 1;

        while (j >= 0 && arr[j]->freq > key->freq) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

static Node* buildHuffmanTree() {
    while (nodeCount > 1) {
        insertionSort(nodes, nodeCount);

        Node* left = nodes[0];
        Node* right = nodes[1];

        Node* merged = (Node*)malloc(sizeof(Node));
        merged->charValue = '\0';
        merged->freq = left->freq + right->freq;
        merged->left = left;
        merged->right = right;

        for (int i = 2; i < nodeCount; i++) {
            nodes[i - 2] = nodes[i];
        }
        nodeCount -= 2;
        nodes[nodeCount++] = merged;
    }

    return nodes[0];
}

static void generateHuffmanCodes(Node* node, char* currentCode, int depth, char codes[256][256]) {
    if (node == NULL) return;

    if (node->charValue != '\0' && node->left == NULL && node->right == NULL) {
        strncpy(codes[(unsigned char)node->charValue], currentCode, depth);
        codes[(unsigned char)node->charValue][depth] = '\0';
    }

    if (node->left) {
        currentCode[depth] = '0';
        generateHuffmanCodes(node->left, currentCode, depth + 1, codes);
    }

    if (node->right) {
        currentCode[depth] = '1';
        generateHuffmanCodes(node->right, currentCode, depth + 1, codes);
    }
}

void huffmanEncoding(const char* word, size_t length, char codes[256][256]) {
    nodeCount = 0;
    memset(nodes, 0, sizeof(nodes));
    calculateFrequencies(word, length);
    Node* root = buildHuffmanTree();

    char currentCode[256];
    generateHuffmanCodes(root, currentCode, 0, codes);
}

void huffmanEncoding2(const int freq[256], char codes[256][256]) {
    nodeCount = 0;
    memset(nodes, 0, sizeof(nodes));

    for (int i = 0; i < 256; i++) {
        if (freq[i] > 0) {
            nodes[nodeCount] = (Node*)malloc(sizeof(Node));
            nodes[nodeCount]->charValue = (char)i;
            nodes[nodeCount]->freq = freq[i];
            nodes[nodeCount]->left = NULL;
            nodes[nodeCount]->right = NULL;
            nodeCount++;
        }
    }

    Node* root = buildHuffmanTree();
    char currentCode[256];
    generateHuffmanCodes(root, currentCode, 0, codes);
}

// void encode_input_with_huffman(const char* input, size_t input_len, char codes[256][256], char* output_bits, size_t* output_bit_len) {
//     size_t bit_index = 0;
//     for (size_t i = 0; i < input_len; i++) {
//         unsigned char byte = (unsigned char)input[i];
//         const char* code = codes[byte];
//         for (int j = 0; code[j] != '\0'; j++) {
//             output_bits[bit_index++] = code[j];
//         }
//     }
//     output_bits[bit_index] = '\0';
//     *output_bit_len = bit_index;
// }

void encode_input_with_huffman(const char* input, size_t input_len, char codes[256][256], char* output_bits, size_t* output_bit_len) {
    size_t bit_index = 0;
    for (size_t i = 0; i < input_len; i++) {
        unsigned char byte = (unsigned char)input[i];
        const char* code = codes[byte];

        if (code == NULL || code[0] == '\0') {
            printf("[ERROR] Missing Huffman code for byte %d (char: '%c')\n", byte, byte);
            continue;
        }

        for (int j = 0; code[j] != '\0'; j++) {
            output_bits[bit_index++] = code[j];

            if (bit_index >= MAX_INPUT_SIZE * 20 - 1) {
                printf("[WARN] Output bitstream buffer full!\n");
                break;
            }
        }
    }

    output_bits[bit_index] = '\0';
    *output_bit_len = bit_index;
    printf("[DEBUG] Huffman bitstream length: %zu\n", bit_index);
}
