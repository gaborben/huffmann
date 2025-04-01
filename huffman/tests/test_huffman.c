#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "huffman.h"

void test_encode_basic(void) {
    char input[] = "ab";
    size_t input_len = 2;
    char codes[256][256] = {{0}};
    huffmanEncoding(input, input_len, codes);

    char output_bits[256] = {0};
    size_t bit_len = 0;
    encode_input_with_huffman(input, input_len, codes, output_bits, &bit_len);

    assert_true(bit_len > 0);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_encode_basic),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
