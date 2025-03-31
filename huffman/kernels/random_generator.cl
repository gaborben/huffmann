__kernel void generate_random_kernel(__global uchar* output, ulong seed, ulong length) {
    ulong gid = get_global_id(0);
    ulong stride = get_global_size(0);

    // Lineáris kongruens generátor (LCG)
    ulong a = 1664525;
    ulong c = 1013904223;

    ulong local_seed = seed ^ (gid * 0x5DEECE66DULL + 0xB);

    // Warm up (csak keverés)
    for (int i = 0; i < 10; i++) {
        local_seed = a * local_seed + c;
    }

    for (ulong i = gid; i < length; i += stride) {
        local_seed = a * local_seed + c;

        // Bit mixing
        ulong mixed = local_seed;
        mixed ^= mixed >> 33;
        mixed ^= mixed << 21;
        mixed ^= mixed >> 11;

        //output[i] = (uchar)(mixed & 0xFF);
        output[i] = (uchar)((mixed % 255) + 1);

    }
}
