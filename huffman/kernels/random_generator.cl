__kernel void generate_random_kernel(__global uchar* output, ulong seed, ulong length) {
    ulong gid = get_global_id(0);
    ulong stride = get_global_size(0);

    // ulong array[256*128];
    // int e = 1;
    // for (int i = 0; i < 256; i++) {
    //     for (int j = 0; j < e; j++) {
    //         array[i+j] = e;
    //     }
    //     e++;
    // }


    

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

        float r = (float)(mixed & 0xFFFFFFFF) / 4294967295.0f;

        float r_updated = pow(r, 2.5f);

        uchar val = (uchar)(r_updated * 254.0f) + 1;
        output[i] = val;

        //output[i] = (uchar)(mixed & 0xFF);
        //output[i] = (uchar)((mixed % 255) + 1);

        // ulong array[256*128];
        // int e = 1;
        // for (int i = 0; i < 256; i++) {
        //     for (int j = 0; j < e; j++) {
        //         array[i+j] = e;
        //     }
        //     e++;
        // }
        // output[i] = (uchar)(array[(mixed % (256 * 128)) + 1]);

    }
}
