__kernel void generate_random_kernel(__global uchar* output, ulong seed, ulong length) {
    ulong gid = get_global_id(0);
    ulong stride = get_global_size(0);

    // Lineáris kongruens generátor (LCG)
    ulong a = 1664525;
    ulong c = 1013904223;

    // Szálanként más seed
    ulong local_seed = seed ^ gid;

    for (ulong i = gid; i < length; i += stride) {
        local_seed = (a * local_seed + c);
        output[i] = (uchar)(local_seed & 0xFF); // csak 0–255 lehetS
    }
}
