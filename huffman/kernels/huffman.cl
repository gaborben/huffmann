__kernel void byte_frequency_kernel(__global const char* input,
                                    __global int* frequency,
                                    const uint length) {
    int gid = get_global_id(0);
    if (gid >= length) return;

    uchar byte_val = (uchar)input[gid];
    atomic_inc(&frequency[byte_val]);
}
