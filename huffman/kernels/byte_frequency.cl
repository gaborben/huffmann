__kernel void byte_frequency_kernel(__global const char* input,
                                    __global int* global_freq,
                                    const ulong length) {
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    const int global_id = get_global_id(0);
    const int global_size = get_global_size(0);

    __local int local_freq[256];
    
    // Csak az első 256 szál nullázza a tömböt
    if (local_id < 256) {
        local_freq[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (ulong i = global_id; i < length; i += global_size) {
        uchar byte_val = (uchar)input[i];
        atomic_inc(&local_freq[byte_val]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 256) {
        atomic_add(&global_freq[local_id], local_freq[local_id]);
    }
}
