#include "kernel_loader.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_INPUT_SIZE 100000000
#define CHUNK_SIZE 1024

int compare_freq(const void* a, const void* b) {
    const int* fa = (const int*)a;
    const int* fb = (const int*)b;
    return fb[1] - fa[1];
}

int main(void) {
    cl_int err;
    int error_code;

    char* input = (char*)malloc(MAX_INPUT_SIZE);
    if (!input) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    size_t input_len = 0;

    printf("Choose input method:\n");
    printf("1. Load from file (input.txt)\n");
    printf("2. Generate random input (%d bytes) with OpenCL\n", MAX_INPUT_SIZE);
    printf("Enter choice [1/2]: ");
    int choice = 0;
    scanf("%d", &choice);

    // OpenCL early setup
    cl_platform_id platform_id;
    cl_uint n_platforms;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);

    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);

    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

    if (choice == 1) {
        FILE* fp = fopen("input.txt", "rb");
        if (!fp) {
            perror("Cannot open input.txt");
            free(input);
            return 1;
        }
        input_len = fread(input, 1, MAX_INPUT_SIZE, fp);
        fclose(fp);
        printf("Loaded %zu bytes from file.\n", input_len);
    } else {
        input_len = MAX_INPUT_SIZE;
        // random_generator.cl betöltése
        const char* rand_kernel_code = load_kernel_source("kernels/random_generator.cl", &error_code);
        if (error_code != 0) {
            fprintf(stderr, "Random kernel load error!\n");
            free(input);
            return 1;
        }

        cl_program rand_program = clCreateProgramWithSource(context, 1, &rand_kernel_code, NULL, NULL);
        err = clBuildProgram(rand_program, 1, &device_id, "", NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(rand_program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char* build_log = (char*)malloc(log_size);
            clGetProgramBuildInfo(rand_program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
            fprintf(stderr, "Random kernel build error:\n%s\n", build_log);
            free(build_log);
            return 1;
        }

        cl_kernel rand_kernel = clCreateKernel(rand_program, "generate_random_kernel", NULL);
        cl_mem rand_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input_len, NULL, NULL);

        cl_ulong seed = (cl_ulong)time(NULL);
        clSetKernelArg(rand_kernel, 0, sizeof(cl_mem), &rand_buffer);
        clSetKernelArg(rand_kernel, 1, sizeof(cl_ulong), &seed);
        clSetKernelArg(rand_kernel, 2, sizeof(cl_ulong), &input_len);

        size_t local_size = 256;
        size_t global_size = ((input_len + local_size - 1) / local_size) * local_size;

        clEnqueueNDRangeKernel(command_queue, rand_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        clFinish(command_queue);
        clEnqueueReadBuffer(command_queue, rand_buffer, CL_TRUE, 0, input_len, input, 0, NULL, NULL);

        clReleaseKernel(rand_kernel);
        clReleaseProgram(rand_program);
        clReleaseMemObject(rand_buffer);

        printf("Generated %zu random bytes with OpenCL.\n", input_len);
    }

    // CPU megoldás
    clock_t start_seq = clock();
    int freq_seq[256] = {0};
    for (size_t i = 0; i < input_len; i++) {
        unsigned char byte = (unsigned char)input[i];
        freq_seq[byte]++;
    }
    clock_t end_seq = clock();
    double time_seq = (double)(end_seq - start_seq) / CLOCKS_PER_SEC;

    // OpenCL kernel betöltése
    const char* kernel_code = load_kernel_source("kernels/byte_frequency.cl", &error_code);
    if (error_code != 0) {
        fprintf(stderr, "Kernel source load error!\n");
        free(input);
        return 1;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    err = clBuildProgram(program, 1, &device_id, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* build_log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        fprintf(stderr, "Build error:\n%s\n", build_log);
        free(build_log);
        free(input);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "byte_frequency_kernel", NULL);
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, input_len, NULL, NULL);
    cl_mem freq_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(int), NULL, NULL);

    clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0, input_len, input, 0, NULL, NULL);
    clEnqueueFillBuffer(command_queue, freq_buffer, &(int){0}, sizeof(int), 0, 256 * sizeof(int), 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &freq_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_ulong), &input_len);

    size_t num_chunks = (input_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
    size_t local_work_size = 256;
    size_t global_work_size = ((num_chunks + local_work_size - 1) / local_work_size) * local_work_size;

    cl_event event;
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                           &global_work_size, &local_work_size,
                           0, NULL, &event);
    clFinish(command_queue);

    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double time_gpu = (end - start) / 1e9;

    int freq_gpu[256];
    clEnqueueReadBuffer(command_queue, freq_buffer, CL_TRUE, 0, 256 * sizeof(int), freq_gpu, 0, NULL, NULL);

    int freq_seq_top[256][2];
    int freq_gpu_top[256][2];
    for (int i = 0; i < 256; i++) {
        freq_seq_top[i][0] = i;
        freq_seq_top[i][1] = freq_seq[i];

        freq_gpu_top[i][0] = i;
        freq_gpu_top[i][1] = freq_gpu[i];
    }

    qsort(freq_seq_top, 256, sizeof(freq_seq_top[0]), compare_freq);
    qsort(freq_gpu_top, 256, sizeof(freq_gpu_top[0]), compare_freq);

    printf("\nSeq: Top 10 byte frequencies:\n");
    for (int i = 0; i < 10; i++) {
        printf("Byte %3d: %d\n", freq_seq_top[i][0], freq_seq_top[i][1]);
    }
    printf("Seq: Runtime: %.6f sec\n", time_seq);

    printf("\nOpenCL: Top 10 byte frequencies:\n");
    for (int i = 0; i < 10; i++) {
        printf("Byte %3d: %d\n", freq_gpu_top[i][0], freq_gpu_top[i][1]);
    }
    printf("OpenCL: Runtime: %.6f sec\n", time_gpu);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(freq_buffer);
    clReleaseCommandQueue(command_queue);
    free(input);

    return 0;
}
