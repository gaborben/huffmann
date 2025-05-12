#include "kernel_loader.h"
#include "huffman.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#define MAX_INPUT_SIZE 100000000//100000000
#define CHUNK_SIZE 2048

void generate_random_seq(unsigned char *output, uint64_t length) {
    const uint64_t a = 1664525ULL;
    const uint64_t c = 1013904223ULL;
    cl_ulong seed = (cl_ulong)time(NULL);
    
    for (uint64_t i = 0; i < length; ++i) {;
        uint64_t local_seed = seed ^ (i * 0x5DEECE66DULL + 0xBULL);
        for (int j = 0; j < 10; ++j) {
            local_seed = a * local_seed + c;
        }
        local_seed = a * local_seed + c;
        uint64_t mixed = local_seed;
        mixed ^= mixed >> 33;
        mixed ^= mixed << 21;
        mixed ^= mixed >> 11;
        float r = (float)(mixed & 0xFFFFFFFFULL) / 4294967295.0f;
        float r_updated = powf(r, 2.5f);
        unsigned char val = (unsigned char)(r_updated * 254.0f) + 1;
        output[i] = val;
    }
}

int compare_freq(const void* a, const void* b) {
    const int* fa = (const int*)a;
    const int* fb = (const int*)b;
    return fb[1] - fa[1];
}

int manual(int input_size) {
    cl_int err;
    int error_code;

    char* input = malloc(input_size);
    if (!input) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    size_t input_len = 0;
    printf("Choose input method:\n");
    printf("1. Load from file (input.txt)\n");
    printf("2. Generate random input (%d bytes) with OpenCL\n", input_size);
    printf("3. Enter text manually\n");
    printf("Enter choice [1/2/3]: ");
    int choice;
    scanf("%d", &choice);
    getchar();

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
        input_len = fread(input, 1, input_size, fp);
        fclose(fp);
        printf("Loaded %zu bytes from file.\n", input_len);
    } else if (choice == 2) {

        //Seq
        unsigned char *input_seq = malloc(input_size);
        clock_t start_seq = clock();
        generate_random_seq(input_seq, input_size);
        clock_t end_seq = clock();
        double time_seq = (double)(end_seq - start_seq) / CLOCKS_PER_SEC;
        printf("Seq generation time: %.4f sec\n", time_seq);

        //OpenCL
        input_len = input_size;
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

        clock_t start_gpu = clock();
        clEnqueueNDRangeKernel(command_queue, rand_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        clFinish(command_queue);
        clEnqueueReadBuffer(command_queue, rand_buffer, CL_TRUE, 0, input_len, input, 0, NULL, NULL);
        clock_t end_gpu = clock();
        double time_gpu = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
        printf("OpenCL generation time: %.4f sec\n", time_gpu);

        clReleaseKernel(rand_kernel);
        clReleaseProgram(rand_program);
        clReleaseMemObject(rand_buffer);

        //printf("Generated %zu random bytes with OpenCL.\n", input_len);
    } else {
        printf("Enter text: ");
        fgets(input, input_size, stdin);
        input_len = strlen(input);
        if (input[input_len - 1] == '\n') {
            input[--input_len] = '\0';
        }
    }

    // Seq
    clock_t start_seq = clock();
    int freq_seq[256] = {0};
    for (size_t i = 0; i < input_len; i++) {
        unsigned char byte = (unsigned char)input[i];
        freq_seq[byte]++;
    }
    clock_t end_seq = clock();
    double time_seq = (double)(end_seq - start_seq) / CLOCKS_PER_SEC;

    // OpenCL
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

    char codes[256][256] = {{0}};
    //huffmanEncoding(input, input_len, codes);
    huffmanEncoding2(freq_gpu, codes);
    
	printf("\nSeq: Top 10 byte frequencies:\n");
	for (int i = 0; i < 10; i++) {
		int byteVal = freq_seq_top[i][0];
		int count = freq_seq_top[i][1];
		printf("Byte %3d: %d\n", byteVal, count);
		if (codes[byteVal][0] != '\0') {
			printf("Code: %s\n", codes[byteVal]);
		}
	}
	printf("Seq: Runtime: %.6f sec\n", time_seq);

	printf("\nOpenCL: Top 10 byte frequencies:\n");
	for (int i = 0; i < 10; i++) {
		int byteVal = freq_gpu_top[i][0];
		int count = freq_gpu_top[i][1];
		printf("Byte %3d: %d\n", byteVal, count);
		if (codes[byteVal][0] != '\0') {
			printf("Code: %s\n", codes[byteVal]);
		}
	}
	printf("OpenCL: Runtime: %.6f sec\n", time_gpu);

	// Huffman seq

	//char* encoded_bits_seq = malloc(MAX_INPUT_SIZE * 20);
    //char encoded_bits_seq[4096];

    size_t total_bits = 0;
    for (size_t i = 0; i < input_len; i++) {
        total_bits += strlen(codes[(unsigned char)input[i]]);
    }
    char* encoded_bits_seq = malloc(total_bits + 1);
    if (!encoded_bits_seq) {
        fprintf(stderr, "Memory allocation failed for encoded bits!\n");
        free(input);
        return 1;
    }

	size_t bitlen_seq = 0;
	clock_t start_huff_seq = clock();
	encode_input_with_huffman(input, input_len, codes, encoded_bits_seq, &bitlen_seq);
	clock_t end_huff_seq = clock();
	double time_huff_seq = (double)(end_huff_seq - start_huff_seq) / CLOCKS_PER_SEC;

    printf("Huffman encoding runtime: %.6f sec\n", time_huff_seq);

    size_t compressed_bytes = (bitlen_seq + 7) / 8; // byte-ra kerekítés felfelé
    double compression_ratio = (double)compressed_bytes / (double)input_len;
    double saving = 100.0 - (compression_ratio * 100.0);

    printf("Original size: %zu bytes\n", input_len);
    printf("Compressed size: %zu bytes (%.0f bits)\n", compressed_bytes, (double)bitlen_seq);
    printf("Compression ratio: %.2f%%\n", compression_ratio * 100.0);
    printf("Space saved: %.2f%%\n", saving);

    printf("\nIn which do you want to get the first 100 bits of the Huffman code?\n");
	printf("1. Print to screen\n");
	printf("2. Save to output.txt\n");
	printf("Enter choice [1/2]: ");
	int output_choice;
	scanf("%d", &output_choice);
	getchar();

	if (output_choice == 1) {
		printf("First 100 bits:\n");
		for (int i = 0; i < 100 && i < bitlen_seq; i++) {
			putchar(encoded_bits_seq[i]);
		}
		putchar('\n');
	} else {
		FILE* out = fopen("output.txt", "w");
		if (out) {
			fwrite(encoded_bits_seq, 1, (bitlen_seq > 100 ? 100 : bitlen_seq), out);
			fclose(out);
			printf("First 100 bits written to output.txt\n");
		} else {
			perror("Failed to open output.txt for writing");
		}
	}

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

int test(size_t input_size, FILE *f_gen,FILE *f_freq, FILE *f_comp) {
    cl_int err;
    int error_code;

    char* input = malloc(input_size);
    if (!input) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // OpenCL early setup
    cl_platform_id platform_id;
    cl_uint n_platforms;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);

    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);

    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);
    
    #pragma region Generation
     //Generation start

    //Seq
    unsigned char *input_seq = malloc(input_size);
    clock_t start_gen_seq = clock();
    generate_random_seq(input_seq, input_size);
    clock_t end_gen_seq = clock();
    double time_gen_seq = (double)(end_gen_seq - start_gen_seq) / CLOCKS_PER_SEC;
    //printf("Seq generation time: %.4f sec\n", time_seq);

    //OpenCL
    cl_ulong input_len = (cl_ulong)input_size;
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

    clock_t start_gen_gpu = clock();
    clEnqueueNDRangeKernel(command_queue, rand_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    clFinish(command_queue);
    clEnqueueReadBuffer(command_queue, rand_buffer, CL_TRUE, 0, input_len, input, 0, NULL, NULL);
    clock_t end_gen_gpu = clock();
    double time_gen_gpu = (double)(end_gen_gpu - start_gen_gpu) / CLOCKS_PER_SEC;
    //printf("OpenCL generation time: %.4f sec\n", time_gpu);

    fprintf(f_gen, "%zu,%.4f,%.4f\n", input_size, time_gen_seq, time_gen_gpu);

    //Generation end
    #pragma endregion

    #pragma region Byte frequency

    clReleaseKernel(rand_kernel);
    clReleaseProgram(rand_program);
    clReleaseMemObject(rand_buffer);

    // Seq
    clock_t start_seq = clock();
    int freq_seq[256] = {0};
    for (size_t i = 0; i < input_len; i++) {
        unsigned char byte = (unsigned char)input[i];
        freq_seq[byte]++;
    }
    clock_t end_seq = clock();
    double time_seq = (double)(end_seq - start_seq) / CLOCKS_PER_SEC;

    // OpenCL 
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

    char codes[256][256] = {{0}};
    huffmanEncoding2(freq_gpu, codes);
    
	//printf("\nSeq: Top 10 byte frequencies:\n");
	for (int i = 0; i < 10; i++) {
		int byteVal = freq_seq_top[i][0];
		int count = freq_seq_top[i][1];
		// printf("Byte %3d: %d\n", byteVal, count);
		// if (codes[byteVal][0] != '\0') {
		// 	printf("Code: %s\n", codes[byteVal]);
		// }
	}
	//printf("Seq: Runtime: %.6f sec\n", time_seq);

	//printf("\nOpenCL: Top 10 byte frequencies:\n");
	for (int i = 0; i < 10; i++) {
		int byteVal = freq_gpu_top[i][0];
		int count = freq_gpu_top[i][1];
		// printf("Byte %3d: %d\n", byteVal, count);
		// if (codes[byteVal][0] != '\0') {
		// 	//printf("Code: %s\n", codes[byteVal]);
		// }
	}
	//printf("OpenCL: Runtime: %.6f sec\n", time_gpu);

    fprintf(f_freq, "%zu,%.4f,%.4f\n", input_size, time_seq, time_gpu);

    #pragma endregion

    #pragma region Huffman
    
    size_t total_bits = 0;
    for (size_t i = 0; i < input_len; i++) {
        total_bits += strlen(codes[(unsigned char)input[i]]);
    }
    char* encoded_bits_seq = malloc(total_bits + 1);
    if (!encoded_bits_seq) {
        fprintf(stderr, "Memory allocation failed for encoded bits!\n");
        free(input);
        return 1;
    }

	size_t bitlen_seq = 0;
	clock_t start_huff_seq = clock();
	encode_input_with_huffman(input, input_len, codes, encoded_bits_seq, &bitlen_seq);
	clock_t end_huff_seq = clock();
	double time_huff_seq = (double)(end_huff_seq - start_huff_seq) / CLOCKS_PER_SEC;

    //printf("Huffman encoding runtime: %.6f sec\n", time_huff_seq);

    #pragma endregion

    #pragma region Compression
    
    size_t compressed_bytes = (bitlen_seq + 7) / 8; 
    double compression_ratio = (double)compressed_bytes / (double)input_len;
    double saving = 100.0 - (compression_ratio * 100.0);

    // printf("Original size: %zu bytes\n", input_len);
    // printf("Compressed size: %zu bytes (%.0f bits)\n", compressed_bytes, (double)bitlen_seq);
    // printf("Compression ratio: %.2f%%\n", compression_ratio * 100.0);
    // printf("Space saved: %.2f%%\n", saving);

    fprintf(f_comp, "%zu,%.4f\n", input_size, compression_ratio);

    #pragma endregion

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

int main() {
    printf("Select mode:\n");
    printf("1. Manual mode\n");
    printf("2. Test mode\n");
    printf("Enter choice [1/2]: ");
    int mode_choice;
    if (scanf("%d", &mode_choice) != 1) {
        fprintf(stderr, "Invalid input.\n");
        return 1;
    }
    getchar();

    if (mode_choice == 1) {
        return manual(MAX_INPUT_SIZE);
    } 
    else if (mode_choice == 2) {
        FILE *f_gen  = fopen("generation_results.txt", "w");
        FILE *f_freq = fopen("byte_frequencies_results.txt", "w");
        FILE *f_comp = fopen("compression_results.txt", "w");
        if (!f_gen || !f_freq || !f_comp) {
            perror("Failed to open result files");
            return 1;
        }

        fprintf(f_gen,  "Size,SeqGenTime,OpenCLGenTime\n");
        fprintf(f_freq, "Size,SeqFreqTime,OpenCLFreqTime\n");
        fprintf(f_comp, "Size,CompressionRatio%%\n");

        for (size_t s = 100; s <= MAX_INPUT_SIZE; s *= 10) {
            printf("Current size under processing: %zu\n", s);
            test(s, f_gen, f_freq, f_comp);
        }

        fclose(f_gen);
        fclose(f_freq);
        fclose(f_comp);
        return 0;
    }
    else {
        fprintf(stderr, "Invalid mode.\n");
        return 1;
    }
}