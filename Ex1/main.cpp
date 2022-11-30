/*
	Simple program made as an exercises to calculate sum of two
    vectors and putting the result into third vector using openCL API
	Be aware that this has no error-handling!
*/

#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#include <CL/cl.h>
#include <typeinfo>
#include <iostream>
#include <chrono>

#define VEC_SIZE 1024

char *read_kernel_file(const char *filename, size_t *program_size);

int main()
{
	const cl_platform_info plat_param_arr[3] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
												 CL_PLATFORM_VERSION };
	const cl_device_info device_param_arr[5] = { CL_DEVICE_NAME, CL_DEVICE_VENDOR,
												 CL_DRIVER_VERSION, CL_DEVICE_VERSION,
												 CL_DEVICE_OPENCL_C_VERSION };
	int vec1[VEC_SIZE], vec2[VEC_SIZE];
	int** vec3;
	cl_platform_id* platforms;
	cl_uint num_platforms;
	char* platform_info;
	char* device_info;
	size_t platform_info_size, device_info_size;
	cl_uint* num_devices;
	cl_device_id** devices;
	int chosen_plat, chosen_dev;
	cl_context* context_ids;
	cl_command_queue** command_queues;
	cl_int error_code;
	cl_mem* vec1_mem_obj, * vec2_mem_obj, * vec3_mem_obj;
	cl_program *programs;
	char* source_file;
	size_t program_size;
	cl_kernel* kernels;
	size_t global_work_size = VEC_SIZE, local_work_size = 1;

	for (int i = 0; i < VEC_SIZE; i++) {
		vec1[i] = i + 1;
		vec2[i] = i + 1;
	}

	source_file = read_kernel_file("vector.cl", &program_size);

	clGetPlatformIDs(0, NULL, &num_platforms);

	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	num_devices = (cl_uint*)malloc(sizeof(cl_uint) * num_platforms);
	devices = (cl_device_id**)malloc(sizeof(cl_device_id*) * num_platforms);
	context_ids = (cl_context*)malloc(sizeof(cl_context) * num_platforms);
	command_queues = (cl_command_queue**)malloc(sizeof(cl_command_queue*) * num_platforms);
	vec3 = (int**)malloc(sizeof(int*) * num_platforms);
	vec1_mem_obj = (cl_mem*)malloc(sizeof(cl_mem) * num_platforms);
	vec2_mem_obj = (cl_mem*)malloc(sizeof(cl_mem) * num_platforms);
	vec3_mem_obj = (cl_mem*)malloc(sizeof(cl_mem) * num_platforms);
	programs = (cl_program*)malloc(sizeof(cl_program) * num_platforms);
	kernels = (cl_kernel*)malloc(sizeof(cl_kernel) * num_platforms);
	// Get all platform IDs on current host machine
	clGetPlatformIDs(num_platforms, platforms, NULL);

	std::cout << "Number of available platforms: " << num_platforms << std::endl;
	std::cout << std::endl;

	for (unsigned int i = 0; i < num_platforms; i++) {
		for (int j = 0; j < 3; j++) {
			// Print informations specified in plat_param_arr array
			clGetPlatformInfo(platforms[i], plat_param_arr[j], 0, NULL, &platform_info_size);
			platform_info = (char*)malloc(sizeof(char) * platform_info_size);
			clGetPlatformInfo(platforms[i], plat_param_arr[j], platform_info_size,
							  platform_info, NULL);
			std::cout << platform_info << std::endl;
			free(platform_info);
		}
		
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices[i]);
		devices[i] = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices[i]);
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices[i], devices[i], NULL);
		context_ids[i] = clCreateContext(NULL, num_devices[i], devices[i], NULL, NULL, &error_code);

		vec3[i] = (int*)malloc(sizeof(int) * VEC_SIZE);
		command_queues[i] = (cl_command_queue*)malloc(sizeof(cl_command_queue) * num_devices[i]);
		vec1_mem_obj[i] = clCreateBuffer(context_ids[i],
						  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						  sizeof(int) * VEC_SIZE, vec1, &error_code);
		vec2_mem_obj[i] = clCreateBuffer(context_ids[i],
						  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						  sizeof(int) * VEC_SIZE, vec2, &error_code);
		vec3_mem_obj[i] = clCreateBuffer(context_ids[i],
						  CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
						  sizeof(int) * VEC_SIZE, vec3, &error_code);

		programs[i] = clCreateProgramWithSource(context_ids[i], 1, (const char **)&source_file, &program_size, &error_code);

		for (int j = 0; j < num_devices[i]; j++) {
			command_queues[i][j] = clCreateCommandQueueWithProperties(context_ids[i], devices[i][j], 0, &error_code);
			clEnqueueWriteBuffer(command_queues[i][j], vec1_mem_obj[i],
								 CL_TRUE, 0, VEC_SIZE * sizeof(int), vec1, 0, NULL, NULL);
			clEnqueueWriteBuffer(command_queues[i][j], vec2_mem_obj[i],
								 CL_TRUE, 0, VEC_SIZE * sizeof(int), vec2, 0, NULL, NULL);
		}

		clBuildProgram(programs[i], 1, devices[i], NULL, NULL, NULL);
		kernels[i] = clCreateKernel(programs[i], "vector", &error_code);
		clSetKernelArg(kernels[i], 0, sizeof(cl_mem), &vec1_mem_obj[i]);
		clSetKernelArg(kernels[i], 1, sizeof(cl_mem), &vec2_mem_obj[i]);
		clSetKernelArg(kernels[i], 2, sizeof(cl_mem), &vec3_mem_obj[i]);

		std::cout << "Number of the devices available on the platform: " << num_devices[i] << std::endl;
		std::cout << std::endl;
	}

	// Separate nested loop for time measurements
	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < num_platforms; i++) {
		for (int j = 0; j < num_devices[i]; j++) {
			clEnqueueNDRangeKernel(command_queues[i][j], kernels[i], 1,
								   NULL, &global_work_size, &local_work_size,
								   0, NULL, NULL);
			clFinish(command_queues[i][j]);
		}
	}
	auto end = std::chrono::steady_clock::now();

	for (int i = 0; i < num_platforms; i++) {
		for (int j = 0; j < num_devices[i]; j++) {
			clEnqueueReadBuffer(command_queues[i][j], vec3_mem_obj[i],
								CL_TRUE, 0, VEC_SIZE * sizeof(int),
								vec3[i], 0, NULL, NULL);
		}
	}

	std::chrono::duration<double> elapsed_seconds = end - start;
	auto elapsed_miliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds).count();
	std::cout << "Calculated time of vector addition for all platforms and devices: " << elapsed_seconds.count()
			  << "s || " << elapsed_miliseconds << "ms" << std::endl;
	
	std::cout << "Choose platform number: " << std::endl;
	std::cin >> chosen_plat;

	std::cout << "Devices informations on chosen platform: " << std::endl;
	for (unsigned int i = 0; i < num_devices[chosen_plat - 1]; i++) {
		for (int j = 0; j < 5; j++) {
			clGetDeviceInfo(devices[chosen_plat - 1][i], device_param_arr[j], 0, NULL, &device_info_size);
			device_info = (char*)malloc(sizeof(char) * device_info_size);
			clGetDeviceInfo(devices[chosen_plat - 1][i], device_param_arr[j], device_info_size, device_info, NULL);
			std::cout << device_info << std::endl;

			free(device_info);
		}
		
		std::cout << std::endl;
	}

	std::cout << "Choose device number to calculate sum of vectors on it: " << std::endl;
	std::cin >> chosen_dev;

	start = std::chrono::steady_clock::now();
	clEnqueueNDRangeKernel(command_queues[chosen_plat- 1][chosen_dev - 1],
						   kernels[chosen_plat - 1], 1, NULL,
						   &global_work_size, &local_work_size, 0, NULL, NULL);
	clFinish(command_queues[chosen_plat - 1][chosen_dev - 1]);
	end = std::chrono::steady_clock::now();

	clEnqueueReadBuffer(command_queues[chosen_plat - 1][chosen_dev - 1],
						vec3_mem_obj[chosen_plat - 1], CL_TRUE, 0, VEC_SIZE * sizeof(int),
						vec3[chosen_plat - 1], 0, NULL, NULL);

	elapsed_seconds = end - start;
	elapsed_miliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds).count();
	std::cout << "Calculated time of vector addition for chosen platform and device: " << elapsed_seconds.count()
		<< "s || " << elapsed_miliseconds << "ms" << std::endl;

	// Decrement reference count and release all objects attached and also
	// free memory allocated from heap
	for (int i = 0; i < num_platforms; i++) {
		clReleaseKernel(kernels[i]);
		clReleaseProgram(programs[i]);
		clReleaseMemObject(vec1_mem_obj[i]);
		clReleaseMemObject(vec2_mem_obj[i]);
		clReleaseMemObject(vec3_mem_obj[i]);
		clReleaseContext(context_ids[i]);

		for (int j = 0; j < num_devices[i]; j++) {
			clReleaseCommandQueue(command_queues[i][j]);
			clReleaseDevice(devices[i][j]);
		}

		free(command_queues[i]);
		free(devices[i]);
		free(vec3[i]);
	}
	
	free(kernels);
	free(programs);
	free(vec1_mem_obj);
	free(vec2_mem_obj);
	free(vec3_mem_obj);
	free(vec3);
	free(context_ids);
	free(devices);
	free(command_queues);
	free(platforms);
	free(num_devices);
	free(source_file);
}

char * read_kernel_file(const char *filename, size_t *program_size)
{
	FILE* program_handle;
	char* program_buffer;

	fopen_s(&program_handle, filename, "rb");
	fseek(program_handle, 0, SEEK_END);
	*program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(sizeof(char) * (* program_size + 1));
	program_buffer[*program_size] = '\0';
	fread(program_buffer, sizeof(char), *program_size, program_handle);
	
	fclose(program_handle);

	return program_buffer;
}
