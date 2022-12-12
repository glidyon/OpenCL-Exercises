/*	Simple program made to test out matrix multiplication and
	transposition speeds using openCL */


#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#include <CL/cl.h>
#include <typeinfo>
#include <iostream>
#include <chrono>
#include <algorithm>
#include "error_handling.h"
#include "matrices_ops.h"


int main(void)
{
	std::srand(0);
	size_t param_size, workgroup_size, vendor_name_size, program_size_mult, program_size_transpose;
	int width_first, height_first, width_second, height_second;
	int total_size_first, total_size_second, total_size_result;
	int* matrix_first, * matrix_second, * matrix_result, * matrix_transpose_res;
	std::chrono::duration<double> elapsed_seconds_matrix_mul;
	std::chrono::duration<double> elapsed_seconds_matrix_trans;
	cl_uint num_platforms, num_devices;
	size_t global_work_size;
	size_t local_work_size;

	cl_mem mem_obj1 = NULL, mem_obj2 = NULL, mem_obj3 = NULL, mem_obj4 = NULL;
	char* file_buffer_mult = NULL, * file_buffer_transpose = NULL;
	cl_int err_code = CL_SUCCESS, err_code_finish = CL_SUCCESS;
	cl_program program = NULL, program_transpose = NULL;
	cl_kernel kernel = NULL, kernel_transpose = NULL;
	cl_command_queue* command_queue = NULL;
	cl_platform_id* platforms_arr = NULL;
	cl_device_id* devices_arr = NULL;
	cl_context context = NULL;

	initialize_matrices(&matrix_first, &matrix_second, &matrix_result,
						&matrix_transpose_res,
						&width_first, &height_first, &width_second,
						&height_second);
	total_size_first = width_first * height_first;
	total_size_second = width_second * height_second;
	total_size_result = height_first * width_second;

	file_buffer_mult = read_kernel_file("matrix.cl", &program_size_mult);
	file_buffer_transpose = read_kernel_file("transpose.cl", &program_size_transpose);

	clGetPlatformIDs(0, NULL, &num_platforms);
	platforms_arr = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	clGetPlatformIDs(num_platforms, platforms_arr, NULL);
	
	for (int i = 0; i < num_platforms; i++) {

		err_code = clGetDeviceIDs(platforms_arr[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		devices_arr = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
		command_queue = (cl_command_queue*)malloc(sizeof(cl_command_queue) * num_devices);
		err_code = clGetDeviceIDs(platforms_arr[i], CL_DEVICE_TYPE_ALL, num_devices, devices_arr, 0);
		if (err_code != CL_SUCCESS)
			goto release_devices;

		context = clCreateContext(NULL, num_devices, devices_arr, NULL, NULL, &err_code);
		if (err_code != CL_SUCCESS)
			goto release_context;

		program = clCreateProgramWithSource(context, 1, (const char**)&file_buffer_mult, &program_size_mult, &err_code);
		if (err_code != CL_SUCCESS)
			goto release_program_mul;

		program_transpose = clCreateProgramWithSource(context, 1, (const char**)&file_buffer_transpose,
													  &program_size_transpose, &err_code);
		if (err_code != CL_SUCCESS)
			goto release_program_trans;

		err_code = clBuildProgram(program_transpose, num_devices, devices_arr, NULL, NULL, NULL);
		if (err_code != CL_SUCCESS)
			goto release_program_trans;

		err_code = clBuildProgram(program, num_devices, devices_arr, NULL, NULL, NULL);
		if (err_code != CL_SUCCESS)
			goto release_program_trans;

		kernel = clCreateKernel(program, "matrix_mul", &err_code);
		if (err_code != CL_SUCCESS)
			goto release_kernel_mul;

		kernel_transpose = clCreateKernel(program_transpose, "transpose", &err_code);
		if (err_code != CL_SUCCESS)
			goto release_kernel_trans;

		mem_obj1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
								  sizeof(int) * total_size_first, matrix_first, &err_code);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj1;

		mem_obj2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
								  sizeof(int) * total_size_second, matrix_second, &err_code);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj2;

		mem_obj3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
								  sizeof(int) * total_size_result, matrix_result, &err_code);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj3;

		mem_obj4 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
			sizeof(int) * total_size_first, matrix_transpose_res, &err_code);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj4;

		err_code = clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), &mem_obj1);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj4;

		err_code = clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), &mem_obj4);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj4;

		err_code = clSetKernelArg(kernel_transpose, 2, sizeof(int), &height_first);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj4;

		err_code = clSetKernelArg(kernel_transpose, 3, sizeof(int), &width_first);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj4;

		err_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_obj1);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj4;

		err_code |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_obj2);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj4;

		err_code |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_obj3);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj4;

		err_code |= clSetKernelArg(kernel, 3, sizeof(int), &height_second);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj4;

		err_code |= clSetKernelArg(kernel, 4, sizeof(int), &width_first);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj4;

		err_code |= clSetKernelArg(kernel, 5, sizeof(int), &width_second);
		if (err_code != CL_SUCCESS)
			goto release_mem_obj4;

		global_work_size = height_first * width_first;
		for (int j = 0; j < num_devices; j++) {
			command_queue[j] = clCreateCommandQueueWithProperties(context, devices_arr[j], 0, &err_code);
			if (err_code != CL_SUCCESS)
				goto release_command_queue;

			clGetKernelWorkGroupInfo(kernel, devices_arr[j], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_work_size),
				&local_work_size, NULL);

			local_work_size = std::min(local_work_size, global_work_size);

			auto start = std::chrono::steady_clock::now();
			err_code = clEnqueueNDRangeKernel(command_queue[j], kernel, 1, NULL, &global_work_size,
				 				   &local_work_size, 0, NULL, NULL);
			err_code_finish = clFinish(command_queue[j]);
			auto end = std::chrono::steady_clock::now();
			if (err_code != CL_SUCCESS || err_code_finish != CL_SUCCESS)
				goto release_command_queue;

			elapsed_seconds_matrix_mul = end - start;

			err_code = clEnqueueReadBuffer(command_queue[j], mem_obj3, CL_TRUE,
										   0, (height_first*width_second) * sizeof(int),
										   matrix_result, 0, NULL, NULL);
			if (err_code != CL_SUCCESS)
				goto release_command_queue;

			start = std::chrono::steady_clock::now();
			err_code = clEnqueueNDRangeKernel(command_queue[j], kernel_transpose, 1, NULL, &global_work_size,
				&local_work_size, 0, NULL, NULL);
			err_code_finish = clFinish(command_queue[j]);
			end = std::chrono::steady_clock::now();
			if (err_code != CL_SUCCESS || err_code_finish != CL_SUCCESS)
				goto release_command_queue;

			elapsed_seconds_matrix_trans = end - start;

			err_code = clEnqueueReadBuffer(command_queue[j], mem_obj4, CL_TRUE,
				0, (height_first * width_first) * sizeof(int),
				matrix_transpose_res, 0, NULL, NULL);

			if (err_code != CL_SUCCESS)
				goto release_command_queue;

			std::cout << std::endl;
			std::cout << "Platform: " << i + 1 << " Device: " << j + 1 << std::endl;
			std::cout << "Elapsed time for matrix multiplication: " << 
						(elapsed_seconds_matrix_mul.count())*1000 << "ms" <<
						"\nElapsed seconds matrix transposition: " << 
						(elapsed_seconds_matrix_trans.count())*1000 << "ms" << std::endl;
			std::cout << std::endl;
		}

	}

release_command_queue:
	for (int i = 0; i < num_devices; i++)
		clReleaseCommandQueue(command_queue[i]);
release_mem_obj4:
	clReleaseMemObject(mem_obj4);
release_mem_obj3:
	clReleaseMemObject(mem_obj3);
release_mem_obj2:
	clReleaseMemObject(mem_obj2);
release_mem_obj1:
	clReleaseMemObject(mem_obj1);
release_kernel_trans:
	clReleaseKernel(kernel_transpose);
release_kernel_mul:
	clReleaseKernel(kernel);
release_program_trans:
	clReleaseProgram(program_transpose);
release_program_mul:
	clReleaseProgram(program);
release_context:
	clReleaseContext(context);
release_devices:
	for (int i = 0; i < num_devices; i++)
		clReleaseDevice(devices_arr[i]);
free_all:
	free(platforms_arr);
	free(devices_arr);
	free(command_queue);
	free(matrix_first);
	free(matrix_second);
	free(matrix_result);
	free(file_buffer_mult);
	free(matrix_transpose_res);
	free(file_buffer_transpose);
	if(err_code_finish != CL_SUCCESS)
		std::cerr << handle_errors(err_code_finish) << std::endl;
	if (err_code != CL_SUCCESS)
		std::cerr << handle_errors(err_code) << std::endl;
	return err_code;
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
