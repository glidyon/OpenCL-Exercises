#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#include <CL/cl.h>
#include <typeinfo>
#include <iostream>
#include <chrono>
#include "error_handling.h"
#include "matrices_ops.h"

int main(void)
{
	std::srand(0);
	size_t param_size, workgroup_size, vendor_name_size, program_size;
	int width_first, height_first, width_second, height_second;
	int* matrix_first, * matrix_second, * matrix_result;
	int total_size_first, total_size_second, total_size_result;
	cl_mem mem_obj1, mem_obj2, mem_obj3;
	cl_uint num_platforms, num_devices;
	cl_command_queue* command_queue;
	cl_platform_id* platforms_arr;
	cl_device_id* devices_arr;
	cl_context context;
	cl_program program;
	char* file_buffer;
	cl_kernel kernel;
	cl_int err_code;

	initialize_matrices(&matrix_first, &matrix_second, &matrix_result,
						&width_first, &height_first, &width_second,
						&height_second);
	total_size_first = width_first * height_first;
	total_size_second = width_second * height_second;
	total_size_result = height_first * width_second;

	// Lambda made to free all of the resources
	auto free_all =
		[&](cl_uint num_devices) { 
		if (err_code) {
			for(int i = 0; i < num_devices; i++)
				clReleaseDevice(devices_arr[i]);
			clReleaseMemObject(mem_obj1);
			clReleaseMemObject(mem_obj2);
			clReleaseMemObject(mem_obj3);
			clReleaseContext(context);
			clReleaseProgram(program);
			clReleaseKernel(kernel);
			free(platforms_arr);
			free(devices_arr);
			free(command_queue);
			free(matrix_first);
			free(matrix_second);
			free(matrix_result);
		}};
	file_buffer = read_kernel_file("matrix.cl", &program_size);

	clGetPlatformIDs(0, NULL, &num_platforms);
	platforms_arr = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	clGetPlatformIDs(num_platforms, platforms_arr, NULL);
	
	for (int i = 0; i < num_platforms; i++) {
		err_code = clGetDeviceIDs(platforms_arr[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		devices_arr = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
		command_queue = (cl_command_queue*)malloc(sizeof(cl_command_queue) * num_devices);
		err_code = clGetDeviceIDs(platforms_arr[i], CL_DEVICE_TYPE_ALL, num_devices, devices_arr, 0);
		if (err_code != CL_SUCCESS) {
			free_all(num_devices);
			return err_code;
		}

		context = clCreateContext(NULL, num_devices, devices_arr, NULL, NULL, &err_code);
		if (err_code != CL_SUCCESS) {
			free_all(num_devices);
			return err_code;
		}

		program = clCreateProgramWithSource(context, 1, (const char**)&file_buffer, &program_size, &err_code);
		if (err_code != CL_SUCCESS) {
			free_all(num_devices);
			return err_code;
		}

		err_code = clBuildProgram(program, num_devices, devices_arr, NULL, NULL, NULL);
		if (err_code != CL_SUCCESS) {
			free_all(num_devices);
			return err_code;
		}

		kernel = clCreateKernel(program, "matrix_mul", &err_code);
		if (err_code != CL_SUCCESS) {
			free_all(num_devices);
			return err_code;
		}

		mem_obj1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
								  sizeof(int) * total_size_first, matrix_first, &err_code);
		if (err_code != CL_SUCCESS) {
			free_all(num_devices);
			return err_code;
		}

		mem_obj2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
								  sizeof(int) * total_size_second, matrix_second, &err_code);
		if (err_code != CL_SUCCESS) {
			free_all(num_devices);
			return err_code;
		}

		mem_obj3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
								  sizeof(int) * total_size_result, matrix_result, &err_code);
		if (err_code != CL_SUCCESS) {
			free_all(num_devices);
			return err_code;
		}

		err_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_obj1);
		err_code |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_obj2);
		err_code |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_obj3);
		if (err_code != CL_SUCCESS) {
			free_all(num_devices);
			return err_code;
		}

		for (int j = 0; j < num_devices; j++) {
			command_queue[j] = clCreateCommandQueueWithProperties(context, devices_arr[j], 0, &err_code);

			if (err_code != CL_SUCCESS) {
				free_all(num_devices);
				return err_code;
			}
		}

		//clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &param_size);
		//clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, param_size, &workgroup_size, 0);
		//clGetDeviceInfo(devices[0], CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &vendor_name_size);
		//vendor = (char*)malloc(sizeof(char) * vendor_name_size);
		//clGetDeviceInfo(devices[0], CL_KERNEL_WORK_GROUP_SIZE, vendor_name_size, vendor, NULL);
		/*std::cout << vendor << std::endl;
		std::cout << workgroup_size << std::endl;*/
	}

	//clReleaseDevice(devices_arr[0]);
	//free(platforms_arr);
	//free(devices_arr);
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
