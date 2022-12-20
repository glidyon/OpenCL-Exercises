/* Simple openCL program for rotating a square NxN image
	by 90 degrees
*/


#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#include <CL/cl.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include "error_handling.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main(void)
{
	int width, height, bpp;
	uint8_t* rgb_image = stbi_load("image.png", &width, &height, &bpp, 3);
	size_t total_size = width * height * 3;
	uint8_t* dest_data = (uint8_t*)malloc(sizeof(uint8_t)* total_size);

	cl_mem mem_obj1 = NULL, mem_obj2 = NULL;
	std::chrono::duration<double> elapsed_seconds_rotate;
	cl_command_queue command_queue = NULL;
	cl_platform_id* platforms_arr = NULL;
	size_t program_size, local_work_size;
	cl_uint num_platforms, num_devices;
	cl_device_id* devices_arr = NULL;
	cl_int err_code = CL_SUCCESS;
	cl_program program = NULL;
	cl_context context = NULL;
	char* file_buffer = NULL;
	cl_kernel kernel = NULL;
	char* program_log;
	size_t log_size;
	size_t dims = width * height;

	file_buffer = read_kernel_file("rotate.cl", &program_size);

	clGetPlatformIDs(0, NULL, &num_platforms);
	platforms_arr = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	clGetPlatformIDs(num_platforms, platforms_arr, NULL);

	err_code = clGetDeviceIDs(platforms_arr[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	devices_arr = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
	err_code = clGetDeviceIDs(platforms_arr[0], CL_DEVICE_TYPE_GPU, num_devices, devices_arr, 0);

	context = clCreateContext(NULL, num_devices, devices_arr, NULL, NULL, &err_code);
	program = clCreateProgramWithSource(context, 1, (const char**)&file_buffer, &program_size, &err_code);
	err_code = clBuildProgram(program, num_devices, devices_arr, NULL, NULL, NULL);

	clGetProgramBuildInfo(program, devices_arr[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	program_log = (char*)calloc(log_size + 1, sizeof(char));
	clGetProgramBuildInfo(program, devices_arr[0], CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);

	kernel = clCreateKernel(program, "rotate_img", &err_code);

	mem_obj1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				sizeof(uint8_t) * total_size, rgb_image, &err_code);
	mem_obj2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		sizeof(uint8_t) * total_size, dest_data, &err_code);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_obj1);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_obj2);
	clSetKernelArg(kernel, 2, sizeof(int), &height);
	clSetKernelArg(kernel, 3, sizeof(int), &width);

	command_queue = clCreateCommandQueueWithProperties(context, devices_arr[0], 0, &err_code);

	clGetKernelWorkGroupInfo(kernel, devices_arr[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_work_size),
		&local_work_size, NULL);

	local_work_size = std::min(local_work_size, dims);

	auto start = std::chrono::steady_clock::now();
	err_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &dims,
		&local_work_size, 0, NULL, NULL);
	clFinish(command_queue);
	auto end = std::chrono::steady_clock::now();

	elapsed_seconds_rotate = end - start;

	err_code = clEnqueueReadBuffer(command_queue, mem_obj2, CL_TRUE,
		0, total_size * sizeof(uint8_t),
		dest_data, 0, NULL, NULL);

	std::cout << "Elapsed time image rotation for 90 degrees: " <<
		(elapsed_seconds_rotate.count()) * 1000 << "ms" << std::endl;

	stbi_write_png("image2.png", width, height, 3, dest_data, width * 3);

	clReleaseKernel(kernel);
	clReleaseContext(context);
	clReleaseCommandQueue(command_queue);
	for(int i = 0; i < num_devices; i++)
		clReleaseDevice(devices_arr[i]);
	clReleaseMemObject(mem_obj1);
	clReleaseMemObject(mem_obj2);
	stbi_image_free(rgb_image);
	stbi_image_free(dest_data);
	free(file_buffer);
	free(platforms_arr);
	free(devices_arr);
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
