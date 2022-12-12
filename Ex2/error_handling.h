#pragma once
#include <CL/Cl.h>

char* read_kernel_file(const char* filename, size_t* program_size);

const char* handle_errors(cl_int err_code);