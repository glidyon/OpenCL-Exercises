#pragma once
#include <CL/Cl.h>

char* read_kernel_file(const char* filename, size_t* program_size);

void handle_create_ctxt_err(cl_int err_code);