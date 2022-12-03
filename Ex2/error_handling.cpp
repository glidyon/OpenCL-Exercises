#include <iostream>
#include <CL/cl.h>
#include "error_handling.h"

// TODO: finish error handling


void handle_get_dev_ids_err(cl_int err_code) {
	switch (err_code) {
		case CL_INVALID_PLATFORM:
			std::cerr << "Provided platform is not a valid platform" << std::endl;
			break;
		case CL_INVALID_DEVICE_TYPE:
			std::cerr << "Provided device_type value is not a valid value" << std::endl;
			break;
		case CL_INVALID_VALUE:
			std::cerr << "num_entries value is equal to zero and device_type is "
						 "not NULL or both num_devices and device_type are NULL" << std::endl;
			break;
		case CL_DEVICE_NOT_FOUND:
			std::cerr << "No OpenCL devices that matched device_type parameter matches" << std::endl;
			break;
		default:
			std::cerr << "Undefined error" << std::endl;
	}
}

void handle_create_ctxt_err(cl_int err_code) {
	switch (err_code) {
		case CL_INVALID_PLATFORM:
			std::cerr << "Provided properties are NULL and no platform could be selected or platform value "
						 "specified in properties is not a valid platform" << std::endl;
			break;
		case CL_INVALID_VALUE:
			std::cerr << "Context property name is not a supported property name or devices variable is NULL, or num_devices" << std::endl;
			break;
		default:
			std::cerr << "Undefined error" << std::endl;
	}
}