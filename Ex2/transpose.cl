__kernel void transpose(__global int *matrix, __global int *matrix_res,
					 int height, int width) {
	int i = get_global_id(0) / width;
	int j = get_global_id(0) % width;

	matrix_res[j*height + i] = matrix[i*width + j];
}
