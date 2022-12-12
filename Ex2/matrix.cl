__kernel void matrix_mul(__global int *matrix1, __global int *matrix2, __global int *matrix_res,
					 int height_second, int width_first, int width_second) {
	int i = get_global_id(0) / width_first;
	int j = get_global_id(0) % width_second;

	int sum = 0;
	for(int k = 0; k < height_second; k++)
		sum += matrix1[k + i*width_first] * matrix2[j + k*width_second];

	matrix_res[i*width_second + j] = sum;
}
