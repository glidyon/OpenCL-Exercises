__kernel void rotate_img(__global unsigned char *src_img, __global unsigned char *res_img,
					 int height, int width) {
	int i = get_global_id(0) / width;
	int j = get_global_id(0) % width;

	for(int k = 0; k < 3; k++)
		res_img[(j * width * 3) + ((height - i - 1) * 3) + k] = src_img[(i * width * 3) + (j*3) + k];
}
