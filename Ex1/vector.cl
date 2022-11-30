__kernel void vector(__global int *vec1, __global int *vec2, __global int *vec3) {
	int i = get_global_id(0);
	vec3[i] = vec1[i] + vec2[i];
}