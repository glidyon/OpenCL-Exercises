#include <iostream>
#include "matrices_ops.h"

void initialize_matrices(int** matrix_first, int** matrix_second, int** matrix_result,
						 int **matrix_transpose_res, int* width_first,
						 int* height_first, int* width_second,
						 int* height_second)
{
	std::cout << "Enter width of the first matrix: " << std::endl;
	std::cin >> *width_first;
	std::cout << "Enter height of the first matrix: " << std::endl;
	std::cin >> *height_first;

	*matrix_first = (int*)malloc(sizeof(int) * (*width_first * *height_first));
	*matrix_transpose_res = (int*)malloc(sizeof(int) * (*width_first * *height_first));;

	std::cout << "Enter width of the second matrix: " << std::endl;
	std::cin >> *width_second;
	std::cout << "Enter height of the second matrix: " << std::endl;
	std::cin >> *height_second;

	*matrix_second = (int*)malloc(sizeof(int) * (*width_second * *height_second));
	*matrix_result = (int*)malloc(sizeof(int) * (*width_second * *height_first));

	randomize_matrix(*matrix_first, height_first, width_first);
	//for (int i = 0; i < *height_first; i++) {
	//	for (int j = 0; j < *width_first; j++) {
	//		std::cout << (*matrix_first)[j + i * (*width_first)] << " ";
	//	}
	//	std::cout << std::endl;
	//}
	randomize_matrix(*matrix_second, height_second, width_second);
}

void randomize_matrix(int* matrix, const int* height, const int* width)
{
	for (int i = 0; i < *height; i++) {
		for (int j = 0; j < *width; j++) {
			matrix[j + i * (*width)] = std::rand() % 5;
		}
	}
}
