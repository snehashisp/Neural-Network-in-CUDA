#include"kernels.cu"
#include"matrix.cpp"

#ifndef FUNCTIONS
#define FUNCTIONS

void cuda_matmul(matrix *mat1,matrix *mat2,matrix *mat3,bool update = false) {

	if(!mat1 -> cudaMat) mat1 -> storeCuda();
	if(!mat2 -> cudaMat) mat2 -> storeCuda();
	if(!mat3 -> mat) mat3 -> init(mat1 -> height,mat2 -> width);
	if(!mat3 -> cudaMat) mat3 -> storeCuda();
	matmul_kernel<<<mat1->height,mat2->width>>>(mat1->width,mat1->cudaMat,mat2->cudaMat,mat3->cudaMat);
	if(update) mat3 -> updateCuda();
}

void cuda_transpose(matrix *mat1,matrix *tr_mat,bool update = false) {
	
	if(!mat1 -> cudaMat) mat1 -> storeCuda();
	if(!tr_mat-> mat) tr_mat -> init(mat1 -> width,mat1 -> height);
	if(!tr_mat -> cudaMat) tr_mat -> storeCuda();
	transpose_kernel<<<mat1->height,mat1->width>>>(mat1 -> cudaMat,tr_mat -> cudaMat);
	if(update) tr_mat -> updateCuda();
}

void cuda_hadamard(matrix *mat1,matrix *mat2,matrix *mat3,bool update = false) {

	if(!mat1 -> cudaMat) mat1 -> storeCuda();
	if(!mat2 -> cudaMat) mat2 -> storeCuda();
	if(!mat3 -> mat) mat3 -> init(mat1 -> height,mat2 -> width);
	if(!mat3 -> cudaMat) mat3 -> storeCuda();
	hadamard_kernel<<<mat1->height,mat2->width>>>(mat1->width,mat1->cudaMat,mat2->cudaMat,mat3->cudaMat);
	if(update) mat3 -> updateCuda();

}

#endif 