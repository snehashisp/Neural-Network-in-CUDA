#include"kernels.cu"
#include"matrix.cpp"

#ifndef FUNCTIONS
#define FUNCTIONS


void cuda_matmul(matrix *mat1,matrix *mat2,matrix *mat3,bool update = false) {

	if(mat1 -> width != mat2 -> height) {
		printf("Multiply Dim mismatch %d %d\n",mat1 -> height,mat2 -> width);
		return;
	}

	if(!mat1 -> cudaMat) mat1 -> storeCuda();
	if(!mat2 -> cudaMat) mat2 -> storeCuda();
	if(!mat3 -> mat) mat3 -> init(mat1 -> height,mat2 -> width);
	if(!mat3 -> cudaMat) mat3 -> storeCuda();
	gaxpy_kernel<<<mat1->height,mat2->width>>>(mat1->width,mat1->cudaMat,mat2->cudaMat,mat3->cudaMat);
	if(update) mat3 -> updateCuda();
	mat3 -> isUpdated = update;

}

void cuda_transpose(matrix *mat1,matrix *tr_mat,bool update = false) {
	
	if(!mat1 -> cudaMat) mat1 -> storeCuda();
	if(!tr_mat-> mat) tr_mat -> init(mat1 -> width,mat1 -> height);
	if(!tr_mat -> cudaMat) tr_mat -> storeCuda();
	transpose_kernel<<<mat1->height,mat1->width>>>(mat1 -> cudaMat,tr_mat -> cudaMat);
	if(update) tr_mat -> updateCuda();
	tr_mat -> isUpdated = update;
}

void cuda_hadamard(matrix *mat1,matrix *mat2,matrix *mat3,bool update = false) {

	if(mat1 -> width != mat2 -> width || mat1->height != mat2->height) {
		printf("Hadamard Dim mismatch %d %d %d %d\n",mat1 -> height,mat1 -> width,
			mat2->height,mat2->width);
		return;
	}

	if(!mat1 -> cudaMat) mat1 -> storeCuda();
	if(!mat2 -> cudaMat) mat2 -> storeCuda();
	if(!mat3 -> mat) mat3 -> init(mat1 -> height,mat2 -> width);
	if(!mat3 -> cudaMat) mat3 -> storeCuda();
	hadamard_kernel<<<mat1->height,mat1->width>>>
		(mat1->cudaMat,mat2->cudaMat,mat3->cudaMat);
	if(update) mat3 -> updateCuda();
	mat3 -> isUpdated = update;


}

//matrix multiplication with a bias 
void cuda_matmul(matrix *mat1,matrix *mat2,matrix *bias,matrix *mat3,bool update = false) {

	if(mat1 -> width != mat2 -> height) {
		printf("Multiply Dim mismatch %d %d\n",mat1 -> height,mat2 -> width);
		return;
	}
	if(bias -> mat && (bias -> height == mat1 -> height && bias -> width == mat2 -> width) || 
		(bias -> height == mat1 -> height && bias -> width == 1) ||
		(bias -> width == mat2 -> width && bias -> height == 1)) {
		if(!mat1 -> cudaMat) mat1 -> storeCuda();
		if(!mat2 -> cudaMat) mat2 -> storeCuda();
		if(!bias -> cudaMat) bias -> storeCuda();
		if(!mat3 -> mat) mat3 -> init(mat1 -> height,mat2 -> width);
		if(!mat3 -> cudaMat) mat3 -> storeCuda();
		gaxpy_kernel<<<mat1->height,mat2->width>>>
			(mat1->width,mat1->cudaMat,mat2->cudaMat,mat3->cudaMat,
				bias->cudaMat,bias->height,bias->width);
		if(update) mat3 -> updateCuda();
		mat3 -> isUpdated = update;
	}
	else 
		printf("Dimension mismatch bias %d %d \n ",bias -> height,bias -> width);

}

void cuda_vecMSE(matrix *mat1,matrix *mat2,matrix *mat3,bool update = false) {

	if(mat1->height != mat2 -> height && mat1 -> width != mat2 -> width) {
		printf("MSE Dim mismatch %d %d %d %d\n",mat1 -> height,mat1 -> width,
			mat2->height,mat2->width);
	}
	if(!mat1 -> cudaMat) mat1 -> storeCuda();
	if(!mat2 -> cudaMat) mat2 -> storeCuda();
	if(!mat3 -> mat) mat3 -> init(mat1 -> height,mat2 -> width);
	if(!mat3 -> cudaMat) mat3 -> storeCuda();
	saxpy_kernel<<<mat1->height,mat1->width>>>
		(mat1->cudaMat,mat2->cudaMat,mat3->cudaMat,1);
	hadamard_kernel<<<mat3->height,mat1->width>>>
		(mat3->cudaMat,mat3->cudaMat,mat3->cudaMat);
	if(update) mat3 -> updateCuda();
	mat3 -> isUpdated = update;

}

void cuda_vecDiff(matrix *mat1,matrix *mat2,matrix *mat3,bool update = false) {

	if(mat1->height != mat2 -> height && mat1 -> width != mat2 -> width) {
		printf("MSE Dim mismatch %d %d %d %d\n",mat1 -> height,mat1 -> width,
			mat2->height,mat2->width);
	}
	if(!mat1 -> cudaMat) mat1 -> storeCuda();
	if(!mat2 -> cudaMat) mat2 -> storeCuda();
	if(!mat3 -> mat) mat3 -> init(mat1 -> height,mat1 -> width);
	if(!mat3 -> cudaMat) mat3 -> storeCuda();

	operate_kernel<<<mat3->height,mat3->width>>>
		(mat2->cudaMat,mat3->cudaMat,-1,OP_MUL);

	saxpy_kernel<<<mat3->height,mat3->width>>>
		(mat1->cudaMat,mat3->cudaMat,mat3->cudaMat,1);
	if(update) mat3 -> updateCuda();
	mat3 -> isUpdated = update;

}

void cuda_function(matrix *mat1,matrix *mat2,int fn,bool update = false) {

	if(!mat1 -> cudaMat) mat1 -> storeCuda();
	if(!mat2 -> mat) mat2 -> init(mat1 -> height, mat1 -> width);
	if(!mat2 -> cudaMat) mat2 -> storeCuda();
	function_kernel<<<mat1 -> height,mat1 -> width>>>
		(mat1 -> cudaMat,mat2 -> cudaMat,fn);
	if(update) mat2 -> updateCuda();
	mat2 -> isUpdated = update;

}

void cuda_operation(matrix *mat1,matrix *mat2,double a,int op,bool update = false) {

	if(!mat1 -> cudaMat) mat1 -> storeCuda();
	if(!mat2 -> mat) mat2 -> init(mat1 -> height, mat1 -> width);
	if(!mat2 -> cudaMat) mat2 -> storeCuda();
	operate_kernel<<<mat1 -> height,mat1 -> width>>>
		(mat1 -> cudaMat,mat2 -> cudaMat,a,op);
	if(update) mat2 -> updateCuda();
	mat2 -> isUpdated = update;

}

void cuda_reduce(matrix *mat1,matrix *res,int op,int axis,bool update = false) {

	if(!mat1 -> cudaMat) mat1 -> storeCuda();
	if(axis == 3) {
		if(!res -> mat) res -> init(1,1);
		else if (res -> height != 1 || res -> width != 1) {
			printf("Reduction result not match dimension %d %d",res -> height,res ->width);
			return;
		}
		if(!res -> cudaMat) res -> storeCuda();
		reduction_kernel<<<1,mat1->height,mat1->height>>>
			(mat1 -> cudaMat,res -> cudaMat,op,mat1->width,axis);
	}
	else if(axis == 2) {
		if(!res -> mat) res -> init(mat1 -> height,1);
		else if(res -> height != mat1 -> height || res -> width != 1) {
			printf("Reduction result not match dimension %d %d",res -> height,res ->width);
			return;	 
		}
		if(!res -> cudaMat) res -> storeCuda();
		reduction_kernel<<<1,mat1->height>>>
			(mat1 -> cudaMat,res -> cudaMat,op,mat1->width,axis);
	}
	else if(axis == 1) {
		if(!res -> mat) res -> init(1,mat1 -> width);
		else if(res -> height != 1 || res -> width != mat1 -> width) {
			printf("Reduction result not match dimension %d %d",res -> height,res ->width);
			return;	 
		}
		if(!res -> cudaMat) res -> storeCuda();
		reduction_kernel<<<1,mat1->width>>>
			(mat1 -> cudaMat,res -> cudaMat,op,mat1->height,axis);
	}
	if(update) res -> updateCuda();
	res -> isUpdated = update;
}

#endif 