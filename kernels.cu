#include<cuda.h>
#include<cstdlib>
#include<cstdio>

#ifndef KERNELS
#define KERNELS

#define OP_NON 0
#define OP_ADD 1
#define OP_SUB 2
#define OP_MUL 3
#define OP_DIV 4

#define FN_SIGM 1 //sigmoid
#define FN_RELU 2 //relu
#define FN_DSIGM 3 //diffrentiation of sigmoid
#define FN_DRELU 4 //diffrentiation of relu

__device__ double operation(int op,double elem1,double elem2) {

	if (op == OP_ADD) return elem1 + elem2;
	else if(op == OP_SUB) return elem1 - elem2;
	else if(op == OP_MUL) return elem1 * elem2;
	else if(op == OP_DIV) return elem1 / (elem2 + 0.00000000001);
	else return elem1;
}

__device__ double function(int fn,double elem) {

	if (fn == FN_SIGM) return 1/(1 + exp(-1*elem));
	else if(fn == FN_RELU) return (elem > 0.00001 ? elem: 0);
	else if(fn == FN_DSIGM) {
		double sig = 1/(1 + exp(-1*elem));
		return sig*(1 - sig);
	}
	else if(fn == FN_DRELU) return(elem > 0.00001 ? 1 : 0);
	else return elem;

}

__global__ void gaxpy_kernel(int k_dim,double *mat1,double *mat2,double *res,double *matc = NULL, int c_row = 0, int c_col = 0) {

	int ri = blockIdx.x,rj = threadIdx.x;
	//printf("%d %d\n",blockDim.x,gridDim.x);
	double *p1 = mat1 + ri*k_dim, *p2 = mat2 + rj;
	double sum = 0;
	for(int k = 0; k < k_dim; k++) {
		//printf("%d %d %d %d %lf %lf \n",ri,k,k,rj,*p1,*p2);
		sum += (*p1) * (*p2);
		p1++;
		p2 += blockDim.x;
	}
	//printf("%d %d %lf\n",ri,rj,sum);
	if(matc == NULL) 
		res[ri * blockDim.x + rj] = sum;
	else {
		if(c_row == 1) res[ri * blockDim.x + rj] = sum + matc[rj];
		else if(c_col == 1) res[ri * blockDim.x + rj] = sum + matc[ri];
		else res[ri * blockDim.x + rj] = sum + matc[ri*blockDim.x + rj];
	}
}

__global__ void transpose_kernel(double *mat,double *tr_mat) {

	tr_mat[threadIdx.x*gridDim.x + blockIdx.x] = mat[blockIdx.x*blockDim.x + threadIdx.x];
}

__global__ void hadamard_kernel(double *mat1,double *mat2,double *hmat) {

	hmat[blockIdx.x*blockDim.x + threadIdx.x] = 
		mat1[blockIdx.x*blockDim.x + threadIdx.x] * mat2[blockIdx.x*blockDim.x + threadIdx.x];
		
}

__global__ void saxpy_kernel(double *mat1,double *mat2,double *res,double a = 1) {

	res[blockIdx.x*blockDim.x + threadIdx.x] = 
		a*mat1[blockIdx.x*blockDim.x + threadIdx.x] + mat2[blockIdx.x*blockDim.x + threadIdx.x];
}

__global__ void operate_kernel(double *mat1,double *res,double a,int op) {

	res[blockIdx.x*blockDim.x + threadIdx.x] = 
		operation(op,mat1[blockIdx.x*blockDim.x + threadIdx.x],a);
}

__global__ void function_kernel(double *mat1,double *res,int fn) {

	res[blockIdx.x*blockDim.x + threadIdx.x] = function(fn,mat1[blockIdx.x*blockDim.x + threadIdx.x]);
}

__global__ void reduction_kernel(double *mat1,double *res,int op,int dim,int axis) {

	double sum = 0;
	if(axis == 1) {
		for(int i = 0; i < dim; i++) 
			sum = operation(op,sum,mat1[i*blockDim.x + threadIdx.x]);
		res[threadIdx.x] = sum;
	}
	else {
		for(int i = 0; i < dim; i++)
			sum = operation(op,sum,mat1[threadIdx.x*dim + i]);
		if(axis == 3) {
			extern __shared__ double s[];
			__syncthreads();
			s[threadIdx.x] = sum;
			__syncthreads();
			if(threadIdx.x == 0) {
				sum = 0;
				for(int i = 0; i < blockDim.x; i++)
					sum = operation(op,sum,s[i]);
				res[0] = sum;
				//printf("%lf",sum);
			}
			__syncthreads();
			return;
		}
		res[threadIdx.x] = sum;
	}

}


#endif