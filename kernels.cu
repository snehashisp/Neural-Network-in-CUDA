#include<cuda.h>
#include<cstdlib>
#include<cstdio>

#ifndef KERNELS
#define KERNELS

__global__ void matmul_kernel(int k_dim,double *mat1,double *mat2,double *res) {

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
	res[ri * blockDim.x + rj] = sum;
}

__global__ void transpose_kernel(double *mat,double *tr_mat) {

	tr_mat[threadIdx.x*gridDim.x + blockIdx.x] = mat[blockIdx.x*blockDim.x + threadIdx.x];
	//printf("%lf ",tr_mat[threadIdx.x*gridDim.x + blockIdx.x]);
}

__global__ void hadamard_kernel(double *mat1,double *mat2,double *hmat) {

	hmat[blockIdx.x*blockDim.x + threadIdx.x] = 
		mat1[blockIdx.x*blockDim.x + threadIdx.x] * mat2[blockIdx.x*blockDim.x + threadIdx.x];
		
}
#endif