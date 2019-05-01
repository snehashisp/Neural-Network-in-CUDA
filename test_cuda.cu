#include<cuda.h>
#include<cstdio>
#include<iostream>
#include<cstdlib>
#include"functions.cu"




int main(int argc,char *argv[]) {

	matrix *mat1 = loadFromFile(argv[1]), *mat2 = loadFromFile(argv[2]);
	//mat1 -> print();
	mat2 -> print();
	//matrix *matm = matrix_multi(mat1,mat2);
	//matm -> print();
	matrix *mat3 = new matrix;
	//cuda_matmul(mat1,mat2,mat3,true);
	cuda_transpose(mat2,mat3,true);
	mat3 -> print();
	cudaDeviceSynchronize();
	//while(1);
}