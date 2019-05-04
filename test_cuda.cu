#include<cuda.h>
#include<cstdio>
#include<iostream>
#include<cstdlib>
#include"neuralnet.cu"


using namespace std;

int main(int argc,char *argv[]) {

	matrix *mat1 = loadFromFile(argv[1]), *mat2 = loadFromFile(argv[2]);
	//matrix *mat3 = loadFromFile(argv[3]);
	mat1 -> print();
	cout << endl;
	mat2 -> print();
	cout << endl;
	// mat3 -> print();
	// cout << endl;
	//matrix *matm = matrix_multi(mat1,mat2);
	//matm -> print();
	matrix *mat4 = new matrix;
	// cuda_matmul(mat1,mat2,mat4,true);
	// mat4 -> print();
	// cout << endl;
	// cuda_matmul(mat1,mat2,mat3,mat4,true);
	// mat4 -> print();

	//cuda_function(mat2,mat4,FN_DSIGM,true);
	mat4 -> init(5,4);
	gaussianInitializer(mat4,0,1);
	mat4 -> print();
	//cuda_function(mat2,mat4,FN_DSIGM,true);
	//mat4 -> print();
	cudaDeviceSynchronize();
	//while(1);
}	

