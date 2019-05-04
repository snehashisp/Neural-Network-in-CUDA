#include<cuda.h>
#include<cstdio>
#include<iostream>
#include<cstdlib>
#include"neuralnet.cu"



using namespace std;

// int main(int argc,char *argv[]) {

// 	matrix *mat1 = loadFromFile(argv[1]), *mat2 = loadFromFile(argv[2]);
// 	//matrix *mat3 = loadFromFile(argv[3]);
// 	mat1 -> print();
// 	cout << endl;
// 	mat2 -> print();
// 	cout << endl;

// 	// cout << endl;
// 	//matrix *matm = matrix_multi(mat1,mat2);
// 	//matm -> print();
// 	matrix *mat4 = new matrix;
// 	mat4 -> init(6,6);
// 	cuda_transpose(mat1,mat4,true);
// 	mat4 -> height = mat1 -> width;
// 	mat4 -> width = mat1 -> height;
// 	mat4 -> print();
// 	cuda_transpose(mat2,mat4,true);
// 	mat4 -> height = mat2 -> width;
// 	mat4 -> width = mat2 -> height;
// 	mat4 -> print();
// 	matrix *mat5 = new matrix;
// 	cuda_matmul(mat2,mat4,mat5,true);
// 	mat5 -> print();
// 	// cout << endl;
// 	// cuda_matmul(mat1,mat2,mat3,mat4,true);
// 	// mat4 -> print();

// 	// //cuda_function(mat2,mat4,FN_DSIGM,true);
// 	// mat4 -> init(5,4);
// 	// gaussianInitializer(mat4,0,1);
// 	// mat4 -> print();
// 	// //cuda_function(mat2,mat4,FN_DSIGM,true);
// 	// //mat4 -> print();
// 	// cudaDeviceSynchronize();
// 	// //while(1);
// }	

// int main() {

// 	neural_network nn;
// 	std :: vector<int> weights = {2,1,2};
// 	nn.init(weights,2);
// 	nn.print_weights();
// 	nn.print_biases();
// 	nn.print_outputs();
// 	nn.print_activations();

// 	matrix *inp = new matrix;
// 	inp->init(2,2);
// 	gaussianInitializer(inp,0,1);
// 	inp -> print();
// 	nn.forward(inp,true);
// 	nn.print_outputs();
// 	nn.print_activations();
// 	nn.MSELossDiff(inp,true);
// 	printf("Loss %lf\n",nn.returnSingleLoss());
// 	nn.printLossMat();
// 	printf("\n");

// 	gaussianInitializer(inp,0,1);
// 	inp -> print();
// 	nn.forward(inp,true);
// 	nn.print_outputs();
// 	nn.print_activations();
// 	nn.MSELossDiff(inp,true);
// 	printf("Loss %lf",nn.returnSingleLoss());
// 	nn.printLossMat();
// }

int main() {

	matrix *mat1 = new matrix, *mat2 = new matrix;
	readCSV(mat1,mat2,60000,785,false);

	matrix *mat3 = mat1 -> rowSlice(9,10);
	mat2 -> print_shape();
	mat3 -> print();
}