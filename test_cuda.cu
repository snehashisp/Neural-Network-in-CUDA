#include<cuda.h>
#include<cstdio>
#include<iostream>
#include<cstdlib>
#include"neuralnet.cu"



using namespace std;

// int main(int argc,char *argv[]) {

// 	// matrix *mat1 = loadFromFile(argv[1]), *mat2 = loadFromFile(argv[2]);
// 	// //matrix *mat3 = loadFromFile(argv[3]);
// 	// mat1 -> print();
// 	// cout << endl;
// 	// mat2 -> print();
// 	// cout << endl;

// 	// cout << endl;
// 	//matrix *matm = matrix_multi(mat1,mat2);
// 	//matm -> print();
// 	matrix *mat4 = new matrix;
// 	mat4 -> init(100,784);
// 	gaussianInitializer(mat4,0,1);
// 	mat4 -> print();
// 	// cuda_transpose(mat1,mat4,true);
// 	// mat4 -> height = mat1 -> width;
// 	// mat4 -> width = mat1 -> height;
// 	// mat4 -> print();
// 	// cuda_transpose(mat2,mat4,true);
// 	// mat4 -> height = mat2 -> width;
// 	// mat4 -> width = mat2 -> height;
// 	// mat4 -> print();
// 	matrix *mat5 = new matrix;
// 	cuda_reduce(mat4,mat5,OP_ADD,1,true);
// 	printf("\nF");
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

//int main() {

// 	neural_network nn;
// 	std :: vector<int> weights = {784,10,784};
// 	nn.init(weights,100);
// 	//nn.print_weights();
// 	//nn.print_biases();
// 	//nn.print_outputs();
// 	//nn.print_activations();

// 	matrix *inp = new matrix;
// 	inp->init(100,784);
// 	gaussianInitializer(inp,5,1);
// 	//inp -> print();
// 	int k = 10;
// 	while(k--) {
// 		nn.forward(inp);
// 		//nn.print_outputs();
// 		//nn.print_activations();
// 		nn.MSELossDiff(inp);
// 		printf("Loss %lf\n",nn.returnSingleLoss());
// 		nn.backprop(inp,0.01);
// 	}
// 	//nn.print_weights();
// 	//nn.print_biases();
// 	// //nn.printLossMat();
// 	// printf("\n");

// }

int main() {

	matrix *mat1 = new matrix, *mat2 = new matrix;
	readCSV(mat1,mat2,60000,785,false);
	cuda_operation(mat1,mat1,255,OP_DIV,true);
	neural_network nn;
	std :: vector<int> weights = {784,50,784};
	nn.init(weights,1000,false);
	nn.trainModel(mat1,mat1,100,0.01,2);
	matrix *emat = nn.encode(mat1);
	storeAsCSV(emat,"results.csv");
	// matrix *mat3 = new matrix, *mat4 = new matrix;
	// mat1 -> rowSlice(mat3,0,100);
	// neural_network nn;
	// std :: vector<int> weights = {784,200,784};
	// nn.init(weights,1000,false,0,1);
	// int k = 10;
	// while(k--) {
	// 	nn.forward(mat3);
	// 	//nn.print_activations();
	// 	nn.MSELossDiff(mat3,true);
	// 	printf("Loss %6.10lf\n",nn.returnSingleLoss());
	// 	//nn.printLossMat();
	// 	//break;
	// 	nn.backprop(mat3,0.1);
	// }
	// cudaDeviceSynchronize();
}