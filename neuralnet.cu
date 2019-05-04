#include"functions.cu"
#include<vector>

#ifndef NNET
#define NNET

/*
input is of dimension N x D : N = batch size and D = data dimension
*/

class neural_network {


	std :: vector<matrix *> weight_arr;
	std :: vector<matrix *> biases_arr;
	std :: vector<matrix *> output_arr;
	std :: vector<matrix *> activation_arr;

	int batch_size, input_size;
	int activation = FN_SIGM;
	bool use_bias = false;

	public:

	void forward(matrix *data,bool updates = false) {


		if(data -> height != batch_size || data -> width != input_size) {
			printf(" Data dimension mismatch required %d %d given %d %d",
				batch_size,input_size,data -> height,data -> width);
		}
		int i = 0;

		if(use_bias) 
			cuda_matmul(data,weight_arr[i],biases_arr[i],output_arr[i],updates);
		else
			cuda_matmul(data,weight_arr[i],output_arr[i],updates);
		cuda_function(output_arr[i],activation_arr[i],activation,updates);
		
		for(i += 1; i < weight_arr.size(); i++) {
			if(use_bias) 
				cuda_matmul(activation_arr[i-1],weight_arr[i],biases_arr[i],output_arr[i],updates);
			else
				cuda_matmul(activation_arr[i-1],weight_arr[i],output_arr[i],updates);
			cuda_function(output_arr[i],activation_arr[i],activation,updates);
		}	
	}


	//public:

	void init(std::vector<int> nodeList,int bsize,int activation = FN_SIGM, 
		bool use_bias=true,double mean = 0,double std = 1) {

		for(int i = 0; i < weight_arr.size(); i++) {

			weight_arr[i] -> freeCuda();
			weight_arr[i] -> ~matrix();
			biases_arr[i] -> freeCuda();
			biases_arr[i] -> ~matrix();
			output_arr[i] -> freeCuda();
			output_arr[i] -> ~matrix();
			activation_arr[i] -> freeCuda();
			activation_arr[i] -> ~matrix();
			this -> use_bias = use_bias;
		}

		weight_arr.clear();
		output_arr.clear();
		activation_arr.clear();

		for(int i = 0; i < nodeList.size()-1; i++) {

			matrix *mat = new matrix, *omat = new matrix, *amat = new matrix;
			mat -> init(nodeList[i],nodeList[i+1]);
			gaussianInitializer(mat,mean,std);

			if(use_bias == true) {
				matrix *bmat = new matrix;
				bmat -> init(1,nodeList[i+1]);
				gaussianInitializer(bmat,mean,std);
				biases_arr.push_back(bmat);
			}

			weight_arr.push_back(mat);
			output_arr.push_back(omat);
			activation_arr.push_back(amat);
		}

		batch_size = bsize;
		this -> activation = activation;
		input_size = nodeList[0];

	}



};

#endif