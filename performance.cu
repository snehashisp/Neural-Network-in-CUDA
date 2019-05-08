#include"neuralnet.cu"
#include<chrono>

using namespace std;
double *timeMulti(matrix *mat1,matrix *mat2) {

	double *timearr = new double[2];
	auto t_start = std::chrono::high_resolution_clock::now();
	matrix *tmat = matrix_multi(mat1,mat2);
	auto t_end = std::chrono::high_resolution_clock::now();
	timearr[0] = std::chrono::duration<double, std::milli>(t_end-t_start).count();

	t_start = std::chrono::high_resolution_clock::now();
	matrix *cmat = new matrix();
	cuda_matmul(mat1,mat2,cmat);
	t_end = std::chrono::high_resolution_clock::now();
	cmat -> updateCuda();
	timearr[1] = std::chrono::duration<double, std::milli>(t_end-t_start).count();

	cmat -> freeCuda();
	cmat -> ~matrix();
	tmat -> ~matrix();
	return timearr;
}



double *timeHadamard(matrix *mat1,matrix *mat2) {

	double *timearr = new double[2];
	auto t_start = std::chrono::high_resolution_clock::now();
	matrix *tmat = point_multi(mat1,mat2);
	auto t_end = std::chrono::high_resolution_clock::now();
	timearr[0] = std::chrono::duration<double, std::milli>(t_end-t_start).count();

	t_start = std::chrono::high_resolution_clock::now();
	matrix *cmat = new matrix();
	cuda_hadamard(mat1,mat2,cmat);
	t_end = std::chrono::high_resolution_clock::now();
	cudaDeviceSynchronize();
	//cmat -> updateCuda();
	timearr[1] = std::chrono::duration<double, std::milli>(t_end-t_start).count();

	cmat -> freeCuda();
	cmat -> ~matrix();
	tmat -> ~matrix();
	return timearr;
}

int main(int s,char *argv[]) {

	int a = atoi(argv[1]),b = atoi(argv[2]),c = atoi(argv[3]);
	matrix *results = new matrix;
	results -> init(b + 1,3);


	// for(int i = a; i <= a + b; i += c) {
	// 	matrix *mat1 = new matrix, *mat2 = new matrix;
	// 	mat1 -> init(i,i);
	// 	mat2 -> init(i,i);
	// 	gaussianInitializer(mat1);
	// 	gaussianInitializer(mat2);
	// 	double *res = timeMulti(mat1,mat2);
	// 	cout << i <<","<<res[0]<<","<<res[1]<<endl;
	// 	results -> mat[(i - a)*2] = i;
	// 	results -> mat[(i - a)*2 + 1] = res[0];
	// 	results -> mat[(i - a)*2 + 2] = res[1];
	// 	mat1 -> ~matrix();
	// 	mat2 -> ~matrix();
	// }


	for(int i = a; i <= a + b; i += c) {
		matrix *mat1 = new matrix, *mat2 = new matrix;
		mat1 -> init(i,i);
		mat2 -> init(i,i);
		gaussianInitializer(mat1);
		gaussianInitializer(mat2);
		double *res = timeHadamard(mat1,mat2);
		cout << i <<","<<res[0]<<","<<res[1]<<endl;
		results -> mat[(i - a)*2] = i;
		results -> mat[(i - a)*2 + 1] = res[0];
		results -> mat[(i - a)*2 + 2] = res[1];
		mat1 -> ~matrix();
		mat2 -> ~matrix();
	}

}
