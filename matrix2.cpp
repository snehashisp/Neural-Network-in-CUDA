#include <bits/stdc++.h>
using namespace std;
class matrix {

	public:
	double *mat = NULL;
	int height,width;
	// bool isUpdated = true;
	double *cudaMat = NULL;

	void init(int height,int width,double **imat = NULL) {
		this -> height = height;
		this -> width = width;
		mat = (double *)calloc(height*width,sizeof(double));
		if(imat) {
			for(int i = 0; i < height; i++) {
				for(int j = 0; j < width; j++) 
					mat[i*width + j] = imat[i][j];
			}
		}

	}
	void print() {
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {
				printf(" %6.2lf ",mat[i*width + j]);
			}
			printf("\n");
		}
	}
};

matrix *readCSV(){
	FILE* f1 = fopen("apparel-trainval.csv","r");

	char rec[2048];
	bool flag=false;
	matrix *mat = new matrix;
	int width=785;
	int height=60000;

	mat -> init(height,width);
	
	int i=0;
	while(fscanf(f1, "%s", rec) != EOF){
		cout<<i<<" ";
		if(!flag){
			flag=true;
			continue;
		}
        char *p = strtok (rec, ",");
        int j=0;
        while (p != NULL){
			mat->mat[i*width + j]=atoi(p);
	        p = strtok (NULL, ",");
	        j++;
	    }
	    i++;
	}
	return mat;

}
int main(int argc, char const *argv[])
{
	
	readCSV();
	return 0;
}
