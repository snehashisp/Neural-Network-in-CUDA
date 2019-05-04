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

void readCSV(matrix *mat , matrix *out_mat){
	FILE* f1 = fopen("apparel-trainval.csv","r");

	char rec[4096];
	bool flag=false;
	int width=785;
	int height=60000;

	mat -> init(height,width);
	out_mat -> init(height,1);
	
	int i=0;
	while(fscanf(f1, "%s", rec) != EOF){
		cout<<i<<endl;
		if(!flag){
			flag=true;
			continue;
		}
        char *p = strtok (rec, ",");
        int j=0;
        while (p != NULL){
        	if(j==0){
        		out_mat->mat[i];
        		j++;
        		continue;
        	}
			mat->mat[i*width + j]=atoi(p);
	        p = strtok (NULL, ",");
	        j++;
	    }
	    i++;
	}
	cout<<"done\n";


}

int main(int argc, char const *argv[])
{
	
	readCSV();
	
}
