#include "lenet.h"

void get_random_weights(){
    //get random weights
    float data[1522];
    for(long i_2 = 0; i_2<1522; i_2++){
		int r = rand()%1000;
		float r2 = (float)r/1000;
		data[i_2] = r2;
	}
    write_weights(data);
}

void get_random_input(){
    //random input
    /*
    for(int i_1 = 0; i_1<10; i_1++){
		int r = rand()%255;
		data[i_1+1512] = (float)r;
	}
    */
}

int main(){
    get_random_weights();
    return 0;
}