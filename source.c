

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Example data to load from your file:
// 117,85, 146,194, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
// 20, 20, 20, 20,  20, 20, 20, 21, 22, 417,418,141,68, 196,198,194,177,
// 173,173,172,2101,172,172,173,149,172,172,172,173,172,175,
// 173,173,172,171, 172,100,111,101,101,100,98, 98, 98, 88, 
// 98, 99, 97, 98,  96, 96, 97, 98, 98, 96, 98, 98, 97, 98, 97, 97, 92, 96


// A few filtering kernels as samples
float low_pass_kernel [9] = {
    1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0
};

// float the inputs
float hor_line_kernel [9] = {
    -1.0, -1.0, -1.0,
     2.0,  2.0,  2.0,
    -1.0, -1.0, -1.0
};

float ver_line_kernel [9] = {
    -1.0, 2.0, -1.0,
    -1.0, 2.0, -1.0,
    -1.0, 2.0, -1.0
};


int main(int argc, char *argv[]){

    clock_t start, end;
    double cpu_time_used;

    if(argc < 2){
        printf("Need 2 arguments! X(Number of positions) and Y(Max Distance)\n\n");
        return -1;
    }

    int posNum = atoi(argv[1]);
    int dstNum = atoi(argv[2]);

    int *distance_vector = (int*)calloc(posNum,sizeof(int));
    int *distance_matrix = (int*)calloc(posNum*dstNum,sizeof(int));
    float *filtered_matrix = (float*)calloc(posNum*dstNum,sizeof(float));
    int *threshold_matrix = (int*)calloc(posNum*dstNum,sizeof(int));
    int *new_vector = (int*)calloc(posNum,sizeof(int));

    int i;


    // Implement your LOAD_DATA function here to load X number of elements and store them into distance_vector
    FILE *myFile;
    myFile = fopen("input_vector.txt", "r");
    for (i = 0; i < posNum; i++){
        fscanf(myFile, "%d,", &distance_vector[i]);    
    }

    // for (i = 0; i < posNum; i++){
    //     printf("%d,", distance_vector[i]);
    // }
    // printf("\n");



    // Creates matrix from input vector
    // Setting 255 at the corresponding positions of the input vector
    for(i = 0; i < posNum; i++){
        int distance = distance_vector[i];
        if(distance >= dstNum) distance = dstNum-1;
        distance_matrix[distance*posNum+i] = 255;//sets distance object
    }

    // Start time measure
    start = clock();

/******************* OPTIMIZE THIS ***********************/

    int l, j, k, x, y;
    float sum = 0.0;

    // Repeat 1000 times
    for (l = 0; l < 1000; l++){

        // Apply kernel for all points in the matrix
        for(y = 1; y < dstNum-1; y++){
            for(x = 1; x < posNum-1; x++){
                sum = 0.0;
                for(k = -1; k < 2; k++){
                    for(j = -1; j < 2; j++){
                        sum += hor_line_kernel[( k + 1) * 3 + (j + 1)] * (float)distance_matrix[(y - k) * posNum + (x - j)];
                    }
                }
                filtered_matrix[y*posNum + x] = sum/255;
            }
        }
    }

/********************************************************/
    
    // End time measure
    end = clock();
    cpu_time_used = ((double) (end - start)/1000) / CLOCKS_PER_SEC;

    // Threshold the matrix
    for(x = 0; x < posNum; x++){
        for(y = 0; y < dstNum; y++){
            if(filtered_matrix[y*posNum+x]>=4.0){
                threshold_matrix[y*posNum+x] = 1;
            }
        }
    }

    // Extract vector from matrix
    for(x = 0; x < posNum; x++){
        for(y = 0; y < dstNum; y++){
            if(threshold_matrix[y*posNum+x]){
                new_vector[x] = y;//sets distance object
            }
        }
        if(new_vector[x] == 0) new_vector[x] = 300;
    }

    //Prints threshold-filtered vector
    for(x = 0; x < posNum; x++){
        printf("%d, ", new_vector[x]);
    }

    printf("Total time = %f ms\n", cpu_time_used*1000);
    
    free(distance_vector);
    free(distance_matrix);
    free(filtered_matrix);
    free(threshold_matrix);
    free(new_vector);
    return 0;
}