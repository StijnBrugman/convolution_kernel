#define HALF_FILTER_SIZE 1

__kernel void convolutional_kernel(
      __global float* input, 
      __global float* output, 
      __constant float* filter) 
{
   const int INPUT_MATRIX_WIDTH  = get_global_size(0);
   const int INPUT_MATRIX_HEIGHT = get_global_size(1);
   const int column_n            = get_global_id(0);
   const int row_n               = get_global_id(1);

   int index = column_n +  row_n * INPUT_MATRIX_WIDTH;

    if(
       column_n < HALF_FILTER_SIZE || 
       column_n > INPUT_MATRIX_WIDTH - HALF_FILTER_SIZE - 1 || 
       row_n < HALF_FILTER_SIZE ||
       row_n > INPUT_MATRIX_HEIGHT - HALF_FILTER_SIZE - 1
    ){return;}

   float sum = 0.0;
   int i,j;
   for(i = 0; i < 3; i++){
      for(j = 0; j < 3; j++){
         sum += input[index + (i - 1) * INPUT_MATRIX_WIDTH + j - 1] * filter[3 * i + j];
      }
   }

   // sum += input[index + -1 * get_global_size(0) - HALF_FILTER_SIZE] * filter[0];
   // sum += input[index + -1 * get_global_size(0)                   ] * filter[1];
   // sum += input[index + -1 * get_global_size(0) + HALF_FILTER_SIZE] * filter[2];

   // sum += input[index + 0 * get_global_size(0) - HALF_FILTER_SIZE ] * filter[3];
   // sum += input[index + 0 * get_global_size(0)                    ] * filter[4];
   // sum += input[index + 0 * get_global_size(0) + HALF_FILTER_SIZE ] * filter[5];

   // sum += input[index + 1 * get_global_size(0) - HALF_FILTER_SIZE ] * filter[6];
   // sum += input[index + 1 * get_global_size(0)                    ] * filter[7];
   // sum += input[index + 1 * get_global_size(0) + HALF_FILTER_SIZE ] * filter[8];
   output[index] = sum  / 255.0;
}
