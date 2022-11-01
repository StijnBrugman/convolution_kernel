#define HALF_FILTER_SIZE     1
#define INPUT_MATRIX_WIDTH   80
#define INPUT_MATRIX_HEIGHT  300
   

__kernel void convolutional_kernel(
      __global float* input, 
      __global float* output, 
      __global float* filter) 
{

   int index = get_global_id(0) +  get_global_id(1) * INPUT_MATRIX_WIDTH;
   
    if(
       get_global_id(0) < HALF_FILTER_SIZE || 
       get_global_id(0) > INPUT_MATRIX_WIDTH - HALF_FILTER_SIZE - 1 || 
       get_global_id(1) < HALF_FILTER_SIZE ||
       get_global_id(1) > INPUT_MATRIX_HEIGHT - HALF_FILTER_SIZE - 1
    ){return;}

   float sum = 0.0;
   int i,j;
   for(i = 0; i < 3; i++){
      for(j = 0; j < 3; j++){
         sum += input[index + (i - 1) * INPUT_MATRIX_WIDTH + j - 1] * filter[3 * i + j];
      }
   }

   output[index] = sum / 255.0;
}
