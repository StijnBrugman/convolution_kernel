#define KERNEL_FUNC "convolutional_kernel"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define N_REPEAT 50000

cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   // Access a GPU-device
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      // CPU
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
      perror("CPU selected");
      exit(1);   
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   program = clCreateProgramWithSource(ctx, 1, 
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}


#define FILTER_SIZE 9
// float the inputs
float hor_line_kernel [FILTER_SIZE] = {
    -1.0, -1.0, -1.0,
     2.0,  2.0,  2.0,
    -1.0, -1.0, -1.0
};

int main(int argc, char *argv[]) {
   // Insufficient amount of arguments were provided
   if(argc < 2){
     printf("Need 2 arguments! X(Number of positions) and Y(Max Distance)\n\n");
     return -1;
   }

   /* ===================================================== 
   INIT Matrix Params 
   ===================================================== */
   clock_t start, end;
   long double cpu_time_used;

   const int MATRIX_WIDTH  = atoi(argv[1]);
   const int MATRIX_HEIGHT = atoi(argv[2]);

   const int MATRIX_SIZE   = MATRIX_WIDTH * MATRIX_HEIGHT;

   float *distance_vector  = (float*)calloc(MATRIX_WIDTH,sizeof(float));
   float *distance_matrix  = (float*)calloc(MATRIX_SIZE, sizeof(float));
   float *filtered_matrix  = (float*)calloc(MATRIX_SIZE, sizeof(float));
   float *threshold_matrix = (float*)calloc(MATRIX_SIZE, sizeof(float));
   float *new_vector       = (float*)calloc(MATRIX_WIDTH,sizeof(float));
   float *new_real_vector  = (float*)calloc(MATRIX_WIDTH,sizeof(float));

   
   /* ===================================================== 
   INIT Distance Matrix
   ===================================================== */

   // load the distance vector
   FILE *myFile;
   myFile = fopen("input_vector.txt", "r");
   for (int i = 0; i < MATRIX_WIDTH; i++){
     fscanf(myFile, "%f,", &distance_vector[i]);    
   }

   // Create Distance Matrix
   for(int i = 0; i < MATRIX_WIDTH; i++){
     int distance = distance_vector[i];
     if(distance >= MATRIX_HEIGHT) distance = MATRIX_HEIGHT-1;
     distance_matrix[distance*MATRIX_WIDTH+i] = 255;//sets distance object
   }

   /* ===================================================== 
   INIT Kernel Strucuters
   ===================================================== */
   //OpenCL structures
   cl_device_id      device;
   cl_context        context;
   cl_program        program;
   cl_kernel         kernel;
   cl_command_queue  queue;
   cl_int            i, j, err;
   cl_mem            input_buffer, out_buffer, filter_buffer;

   const int DIMENSION = 2;
   size_t global_size[2] = {MATRIX_WIDTH, MATRIX_HEIGHT}; // 80, 300 
   size_t local_size[2]  = {8, 30};
   size_t local_size_naive[2]  = {1, 1};
   
   /* ===================================================== 
   SETUP kernal strucutres
   ===================================================== */
   device = create_device();

   /* Create Context */
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   program = build_program(context, device, PROGRAM_FILE);
 
   input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_COPY_HOST_PTR, MATRIX_SIZE * sizeof(float), distance_matrix, &err); 
   out_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_COPY_HOST_PTR, MATRIX_SIZE * sizeof(float), filtered_matrix, &err); 
   filter_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_COPY_HOST_PTR, FILTER_SIZE * sizeof(float), hor_line_kernel, &err); 

   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };

   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   start = clock();

   err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_buffer); 
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&out_buffer); 
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&filter_buffer); 

   if(err < 0) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

   cl_event event[N_REPEAT];
   for (int i =0; i < N_REPEAT; i++ ){
        err = clEnqueueNDRangeKernel(queue, kernel, DIMENSION, NULL, global_size, 
             NAIVE ? local_size : local_size_naive, 0, NULL, &event[i]); 
       
        if(err < 0) {
            perror("Couldn't enqueue the kernel");
            exit(1);
        }
    }
   
   cl_ulong time_start;
   cl_ulong time_end;
   double profile_timing = 0;
   clWaitForEvents(N_REPEAT, &event[0]);
   // clWaitForEvents(1, &event);
   clFinish(queue);

   for (int i =0; i < N_REPEAT; i++ ){
        clGetEventProfilingInfo(event[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        profile_timing += (time_end-time_start);
    }
    profile_timing = (profile_timing / N_REPEAT) / 1000.0;


   /* Read the kernel's output    */
   err = clEnqueueReadBuffer(queue, out_buffer, CL_TRUE, 0, 
         MATRIX_SIZE * sizeof(float), (void *) filtered_matrix, 0, NULL, NULL); // <=====GET OUTPUT
   //end time measure
   end = clock();
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }
   // }

   
   cpu_time_used = ((long double) (end - start)  * 1000000/ N_REPEAT) / CLOCKS_PER_SEC;
   printf("----GPU MEASUREMENT----\r\n");
   printf("Total time = %.3f us\n", profile_timing);


   /* ===================================================== 
   CHECK the result
   ===================================================== */

   // Threshold the matrix
    for(int x = 0; x < MATRIX_WIDTH; x++){
        for(int y = 0; y < MATRIX_HEIGHT; y++){
            if(filtered_matrix[y*MATRIX_WIDTH+x]>=3.95){
                threshold_matrix[y*MATRIX_WIDTH+x] = 1;
            }
        }
    }

    // Extract vector from matrix
    for(int x = 0; x < MATRIX_WIDTH; x++){
        for(int y = 0; y < MATRIX_HEIGHT; y++){
            if(threshold_matrix[y*MATRIX_WIDTH+x]){
                new_vector[x] = y;//sets distance object
            }
        }
        if(new_vector[x] == 0) new_vector[x] = MATRIX_HEIGHT;
    }

    printf("\r\n ----REAL MEASUREMENT -----\r\n");
    int l, k, x, y;

    // Clear filtered matrix
    for(x = 0; x < MATRIX_WIDTH; x++){
        for(y = 0; y < MATRIX_HEIGHT; y++){
            filtered_matrix[y*MATRIX_WIDTH+x] = 0;
        }
    }

    // Clear Threshold matrix
    for(x = 0; x < MATRIX_WIDTH; x++){
        for(y = 0; y < MATRIX_HEIGHT; y++){
            threshold_matrix[y*MATRIX_WIDTH+x] = 0;
        }
    }

    // Start time measure
    start = clock();

/******************* OPTIMIZE THIS ***********************/

    
    float sum = 0.0;

    // Repeat 1000 times
    for (l = 0; l < N_REPEAT; l++){

        // Apply kernel for all points in the matrix
        for(y = 1; y < MATRIX_HEIGHT-1; y++){
            for(x = 1; x < MATRIX_WIDTH-1; x++){
                sum = 0.0;
                for(k = -1; k < 2; k++){
                    for(int j = -1; j < 2; j++){
                        sum += hor_line_kernel[( k + 1) * 3 + (j + 1)] * (float)distance_matrix[(y - k) * MATRIX_WIDTH + (x - j)];
                    }
                }
                filtered_matrix[y*MATRIX_WIDTH + x] = sum/255.0;
            }
        }
    }

/********************************************************/
    
    // End time measure
    end = clock();
    cpu_time_used = ((long double) (end - start) * 1000000 /N_REPEAT) / CLOCKS_PER_SEC;

    // Threshold the matrix
    for(x = 0; x < MATRIX_WIDTH; x++){
        for(y = 0; y < MATRIX_HEIGHT; y++){
            if(filtered_matrix[y*MATRIX_WIDTH+x]>=4.0){
                threshold_matrix[y*MATRIX_WIDTH+x] = 1;
            }
        }
    }

    // Extract vector from matrix
    for(x = 0; x < MATRIX_WIDTH; x++){
        for(y = 0; y < MATRIX_HEIGHT; y++){
            if(threshold_matrix[y*MATRIX_WIDTH+x]){
                new_real_vector[x] = y;//sets distance object
            }
        }
        if(new_real_vector[x] == 0) new_real_vector[x] = 300;
    }
    printf("Total time = %.3Lf us\n", cpu_time_used);

   /* Check result */
   int succes_flag = 1;
   for(i=0; i<MATRIX_WIDTH; i++) {
      if (new_real_vector[i] != new_vector[i]){
        succes_flag = 0;
      }
   }

   if(!succes_flag){
      printf("Check failed.\n");
   }
   else{
      printf("Check passed. Improvement of %.3Lf\n", cpu_time_used / profile_timing);
   }

   /* Deallocate resources */
   clReleaseKernel(kernel);
   clReleaseMemObject(out_buffer);
   clReleaseMemObject(input_buffer);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   for(int i =0; i < N_REPEAT; i++){
    clReleaseEvent(event[i]);
    }

   /* Deallocate vectors & matrix */
   free(distance_vector);
   free(distance_matrix);
   free(filtered_matrix);
   free(threshold_matrix);
   free(new_vector);
   free(new_real_vector);

   return 0;
}
