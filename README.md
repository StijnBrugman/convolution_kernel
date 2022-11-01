# Convolution Kernal on 2D matrix



## Details
* Kernal size: 3x3
* Matrix size: 80x300

## Implementation steps
- [x] Load distance_vector data from file
- [x] Run average execution time
- [x] Improve functionality with GPU
- [x] Quantify performance

## Running the software
```c
$ ./run.sh 0
```

## Flags
A flag can be added to specify the mode in the bash file
Naive
```c
$ ./run.sh 0
```

Optimized
```c
$ ./run.sh 1
```

flags:
* -O3 : -Ofast | Speeds up the program signifanctly 
* -g | Debug mode

## Useful sources
- https://github.com/Xilinx/Vitis-HLS-Introductory-Examples/blob/master/

## Contact
For more information about the project, you can contact Stijn Brugman ([s.r.d.brugman@student.utwente.nl](mailto:s.r.d.brugman@student.utwente.nl)).
