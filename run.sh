while getopts "m:" arg; do
	case $arg in
		m) mode=$OPTARG;;
	esac
done	


if [ $mode == 0 ]; then
		gcc convolutional_kernel.c -o program -framework OpenCL -D PROGRAM_FILE='"convolutional_kernel_naive.cl"' -D NAIVE=0
elif [ $mode == 1 ]; then
		gcc convolutional_kernel.c -o program -framework OpenCL -D PROGRAM_FILE='"convolutional_kernel.cl"' -D NAIVE=1
else
	gcc convolutional_kernel.c -o program -framework OpenCL -O3 -D  PROGRAM_FILE='"convolutional_kernel.cl"' -D NAIVE=1
	echo ""
	echo ""
	echo "                                                -----"
	echo "                                              /      \\"
	echo "                                              )      |"
	echo "       :================:                      :    )/"
	echo "      /||              ||                      )_ /*"
	echo "     / ||    System    ||                          *"
	echo "    |  ||     Down     ||                   (=====~*~======)"
	echo "     \\ || Please wait  ||                  0      \\ /       0"
	echo "       ==================                //   (====*====)   ||"
	echo "........... /      \\.............       //         *         ||"
	echo ":\\        ############            \\    ||    (=====*======)  ||"
	echo ": ---------------------------------     V          *          V"
	echo ": |  *   |__________|| ::::::::::  |    o   (======*=======) o"
	echo "\\ |      |          ||   .......   |    \\\\         *         ||"
	echo "  --------------------------------- 8   ||   (=====*======)  //"
	echo "                                     8   V         *         V"
	echo "  --------------------------------- 8   =|=;  (==/ * \\==)   =|="
	echo "  \\   ###########################  \\   / ! \\     _ * __    / | \\"
	echo "   \\  +++++++++++++++++++++++++++   \\  ! !  !  (__/ \\__)  !  !  !"
	echo "    \\ ++++++++++++++++++++++++++++   \\        0 \\ \\V/ / 0"
	echo "     \\________________________________\\     ()   \\o o/   ()"
	echo "      *********************************     ()           ()"
	echo ""
	echo ""
fi
./program 80 300