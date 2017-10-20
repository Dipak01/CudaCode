//Author: Dhruv Dogra; dd798

Follow the steps below to run the project:
1. Compile project as follow:
	make spmv
2. Run the project:
	./spmv -mat <Matrix Name> -ivec <Vector Name> -alg segment -blksize 256 -blknum 8

NOTE:
1. In Makefile I added "spmv" command which has code similar to "all" command present there.

2. In main.c change made in the if-else block where verify function is called. In the else block incorrect statement "Test Failed"  was getting printed and value 1 was being returned. I updated the statement and set return to 0.

3. "design" part has not been implemented.

4. If required, create a vector file as follow:
 	./vector_generator.sh <Number of cols in matrix>  >> vector_file_name.txt

