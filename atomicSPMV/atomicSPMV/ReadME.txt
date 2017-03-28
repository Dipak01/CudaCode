Compile the files using command
make -B all

Run the project using command
./spmv -mat <Matrix Name> -ivec <Vector Name> -alg <atomic, segment, design> -blockSize 16 -blockNum 512

NOTE:
To create a vector file, run vector_generator as follows::
1. Edit the "vector_generator" file by adding file name into which we shall copy the vector generated. For example :
awk -v min=0.1 -v max=3.5 -v num=$1 'BEGIN{ srand(); for (i = 1; i <= num; i++)   print (min+rand()*(max-min+1))}' >> vecFC.txt
This will create a text file by the name "vecFC.txt".

2. Run ./vector_generator.sh <Number of cols in matrix>

3. Now open vecFC.txt and on the first row add the total number of cols the matrix has.

4. Save and close the file