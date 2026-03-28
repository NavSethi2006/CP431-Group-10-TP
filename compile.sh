# JUST FOR LOCAL TESTING

#create julia set
cd MPI
mpicc main.c -o julia -lm
mpirun -np 4 ./julia

cd ../Opengl
gcc julia_viewer.c -lglut -lGLEW -lGL -lGLU -lm -o julia_viewer

cd ../
Opengl/julia_viewer MPI/julia_set.bin
