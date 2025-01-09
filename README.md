##Graph Optimization on GPUs

I have implemented 2 CUDA kernels (Breadth First Search and Depth First Search) for graph algorithms and compared the performance improvement from CPU to GPU for the project. 

To run this code, you will need to do the following commands:

1) Open the project directory on Visual Studio Code
2) run the command: nvcc -std=c++11 -o BFS_implementation.exe BFS_implementation.cu (this command will compile the code and generate the executable)
3) run the executable: ./BFS_implementation.exe (I have attached the screenshot in the zip file to show some results and command implementation)
4) Repeat step 2 and 3 to run the DFS_implementation CUDA kernel. 
