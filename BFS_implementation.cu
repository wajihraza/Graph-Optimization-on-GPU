#include <cstdio>
#include <cstdlib>
#include <vector>
#include <queue>
#include <iostream>
#include <cuda.h>
#include <chrono>
#include <random>

// Error checking macro
#define CUDA_CHECK(err) do { \
    cudaError_t e = err;     \
    if (e != cudaSuccess) {  \
        fprintf(stderr, "CUDA Error %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(-1); \
    } \
} while(0)

struct Graph {
    int numNodes;
    std::vector<int> edges;
    std::vector<int> offset; 
};


// CPU implementation of the code (to compare with GPU optimization)
void bfsCPU(const Graph& graph, int startNode, std::vector<int>& levels) {
    std::vector<bool> visited(graph.numNodes, false);
    for (int i = 0; i < graph.numNodes; i++) levels[i] = -1;
    levels[startNode] = 0;
    visited[startNode] = true;

    std::queue<int> q;
    q.push(startNode);

    while (!q.empty()) {
        int node = q.front(); q.pop();
        for (int i = graph.offset[node]; i < graph.offset[node+1]; ++i) {
            int neighbor = graph.edges[i];
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                levels[neighbor] = levels[node] + 1;
                q.push(neighbor);
            }
        }
    }
}

__global__ void bfsKernel(
    const int* d_edges, const int* d_offset, int* d_visited,
    int* d_levels, int currentLevel, int* d_nextLevelSize, int numNodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    if (d_levels[idx] == currentLevel) {
        int start = d_offset[idx];
        int end = d_offset[idx+1];
        for (int i = start; i < end; ++i) {
            int neighbor = d_edges[i];
            if (atomicCAS(&d_visited[neighbor], 0, 1) == 0) {
                d_levels[neighbor] = currentLevel + 1;
                atomicAdd(d_nextLevelSize, 1);
            }
        }
    }
}

void bfsCUDA(const Graph& graph, int startNode, std::vector<int>& levels) {
    int numNodes = graph.numNodes;

    int *d_edges = 0;
    int *d_offset = 0;
    int *d_visited = 0;
    int *d_levels = 0;
    int *d_nextLevelSize = 0;

    CUDA_CHECK(cudaMalloc(&d_edges, graph.edges.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset, graph.offset.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited, numNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_levels, numNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nextLevelSize, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_edges, graph.edges.data(), graph.edges.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset, graph.offset.data(), graph.offset.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_visited, 0, numNodes * sizeof(int)));
    // Initialize levels to -1
    CUDA_CHECK(cudaMemset(d_levels, -1, numNodes * sizeof(int)));

    int zeroVal = 0;
    int oneVal = 1;
    CUDA_CHECK(cudaMemcpy(d_levels + startNode, &zeroVal, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited + startNode, &oneVal, sizeof(int), cudaMemcpyHostToDevice));

    levels.assign(numNodes, -1);
    levels[startNode] = 0;

    int blockSize = 256;
    int numBlocks = (numNodes + blockSize - 1) / blockSize;

    int nextLevelSize = 1;
    int currentLevel = 0;
    while (nextLevelSize > 0) {
        nextLevelSize = 0;
        CUDA_CHECK(cudaMemcpy(d_nextLevelSize, &nextLevelSize, sizeof(int), cudaMemcpyHostToDevice));

        bfsKernel<<<numBlocks, blockSize>>>(d_edges, d_offset, d_visited, d_levels, currentLevel, d_nextLevelSize, numNodes);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&nextLevelSize, d_nextLevelSize, sizeof(int), cudaMemcpyDeviceToHost));

        currentLevel++;
    }

    CUDA_CHECK(cudaMemcpy(&levels[0], d_levels, numNodes * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_edges);
    cudaFree(d_offset);
    cudaFree(d_visited);
    cudaFree(d_levels);
    cudaFree(d_nextLevelSize);
}

int main() {
    // Increase graph size to see GPU advantage
    int numNodes = 100000;   // Adjust if too large for your GPU (if you want to test GPU performance on varying graph sizes)
    int edgesPerNode = 10;   // Each node will have ~10 edges
    int startNode = 0;

    // Random graph generation
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, numNodes - 1);

    std::vector<int> outDegree(numNodes, edgesPerNode);

    std::vector<int> offset(numNodes + 1, 0);
    for (int i = 1; i <= numNodes; i++) {
        offset[i] = offset[i - 1] + outDegree[i - 1];
    }

    std::vector<int> edges(offset[numNodes]);
    // Fill edges with random targets
    for (int i = 0; i < numNodes; i++) {
        int startIdx = offset[i];
        int endIdx = offset[i + 1];
        for (int e = startIdx; e < endIdx; e++) {
            edges[e] = dist(rng);
        }
    }

    Graph graph;
    graph.numNodes = numNodes;
    graph.edges = edges;
    graph.offset = offset;

    // Prepare level arrays
    std::vector<int> levelsCPU(graph.numNodes, -1);
    std::vector<int> levelsGPU(graph.numNodes, -1);

    // Timing CPU BFS
    auto cpu_start = std::chrono::high_resolution_clock::now();
    bfsCPU(graph, startNode, levelsCPU);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // Timing GPU BFS
    float gpu_time = 0.0f;
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    CUDA_CHECK(cudaEventRecord(startEvent, 0));
    bfsCUDA(graph, startNode, levelsGPU);
    CUDA_CHECK(cudaEventRecord(stopEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, startEvent, stopEvent));

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // Print timing and correctness check
    std::cout << "CPU Time: " << cpu_time << " ms\n";
    std::cout << "GPU Time: " << gpu_time << " ms\n";

    bool correct = true;
    for (int i = 0; i < graph.numNodes; i++) {
        if (levelsCPU[i] != levelsGPU[i]) {
            correct = false;
            break;
        }
    }

    std::cout << "Correctness: " << (correct ? "PASSED" : "FAILED") << "\n";

    if (correct && gpu_time > 0) {
        double speedup = cpu_time / gpu_time;
        std::cout << "Speedup (CPU/GPU): " << speedup << "x\n";
    }

    return 0;
}
