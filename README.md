# gQM3d
3D Constrained Delaunay Refinement on the GPU

Project Website: https://www.comp.nus.edu.sg/~tants/gqm3d.html

Paper:  
1. Computing Three-dimensional constrained Delaunay Refinement Using the GPU. Z. Chen and T.S. Tan. The 28th International Conference on Parallel Architectures and Compilation Techniques, 21-25 September, 2019, Seattle, WA, USA. (<a href="https://www.comp.nus.edu.sg/~tants/gqm3d_files/gqm3d.pdf">PDF</a>)
2. On Designing GPU Algorithms with Applications to Mesh Refinement. Z. Chen, T.S. Tan, and H.Y. Ong. arXiv, 2020. (<a href="https://arxiv.org/abs/2007.00324">PDF</a>)

* A NVIDIA GPU is required since this project is implemented using CUDA.
* The development environment: Visual Studio 2017 and CUDA 9.0. (Please use x64 and Release mode.)
* TetGen 1.5 is used. Check http://wias-berlin.de/software/index.jsp?id=TetGen&lang=1 for more information.
* CGAL 4.13 is used for comparision. Check https://www.cgal.org/index.html for more information. You might remove Experiment_CGAL.h and Experiment_CGAL.cpp if you don't want to use CGAL.

--------------------------------------------------------------
GPU Refinement Routine (located in GPU_Refine_3D/MeshRefine.h and GPU_Refine_3D/MeshRefine.cpp):  
void <b>refineInputFileOnGPU</b>(  
&nbsp;&nbsp;&nbsp;&nbsp; char* infile,  
&nbsp;&nbsp;&nbsp;&nbsp; MESHBH* input_behavior,  
&nbsp;&nbsp;&nbsp;&nbsp; char* outmesh,  
&nbsp;&nbsp;&nbsp;&nbsp; char* outdata)

This routine calls both TetGen (GPU_Refine_3D/tetgen.h and GPU_Refine_3D/tetgen.cpp) and the GPU refinement pipeline (GPU_Refine_3D/CudaRefine.h and GPU_Refine_3D/CudaRefine.cu) to refine the input mesh, and output the quality mesh and its statistic.

char* infile:  
The path for input mesh.

MESHBH* input_behavior:  
Different behaviors used to control the refinement process; see GPU_Refine_3D/MeshStructure.h for more details.
	
char* outmesh:  
The path for output mesh.

char* outdata:  
The path for output mesh statistic. This includes the final element numbers, timing for different stages, and the dihedral angle distribution.

--------------------------------------------------------------
Experiment

All experiments were conducted on a PC with an Intel i7-7700k 4.2GHz CPU, 32GB of DDR4 RAM and a GTX1080 Ti graphics card with 11GB of video memory.

* Synthetic dataset:  
The synthetic data was generated by our input generator (InputGenerator.h and InputGenerator.cpp) and might have small angles. Some samples and result statistics by TetGen, CGAL and this software are provided in GPU_Refine_3D/input/.

* Real-world dataset:  
3D printing models from the <a href="https://ten-thousand-models.appspot.com/">Thingi10K</a> dataset were used. Some samples and result statistics by TetGen and this software are provided in GPU_Refine_3D/input_real/.
--------------------------------------------------------------

Proceed to GPU_Refine_3D/main.cpp to check how to call gpu and cpu refinement routines properly.