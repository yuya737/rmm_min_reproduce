
#include <algorithm>
#include <iostream>
#include <numeric>
#include <array>
#include <vector>
#include <stdlib.h>
#include <random>
#include <thread>

#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include "thrust_rmm_allocator.h"


#include "cClipping.h"

typedef rmm::device_vector<float>::iterator IterFloat;
typedef rmm::device_vector<int>::iterator IterInt;

int main(int argc, char *argv[]){

  size_t sx, sy, sz;
  
  std::vector<float> pos;
  rmm::device_vector<float> d_pos;

  // Types of allocations:
	// CudaDefaultAllocation
	// PoolAllocation
	// CudaManagedMemory

  rmmOptions_t options{static_cast<rmmAllocationMode_t>(PoolAllocation | CudaManagedMemory), 0, true};
  rmmInitialize(&options);
    
  cudaStream_t stream;
	if (cudaStreamCreate (&stream) !=  cudaSuccess){
    std::cout<< "stream error";
  }

  if (argc < 4){
    std::cout << "Usage: clipping x_size y_size z_size" << std::endl;
    return 1;
  }
	sx = std::stoll (std::string(argv[1]));
	sy = std::stoll (std::string(argv[2]));
	sz = std::stoll (std::string(argv[3]));
  std::cout << "Initializing dataset..." << std::endl;
  initDataset(&pos, sx, sy, sz);
  std::cout << "done!" << std::endl;
  d_pos = pos;
  float normal[3], d = 0.0f;
  normal[0] = 0.5f;
  normal[1] = 0.5f;
  normal[2] = 0.5f;
  
  plane_clippingPDBver2 clip	(normal, d);

	strided_range<IterFloat> X		( d_pos.begin()  , d_pos.end(), 4);
	strided_range<IterFloat> Y		( d_pos.begin()+1, d_pos.end(), 4);
	strided_range<IterFloat> Z		( d_pos.begin()+2, d_pos.end(), 4);
  strided_range<IterFloat> W		( d_pos.begin()+3, d_pos.end(), 4);
  
	size_t new_size = thrust::remove_if(rmm::exec_policy(stream)->on(stream), thrust::make_zip_iterator ( thrust::make_tuple( X.begin(), Y.begin(), Z.begin(), W.begin() )),
			   	   	   	   	   	   	   	 thrust::make_zip_iterator ( thrust::make_tuple( X.end(),Y.end(), Z.end(), W.end() )),
			   	   	   	   	   	   	   	 clip )
                                      - thrust::make_zip_iterator(thrust::make_tuple(X.begin(), Y.begin(), Z.begin(), W.begin()));
  return 0;
}
