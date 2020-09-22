#include "fNablaEngineCuda.cuh"

#ifdef __INTELLISENSE__
#pragma diag_suppress 20, 29 //supress rhypot and kernel call errors
#endif

inline void  fNablaEngineCuda::_gpuErrorCheck(cudaError_t code, const char* file, int line){
	if (code != cudaSuccess) {
		throw std::runtime_error(std::string(cudaGetErrorString(code)) + file + std::to_string(line));
	}
}

void fNablaEngineCuda::CheckCUDACapable() {
	int cudaDevices = 0;
	CUDA_ERROR_CHECK(cudaGetDeviceCount(&cudaDevices));
	if (cudaDevices == 0) {
		throw NoCUDAGPU();
	}
}

__global__ void fNablaEngineCuda::_pre_kernel(
	double2* directions,
	const uint samples
) {
	const uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < samples) {
		double theta = 2.0 * _PI * ((double)id / (double)samples) - 2.0 * _PI_2; // -pi/2 to pi/2
		directions[id].y = sin(theta);
		directions[id].x = cos(theta);
	}
}

__device__ __forceinline__ double fNablaEngineCuda::IntegrateArc(double2 h, double n, double sin_n, double cos_n) {
	return 0.25 * cos(n + h.x + h.x) + 0.25 * cos(n + h.y + h.y) + 1.0 * (_PI - h.x - h.y) * sin_n + 0.5 * cos_n;
}

__device__ __forceinline__ void fNablaEngineCuda::TestHorizon(
	double* displacement,
	dim3 shape,
	int2 pos,
	double distance,
	double height_ref,
	double& horizon,
	Edge_mode mode,
	bool invert
) {
	switch (mode) {
		case Edge_mode::Tile:
			pos.y += int(shape.y) * ((pos.y < 0) - (pos.y >= int(shape.y)));
			pos.x += int(shape.x) * ((pos.x < 0) - (pos.x >= int(shape.x)));
		case Edge_mode::Repeat:
			pos.y = max(min(pos.y, shape.y-1), 0);
			pos.x = max(min(pos.x, shape.x-1), 0);
	}

	horizon = fmax(horizon, (invert ? -1.0 : 1.0) * (displacement[pos.y * shape.x + pos.x] - height_ref) / distance);
}

__global__ void fNablaEngineCuda::_AO_kernel(
	double* displacement,
	double3* normal,
	double2* directions,
	double* output,
	dim3 shape,
	const double radius,
	const double step_size,
	const double depth
) {
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if ((x < shape.x) && (y < shape.y) && (z < shape.z)){
		const double displacement_xy = displacement[y * shape.x + x];
		const double3 normal_xy = normal[y * shape.x + x];

		//double theta = M_PI * (((double)z) / (double)samples) - M_PI_2; //-pi/2 to pi/2
		const double sin_theta = directions[z].y; // sin(theta);
		const double cos_theta = directions[z].x; // cos(theta);

		//D={ sin(theta), cos(theta), 0 };
		//V={0, 0, 1};
		//SN = cross(D, V)
		//N = normal_ij;
		//PN = N - SN * NdotSN = {sin_theta * NdotSN, cos_theta * NdotSN, normal_ij.x}

		const double NdotSN = normal_xy.y * cos_theta + normal_xy.z * sin_theta;
		const double rproj_length = rhypot(NdotSN, normal_xy.x);

		double2 Horizons{-_PI, -_PI};

		for (double length = step_size; length <= radius; length += step_size) {

			const int2 step{lrint(sin_theta * length), lrint(cos_theta * length)};

			TestHorizon(displacement, shape, {int(x) + step.x, int(y) + step.y}, length, displacement_xy, Horizons.x);
			TestHorizon(displacement, shape, {int(x) - step.x, int(y) - step.y}, length, displacement_xy, Horizons.y);
		}
		Horizons.x = atan(depth * Horizons.x);
		Horizons.y = atan(depth * Horizons.y);

		const double n = atan2(normal_xy.x, NdotSN);
		//clamp to normal hemisphere
		Horizons.x = n + fmin(Horizons.x - n, _PI_2);
		Horizons.y = n + fmin(Horizons.y - n, _PI_2);

		output[y * shape.x * shape.z + x * shape.z + z] = (IntegrateArc(Horizons, n, normal_xy.x * rproj_length, NdotSN * rproj_length) - 1.0) / rproj_length + 1.0;
	}
}


__global__ void fNablaEngineCuda::reduce_kernel(
	double* input,
	double* output,
	const dim3 shape
) {
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < shape.x) && (y < shape.y)) {
		output[y * shape.x + x] = 0.0;
		for (size_t z = 0; z < shape.z; z++) {
			output[y * shape.x + x] += input[y * shape.x * shape.z + x * shape.z + z] / shape.z;
		}
	}
}

void fNablaEngineCuda::ComputeAOCuda(
	cv::Mat& displacement,
	cv::Mat& normal,
	cv::Mat& ao,
	const uint samples,
	const double distance,
	const double depth,
	std::string& status
) {
	CheckCUDACapable();

	const dim3 shape(ao.cols, ao.rows, samples);
	const double scale = double(std::min(shape.x, shape.y));
	const double radius = std::max(distance * scale, 1.0);

	GPUMat<double> gpu_calc(shape);

	GPUMat<double> gpu_displacement(displacement);

	GPUMat<double3> gpu_normal(normal);

	GPUMat<double2> gpu_directions({samples});
	const dim3 block_dir(8);
	const dim3 grid_dir((samples + block_dir.x - 1) / block_dir.x);

	CUDA_LAUNCH(
		_pre_kernel,
		grid_dir,
		block_dir,
		gpu_directions.data,
		samples
	);
	CUDA_ERROR_CHECK(cudaPeekAtLastError());

	CUDA_ERROR_CHECK(cudaDeviceSynchronize());

	status = "Computing Ambient Occlusion";

	const dim3 block_calc(4, 4, 4);
	const dim3 grid_calc(
		(shape.x + block_calc.x - 1) / block_calc.x,
		(shape.y + block_calc.y - 1) / block_calc.y,
		(shape.z + block_calc.z - 1) / block_calc.z
	);
	CUDA_LAUNCH(
		_AO_kernel,
		grid_calc,
		block_calc,
		gpu_displacement.data,
		gpu_normal.data,
		gpu_directions.data,
		gpu_calc.data,
		shape,
		radius,
		radius / (samples * 16),
		depth * scale
	);

	CUDA_ERROR_CHECK(cudaPeekAtLastError());

	CUDA_ERROR_CHECK(cudaDeviceSynchronize());

	status = "Finalizing Ambient Occlusion";

	GPUMat<double> gpu_output(dim3(shape.x, shape.y));

	const dim3 block_reduce(8, 8);
	const dim3 grid_reduce(
		(shape.x + block_reduce.x - 1) / block_reduce.x,
		(shape.y + block_reduce.y - 1) / block_reduce.y
	);
	CUDA_LAUNCH(
		reduce_kernel,
		grid_reduce,
		block_reduce,
		gpu_calc.data,
		gpu_output.data,
		shape
	);

	CUDA_ERROR_CHECK(cudaPeekAtLastError());

	CUDA_ERROR_CHECK(cudaDeviceSynchronize());

	gpu_output.Download(ao);
}