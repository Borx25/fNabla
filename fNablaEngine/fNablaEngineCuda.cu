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
	const unsigned int samples
) {
	const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < samples) {
		double theta = 2.0 * _PI * ((double)id / (double)samples) - 2.0 * _PI_2; // -pi/2 to pi/2
		directions[id].y = sin(theta);
		directions[id].x = cos(theta);
	}
}

__device__ inline double fNablaEngineCuda::IntegrateArc(double l, double r, double n, double sin_n, double cos_n) {
	//simplified formula that's very close including lerp:
	//ao_ij += normal_ij.x * (_PI - l - r) - hypot(NdotSN, normal_ij.x) + 1.0;
	return 0.25 * cos(n + l + l) + 0.25 * cos(n + r + r) + 1.0 * (_PI - l - r) * sin_n + 0.5 * cos_n;
}

__global__ void fNablaEngineCuda::_AO_kernel(
	double* displacement,
	double3* normal,
	double2* directions,
	double* output,
	dim3 shape,
	const double radius,
	const double depth
) {
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if ((x < shape.x) && (y < shape.y) && (z < shape.z)){
		const double displacement_xy = displacement[y * shape.x + x];
		double3 normal_xy = normal[y * shape.x + x];

		//double theta = M_PI * (((double)z) / (double)samples) - M_PI_2; //-pi/2 to pi/2
		const double sin_theta = directions[z].y; // sin(theta);
		const double cos_theta = directions[z].x; // cos(theta);

		//D={ sin(theta), cos(theta), 0 }; V={0, 0, 1}; SN = cross(D, V)
		//N = normal_ij; PN = N - SN * NdotSN = {sin_theta * NdotSN, cos_theta * NdotSN, normal_ij.x}

		double NdotSN = normal_xy.y * cos_theta + normal_xy.z * sin_theta;
		double rproj_length = rhypot(NdotSN, normal_xy.x);
		double n = atan2(normal_xy.x, NdotSN);
		double cos_n = NdotSN * rproj_length;
		double sin_n = normal_xy.x * rproj_length;

		double max_left = 0.0;
		double max_right = 0.0;

		//16 line samples per radial sample; 32 radial => 512 line which for up to 4k with the average of 0.25 distance needed means we are only skipping 1 pixel
		double step_size = radius / (shape.z * 16);
		//for (double length = 1.0; length <= radius; length++) {
		for (double length = step_size; length <= radius; length += step_size) {
			int x_step = lrint(sin_theta * length);
			int y_step = lrint(cos_theta * length);
			int rows = shape.y;
			int cols = shape.x;

			{
				int left_y = int(y) + y_step;
				int left_x = int(x) + x_step;

				//edge repeat
				//int left_i = max(min(left_i, rows-1), 0);
				//int left_j = max(min(left_j, cols-1), 0);
				//mirror
				left_y += rows * ((left_y < 0) - (left_y >= rows));
				left_x += cols * ((left_x < 0) - (left_x >= cols));

				double diff_left = (displacement[left_y * shape.x + left_x] - displacement_xy);

				max_left = fmax(max_left, diff_left / length);
			}

			{
				int right_y = int(y) - y_step;
				int right_x = int(x) - x_step;

				//edge repeat
				//right_i = max(min(right_i, rows-1), 0);
				//right_j = max(min(right_j, cols-1), 0);
				//mirror
				right_y += rows * ((right_y < 0) - (right_y >= rows));
				right_x += cols * ((right_x < 0) - (right_x >= cols));

				double diff_right = (displacement[right_y * shape.x + right_x] - displacement_xy);

				max_right = fmax(max_right, diff_right / length);
			}
		}
		double l = atan(depth * max_left);
		double r = atan(depth * max_right);

		//clamp to normal hemisphere
		l = n + fmin(l - n, _PI_2);
		r = n + fmin(r - n, _PI_2);

		output[y * shape.x * shape.z + x * shape.z + z] = (IntegrateArc(l, r, n, sin_n, cos_n) - 1.0) / rproj_length + 1.0;
	}
}

__global__ void fNablaEngineCuda::reduce_kernel(
	double* input,
	double* output,
	const dim3 shape
) {
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < shape.x) && (y < shape.y)) {
		output[y * shape.x + x] = 0.0;
		for (size_t z = 0; z < shape.z; z++) {
			output[y * shape.x + x] += input[y * shape.x * shape.z + x * shape.z + z] / shape.z;
		}
	}
}

void fNablaEngineCuda::ComputeAOCuda(cv::Mat& displacement, cv::Mat& normal, cv::Mat& ao, const unsigned int samples, const double distance, const double depth) {
	CheckCUDACapable();

	const dim3 shape(ao.cols, ao.rows, samples);
	const double max_size = double(std::min(shape.x, shape.y));

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
		std::max(distance * max_size, 1.0),
		depth * max_size
	);

	CUDA_ERROR_CHECK(cudaPeekAtLastError());

	CUDA_ERROR_CHECK(cudaDeviceSynchronize());


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