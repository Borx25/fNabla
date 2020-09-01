#include "fNablaEngine.h"

//DLL Entry point
BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved){
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		// Initialize once for each new process.
		// Return FALSE to fail DLL load.  
		break;
	case DLL_THREAD_ATTACH:
		// Do thread-specific initialization.
		break;
	case DLL_THREAD_DETACH:
		// Do thread-specific cleanup.       
		break;
	case DLL_PROCESS_DETACH:
		// Perform any necessary cleanup.
		break;
	}
	return TRUE;
}

std::tuple<double, double> fNablaEngine::GetAlphaBeta(const int CV_Depth, const double lower, const double upper, const bool inverse) {
	double input_lower = 0.0;
	double input_upper = 1.0;
	switch (CV_Depth) {
	case CV_8U:
		input_upper = 255.0;
		break;
	case CV_16U:
		input_upper = 65535.0;
		break;
	}
	double alpha, beta;
	if (!inverse) {
		alpha = (upper - lower) / (input_upper - input_lower);
		beta = upper - input_upper * alpha;
	}
	else {
		alpha = (input_upper - input_lower) / (upper - lower);
		beta = (input_lower * upper - input_upper * lower) / (upper - lower);
	}
	return { alpha, beta };
}

//MESHMAP
void fNablaEngine::MeshMap::AllocateMat(const cv::Size shape) {
	Mat = cv::Mat(shape, Type);
}

void fNablaEngine::MeshMap::Import(const cv::Mat& input, const double factor){
	auto [alpha, beta] = GetAlphaBeta(input.depth(), RangeLower, RangeUpper, false);
	input.convertTo(Mat, Type, alpha, beta);
	if (factor != 1.0) {
		cv::resize(Mat, Mat, cv::Size(), factor, factor, cv::INTER_AREA);
	}
}

cv::Mat fNablaEngine::MeshMap::Export(const int depth, const bool postprocess, const double factor) {
	auto [alpha, beta] = GetAlphaBeta(depth, RangeLower, RangeUpper, true);
	cv::Mat output = (postprocess ? Postprocess() : Mat.clone());
	if (factor != 1.0) {
		cv::resize(output, output, cv::Size(), factor, factor, cv::INTER_AREA);
	}
	output.convertTo(output, depth, alpha, beta);
	return output;
}

void fNablaEngine::MeshMap::Normalize() {
	cv::normalize(Mat, Mat, RangeLower, RangeUpper, cv::NORM_MINMAX);
}

cv::Mat fNablaEngine::MeshMap::Postprocess() {
	return Mat.clone();
}


//SURFACE MAPS

cv::Mat fNablaEngine::SurfaceMap::AllocateSpectrum(cv::Size shape) {
	return cv::Mat_<std::complex<double>>(shape, std::complex<double>(0.0, 0.0));
}

void fNablaEngine::SurfaceMap::CalculateSpectrum(cv::Mat& Spectrum) {
	int fromTo[] = { 0, 0, -1, 1 };
	cv::mixChannels(&Mat, 1, &Spectrum, 1, fromTo, 2);
	fft2(Spectrum);
}

void fNablaEngine::SurfaceMap::ReconstructFromSpectrum(cv::Mat& Spectrum) {
	ifft2(Spectrum);
	int fromTo[] = { 0, 0 };
	cv::mixChannels(&Spectrum, 1, &Mat, 1, fromTo, 1);
}

void fNablaEngine::SurfaceMap::ComputeSpectrums(
	std::array<cv::Mat, 3>& spectrums,
	const cv::Size shape,
	const int process_flags,
	Config& config,
	const double scale_factor = 1.0
) {
	const double aspect_ratio = double(shape.height) / double(shape.width);
	const double inv_aspect_ratio = double(shape.width) / double(shape.height);
	const double effective_scale_factor = _4_PI_PI / (scale_factor * scale_factor);
	const double effective_high_pass = 1.0 / (config.integration_window.Get() * effective_scale_factor);

	const cv::Point2d sigma_integration(effective_high_pass * inv_aspect_ratio, effective_high_pass * aspect_ratio);

	spectrums[(process_flags & INPUT_MASK) >> (NUM_OUTPUTS + 1)].forEach<std::complex<double>>([&](std::complex<double>& input, const int* pos) {
		const double x = (double)pos[1] / (double)shape.width;
		const double y = (double)pos[0] / (double)shape.height;
		//2 * pi * (x - floor(2x)) + i * 2 * pi * (y - floor(2y)), x = 0..1, y = 0..1
		const double ramp_x = _2_PI * (x - floor(2 * x));
		const double ramp_y = _2_PI * (y - floor(2 * y));
		std::complex<double> ramp(ramp_x, ramp_y);

		//8.0 * cos^2(0.5 * 2*pi*(y-floor(2y))) * sin(2*pi*(x-floor(2x))) + i * 8.0 * cos^2(0.5 * 2*pi*(x-floor(2x))) * sin(2*pi*(y-floor(2y))), x=0..1, y=0..1
		std::complex<double> smooth_operator(
			8.0 * pow(cos(0.5 * ramp_y), 2.0) * sin(ramp_x),
			8.0 * pow(cos(0.5 * ramp_x), 2.0) * sin(ramp_y)
		);

		const double integration_window = 1.0 - exp(-(ramp_x * ramp_x * sigma_integration.x + ramp_y * ramp_y * sigma_integration.y));
		const double sq_freq = norm(ramp);

		if (sq_freq != 0.0) {
			if (process_flags & OUTPUT_DISPLACEMENT) {
				std::complex<double>& h = spectrums[DISPLACEMENT].at<std::complex<double>>(pos);
				if (process_flags & INPUT_NORMAL) {
					h = input / ramp;
				}
				else if (process_flags & INPUT_CURVATURE) {
					h = input / sq_freq;
				}
				h *= integration_window;
			}

			if (process_flags & OUTPUT_NORMAL) {
				std::complex<double>& n = spectrums[NORMAL].at<std::complex<double>>(pos);
				if (process_flags & INPUT_DISPLACEMENT) {
					n = input * ramp;
				}
				else if (process_flags & INPUT_CURVATURE) {
					n = (input / conj(ramp)) * integration_window;
				}
			}

			if (process_flags & OUTPUT_CURVATURE) {
				std::complex<double>& c = spectrums[CURVATURE].at<std::complex<double>>(pos);
				if (process_flags & INPUT_DISPLACEMENT) {
					c = input * norm(smooth_operator);
				}
				else if (process_flags & INPUT_NORMAL) {
					c = input * conj(smooth_operator);
				}
			}
		}
	});
}

void fNablaEngine::fft2(cv::Mat& input, bool inverse) {
	DFTI_DESCRIPTOR_HANDLE descriptor = NULL;
	int status;
	int dim_sizes[2] = { input.rows, input.cols };

	status = DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim_sizes); //complex doubles, 2D
	if (inverse)
	{
		status = DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / (dim_sizes[0] * dim_sizes[1])); //Scale down the output
		status = DftiCommitDescriptor(descriptor);
		status = DftiComputeBackward(descriptor, input.data);
	}
	else
	{
		status = DftiCommitDescriptor(descriptor);
		status = DftiComputeForward(descriptor, input.data);
	}
	status = DftiFreeDescriptor(&descriptor);
}

void fNablaEngine::ifft2(cv::Mat& input) {
	fft2(input, true);
}

void fNablaEngine::Compute(MeshMapArray& Maps, int process_flags, Config& config, double scale_factor) {
	const int input_index = (process_flags & INPUT_MASK) >> (NUM_OUTPUTS + 1);
	const cv::Size shape = Maps[input_index]->Mat.size();

	if (process_flags & OUTPUT_AO) //make sure we are computing or already have as input a heightmap and normalmap
	{
		if (input_index != NORMAL) {
			process_flags = process_flags | (OUTPUT_NORMAL);
		}
		if (input_index != DISPLACEMENT) {
			process_flags = process_flags | (OUTPUT_DISPLACEMENT);
		}
	}

	Maps[input_index]->Normalize();

	for (int i = 0; i < NUM_OUTPUTS; i++) {
		if ((i != input_index) && (process_flags & (1 << i))) {
			Maps[i]->AllocateMat(shape);
		}
	}

	if (process_flags & OUTPUT_SURFACEMAPS)
	{
		std::array<cv::Mat, 3> spectrums = {
			((process_flags & OUTPUT_DISPLACEMENT) || (process_flags & INPUT_DISPLACEMENT) ? SurfaceMap::AllocateSpectrum(shape) : cv::Mat()),
			((process_flags & OUTPUT_NORMAL) || (process_flags & INPUT_NORMAL) ? SurfaceMap::AllocateSpectrum(shape) : cv::Mat()),
			((process_flags & OUTPUT_CURVATURE) || (process_flags & INPUT_CURVATURE) ? SurfaceMap::AllocateSpectrum(shape) : cv::Mat()),
		};

		dynamic_cast<SurfaceMap*>(Maps[input_index].get())->CalculateSpectrum(spectrums[input_index]);

		SurfaceMap::ComputeSpectrums(
			spectrums,
			shape,
			process_flags,
			config,
			scale_factor
		);

		for (int i = 0; i < 3; i++) {
			if ((process_flags & (1 << i) && i != input_index))
			{
				dynamic_cast<SurfaceMap*>(Maps[i].get())->ReconstructFromSpectrum(spectrums[i]);
				Maps[i]->Normalize();
			}
		}
	}
	if (process_flags & OUTPUT_AO)
	{
		//ojo, AO requiere normal y displacement, one of them could be input or none.
		dynamic_cast<AmbientOcclusionMap*>(Maps[AO].get())->Compute(Maps[DISPLACEMENT], Maps[NORMAL]);
	}
}
//DISPLACEMENT

cv::Mat fNablaEngine::DisplacementMap::Postprocess() {
	int current_colormap = config.displacement_colormap.Get();
	if (current_colormap == GRAYSCALE)
	{
		return Mat.clone();
	}
	else {
		cv::Mat output(Mat.rows, Mat.cols, CV_64FC3);

		switch (current_colormap)
		{
		case VIRIDIS:
			output.forEach<cv::Point3d>([&](cv::Point3d& display, const int* pos) -> void {
				const double grayscale = Mat.at<double>(pos);
				display = Viridis::at(grayscale);
				});
			break;
		case MAGMA:
			output.forEach<cv::Point3d>([&](cv::Point3d& display, const int* pos) -> void {
				const double grayscale = Mat.at<double>(pos);
				display = Magma::at(grayscale);
				});
			break;
		}
		return output;
	}
}

//NORMAL

void fNablaEngine::NormalMap::Normalize() {
	Mat.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
		double norm = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
		if (norm != 0.0) {
			p.x /= norm;
			p.y /= norm;
			p.z /= norm;
		}
		else {
			p.x = 0.0;
			p.y = 0.0;
			p.z = 0.0;
		}
	});
}

void fNablaEngine::NormalMap::ReconstructZComponent() {
	Mat.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
		p.x = 1.0 - sqrt(p.y * p.y + p.z * p.z);
	});
	Normalize();
}


void fNablaEngine::NormalMap::CalculateSpectrum(cv::Mat& Spectrum) {
	int fromTo[] = { 1, 0, 2, 1 };
	cv::mixChannels(&Mat, 1, &Spectrum, 1, fromTo, 2);
	cv::multiply(Spectrum, config.normal_swizzle.Get(), Spectrum);
	fft2(Spectrum);
}

void fNablaEngine::NormalMap::ReconstructFromSpectrum(cv::Mat& Spectrum) {
	ifft2(Spectrum);
	cv::multiply(Spectrum, config.normal_swizzle.Get(), Spectrum);
	cv::Mat planes[2];
	cv::split(Spectrum, planes);

	double min, max;
	cv::minMaxLoc(planes[0], &min, &max);
	planes[0] /= std::max(abs(min), abs(max));

	cv::minMaxLoc(planes[1], &min, &max);
	planes[1] /= std::max(abs(min), abs(max));

	int build_normal[] = { 1, 2, 0, 1, -1, 0 }; //XY->BGR
	cv::mixChannels(planes, 2, &Mat, 1, build_normal, 3);

	Mat.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
		p.y = p.y;
		p.z = p.z;
		p.x = 1.0 / sqrt(p.x * p.x + p.y * p.y + 1.0);
		p.y *= p.x;
		p.z *= p.x;
		});
}

cv::Mat fNablaEngine::NormalMap::Postprocess() {
	cv::Mat output = Mat.clone();
	//derivative
	//output.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
	//	p.z = -p.z / p.x;
	//	p.y = -p.y / p.x;
	//	p.x = -1.0;
	//});
	//normal pero el blue es demasiado claro. Revisar las normals con el pdf de care of normal vectors
	output.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
		p.y = p.y * config.normal_scale.Get();
		p.z = p.z * config.normal_scale.Get();
		p.x = 1.0 / sqrt(p.x * p.x + p.y * p.y + 1.0);
		p.y *= p.x;
		p.z *= p.x;
		});
	return output;
}

//CURVATURE

void fNablaEngine::CurvatureMap::Normalize() {
	double min, max;
	cv::minMaxLoc(Mat, &min, &max);
	Mat /= std::max(abs(min), max);
}

cv::Mat fNablaEngine::CurvatureMap::Postprocess() {
	int current_mode = config.curvature_mode.Get();
	if (current_mode == CURVATURE_SPLIT)
	{
		cv::Mat output(Mat.rows, Mat.cols, CV_64FC3);
		output.forEach<cv::Point3d>([&](cv::Point3d& curv, const int* pos) -> void {
			const double curv_gray = tanh(Mat.at<double>(pos) * config.curvature_scale.Get());
			curv.z = (curv_gray > 0.0 ? curv_gray : 0.0) * 2.0 - 1.0;
			curv.y = (curv_gray < 0.0 ? -curv_gray : 0.0) * 2.0 - 1.0;
			curv.x = -1.0;
			});
		return output;
	}
	else
	{
		cv::Mat output = Mat.clone();
		output.forEach<double>([&](double& curv, const int* pos) -> void {
			curv = tanh(curv * config.curvature_scale.Get());
			});
		switch (current_mode)
		{
		case CURVATURE_CONVEXITY:
			output.forEach<double>([&](double& curv, const int* pos) -> void {
				curv = (curv > 0.0 ? curv : 0.0) * 2.0 - 1.0;
				});
			break;
		case CURVATURE_CONCAVITY:
			output.forEach<double>([&](double& curv, const int* pos) -> void {
				curv = (curv < 0.0 ? -curv : 0.0) * 2.0 - 1.0;
				});
			break;
		}
		return output;
	}
}

//AO

cv::Mat fNablaEngine::AmbientOcclusionMap::Postprocess() {
	cv::Mat output = Mat.clone();
	output.forEach<double>([&](double& occ, const int* pos) -> void {
		occ = 1.0 - pow(1.0 - tanh(occ), config.ao_power.Get());
	});
	return output;
}

void fNablaEngine::AmbientOcclusionMap::Compute(std::shared_ptr<fNablaEngine::MeshMap> displacement, std::shared_ptr<fNablaEngine::MeshMap> normal) {
	fNablaEngineCuda::ComputeAOCuda(
		displacement->Mat,
		normal->Mat,
		Mat,
		config.ao_samples.Get(),
		config.ao_distance.Get(),
		config.ao_scale.Get()
	);
}

//UTILITIES

cv::Mat fNablaEngine::fftshift(const cv::Mat& input) {
	cv::Mat output = input(cv::Rect(0, 0, input.cols & -2, input.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = output.cols / 2;
	int cy = output.rows / 2;

	cv::Mat q0(output, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(output, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(output, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(output, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	return output;
}

#ifdef _DEBUG
void fNablaEngine::image_show(cv::Mat& input, const std::string& name, bool fft) {
	std::string full_name;
	cv::Mat output;
	if (fft) {
		cv::Mat planes[2];
		cv::split(input, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

		cv::magnitude(planes[0], planes[1], output);

		output += cv::Scalar::all(1);                    // switch to logarithmic scale
		cv::log(output, output);

		output = fftshift(output);

		normalize(output, output, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
		full_name = name + " (FFT)";
		cv::namedWindow(full_name, cv::WINDOW_NORMAL);
		cv::resizeWindow(full_name, 512, 512);
		cv::imshow(full_name, output);
	}
	else {
		switch (input.channels()) {
		case 1:
			cv::normalize(input, output, 0, 1, cv::NORM_MINMAX);
			full_name = name + " (Grayscale)";
			cv::namedWindow(full_name, cv::WINDOW_NORMAL);
			cv::resizeWindow(full_name, 512, 512);
			cv::imshow(full_name, output);
			cv::waitKey();
			break;
		case 3:
			cv::normalize(input, output, 0, 1, cv::NORM_MINMAX);
			full_name = name + " (RGB)";
			cv::namedWindow(full_name, cv::WINDOW_NORMAL);
			cv::resizeWindow(full_name, 512, 512);
			cv::imshow(full_name, output);
			cv::waitKey();
			break;
		default:
			std::vector<cv::Mat> channels(input.channels());
			cv::split(input, channels);
			for (int i = 0; i < input.channels(); i++) {
				cv::normalize(channels[i], channels[i], 0, 1, cv::NORM_MINMAX);
				std::string full_name = name + " (" + std::to_string(i) + ")";
				cv::namedWindow(full_name, cv::WINDOW_NORMAL);
				cv::resizeWindow(full_name, 512, 512);
				cv::imshow(full_name, channels[i]);
			}
			cv::waitKey();
		}
	}
}
#endif