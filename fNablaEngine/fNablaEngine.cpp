#include "fNablaEngine.h"

//DLL Entry point
BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
	switch (ul_reason_for_call) {
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
	} else {
		alpha = (input_upper - input_lower) / (upper - lower);
		beta = (input_lower * upper - input_upper * lower) / (upper - lower);
	}
	return {alpha, beta};
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
	if (Mat.empty()) {
		//return an error code, we'd need to take a mat reference to which to export
	}
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

void fNablaEngine::ExecuteConversion(MeshMapArray& Maps, Configuration& configuration, Descriptor& descriptor, double scale_factor) {
	if ((descriptor.Input >= NUM_INPUTS) || (descriptor.Input < 0) || (descriptor.Output.none())) {
		//invalid descriptor, return an error code that needs to be caught by client
		return;
	}
	const cv::Size shape = Maps[descriptor.Input]->Mat.size();

	Descriptor final_descriptor(descriptor);


	if (final_descriptor.Output[AO]) //handle dependencies, those that need to be computed but not in the original descriptor will be then discarded
	{
		final_descriptor.Output.set(DISPLACEMENT, final_descriptor.Input != DISPLACEMENT);
		final_descriptor.Output.set(NORMAL, final_descriptor.Input != NORMAL);
	}

	Maps[final_descriptor.Input]->Normalize();

	for (int i = 0; i < NUM_OUTPUTS; i++) {
		if ((final_descriptor.Input != i) && (final_descriptor.Output[i])) {
			Maps[i]->AllocateMat(shape);
		}
	}

	std::array<cv::Mat, 3> spectrums = {
		(((final_descriptor.Output[DISPLACEMENT]) || (final_descriptor.Input == DISPLACEMENT)) ? SurfaceMap::AllocateSpectrum(shape) : cv::Mat()),
		(((final_descriptor.Output[NORMAL]) || (final_descriptor.Input == NORMAL)) ? SurfaceMap::AllocateSpectrum(shape) : cv::Mat()),
		(((final_descriptor.Output[CURVATURE]) || (final_descriptor.Input == CURVATURE)) ? SurfaceMap::AllocateSpectrum(shape) : cv::Mat()),
	};

	dynamic_cast<SurfaceMap*>(Maps[final_descriptor.Input].get())->CalculateSpectrum(spectrums[final_descriptor.Input]);

	SurfaceMap::ComputeSpectrums(
		spectrums,
		shape,
		configuration,
		final_descriptor,
		scale_factor
	);

	for (int i = 0; i < 3; i++) {
		if ((final_descriptor.Input != i) && (final_descriptor.Output[i])) {
			dynamic_cast<SurfaceMap*>(Maps[i].get())->ReconstructFromSpectrum(spectrums[i]);
			Maps[i]->Normalize();
		}
	}
	if (final_descriptor.Output[AO]) {
		fNablaEngineCuda::ComputeAOCuda(
			Maps[DISPLACEMENT]->Mat,
			Maps[NORMAL]->Mat,
			Maps[AO]->Mat,
			configuration.ao_samples.Get(),
			configuration.ao_distance.Get(),
			configuration.ao_scale.Get()
		);
	}
}

void fNablaEngine::SurfaceMap::ComputeSpectrums(
	std::array<cv::Mat, 3>& spectrums,
	const cv::Size shape,
	Configuration& configuration,
	Descriptor& descriptor,
	const double scale_factor = 1.0
) {
	const double aspect_ratio = double(shape.height) / double(shape.width);
	const double inv_aspect_ratio = double(shape.width) / double(shape.height);
	const double effective_scale_factor = _4_PI_PI / (scale_factor * scale_factor);
	const double effective_high_pass = 1.0 / (configuration.integration_window.Get() * effective_scale_factor);

	const cv::Point2d sigma_integration(effective_high_pass * inv_aspect_ratio, effective_high_pass * aspect_ratio);

	spectrums[descriptor.Input].forEach<std::complex<double>>([&](std::complex<double>& input, const int* pos) {
		const double x = (double)pos[1] / (double)shape.width;
		const double y = (double)pos[0] / (double)shape.height;
		//plot 2 * pi * (x - floor(2x)) + i * 2 * pi * (y - floor(2y)), x = 0..1, y = 0..1
		const double ramp_x = _2_PI * (x - floor(2 * x));
		const double ramp_y = _2_PI * (y - floor(2 * y));
		std::complex<double> ramp(
			ramp_x,
			ramp_y
		);

		//plot 8.0 * cos^2(0.5 * 2*pi*(y-floor(2y))) * sin(2*pi*(x-floor(2x))) + i * 8.0 * cos^2(0.5 * 2*pi*(x-floor(2x))) * sin(2*pi*(y-floor(2y))), x=0..1, y=0..1
		//4.0 * (1 + cos(ramp_y)) * sin(ramp_x) + i * 4.0 * (1 + cos(ramp_y)) * sin(ramp_x)
		std::complex<double> smooth_operator(
			4.0 * (1.0 + cos(ramp_y)) * sin(ramp_x),
			4.0 * (1.0 + cos(ramp_x)) * sin(ramp_y)
		);

		const double integration_window = 1.0 - exp(-(ramp_x * ramp_x * sigma_integration.x + ramp_y * ramp_y * sigma_integration.y));
		const double sq_freq = norm(ramp);

		if (sq_freq != 0.0) {
			if (descriptor.Output[DISPLACEMENT]) {
				std::complex<double>& h = spectrums[DISPLACEMENT].at<std::complex<double>>(pos);
				if (descriptor.Input == NORMAL) {
					h = input / ramp;
				} else if (descriptor.Input == CURVATURE) {
					h = input / sq_freq;
				}
				h *= integration_window;
			}

			if (descriptor.Output[NORMAL]) {
				std::complex<double>& n = spectrums[NORMAL].at<std::complex<double>>(pos);
				if (descriptor.Input == DISPLACEMENT) {
					n = input * ramp;
				} else if (descriptor.Input == CURVATURE) {
					n = (input / conj(ramp)) * integration_window;
				}
			}

			if (descriptor.Output[CURVATURE]) {
				std::complex<double>& c = spectrums[CURVATURE].at<std::complex<double>>(pos);
				if (descriptor.Input == DISPLACEMENT) {
					c = input * norm(smooth_operator);
				} else if (descriptor.Input == NORMAL) {
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

//DISPLACEMENT

cv::Mat fNablaEngine::DisplacementMap::Postprocess() {
	int current_colormap = config.displacement_colormap.Get();
	if (current_colormap == GRAYSCALE) {
		return Mat.clone();
	} else {
		cv::Mat output(Mat.rows, Mat.cols, CV_64FC3);

		switch (current_colormap) {
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
		} else {
			p.x = 0.0;
			p.y = 0.0;
			p.z = 0.0;
		}
	});
}

void fNablaEngine::NormalMap::ReconstructZComponent() {
	Mat.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
		p.x = sqrt(-p.z * p.z - p.y * p.y + 1.0);
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
		p.x = 1.0 / sqrt(1.0 + p.y * p.y + p.z * p.z);
		if (p.x != 0.0) {
			p.y *= p.x;
			p.z *= p.x;
		} else {
			p.x = 1.0;
			p.y = 0.0;
			p.z = 0.0;
		}
	});
	Normalize();
}

cv::Mat fNablaEngine::NormalMap::Postprocess() {
	cv::Mat output = Mat.clone();
	//derivative
	//output.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
	//	p.z = -p.z / p.x;
	//	p.y = -p.y / p.x;
	//	p.x = -1.0;
	//});
	output.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
		//scale x, y
		p.y = p.y * config.normal_scale.Get();
		p.z = p.z * config.normal_scale.Get();
		//normalize
		p.x = 1.0 / sqrt(1.0 + p.y * p.y + p.z * p.z);
		if (p.x != 0.0) {
			p.y *= p.x;
			p.z *= p.x;
		} else {
			p.x = 1.0;
			p.y = 0.0;
			p.z = 0.0;
		}
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

//UTILITIES

cv::Mat fNablaEngine::fftshift(const cv::Mat& input) {
	cv::Mat output = input(cv::Rect(0, 0, input.cols & ~2, input.rows & ~2)); //round down to even integer

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

		output += cv::Scalar::all(1); // switch to logarithmic scale
		cv::log(output, output);

		output = fftshift(output);

		normalize(output, output, 0, 1, cv::NORM_MINMAX);
		full_name = name + " (FFT)";
		cv::namedWindow(full_name, cv::WINDOW_NORMAL);
		cv::resizeWindow(full_name, 512, 512);
		cv::imshow(full_name, output);
	} else {
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