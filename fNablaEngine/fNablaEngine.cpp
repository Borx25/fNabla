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

void fNablaEngine::GetAlphaBeta(const int CV_Depth, const double lower, const double upper, double& alpha, double& beta, const bool inverse) {
	double input_lower = 0.0;
	double input_upper = 1.0;
	switch (CV_Depth) {
	case CV_8U:
		input_lower = 0.0;
		input_upper = 255.0;
		break;
	case CV_16U:
		input_lower = 0.0;
		input_upper = 65535.0;
		break;
	}
	if (!inverse) {
		alpha = (upper - lower) / (input_upper - input_lower);
		beta = upper - input_upper * alpha;
	}
	else {
		alpha = (input_upper - input_lower) / (upper - lower);
		beta = (input_lower * upper - input_upper * lower) / (upper - lower);
	}
}

void fNablaEngine::MeshMap::Import(const cv::Mat& input, const double scale_factor){
	double alpha, beta;
	fNablaEngine::GetAlphaBeta(input.depth(), this->RangeLower, this->RangeUpper, alpha, beta, false);
	input.convertTo(this->Mat, this->Type, alpha, beta);
	if (scale_factor != 1.0) {
		cv::resize(this->Mat, this->Mat, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);
	}
}

cv::Mat fNablaEngine::MeshMap::Export(const int depth) {
	cv::Mat output = this->Postprocess();
	double alpha, beta;
	fNablaEngine::GetAlphaBeta(depth, this->RangeLower, this->RangeUpper, alpha, beta, true);
	output.convertTo(output, depth, alpha, beta);
	return output;
}

void fNablaEngine::MeshMap::Normalize() {
	cv::normalize(this->Mat, this->Mat, this->RangeLower, this->RangeUpper, cv::NORM_MINMAX);
}

void fNablaEngine::NormalMap::Normalize() {
	this->Mat.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
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
	this->Mat.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
		p.x = 1.0 - sqrt(p.y * p.y + p.z * p.z);
	});
	this->Normalize();
}

void fNablaEngine::CurvatureMap::Normalize() {
	double min, max;
	cv::minMaxLoc(this->Mat, &min, &max);
	this->Mat /= std::max(abs(min), max);
}

cv::Mat* fNablaEngine::MeshMap::AllocateSpectrum() {
	this->Spectrum = cv::Mat_<std::complex<double>>(this->Mat.rows, this->Mat.cols, std::complex<double>(0.0, 0.0));
	return &this->Spectrum;
}

void fNablaEngine::MeshMap::CalculateSpectrum() {
	int fromTo[] = { 0, 0, -1, 1 };
	cv::mixChannels(&this->Mat, 1, &this->Spectrum, 1, fromTo, 2);
	fNablaEngine::fft2(this->Spectrum);
}

void fNablaEngine::NormalMap::CalculateSpectrum() {
	int fromTo[] = { 1, 0, 2, 1 };
	cv::mixChannels(&this->Mat, 1, &this->Spectrum, 1, fromTo, 2);
	cv::multiply(this->Spectrum, this->swizzle_xy_coordinates, this->Spectrum);
	fNablaEngine::fft2(this->Spectrum);
}

void fNablaEngine::MeshMap::ReconstructFromSpectrum() {
	fNablaEngine::ifft2(this->Spectrum);
	int fromTo[] = { 0, 0 };
	cv::mixChannels(&this->Spectrum, 1, &this->Mat, 1, fromTo, 1);
}

void fNablaEngine::NormalMap::ReconstructFromSpectrum() {
	fNablaEngine::ifft2(this->Spectrum);
	cv::multiply(this->Spectrum, this->swizzle_xy_coordinates, this->Spectrum);
	cv::Mat planes[2];
	cv::split(this->Spectrum, planes);

	double min, max;
	cv::minMaxLoc(planes[0], &min, &max);
	planes[0] /= std::max(abs(min), abs(max));

	cv::minMaxLoc(planes[1], &min, &max);
	planes[1] /= std::max(abs(min), abs(max));

	int build_normal[] = { 1, 2, 0, 1, -1, 0 }; //XY->BGR
	cv::mixChannels(planes, 2, &this->Mat, 1, build_normal, 3);

	this->Mat.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
		p.y = p.y;
		p.z = p.z;
		p.x = 1.0 / sqrt(p.x * p.x + p.y * p.y + 1.0);
		p.y *= p.x;
		p.z *= p.x;
	});
}

void fNablaEngine::CurvatureMap::ReconstructFromSpectrum() {
	MeshMap::ReconstructFromSpectrum();
	cv::Mat duplicate = this->Mat.clone();
	//auto size = cv::Size(sqrt(log2(this->Mat.cols)), sqrt(log2(this->Mat.rows)));
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3), cv::Point(-1, -1));
	cv::morphologyEx(duplicate, duplicate, cv::MORPH_CLOSE, kernel);
	cv::morphologyEx(this->Mat, this->Mat, cv::MORPH_OPEN, kernel);
	this->Mat += duplicate;
}

cv::Mat fNablaEngine::DisplacementMap::Postprocess() {
	if (this->mode == fColormaps::GRAYSCALE)
	{
		return this->Mat.clone();
	}
	else {
		cv::Mat output(this->Mat.rows, this->Mat.cols, CV_64FC3);

		switch (this->mode)
		{
		case fColormaps::VIRIDIS:
			output.forEach<cv::Point3d>([&](cv::Point3d& display, const int* pos) -> void {
				const double grayscale = this->Mat.at<double>(pos);
				display = fColormaps::Viridis::at(grayscale);
			});
			break;
		case fColormaps::MAGMA:
			output.forEach<cv::Point3d>([&](cv::Point3d& display, const int* pos) -> void {
				const double grayscale = this->Mat.at<double>(pos);
				display = fColormaps::Magma::at(grayscale);
			});
			break;
		}
		return output;
	}
}

cv::Mat fNablaEngine::NormalMap::Postprocess() {
	cv::Mat output = this->Mat.clone();
	output.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
		p.z = -p.z / p.x;
		p.y = -p.y / p.x;
		p.x = -1.0;
	});
	//output.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
	//	p.y = p.y * this->scale * 5;
	//	p.z = p.z * this->scale * 5;
	//	p.x = 1.0 / sqrt(p.x * p.x + p.y * p.y + 1.0);
	//	p.y *= p.x;
	//	p.z *= p.x;
	//});
	return output;
}

cv::Mat fNablaEngine::AmbientOcclusionMap::Postprocess() {
	cv::Mat output = this->Mat.clone();
	output.forEach<double>([&](double& occ, const int* pos) -> void {
		occ = 1.0 - pow(1.0 - tanh(occ), this->ao_power);
	});
	return output;
}

cv::Mat fNablaEngine::CurvatureMap::Postprocess() {
	if (this->mode == CURVATURE_SPLIT)
	{
		cv::Mat output(this->Mat.rows, this->Mat.cols, CV_64FC3);
		output.forEach<cv::Point3d>([&](cv::Point3d& curv, const int* pos) -> void {
			const double curv_gray = tanh(this->Mat.at<double>(pos) * this->scale * 5.0);
			curv.z = (curv_gray > 0.0 ? curv_gray : 0.0) * 2.0 - 1.0;
			curv.y = (curv_gray < 0.0 ? -curv_gray : 0.0) * 2.0 - 1.0;
			curv.x = -1.0;
			});
		return output;
	}
	else
	{
		cv::Mat output = this->Mat.clone();
		output.forEach<double>([&](double& curv, const int* pos) -> void {
			curv = tanh(curv * this->scale * 5.0);
		});
		switch (this->mode)
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

void fNablaEngine::ComputeSpectrums(
	cv::Mat* spectrums[3],
	const cv::Size shape,
	const int process_flags,
	const double high_pass,
	const double curvature_sharpness,
	const double scale_factor = 1.0
) {
	const double aspect_ratio = double(shape.height) / double(shape.width);
	const double inv_aspect_ratio = double(shape.width) / double(shape.height);
	const double effective_scale_factor = _4_PI_PI / (scale_factor * scale_factor);
	const double effective_high_pass = 1.0 / (high_pass * effective_scale_factor);
	const double effective_curvature_sharpness = 1.0 / (curvature_sharpness * effective_scale_factor);

	const cv::Point2d sigma_integration(effective_high_pass * inv_aspect_ratio, effective_high_pass * aspect_ratio);
	const cv::Point2d sigma_blur(effective_curvature_sharpness * inv_aspect_ratio, effective_curvature_sharpness * aspect_ratio);

	spectrums[(process_flags & INPUT_MASK) >> (NUM_OUTPUTS + 1)]->forEach<std::complex<double>>([&](std::complex<double>& input, const int* pos) {

		std::complex<double> freq(_2_PI * ((double)pos[1] / (double)shape.width - 2 * pos[1] / shape.width), _2_PI * ((double)pos[0] / (double)shape.height - 2 * pos[0] / shape.height));

		const double sq_freq = norm(freq);

		if (sq_freq != 0.0) {
			if (process_flags & fNablaEngine::OUTPUT_DISPLACEMENT) {
				std::complex<double>& h = spectrums[DISPLACEMENT]->at<std::complex<double>>(pos);
				if (process_flags & fNablaEngine::INPUT_NORMAL) {
					h = input / freq;
				}
				else if (process_flags & fNablaEngine::INPUT_CURVATURE) {
					h = input / sq_freq;
				}
				const double h_window = 1.0 - exp(-(freq.real() * freq.real() * sigma_integration.x + freq.imag() * freq.imag() * sigma_integration.y));
				h *= h_window;
			}

			if (process_flags & fNablaEngine::OUTPUT_NORMAL) {
				std::complex<double>& n = spectrums[NORMAL]->at<std::complex<double>>(pos);
				if (process_flags & fNablaEngine::INPUT_DISPLACEMENT) {
					n = input * freq;
				}
				else if (process_flags & fNablaEngine::INPUT_CURVATURE) {
					n = input / conj(freq);

					const double n_window = 1.0 - exp(-(freq.real() * freq.real() * sigma_integration.x + freq.imag() * freq.imag() * sigma_integration.y));
					n *= n_window;
				}
			}

			if (process_flags & fNablaEngine::OUTPUT_CURVATURE) {
				std::complex<double>& c = spectrums[CURVATURE]->at<std::complex<double>>(pos);
				if (process_flags & fNablaEngine::INPUT_DISPLACEMENT) {
					c = input * sq_freq;
				}
				else if (process_flags & fNablaEngine::INPUT_NORMAL) {
					c = input * conj(freq);
				}

				const double c_window = exp(-(freq.real() * freq.real() * sigma_blur.x + freq.imag() * freq.imag() * sigma_blur.y));

				c *= c_window * c_window;
			}
		}
	});
}


void fNablaEngine::AmbientOcclusionMap::Compute(fNablaEngine::MeshMap* displacement, fNablaEngine::MeshMap* normal) {
	fNablaEngineCuda::ComputeAOCuda(
		displacement->Mat,
		normal->Mat,
		this->Mat,
		this->ao_samples,
		this->ao_distance,
		this->scale
	);
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
	fNablaEngine::fft2(input, true);
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

		// crop the spectrum, if it has an odd number of rows or columns
		output = output(cv::Rect(0, 0, output.cols & -2, output.rows & -2));

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

		normalize(output, output, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
		full_name = name + " (FFT)";
		cv::namedWindow(full_name, cv::WINDOW_NORMAL);
		cv::resizeWindow(full_name, 512, 512);
		cv::imshow(full_name, output);
		cv::waitKey();
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
				cv::waitKey();
			}
		}
	}
}
#endif

//void fNablaEngine::ComputeSpectrums(
//	cv::Mat* displacement_spectrum,
//	cv::Mat* normal_spectrum,
//	cv::Mat* curvature_spectrum,
//	const cv::Size shape,
//	const int process_flags,
//	const double high_pass,
//	const double curvature_sharpness,
//	const double scale_factor = 1.0
//){
//	const double aspect_ratio = double(shape.height) / double(shape.width);
//	const double effective_scale_factor = 1.0 / (scale_factor * scale_factor);
//
//	const cv::Point2d sigma_integration(high_pass * effective_scale_factor * aspect_ratio, high_pass * effective_scale_factor / aspect_ratio);
//
//	const cv::Point2d sigma_blur(curvature_sharpness * effective_scale_factor * aspect_ratio, curvature_sharpness * effective_scale_factor / aspect_ratio);
//
//	displacement_spectrum->forEach<cv::Point2d>([&](cv::Point2d& h, const int* pos){
//
//		const cv::Point2d freq(
//			((double)pos[1] / (double)shape.width - 2 * pos[1] / shape.width),
//			((double)pos[0] / (double)shape.height - 2 * pos[0] / shape.height)
//		);
//
//		const cv::Point2d sqfreq(
//			freq.x * freq.x,
//			freq.y * freq.y
//		);
//
//		const double sqmag = M_2_PI * (sqfreq.x + sqfreq.y);
//
//		cv::Point2d& n = normal_spectrum->at<cv::Point2d>(pos);
//		cv::Point2d& c = curvature_spectrum->at<cv::Point2d>(pos);
//
//		if (sqmag != 0.0) {
//			if (process_flags & fNablaEngine::OUTPUT_DISPLACEMENT) {
//				if (process_flags & fNablaEngine::INPUT_NORMAL) {
//					h.x = (-n.x * freq.y - n.y * freq.x) / sqmag;
//					h.y = (n.x * freq.x - n.y * freq.y) / sqmag;
//				}
//				else if (process_flags & fNablaEngine::INPUT_CURVATURE) {
//					h.x = c.x / (2.0 * sqmag);
//					h.y = c.y / (2.0 * sqmag);
//				}
//				const double h_window = 1.0 - exp(-(sqfreq.x / sigma_integration.x + sqfreq.y / sigma_integration.y));
//
//				h.x *= h_window;
//				h.y *= h_window;
//			}
//			
//			if (process_flags & fNablaEngine::OUTPUT_NORMAL) {
//				if (process_flags & fNablaEngine::INPUT_DISPLACEMENT) {
//					n.x = M_2_PI * (-h.x * freq.y + h.y * freq.x);
//					n.y = M_2_PI * (-h.y * freq.y - h.x * freq.x);
//				}
//				else if (process_flags & fNablaEngine::INPUT_CURVATURE) {
//					n.x = (c.y * freq.x - c.x * freq.y) / sqmag;
//					n.y = (-c.y * freq.y - c.x * freq.x) / sqmag;
//
//					const double h_window = 1.0 - exp(-(sqfreq.x / sigma_integration.x + sqfreq.y / sigma_integration.y));
//
//					n.x *= h_window;
//					n.y *= h_window;
//				}
//			}
//			
//			if (process_flags & fNablaEngine::OUTPUT_CURVATURE) {
//				if (process_flags & fNablaEngine::INPUT_DISPLACEMENT) {
//					c.x = M_2_PI * h.x * sqmag;
//					c.y = M_2_PI * h.y * sqmag;
//				}
//				else if (process_flags & fNablaEngine::INPUT_NORMAL) {
//					c.x = M_2_PI * (-n.x * freq.y - n.y * freq.x);
//					c.y = M_2_PI * (-n.y * freq.y + n.x * freq.x);
//				}
//
//				const double c_window = exp(-(sqfreq.x / sigma_blur.x + sqfreq.y / sigma_blur.y));
//
//				c.x *= c_window;
//				c.y *= c_window;
//			}
//		}
//	});
//}

//void fNablaEngine::DisplacementMap::CompleteSpectrumTriad(
//	fNablaEngine::MeshMap* displacement,
//	fNablaEngine::MeshMap* normal,
//	fNablaEngine::MeshMap* curvature) {
//	cv::Size shape = cv::Size(this->Mat.cols, this->Mat.rows);
//	const double high_pass = dynamic_cast<fNablaEngine::DisplacementMap*>(displacement)->integration_window;
//	const double curvature_sharpness = dynamic_cast<fNablaEngine::CurvatureMap*>(curvature)->curvature_sharpness;
//
//	double scale_factor = (1024.0 * 1024.0) / (double(shape.height) * double(shape.width));
//	double aspect_ratio = double(shape.height) / double(shape.width);
//
//	double h_sigma_x = high_pass * aspect_ratio * scale_factor;
//	double h_sigma_y = high_pass * 1.0 / aspect_ratio * scale_factor;
//	double c_sigma_x = curvature_sharpness * aspect_ratio * scale_factor;
//	double c_sigma_y = curvature_sharpness * 1.0 / aspect_ratio * scale_factor;
//
//	this->Spectrum.forEach<cv::Point2d>([&](cv::Point2d& h, const int* pos) -> void { //Captures: [&]  significa que cualquiera que referencie se haga via reference no value ([=])
//		cv::Point2d& n = normal->Spectrum.at<cv::Point2d>(pos);
//		cv::Point2d& c = curvature->Spectrum.at<cv::Point2d>(pos);
//
//		double freq_y = ((double)pos[0] / (double)shape.height - floor(2.0 * (double)pos[0] / (double)shape.height));
//		double freq_x = ((double)pos[1] / (double)shape.width - floor(2.0 * (double)pos[1] / (double)shape.width));
//
//		double h_window = 1.0 - exp(-(freq_x * freq_x / h_sigma_x + freq_y * freq_y / h_sigma_y));
//
//		double c_window = exp(-(freq_x * freq_x / c_sigma_x + freq_y * freq_y / c_sigma_y));
//
//		h.x *= h_window;
//		h.y *= h_window;
//
//		//N = H * -2πF
//		n.x = -M_2_PI * (h.x * freq_y - h.y * freq_x);
//		n.y = -M_2_PI * (h.y * freq_y + h.x * freq_x);
//
//		//C = H * (-2πF)*(-2πF*) = H * 4π^2*|F|^2
//		double squared_sum = freq_y * freq_y + freq_x * freq_x;
//
//		c.x = M_4_PI_PI * h.x * squared_sum;
//		c.y = M_4_PI_PI * h.y * squared_sum;
//
//		c.x *= c_window;
//		c.y *= c_window;
//	});
//}
//
//void fNablaEngine::NormalMap::CompleteSpectrumTriad(
//	fNablaEngine::MeshMap* displacement,
//	fNablaEngine::MeshMap* normal,
//	fNablaEngine::MeshMap* curvature) {
//	cv::Size shape = cv::Size(this->Mat.cols, this->Mat.rows);
//	const double curvature_sharpness = dynamic_cast<fNablaEngine::CurvatureMap*>(curvature)->curvature_sharpness;
//	const double high_pass = dynamic_cast<fNablaEngine::DisplacementMap*>(displacement)->integration_window;
//
//	double scale_factor = (1024.0 * 1024.0) / (double(shape.height) * double(shape.width));
//	double aspect_ratio = double(shape.height) / double(shape.width);
//
//	double h_sigma_x = high_pass * aspect_ratio * scale_factor;
//	double h_sigma_y = high_pass * 1.0 / aspect_ratio * scale_factor;
//	double c_sigma_x = curvature_sharpness * aspect_ratio * scale_factor;
//	double c_sigma_y = curvature_sharpness * 1.0 / aspect_ratio * scale_factor;
//
//	this->Spectrum.forEach<cv::Point2d>([&](cv::Point2d& n, const int* pos) -> void { //Captures: [&]  significa que cualquiera que referencie se haga via reference no value ([=])
//		cv::Point2d& h = displacement->Spectrum.at<cv::Point2d>(pos);
//		cv::Point2d& c = curvature->Spectrum.at<cv::Point2d>(pos);
//
//		double freq_y = ((double)pos[0] / (double)shape.height - floor(2.0 * (double)pos[0] / (double)shape.height));
//		double freq_x = ((double)pos[1] / (double)shape.width - floor(2.0 * (double)pos[1] / (double)shape.width));
//
//		double h_window = 1.0 - exp(-(freq_x * freq_x / h_sigma_x + freq_y * freq_y / h_sigma_y));
//
//		double c_window = exp(-(freq_x * freq_x / c_sigma_x + freq_y * freq_y / c_sigma_y));
//
//		//C = N * -2πF* //F* es el complex conjugate de F, es decir freq_x - freq_y*i
//
//		c.x = -M_2_PI * (n.x * freq_y + n.y * freq_x);
//		c.y = -M_2_PI * (n.y * freq_y - n.x * freq_x);
//
//		//H = N * 1/(-2πF)
//
//		double denominator = M_2_PI * (freq_y * freq_y + freq_x * freq_x);
//		if (denominator != 0.0) {
//			h.x = ((-n.x * freq_y - n.y * freq_x) / denominator) * h_window;
//			h.y = ((n.x * freq_x - n.y * freq_y) / denominator) * h_window;
//		}
//		else {
//			h.x = h.y = 0.0;
//		}
//
//
//		c.x *= c_window;
//		c.y *= c_window;
//	});
//}
//
//void fNablaEngine::CurvatureMap::CompleteSpectrumTriad(
//	fNablaEngine::MeshMap* displacement,
//	fNablaEngine::MeshMap* normal,
//	fNablaEngine::MeshMap* curvature) {
//	cv::Size shape = cv::Size(this->Mat.cols, this->Mat.rows);
//	const double high_pass = dynamic_cast<fNablaEngine::DisplacementMap*>(displacement)->integration_window;
//	const double curvature_sharpness = dynamic_cast<fNablaEngine::CurvatureMap*>(curvature)->curvature_sharpness;
//
//	double scale_factor = (1024.0 * 1024.0) / (double(shape.height) * double(shape.width));
//	double aspect_ratio = double(shape.height) / double(shape.width);
//
//	double h_sigma_x = high_pass * aspect_ratio * scale_factor;
//	double h_sigma_y = high_pass * 1.0/aspect_ratio * scale_factor;
//	double c_sigma_x = curvature_sharpness * aspect_ratio * scale_factor;
//	double c_sigma_y = curvature_sharpness * 1.0/aspect_ratio * scale_factor;
//
//	this->Spectrum.forEach<cv::Point2d>([&](cv::Point2d& c, const int* pos) -> void { //Captures: [&]  significa que cualquiera que referencie se haga via reference no value ([=])
//		cv::Point2d& h = displacement->Spectrum.at<cv::Point2d>(pos);
//		cv::Point2d& n = normal->Spectrum.at<cv::Point2d>(pos);
//
//		double freq_y = ((double)pos[0] / (double)shape.height - floor(2.0 * (double)pos[0] / (double)shape.height));
//		double freq_x = ((double)pos[1] / (double)shape.width - floor(2.0 * (double)pos[1] / (double)shape.width));
//
//		double h_window = 1.0 - exp(-(freq_x * freq_x / h_sigma_x + freq_y * freq_y / h_sigma_y));
//
//		double c_window = exp(-(freq_x * freq_x / c_sigma_x + freq_y * freq_y / c_sigma_y));
//
//		double denominator = M_2_PI * (freq_y * freq_y + freq_x * freq_x);
//		if (denominator != 0.0) {
//			//N = C * 1/(-2πF*)
//			n.x = ((c.y * freq_x - c.x * freq_y) / denominator) * h_window;
//			n.y = -((c.y * freq_y + c.x * freq_x) / denominator) * h_window;
//			//H = C * 1/(-2πF*) * 1/(-2πF) = C * 1/4π^2|F|^2
//			h.x = (c.x / (2.0 * denominator)) * h_window;
//			h.y = (c.y / (2.0 * denominator)) * h_window;
//		}
//		else {
//			n.x = 0.0;
//			n.y = 0.0;
//			h.x = 0.0;
//			h.y = 0.0;
//		}
//
//		c.x *= c_window;
//		c.y *= c_window;
//	});
//}

//void fNablaEngine::HeightToNormal(cv::Mat& displacement, cv::Mat& normal) {
//	cv::Size shape = cv::Size(displacement.cols, displacement.rows);
//	displacement.forEach<cv::Point2d>([&](cv::Point2d& h, const int* pos) -> void {
//		cv::Point2d& n = normal.at<cv::Point2d>(pos);
//
//		double freq_y = ((double)pos[0] / (double)shape.height - floor(2.0 * (double)pos[0] / (double)shape.height));
//		double freq_x = ((double)pos[1] / (double)shape.width - floor(2.0 * (double)pos[1] / (double)shape.width));
//
//		n.x = -PI_DOUBLE * (h.x * freq_y - h.y * freq_x);
//		n.y = -PI_DOUBLE * (h.y * freq_y + h.x * freq_x);
//	});
//}
//
//void fNablaEngine::HeightToCurvature(cv::Mat& displacement, cv::Mat& curvature, double sharpness) {
//	cv::Size shape = cv::Size(displacement.cols, displacement.rows);
//	displacement.forEach<cv::Point2d>([&](cv::Point2d& h, const int* pos) -> void {
//		cv::Point2d& c = curvature.at<cv::Point2d>(pos);
//
//		double freq_y = ((double)pos[0] / (double)shape.height - floor(2.0 * (double)pos[0] / (double)shape.height));
//		double freq_x = ((double)pos[1] / (double)shape.width - floor(2.0 * (double)pos[1] / (double)shape.width));
//
//		double squared_sum = freq_y * freq_y + freq_x * freq_x;
//
//		double c_sigma_x = sharpness * (shape.height / shape.width);
//		double c_sigma_y = sharpness * (shape.width / shape.height);
//		double c_window = exp(-(freq_x * freq_x / c_sigma_x + freq_y * freq_y / c_sigma_y));
//
//		c.x = PI_SQUARED_FOUR * h.x * squared_sum * c_window;
//		c.y = PI_SQUARED_FOUR * h.y * squared_sum * c_window;
//	});
//}
//
//void fNablaEngine::NormalToDisplacement(cv::Mat& normal, cv::Mat& displacement, double high_pass) {
//	cv::Size shape = cv::Size(normal.cols, normal.rows);
//	normal.forEach<cv::Point2d>([&](cv::Point2d& n, const int* pos) -> void {
//		cv::Point2d& h = displacement.at<cv::Point2d>(pos);
//
//		double freq_y = ((double)pos[0] / (double)shape.height - floor(2.0 * (double)pos[0] / (double)shape.height));
//		double freq_x = ((double)pos[1] / (double)shape.width - floor(2.0 * (double)pos[1] / (double)shape.width));
//
//		double h_sigma_x = high_pass * high_pass * (shape.height / shape.width);
//		double h_sigma_y = high_pass * high_pass * (shape.width / shape.height);
//		double h_window = 1.0 - exp(-(freq_x * freq_x / h_sigma_x + freq_y * freq_y / h_sigma_y));
//
//		double denominator = PI_DOUBLE * (freq_y * freq_y + freq_x * freq_x);
//		if (denominator != 0.0) {
//			h.x = ((-n.x * freq_y - n.y * freq_x) / denominator) * h_window;
//			h.y = ((n.x * freq_x - n.y * freq_y) / denominator) * h_window;
//		}
//		else {
//			h.x = h.y = 0.0;
//		}
//	});
//}
//
//void fNablaEngine::NormalToCurvature(cv::Mat& normal, cv::Mat& curvature, double sharpness) {
//	cv::Size shape = cv::Size(normal.cols, normal.rows);
//	normal.forEach<cv::Point2d>([&](cv::Point2d& n, const int* pos) -> void {
//		cv::Point2d& c = curvature.at<cv::Point2d>(pos);
//
//		double freq_y = ((double)pos[0] / (double)shape.height - floor(2.0 * (double)pos[0] / (double)shape.height));
//		double freq_x = ((double)pos[1] / (double)shape.width - floor(2.0 * (double)pos[1] / (double)shape.width));
//
//		double c_sigma_x = sharpness * (shape.height / shape.width);
//		double c_sigma_y = sharpness * (shape.width / shape.height);
//		double c_window = exp(-(freq_x * freq_x / c_sigma_x + freq_y * freq_y / c_sigma_y));
//
//		c.x = -PI_DOUBLE * (n.x * freq_y + n.y * freq_x) * c_window;
//		c.y = -PI_DOUBLE * (n.y * freq_y - n.x * freq_x) * c_window;
//	});
//}
//
//void fNablaEngine::CurvatureToDisplacement(cv::Mat& curvature, cv::Mat& displacement, double high_pass) {
//	cv::Size shape = cv::Size(curvature.cols, curvature.rows);
//	curvature.forEach<cv::Point2d>([&](cv::Point2d& c, const int* pos) -> void {
//		cv::Point2d& h = displacement.at<cv::Point2d>(pos);
//
//		double freq_y = ((double)pos[0] / (double)shape.height - floor(2.0 * (double)pos[0] / (double)shape.height));
//		double freq_x = ((double)pos[1] / (double)shape.width - floor(2.0 * (double)pos[1] / (double)shape.width));
//
//		double s_sigma_x = high_pass * high_pass * (shape.height / shape.width);
//		double s_sigma_y = high_pass * high_pass * (shape.width / shape.height);
//		double s_window = 1.0 - exp(-(freq_x * freq_x / s_sigma_x + freq_y * freq_y / s_sigma_y));
//
//		double denominator = PI_DOUBLE * (freq_y * freq_y + freq_x * freq_x);
//		if (denominator != 0.0) {
//			h.x = (c.x / (2.0 * denominator)) * s_window;
//			h.y = (c.y / (2.0 * denominator)) * s_window;
//		}
//		else {
//			h.x = 0.0;
//			h.y = 0.0;
//		}
//	});
//}
//
//void fNablaEngine::CurvatureToNormal(cv::Mat& curvature, cv::Mat& normal, double high_pass) {
//	cv::Size shape = cv::Size(curvature.cols, curvature.rows);
//	curvature.forEach<cv::Point2d>([&](cv::Point2d& c, const int* pos) -> void {
//		cv::Point2d& n = normal.at<cv::Point2d>(pos);
//
//		double freq_y = ((double)pos[0] / (double)shape.height - floor(2.0 * (double)pos[0] / (double)shape.height));
//		double freq_x = ((double)pos[1] / (double)shape.width - floor(2.0 * (double)pos[1] / (double)shape.width));
//
//		double s_sigma_x = high_pass * high_pass * (shape.height / shape.width);
//		double s_sigma_y = high_pass * high_pass * (shape.width / shape.height);
//		double s_window = 1.0 - exp(-(freq_x * freq_x / s_sigma_x + freq_y * freq_y / s_sigma_y));
//
//		double denominator = PI_DOUBLE * (freq_y * freq_y + freq_x * freq_x);
//		if (denominator != 0.0) {
//			n.x = ((c.y * freq_x - c.x * freq_y) / denominator) * s_window;
//			n.y = -((c.y * freq_y + c.x * freq_x) / denominator) * s_window;
//		}
//		else {
//			n.x = 0.0;
//			n.y = 0.0;
//		}
//	});
//}
