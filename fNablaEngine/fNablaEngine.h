#pragma once

// Windows Header Files
#define WIN32_LEAN_AND_MEAN  // Exclude rarely-used stuff from Windows headers
#include <windows.h>

//MKL
#include "mkl_dfti.h"

//CUDA
#include "fNablaEngineCuda.cuh"

namespace fColormaps
{
	enum Colormap_types {
		GRAYSCALE,
		VIRIDIS,
		MAGMA,
	};

	struct Viridis {
		Viridis(const Viridis&) = delete;
		Viridis() {}
		static Viridis& Get() {
			static Viridis instance;
			return instance;
		}

		static cv::Point3d at(double x) {
			double r = 0.0;
			double g = 0.0;
			double b = 0.0;
			for (int i = 10; i >= 0; i--)
			{
				double power = pow(x, i);
				Viridis& ref = Get();
				r += ref.R_coeffs[i] * power;
				g += ref.G_coeffs[i] * power;
				b += ref.B_coeffs[i] * power;
			}
			return cv::Point3d(b, g, r);
		}

		double R_coeffs[11] = {
			0.2687991305597938,
			0.195695068597545,
			2.069193214801474,
			-32.74409351551069,
			32.00582675786834,
			641.8213302764933,
			-3124.577583332242,
			6561.962859200739,
			-7221.657271442402,
			4071.0458672486193,
			-929.4027186597173,
		};
		double G_coeffs[11] = {
			0.0024819165042698,
			1.6241485718278847,
			-3.525653178812067,
			20.025165047665833,
			-76.0740427442855,
			153.34602434143605,
			-134.14808123604527,
			-27.623014548781104,
			154.33954247241087,
			-117.42885026019304,
			30.368757730257524,
		};
		double B_coeffs[11] = {
			0.3289447267590451,
			1.6020903777881879,
			-3.7756278891864845,
			19.950263860825128,
			-186.39314640723353,
			836.4228957104253,
			-1969.5554553426439,
			2614.45997632588,
			-1948.3737371462446,
			739.1261566689692,
			-103.64168838177879,
		};
	};

	struct Magma {
		Magma(const Magma&) = delete;
		Magma() {}
		static Magma& Get() {
			static Magma instance;
			return instance;
		}

		static cv::Point3d at(double x) {
			double r = 0.0;
			double g = 0.0;
			double b = 0.0;
			for (int i = 10; i >= 0; i--)
			{
				double power = pow(x, i);
				Magma& ref = Get();
				r += ref.R_coeffs[i] * power;
				g += ref.G_coeffs[i] * power;
				b += ref.B_coeffs[i] * power;
			}
			return cv::Point3d(b, g, r);
		}

		double R_coeffs[11] = {
			0.0004062763460296,
			0.2765580349156502,
			6.066078845033149,
			-15.785757784030428,
			141.79465152894602,
			-1010.5284601954093,
			3470.9812106926493,
			-6369.328232133341,
			6464.36126183672,
			-3433.588054405529,
			746.7387907132164,
		};
		double G_coeffs[11] = {
			0.0069230208847109,
			-0.927625044888057,
			45.25493630314179,
			-529.5616637640426,
			2976.8215782803554,
			-9380.063372551465,
			17758.797776420426,
			-20613.71150768797,
			14347.739661261421,
			-5488.721941663776,
			885.3555318129548,
		};
		double B_coeffs[11] = {
			0.0062637587647372,
			2.640656213391912,
			-32.54058593086639,
			548.1248069266028,
			-3997.301754546941,
			15336.155185166343,
			-34548.07232423682,
			47349.901020990794,
			-38851.35544419425,
			17553.826465839295,
			-3360.636277605164,
		};
	};
}

namespace fNablaEngine
{
	enum MapTypes {
		NUM_INPUTS = 3,
		NUM_OUTPUTS = 4,

		DISPLACEMENT = 0,
		NORMAL = 1,
		CURVATURE = 2,
		AO = 3,
	};

	enum ComputeFlags
	{
		INPUT_MASK = ((1 << NUM_INPUTS) - 1) << NUM_OUTPUTS,
		INPUT_DISPLACEMENT = 1 << (DISPLACEMENT + NUM_OUTPUTS),
		INPUT_NORMAL = 1 << (NORMAL + NUM_OUTPUTS),
		INPUT_CURVATURE = 1 << (CURVATURE + NUM_OUTPUTS),

		OUTPUT_MASK = (1 << NUM_OUTPUTS) - 1,
		OUTPUT_DISPLACEMENT = 1 << DISPLACEMENT,
		OUTPUT_NORMAL = 1 << NORMAL,
		OUTPUT_CURVATURE = 1 << CURVATURE,
		OUTPUT_AO = 1 << AO,
	};

	enum CurvatureModes {
		CURVATURE_COMPLETE,
		CURVATURE_SPLIT,
		CURVATURE_CONVEXITY,
		CURVATURE_CONCAVITY
	};

	class __declspec(dllexport) MeshMap {
	public:
		std::string Name;
		cv::Mat Mat;
		cv::Mat Spectrum;
		int Type = CV_64FC1;
		double RangeLower = 0.0;
		double RangeUpper = 1.0;
		int ReadFlags = cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH;
		//variables used by some members, a better way to handle this?
		double scale = 0.0;
		//---

		virtual void Import(const cv::Mat& input, const double scale_factor = 1.0);
		virtual cv::Mat Export(const int depth);
		virtual void Normalize();
		virtual cv::Mat* AllocateSpectrum();
		virtual void CalculateSpectrum();
		virtual void ReconstructFromSpectrum();
		virtual cv::Mat Postprocess() { return this->Mat; }

		~MeshMap() {
			this->Mat.release();
			this->Spectrum.release();
		}
	};

	class __declspec(dllexport) DisplacementMap : public MeshMap {
	public:

		double integration_window;
		int mode = fColormaps::GRAYSCALE;

		DisplacementMap() {
			this->Name = "Displacement Map";
		}
		virtual cv::Mat Postprocess();
	};

	class __declspec(dllexport) NormalMap: public MeshMap {
	public:
		cv::Scalar swizzle_xy_coordinates = cv::Scalar(1.0, 1.0);
		NormalMap() {
			this->Name = "Normal Map";
			this->Type = CV_64FC3;
			this->RangeLower = -1.0;
			this->ReadFlags = cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH;
		}

		virtual void Normalize();
		virtual void ReconstructZComponent();
		virtual void CalculateSpectrum();

		virtual void ReconstructFromSpectrum();
		virtual cv::Mat Postprocess();
	};

	class __declspec(dllexport) CurvatureMap : public MeshMap {
	public:
		double curvature_sharpness;
		int mode = CURVATURE_COMPLETE;

		CurvatureMap() {
			this->Name = "Curvature Map";
			this->RangeLower = -1.0;
		}

		virtual void Normalize();

		virtual void ReconstructFromSpectrum();
		virtual cv::Mat Postprocess();
	};

	class __declspec(dllexport) AmbientOcclusionMap : public MeshMap {
	public:
		double ao_power;
		double ao_distance;
		int ao_samples;


		AmbientOcclusionMap() {
			this->Name = "Ambient Occlusion Map";
		}

		void Compute(fNablaEngine::MeshMap* displacement, fNablaEngine::MeshMap* normal);

		virtual cv::Mat Postprocess();
	};


	void fft2(cv::Mat& input, bool inverse = false);
	void ifft2(cv::Mat& input);

	void GetAlphaBeta(const int CV_Depth, const double lower, const double upper, double& alpha, double& beta, const bool inverse);

	void __declspec(dllexport) ComputeSpectrums(
		cv::Mat* spectrums[3],
		const cv::Size shape,
		const int process_flags,
		const double high_pass,
		const double curvature_sharpness,
		const double scale_factor
	);


#ifdef _DEBUG
	void __declspec(dllexport) image_show(cv::Mat& input, const std::string& name, bool fft = false);
#endif
}