#pragma once
#pragma warning(disable:4251)

//Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

//MKL
#define USE_MKL

#ifdef USE_MKL
#include <mkl_dfti.h>
#endif // USE_MKL


//STD
#include <thread>
#include <future>

//Engine components
#include "fNablaConfig.h"
#include "fNablaEngineCuda.cuh"
#include "fNablaConstants.h"

namespace fNablaEngine
{
	/// Generic MeshMap to be subclassed
	class __declspec(dllexport) TextureMap {
	protected:
		/// Configuration file with operational parameters
		Configuration& config;
		/// Lower bound of domain
		double RangeLower = 0.0;
		/// Upper bound of domain
		double RangeUpper = 1.0;
	public:
		/// Main container of matrix data
		cv::Mat Mat;
		/// CV Type describing depth and number of channels
		int Type = CV_64FC1;
		unsigned int NumChannels = 1;
		unsigned int Depth = CV_64F;
		/// CV flags used when reading an input of this kind
		int ReadFlags = cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH;
		/// Allocates the memory required for the input shape on this->Mat.
		virtual void AllocateMat(const cv::Size shape);
		/// Loads a cv::Mat into the Meshmap
		virtual void Import(const cv::Mat& input, const double factor = 1.0);
		/// Exports a copy of the Mat
		virtual cv::Mat Export(const int depth, const bool postprocess = true, const double factor = 1.0);
		/// Normalize data
		virtual void Normalize();
		/// Applies some final modification to the data that is not computationally intensive
		virtual cv::Mat Postprocess();

		/// Constructor. Takes a configuration object with all tweakable operational parameters.
		TextureMap(Configuration& config_) : config(config_){}

		///Destructor
		~TextureMap() {
			Mat.release();
		}
	};

	/// Subset of MeshMaps which represent the geometry of the surface
	class __declspec(dllexport) SurfaceMap : public TextureMap {
	public:
		/// Allocates a matrix of complex doubles of the given shape
		static cv::Mat AllocateSpectrum(cv::Size shape);
		/// Forward fourier transform of Mat to spectrum
		virtual void CalculateSpectrum(cv::Mat& Spectrum);
		/// Backward fourier transform of spectrum to mat
		virtual void ReconstructFromSpectrum(cv::Mat& Spectrum);

		SurfaceMap(Configuration& config_) : TextureMap(config_){}
	};

	/// Displacement or height map
	class __declspec(dllexport) DisplacementMap : public SurfaceMap {
	public:
		DisplacementMap(Configuration& config_) : SurfaceMap(config_){}
		cv::Mat Postprocess();
	};

	/// Normal map
	class __declspec(dllexport) NormalMap: public SurfaceMap {
	public:
		NormalMap(Configuration& config_) : SurfaceMap(config_){
			Type = CV_64FC3;
			NumChannels = 3;
			RangeLower = -1.0;
			ReadFlags = cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH;
		}

		void Normalize();
		void CalculateSpectrum(cv::Mat& Spectrum);
		void ReconstructFromSpectrum(cv::Mat& Spectrum);
		cv::Mat Postprocess();
	};

	/// Curvature map
	class __declspec(dllexport) CurvatureMap : public SurfaceMap {
	public:
		CurvatureMap(Configuration& config_) : SurfaceMap(config_) {
			RangeLower = -1.0;
		}

		void Normalize();
		cv::Mat Postprocess();
	};

	/// Ambient Occlusion map
	class __declspec(dllexport) AmbientOcclusionMap : public TextureMap {
	public:
		AmbientOcclusionMap(Configuration& config_) : TextureMap(config_) {}
		cv::Mat Postprocess();
	};

	/// Containers for all MeshMap types handled
	using TextureSet = std::array<std::shared_ptr<fNablaEngine::TextureMap>, fNablaEngine::NUM_OUTPUTS>;

	/// Asyncronously executes the conversion defined by the descriptor on the MeshMaps and tracks its progress
	class __declspec(dllexport) ConversionTask {
	public:
		std::string status = "Starting...";
		std::atomic<double> progress = 0.0; ///0 to 1
		std::future<void> output;
		ConversionTask(TextureSet& Maps, Configuration& config, Descriptor& descriptor, double scale_factor = 1.0);
		bool CheckReady();
	private:
		void Run(TextureSet& Maps, Configuration& config, Descriptor& descriptor, double scale_factor = 1.0);
		//Milestone counter
		int m_num_milestones = 0;
		int m_milestone_counter = 0;
		void StartProgress(int num_milestones);
		void NextMilestone(std::string new_status);
	};

	/// Safely checks for a valid GPU
	bool __declspec(dllexport) CheckGPUCompute();

	///---Internal methods---

	/// Obtain Alpha (scale factor) and Beta (offset) values for type conversion
	std::tuple<double, double> GetAlphaBeta(const int CV_Depth, const double out_low, const double out_up, const bool inverse);
	int CVType(const int CV_Depth, const unsigned int num_channels);

	template <typename T>
	int CVType(const int CV_Depth, const unsigned int num_channels) {
		return CVType(cv::DataType<T>::value, num_channels);
	}

	/// Performs forward or backwards fourier transform
	void fft2(cv::Mat& input, bool inverse = false);
	/// Convenience function for inverse fourier transform
	void ifft2(cv::Mat& input);
	/// Shifts cuadrants of fourier transform
	cv::Mat fftshift(const cv::Mat& input);

		/// Colormaps described as an RGB polynomial
	struct Viridis {
		Viridis(const Viridis&) = delete;
		Viridis() {}
		static Viridis& Get();
		static cv::Point3d at(double x);

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
		static Magma& Get();
		static cv::Point3d at(double x);

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