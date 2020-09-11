#include "fNablaConfig.h"

fNablaEngine::Descriptor_Elements::Boolean::Boolean(bool val) {
	Set(val);
}
void fNablaEngine::Descriptor_Elements::Boolean::Set(bool val) {
	m_val = val;
}
bool fNablaEngine::Descriptor_Elements::Boolean::Get() {
	return m_val;
}

fNablaEngine::Descriptor_Elements::ValidatedString::ValidatedString(std::string value, std::string regex) {
	m_validate_regex = regex;
	Set(value);
}

void fNablaEngine::Descriptor_Elements::ValidatedString::Set(std::string new_value) {
	m_val = (std::regex_match(new_value, m_validate_regex) ? new_value : m_val);
}

std::string fNablaEngine::Descriptor_Elements::ValidatedString::Get() {
	return m_val;
}


fNablaEngine::Descriptor_Elements::ChannelSign::ChannelSign(bool x, bool y) {
	Set(x, y);
}

void fNablaEngine::Descriptor_Elements::ChannelSign::Set(bool x, bool y) {
	m_x = x;
	m_y = y;
	m_val = cv::Scalar(pow(-1.0, x), pow(-1.0, y));
}

cv::Scalar fNablaEngine::Descriptor_Elements::ChannelSign::Get() {
	return m_val;
}

bool fNablaEngine::Descriptor_Elements::ChannelSign::Get_x() {
	return m_x;
}

bool fNablaEngine::Descriptor_Elements::ChannelSign::Get_y() {
	return m_y;
}

fNablaEngine::Descriptor_Elements::Export::Export(int format_value, int bitdepth_value, std::string suffix_value) {
	Set_format(format_value);
	Set_bitdepth(bitdepth_value);
	Set_suffix(suffix_value);
}
void fNablaEngine::Descriptor_Elements::Export::Set_format(int new_value) {
	format.Set(new_value);
	int validated_format = format.Get();
	if (!IsDepthSupported[validated_format][bitdepth.Get()]) //we have a depth that's not supported by the new format
	{
		auto start_iter = IsDepthSupported[validated_format].begin();
		bitdepth.Set(std::find(start_iter, IsDepthSupported[validated_format].end(), true) - start_iter); //set bitdepth to first(highest) compatible depth
	}
}
void fNablaEngine::Descriptor_Elements::Export::Set_bitdepth(int new_value) {
	bitdepth.Set(new_value);
	int validated_bitdepth = bitdepth.Get();
	if (!IsDepthSupported[format.Get()][validated_bitdepth]){ //we have a format that doesn't support the new bitdepth 
		for (auto it = IsDepthSupported.begin(); it < IsDepthSupported.end(); it++) {
			//find first format(outer) that contains true in out bitdepth index
			if ((*it)[validated_bitdepth]) {
				format.Set(it - IsDepthSupported.begin());
				return;
			}
		}
	}
}
void fNablaEngine::Descriptor_Elements::Export::Set_suffix(std::string new_value) {
	suffix.Set(new_value);
}
int fNablaEngine::Descriptor_Elements::Export::Get_format() {
	return format.Get();
}
int fNablaEngine::Descriptor_Elements::Export::Get_bitdepth() {
	return bitdepth.Get();
}
int fNablaEngine::Descriptor_Elements::Export::Get_CVdepth() {
	return (int(-2.5 * (double)bitdepth.Get() + 5.0));
}
std::string fNablaEngine::Descriptor_Elements::Export::Get_suffix() {
	return suffix.Get();
}
std::string fNablaEngine::Descriptor_Elements::Export::Get_full_suffix() {
	return suffix.Get() + "." + Extensions[format.Get()];
}

fNablaEngine::Viridis& fNablaEngine::Viridis::Get() {
	static Viridis instance;
	return instance;
}

cv::Point3d fNablaEngine::Viridis::at(double x) {
	double r = 0.0;
	double g = 0.0;
	double b = 0.0;
	Viridis& ref = Viridis::Get();
	for (int i = 10; i >= 0; i--)
	{
		double power = pow(x, i);
		r += ref.R_coeffs[i] * power;
		g += ref.G_coeffs[i] * power;
		b += ref.B_coeffs[i] * power;
	}
	return cv::Point3d(b, g, r);
}

fNablaEngine::Magma& fNablaEngine::Magma::Get() {
	static Magma instance;
	return instance;
}

cv::Point3d fNablaEngine::Magma::at(double x) {
	double r = 0.0;
	double g = 0.0;
	double b = 0.0;
	Magma& ref = Magma::Get();
	for (int i = 10; i >= 0; i--)
	{
		double power = pow(x, i);
		r += ref.R_coeffs[i] * power;
		g += ref.G_coeffs[i] * power;
		b += ref.B_coeffs[i] * power;
	}
	return cv::Point3d(b, g, r);
}