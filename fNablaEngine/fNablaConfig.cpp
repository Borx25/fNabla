#include "fNablaConfig.h"

fNablaEngine::Config_Elements::Boolean::Boolean(bool val) {
	Set(val);
}
void fNablaEngine::Config_Elements::Boolean::Set(bool val) {
	m_val = val;
}
bool fNablaEngine::Config_Elements::Boolean::Get() {
	return m_val;
}

fNablaEngine::Config_Elements::ValidatedString::ValidatedString(std::string value, std::string regex) {
	m_validate_regex = regex;
	Set(value);
}

void fNablaEngine::Config_Elements::ValidatedString::Set(std::string new_value) {
	m_val = (std::regex_match(new_value, m_validate_regex) ? new_value : m_val);
}

std::string fNablaEngine::Config_Elements::ValidatedString::Get() {
	return m_val;
}


fNablaEngine::Config_Elements::ChannelSign::ChannelSign(bool x, bool y) {
	Set(x, y);
}

void fNablaEngine::Config_Elements::ChannelSign::Set(bool x, bool y) {
	m_x = x;
	m_y = y;
	m_val = cv::Scalar(pow(-1.0, x), pow(-1.0, y));
}

cv::Scalar fNablaEngine::Config_Elements::ChannelSign::Get() {
	return m_val;
}

bool fNablaEngine::Config_Elements::ChannelSign::Get_x() {
	return m_x;
}

bool fNablaEngine::Config_Elements::ChannelSign::Get_y() {
	return m_y;
}

bool fNablaEngine::Config_Elements::ExportSettings::CheckCompatibility(bool FormatPreference) {
	if ((m_format == PNG) && (m_bitdepth == BIT32)) {
		if (FormatPreference) {
			m_bitdepth = BIT16;
		} else {
			m_format = TIFF;
		}
		return false;
	}
	return true;
}

fNablaEngine::Config_Elements::ExportSettings::ExportSettings(int format_value, int bitdepth_value, std::string suffix_value) {
	m_format = std::min(std::max(format_value, 0), _NUM_FORMATS - 1);
	m_bitdepth = std::min(std::max(bitdepth_value, 0), _NUM_BITDEPTHS - 1);
	CheckCompatibility();
	m_suffix = ((!suffix_value.empty() && std::regex_match(suffix_value, suffix_regex)) ? suffix_value : "_default");
}

void fNablaEngine::Config_Elements::ExportSettings::Set_format(int new_value) {
	m_format = std::min(std::max(new_value, 0), _NUM_FORMATS - 1);
	CheckCompatibility(true);
}
void fNablaEngine::Config_Elements::ExportSettings::Set_bitdepth(int new_value) {
	m_bitdepth = std::min(std::max(new_value, 0), _NUM_BITDEPTHS - 1);
	CheckCompatibility(false);
}
void fNablaEngine::Config_Elements::ExportSettings::Set_suffix(std::string new_value) {
	m_suffix = (std::regex_match(new_value, suffix_regex) ? new_value : m_suffix);
}
int fNablaEngine::Config_Elements::ExportSettings::Get_format() {
	return m_format;
}
int fNablaEngine::Config_Elements::ExportSettings::Get_bitdepth() {
	return m_bitdepth;
}
int fNablaEngine::Config_Elements::ExportSettings::Get_CVdepth() {
	switch (m_bitdepth) {
		case BIT32:
			return CV_32F;
		case BIT16:
			return CV_16U;
		case BIT8:
			return CV_8U;
		default:
			return CV_32F;
	}
}
std::string fNablaEngine::Config_Elements::ExportSettings::Get_suffix() {
	return m_suffix;
}
std::string fNablaEngine::Config_Elements::ExportSettings::Get_full_suffix() {
	switch (m_format) {
		case TIFF:
			return m_suffix + ".TIFF";
		case PNG:
			return m_suffix + ".PNG";
		default:
			return m_suffix + ".TIFF";
	}
}