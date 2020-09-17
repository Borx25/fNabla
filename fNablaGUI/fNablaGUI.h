#pragma once

#include "fNablaEngine.h" //Conversion Engine, includes OpenCV
#include "ui_fNablaGUI.h" // UI header

//QT headers not covered by the UI header
#include <QCommandLineParser>
#include <QtWidgets/QSplashScreen>
#include <QtWidgets/QScrollBar>
#include <QFileInfo>
#include <qfiledialog.h>
#include <qmessagebox.h>
#include <qdebug.h>
#include <QStandardItemModel>
#include <QSettings>

//STD
#include <memory>
#include <array>

using namespace fNablaEngine;

struct MapInfo {
	QString Name;
	QString SettingCategory;
	QLineEdit* ExportSuffix;
	QComboBox* ExportDepth;
	QComboBox* ExportFormat;
	QAction* ExportAction;
	QCheckBox* EnableSetting;
	QPixmap Pixmap;
};

class fNablaGUI : public QMainWindow
{
	Q_OBJECT

public:
	fNablaGUI(QWidget* parent = Q_NULLPTR);

private slots:
	void on_actionSaveSettings_clicked();

protected:
	void closeEvent(QCloseEvent* event) override;

private:
	void SetLoadedState(bool loaded);
	void Zoom(float factor, bool fit = false);

	void LoadSettings();

	void LoadManager(int i);
	void MonitorProgressAndWait(ConversionTask& conversion);
	void ProcessInput(bool override_work_res = false);
	void ComputeMap(int i);
	void UpdatePixmap(int i);
	void RedrawAll();
	void UpdateLabel();

	void ExportManager(std::bitset<NUM_OUTPUTS> map_selection);

	Ui::fNablaGUIClass ui;
	float UIScaleFactor = 1.0f;
	double WorkingScaleFactor = 1.0;
	bool LoadedState = false;
	bool HasGPU = false;

	QPixmap DefaultImage;
	std::array<MapInfo, NUM_OUTPUTS>MapInfoArray;
	std::unique_ptr<QSettings> settings;

	Configuration configuration;
	Descriptor global_descriptor;

	cv::Mat input_image;
	MeshMapArray Maps;
};