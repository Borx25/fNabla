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

struct MapSlot {
	fNablaEngine::MeshMap* Map;
	QLineEdit* ExportSuffix;
	QComboBox* ExportDepth;
	QComboBox* ExportFormat;
	QAction* ExportAction;
	QLabel* DisplayLabel;
	QScrollArea* DisplayScrollArea;
};

class fNablaGUI : public QMainWindow
{
	Q_OBJECT

public:
	fNablaGUI(QWidget* parent = Q_NULLPTR);

	static const int NumMaps = fNablaEngine::NUM_OUTPUTS;
	MapSlot MapSlots [NumMaps];

private slots:

	void on_actionClear_triggered();
	void on_actionAbout_fNabla_triggered();
	void on_actionSaveSettings_clicked();

	void on_Settings_high_pass_valueChanged(double value);
	void on_Settings_depth_valueChanged(double value);
	void on_Settings_displacement_mode_currentIndexChanged(int index);
	void on_Settings_curvature_sharpness_valueChanged(double value);
	void on_Settings_curvature_mode_currentIndexChanged(int index);
	void on_Settings_ao_power_valueChanged(double value);
	void on_Settings_ao_distance_valueChanged(double value);
	void on_Settings_ao_samples_valueChanged(double value);
	void on_WorkingResolution_currentIndexChanged(int index);

protected:
	void closeEvent(QCloseEvent* event) override;

private:
	Ui::fNablaGUIClass ui;
	QPixmap DefaultImage;
	QRegExpValidator* SuffixValidator;
	QSettings* settings;
	cv::Mat input_image;
	int input_map_type = -1;

	void on_swizzle_updated();
	void CheckAnyExportSelected();
	void ProcessInput(bool override_work_res = false);
	void LoadMap(int i);
	void Export(bool ExportAll);
	void UpdateDisplay(int i);
	void SetLoadedState(bool loaded);
	void Zoom(float factor, bool fit = false);
	float UIScaleFactor = 1.0f;
};