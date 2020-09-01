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

struct MapInfo {
	QString Name;
	QString SettingCategory;
	QLineEdit* ExportSuffix;
	QComboBox* ExportDepth;
	QComboBox* ExportFormat;
	QAction* ExportAction;
	QCheckBox* EnableSetting;
	QLabel* DisplayLabel;
	QScrollArea* DisplayScrollArea;
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
	Ui::fNablaGUIClass ui;
	QPixmap DefaultImage;
	static const int NumMaps = fNablaEngine::NUM_OUTPUTS;
	std::array<MapInfo, NumMaps>MapInfoArray;

	void CheckAnyExportSelected();
	void SetLoadedState(bool loaded);
	bool LoadedState = false;
	void Zoom(float factor, bool fit = false);
	float UIScaleFactor = 1.0f;

	void LoadSettings();
	std::unique_ptr<QSettings> settings;
	fNablaEngine::Config local_config;

	cv::Mat input_image;
	int input_map_type = -1;
	double WorkingScaleFactor = 1.0;
	fNablaEngine::MeshMapArray Maps;

	void LoadMap(int i);
	void ProcessInput(bool override_work_res = false);
	void Compute_AO();
	void Draw(int i);
	void RedrawAll();
	void ReprocessAll();

	void Export(bool ExportAll);

};