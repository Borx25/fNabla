#include "fNablaGUI.h"

int main(int argc, char* argv[]){
	QApplication app(argc, argv);
	QCoreApplication::setApplicationName("fNabla");
	QCoreApplication::setOrganizationName("fNabla");
	QSettings::setDefaultFormat(QSettings::IniFormat);


	QSplashScreen splash(QPixmap(":/fNablaResources/SplashScreen.png"));
	splash.show();

	fNablaGUI window;
	window.setWindowIcon(QIcon(":/fNablaResources/fNabla.ico"));
	window.show();
	splash.finish(&window);
	return app.exec();
}

fNablaGUI::fNablaGUI(QWidget* parent) : QMainWindow(parent) {
	ui.setupUi(this);

	DefaultImage.load(":/fNablaResources/default.png");

	local_config = fNablaEngine::Config();

	MapInfoArray = { {
		{
		QStringLiteral("Displacement Map"),
		QStringLiteral("Displacement"),
		ui.Settings_displacement_suffix,
		ui.Settings_displacement_depth,
		ui.Settings_displacement_format,
		ui.actionSetExport_Displacement,
		ui.Settings_displacement_enable,
		ui.Label_Displacement,
		ui.scrollArea_Displacement
		},
		{
		QStringLiteral("Normal Map"),
		QStringLiteral("Normal"),
		ui.Settings_normal_suffix,
		ui.Settings_normal_depth,
		ui.Settings_normal_format,
		ui.actionSetExport_TSNormal,
		ui.Settings_normal_enable,
		ui.Label_TSNormal,
		ui.scrollArea_TSNormal
		},
		{
		QStringLiteral("Curvature Map"),
		QStringLiteral("Curvature"),
		ui.Settings_curvature_suffix,
		ui.Settings_curvature_depth,
		ui.Settings_curvature_format,
		ui.actionSetExport_Curvature,
		ui.Settings_curvature_enable,
		ui.Label_Curvature,
		ui.scrollArea_Curvature
		},
		{
		QStringLiteral("Ambient Occlusion Map"),
		QStringLiteral("AO"),
		ui.Settings_ao_suffix,
		ui.Settings_ao_depth,
		ui.Settings_ao_format,
		ui.actionSetExport_AO,
		ui.Settings_ao_enable,
		ui.Label_AO,
		ui.scrollArea_AO
		},
	}};

	Maps = { {
		std::make_shared<fNablaEngine::DisplacementMap>(local_config),
		std::make_shared<fNablaEngine::NormalMap>(local_config),
		std::make_shared<fNablaEngine::CurvatureMap>(local_config),
		std::make_shared<fNablaEngine::AmbientOcclusionMap>(local_config),
	} };

	UIScaleFactor = 1.0;

	//CONNECTIONS
	for (int i = 0; i < NumMaps; i++) {
		QObject::connect(MapInfoArray[i].ExportAction, &QAction::triggered, this, &fNablaGUI::CheckAnyExportSelected);

		QObject::connect(MapInfoArray[i].ExportDepth, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this, i](int idx) {
			local_config.export_settings[i].Set_bitdepth(idx);
			int updated_format = local_config.export_settings[i].Get_format();
			if (MapInfoArray[i].ExportFormat->currentIndex() != updated_format) {
				MapInfoArray[i].ExportFormat->setCurrentIndex(updated_format);
			}
		});

		QObject::connect(MapInfoArray[i].ExportFormat, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this, i](int idx) {
			local_config.export_settings[i].Set_format(idx);
			int updated_bitdepth = local_config.export_settings[i].Get_bitdepth();
			if (MapInfoArray[i].ExportDepth->currentIndex() != updated_bitdepth) {
				MapInfoArray[i].ExportDepth->setCurrentIndex(updated_bitdepth);
			}
		});

		QObject::connect(MapInfoArray[i].ExportSuffix, &QLineEdit::editingFinished, this, [this, i]() {
			local_config.export_settings[i].Set_suffix(MapInfoArray[i].ExportSuffix->text().toStdString());
		});

		QRegExpValidator SuffixValidator(QRegExp("[a-zA-Z0-9_-]{2,25}"), this);
		MapInfoArray[i].ExportSuffix->setValidator(&SuffixValidator);

		QObject::connect(MapInfoArray[i].DisplayScrollArea->horizontalScrollBar(), &QScrollBar::valueChanged, this, [this, i](int value) {
			for (int j = 0; j < NumMaps; j++) {
				if (i != j)
				{
					MapInfoArray[j].DisplayScrollArea->horizontalScrollBar()->setValue(value);
				}
			}
		});

		QObject::connect(MapInfoArray[i].DisplayScrollArea->verticalScrollBar(), &QScrollBar::valueChanged, this, [this, i](int value) {
			for (int j = 0; j < NumMaps; j++) {
				if (i != j)
				{
					MapInfoArray[j].DisplayScrollArea->verticalScrollBar()->setValue(value);
				}
			}
		});
	}
	QObject::connect(ui.actionExport_All, &QAction::triggered, this, [this]() {
		Export(true); 
	});
	QObject::connect(ui.actionExport_Selected, &QAction::triggered, this, [this]() {
		Export(false);
	});
	QObject::connect(ui.actionZoom_In, &QAction::triggered, this, [this]() {
		Zoom(1.25);
	});
	QObject::connect(ui.actionZoom_Out, &QAction::triggered, this, [this]() {
		Zoom(0.8);
	});
	QObject::connect(ui.actionFit_Window, &QAction::triggered, this, [this]() {
		Zoom(1.0, true);
	});

	QObject::connect(ui.actionLoad_Displacement, &QAction::triggered, this, [this]() {
		LoadMap(fNablaEngine::DISPLACEMENT);
	});
	QObject::connect(ui.actionLoad_TSNormal, &QAction::triggered, this, [this]() {
		LoadMap(fNablaEngine::NORMAL);
	});
	QObject::connect(ui.actionLoad_Curvature, &QAction::triggered, this, [this]() {
		LoadMap(fNablaEngine::CURVATURE);
	});
	QObject::connect(ui.actionClear, &QAction::triggered, this, [this]() {
		input_map_type = -1;
		ui.ProgressBar->setValue(0);
		ui.Status->setText(QStringLiteral(""));
		for (int i = 0; i < NumMaps; i++) {
			Maps[i]->Mat.release();
			MapInfoArray[i].DisplayLabel->setPixmap(DefaultImage);
		}
		SetLoadedState(false);
	});
	QObject::connect(ui.actionAbout_fNabla, &QAction::triggered, this, [this]() {
		QMessageBox::about(this, "About fNabla", "<b>fNabla</b><br><br>Version 1.0<br>fNabla is a tool for conversion between various mesh maps<br>Copyright (C) 2020 Borja Franco Garcia");
	});
	QObject::connect(ui.actionExit, &QAction::triggered, this, [this]() {
		QCoreApplication::exit(0);
	});
	QObject::connect(ui.Settings_high_pass, &SettingWidget::valueChanged, this, [this](double value) {
		local_config.integration_window.Set(value);
		if (LoadedState) {
			ReprocessAll();
		}
	});
	QObject::connect(ui.Settings_displacement_mode, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this](int idx) {
		local_config.displacement_colormap.Set(idx);
		if (LoadedState && MapInfoArray[0].EnableSetting->isChecked()) {
			Draw(fNablaEngine::DISPLACEMENT);
		}
	});
	QObject::connect(ui.Settings_normal_scale, &SettingWidget::valueChanged, this, [this](double value) {
		local_config.normal_scale.Set(value);
		if (LoadedState && MapInfoArray[1].EnableSetting->isChecked()) {
			Draw(fNablaEngine::NORMAL);
		}
	});
	QObject::connect(ui.swizzle_x, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this](int idx) {
		local_config.normal_swizzle.Set(ui.swizzle_x->currentIndex(), ui.swizzle_y->currentIndex());
		if (LoadedState && MapInfoArray[1].EnableSetting->isChecked()) {
			ReprocessAll();
		}
	});
	QObject::connect(ui.swizzle_y, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this](int idx) {
		local_config.normal_swizzle.Set(ui.swizzle_x->currentIndex(), ui.swizzle_y->currentIndex());
		if (LoadedState && MapInfoArray[1].EnableSetting->isChecked()) {
			ReprocessAll();
		}
	});
	QObject::connect(ui.Settings_curvature_scale, &SettingWidget::valueChanged, this, [this](double value) {
		local_config.curvature_scale.Set(value);
		if (LoadedState && MapInfoArray[2].EnableSetting->isChecked()) {
			Draw(fNablaEngine::CURVATURE);
		}
	});
	QObject::connect(ui.Settings_curvature_mode, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this](int idx) {
		local_config.curvature_mode.Set(idx);
		if (LoadedState && MapInfoArray[2].EnableSetting->isChecked()) {
			Draw(fNablaEngine::CURVATURE);
		}
	});
	QObject::connect(ui.Settings_ao_scale, &SettingWidget::valueChanged, this, [this](double value) {
		local_config.ao_scale.Set(value);
		if (LoadedState && MapInfoArray[3].EnableSetting->isChecked()) {
			Compute_AO();
			Draw(fNablaEngine::AO);
		}
	});
	QObject::connect(ui.Settings_ao_power, &SettingWidget::valueChanged, this, [this](double value) {
		local_config.ao_power.Set(value);
		if (LoadedState && MapInfoArray[3].EnableSetting->isChecked()) {
			Draw(fNablaEngine::AO);
		}
	});
	QObject::connect(ui.Settings_ao_distance, &SettingWidget::valueChanged, this, [this](double value) {
		local_config.ao_distance.Set(value);
		if (LoadedState && MapInfoArray[3].EnableSetting->isChecked()) {
			Compute_AO();
			Draw(fNablaEngine::AO);
		}
	});
	QObject::connect(ui.Settings_ao_samples, &SettingWidget::valueChanged, this, [this](double value) {
		local_config.ao_samples.Set((int)value);
		if (LoadedState && MapInfoArray[3].EnableSetting->isChecked()) {
			Compute_AO();
			Draw(fNablaEngine::AO);
		}
	});
	QObject::connect(ui.WorkingResolution, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this](int idx) {
		WorkingScaleFactor = exp(-log(2.0) * double(idx));

		if (LoadedState) {
			ReprocessAll();
			ui.actionFit_Window->trigger();
		}
	});

	ui.actionSetExportGroup->setExclusive(false);

	LoadSettings();
}

//ProgressBar Macros
#define START_PROGRESSBAR(NUM_MILESTONES)		\
	int _NUM_MILESTONES = NUM_MILESTONES;		\
	int _MILESTONE_COUNTER = 0;					\
	ui.ProgressBar->setValue(0);				\


#define MILESTONE(STATUS)																\
	ui.Status->setText(QStringLiteral(STATUS));											\
	_MILESTONE_COUNTER += 1;															\
	ui.ProgressBar->setValue(int(100.0 * _MILESTONE_COUNTER/_NUM_MILESTONES));			\


void fNablaGUI::closeEvent(QCloseEvent* event) {
	settings->setValue("GUI/size", size());
	settings->setValue("GUI/pos", pos());
}

void fNablaGUI::CheckAnyExportSelected() {
	bool anyChecked = false;
	for (int i = 0; i < NumMaps; i++) {

		anyChecked = (anyChecked || MapInfoArray[i].EnableSetting->isChecked());
	}
	ui.actionExport_Selected->setEnabled(anyChecked && (input_map_type != -1));
}

//SETTINGS

void fNablaGUI::LoadSettings() {
	settings = std::make_unique<QSettings>();

	resize(settings->value("GUI/size", QSize(1350, 900)).toSize());
	move(settings->value("GUI/pos", QPoint(100, 50)).toPoint());

	ui.Settings_high_pass->setValue(settings->value("Global/window", local_config.integration_window.Get_raw()).toDouble());

	ui.Settings_displacement_mode->setCurrentIndex(settings->value("Displacement/colormap", local_config.displacement_colormap.Get_raw()).toInt());

	ui.Settings_normal_scale->setValue(settings->value("Normal/scale", local_config.normal_scale.Get_raw()).toDouble());
	ui.swizzle_x->setCurrentIndex(settings->value("Normal/swizzle_x", local_config.normal_swizzle.Get_x()).toInt());
	ui.swizzle_y->setCurrentIndex(settings->value("Normal/swizzle_y", local_config.normal_swizzle.Get_y()).toInt());

	ui.Settings_curvature_scale->setValue(settings->value("Curvature/scale", local_config.curvature_scale.Get_raw()).toDouble());
	ui.Settings_curvature_mode->setCurrentIndex(settings->value("Curvature/mode", local_config.curvature_mode.Get_raw()).toInt());

	ui.Settings_ao_scale->setValue(settings->value("AO/scale", local_config.ao_scale.Get_raw()).toDouble());
	ui.Settings_ao_samples->setValue(settings->value("AO/samples", local_config.ao_samples.Get_raw()).toDouble());
	ui.Settings_ao_distance->setValue(settings->value("AO/distance", local_config.ao_distance.Get_raw()).toDouble());
	ui.Settings_ao_power->setValue(settings->value("AO/power", local_config.ao_power.Get_raw()).toDouble());

	for (int i = 0; i < NumMaps; i++) {
		MapInfoArray[i].EnableSetting->setChecked(settings->value(MapInfoArray[i].SettingCategory + QStringLiteral("/enable"), local_config.enabled_maps[i].Get()).toBool());
		MapInfoArray[i].ExportFormat->setCurrentIndex(settings->value(MapInfoArray[i].SettingCategory + QStringLiteral("/format"), local_config.export_settings[i].Get_format()).toInt());
		MapInfoArray[i].ExportDepth->setCurrentIndex(settings->value(MapInfoArray[i].SettingCategory + QStringLiteral("/bitdepth"), local_config.export_settings[i].Get_bitdepth()).toInt());
		MapInfoArray[i].ExportSuffix->setText(settings->value(MapInfoArray[i].SettingCategory + QStringLiteral("/suffix"), QString::fromStdString(local_config.export_settings[i].Get_suffix())).toString());
	}
}

void fNablaGUI::on_actionSaveSettings_clicked() {

	settings->setValue("Global/window", ui.Settings_high_pass->Value());

	settings->setValue("Displacement/colormap", ui.Settings_displacement_mode->currentIndex());

	settings->setValue("Normal/scale", ui.Settings_normal_scale->Value());
	settings->setValue("Normal/swizzle_x", ui.swizzle_x->currentIndex());
	settings->setValue("Normal/swizzle_y", ui.swizzle_y->currentIndex());

	settings->setValue("Curvature/scale", ui.Settings_curvature_scale->Value());
	settings->setValue("Curvature/mode", ui.Settings_curvature_mode->currentIndex());

	settings->setValue("AO/scale", ui.Settings_ao_scale->Value());
	settings->setValue("AO/samples", ui.Settings_ao_samples->Value());
	settings->setValue("AO/distance", ui.Settings_ao_distance->Value());
	settings->setValue("AO/power", ui.Settings_ao_power->Value());

	for (int i = 0; i < NumMaps; i++) {
		settings->setValue(MapInfoArray[i].SettingCategory + QStringLiteral("/enable"), MapInfoArray[i].EnableSetting->isChecked());
		settings->setValue(MapInfoArray[i].SettingCategory + QStringLiteral("/format"), MapInfoArray[i].ExportFormat->currentIndex());
		settings->setValue(MapInfoArray[i].SettingCategory + QStringLiteral("/bitdepth"), MapInfoArray[i].ExportDepth->currentIndex());
		settings->setValue(MapInfoArray[i].SettingCategory + QStringLiteral("/suffix"), MapInfoArray[i].ExportSuffix->text());
	}
}

//ADDITIONAL UI FUNCTIONS

void fNablaGUI::Zoom(float factor, bool fit) {
	double OldScale = UIScaleFactor;

	if (fit)
	{
		UIScaleFactor = std::min((float)MapInfoArray[0].DisplayScrollArea->height() / (float)Maps[0]->Mat.rows, (float)MapInfoArray[0].DisplayScrollArea->width() / (float)Maps[0]->Mat.cols);
	}
	else {
		UIScaleFactor *= factor;
	}

	QScrollArea* RefScrollArea = MapInfoArray[0].DisplayScrollArea;

	QPointF Pivot = RefScrollArea->pos() + QPointF(RefScrollArea->size().width() / 2.0, RefScrollArea->size().height() / 2.0);
	QPointF ScrollbarPos = QPointF(RefScrollArea->horizontalScrollBar()->value(), RefScrollArea->verticalScrollBar()->value());
	QPointF DeltaToPos = Pivot / OldScale - RefScrollArea->pos() / OldScale;
	QPointF Delta = DeltaToPos * UIScaleFactor - DeltaToPos * OldScale;

	RedrawAll();

	RefScrollArea->horizontalScrollBar()->setValue(ScrollbarPos.x() + Delta.x());
	RefScrollArea->verticalScrollBar()->setValue(ScrollbarPos.y() + Delta.y());

	ui.actionZoom_In->setEnabled(UIScaleFactor < 10.0); //disable zoom in at 1000%
	ui.actionZoom_Out->setEnabled(UIScaleFactor > 0.1); //disable zoom out at 10%
}

void fNablaGUI::SetLoadedState(bool loaded) {
	ui.actionClear->setEnabled(loaded);
	ui.actionExport_All->setEnabled(loaded);
	CheckAnyExportSelected();
	ui.actionFit_Window->setEnabled(loaded);
	ui.actionZoom_In->setEnabled(loaded);
	ui.actionZoom_Out->setEnabled(loaded);
	LoadedState = loaded;
}

//LOADING
void fNablaGUI::LoadMap(int i) {
	//LOADING
	QString fileName = QFileDialog::getOpenFileName(this,
		QStringLiteral("Load ") + MapInfoArray[i].Name, "",
		QStringLiteral("Image files (*.png *.tiff *.tif *.pbm);;All Files (*)"));
	if (!fileName.isEmpty()) {
		ui.Status->setText(QStringLiteral("Loading"));
		if (input_map_type != -1)
		{
			ui.actionClear->trigger();
		}
		input_map_type = i;
		input_image = cv::imread(fileName.toStdString(), Maps[input_map_type]->ReadFlags);
		//--------------------------
		//PROCESSING
		ProcessInput();
		//-------------------
		//SET LOADED STATE
		ui.actionFit_Window->trigger();
		SetLoadedState(true);
	}
}

//DISPLAY
void fNablaGUI::Draw(int i) {
	cv::Mat mat_8bit = Maps[i]->Export(CV_8U);
	QPixmap pix;
	if (mat_8bit.channels() == 3) {
		pix = QPixmap::fromImage(QImage((unsigned char*)mat_8bit.data, mat_8bit.cols, mat_8bit.rows, mat_8bit.step, QImage::Format_RGB888).rgbSwapped());
	}
	else {
		pix = QPixmap::fromImage(QImage((unsigned char*)mat_8bit.data, mat_8bit.cols, mat_8bit.rows, mat_8bit.step, QImage::Format_Grayscale8));
	}
	if (UIScaleFactor != 1.0) {
		pix = pix.scaled(QSize(mat_8bit.cols * UIScaleFactor, mat_8bit.rows * UIScaleFactor), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	}
	MapInfoArray[i].DisplayLabel->setPixmap(pix);
	MapInfoArray[i].DisplayLabel->adjustSize();
}

void fNablaGUI::RedrawAll() {
	for (int i = 0; i < NumMaps; i++) {
		if (MapInfoArray[i].EnableSetting->isChecked() || i == input_map_type) {
			Draw(i);
		}
	}
}

void fNablaGUI::ReprocessAll() {
	ProcessInput();
	RedrawAll();
}

//PROCESSSING

void fNablaGUI::ProcessInput(bool override_work_res) {
	double scale_factor = (override_work_res ? 1.0 : WorkingScaleFactor);

	Maps[input_map_type]->Import(input_image, scale_factor);

	int compute_plan = 1 << (input_map_type + fNablaEngine::NUM_OUTPUTS) |
		MapInfoArray[0].EnableSetting->isChecked() << fNablaEngine::DISPLACEMENT |
		MapInfoArray[1].EnableSetting->isChecked() << fNablaEngine::NORMAL |
		MapInfoArray[2].EnableSetting->isChecked() << fNablaEngine::CURVATURE |
		MapInfoArray[3].EnableSetting->isChecked() << fNablaEngine::AO;

	fNablaEngine::Compute(Maps, compute_plan, local_config, scale_factor);
}

void fNablaGUI::Compute_AO() {
	dynamic_cast<fNablaEngine::AmbientOcclusionMap*>(Maps[fNablaEngine::AO].get())->Compute(Maps[fNablaEngine::DISPLACEMENT], Maps[fNablaEngine::NORMAL]);
}


//EXPORTING

void fNablaGUI::Export(bool ExportAll) {
	QString MapsSaved;
	// Asumption: All images have been processed since that happens on load. I'll have to make it so clicking this is not available until that's done processing.
	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Export texture set"), "",
		tr("All Files (*)")); //we don't care about extension
	QFileInfo fileInfo(fileName);
	if (!fileName.isEmpty()){
		if (ui.WorkingResolution->currentIndex() != 0)
		{
			ProcessInput(true); //process at full res
		}
		for (int i = 0; i < NumMaps; i++) {
			if ((ExportAll || MapInfoArray[i].EnableSetting->isChecked())) {
				if (MapInfoArray[i].ExportSuffix->hasAcceptableInput())
				{
					QString output = fileInfo.absolutePath() + "/" + fileInfo.baseName() + QString::fromStdString(local_config.export_settings[i].Get_full_suffix());
					cv::Mat save_img = Maps[i]->Export(local_config.export_settings[i].Get_CVdepth(), i!=0); //don't postprocess the displacement (colormap)
					cv::imwrite(output.toStdString(), save_img);
					MapsSaved.append(output + QString("\n"));
				}
				else {
					QMessageBox::warning(this, "Export Failed", "Invalid suffix");
				}
			}
		}
		if (MapsSaved.isEmpty()) {
			QMessageBox::warning(this, "Export Failed", "No output generated");
		}
		else {
			QMessageBox::information(this, "Exported:", MapsSaved);
		}
	}
}