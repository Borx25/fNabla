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

	this->DefaultImage.load(":/fNablaResources/default.png");

	this->MapSlots[0] = {
		new fNablaEngine::DisplacementMap(),
		this->ui.Settings_displacement_suffix,
		this->ui.Settings_displacement_depth,
		this->ui.Settings_displacement_format,
		this->ui.actionSetExport_Displacement,
		this->ui.Label_Displacement,
		this->ui.scrollArea_Displacement
		};
	this->MapSlots[1] = {
		new fNablaEngine::NormalMap(),
		this->ui.Settings_normal_suffix,
		this->ui.Settings_normal_depth,
		this->ui.Settings_normal_format,
		this->ui.actionSetExport_TSNormal,
		this->ui.Label_TSNormal,
		this->ui.scrollArea_TSNormal
		};

	this->MapSlots[2] = {
		new fNablaEngine::CurvatureMap(),
		this->ui.Settings_curvature_suffix,
		this->ui.Settings_curvature_depth,
		this->ui.Settings_curvature_format,
		this->ui.actionSetExport_Curvature,
		this->ui.Label_Curvature,
		this->ui.scrollArea_Curvature
		};

	this->MapSlots[3] = {
		new fNablaEngine::AmbientOcclusionMap(),
		this->ui.Settings_ao_suffix,
		this->ui.Settings_ao_depth,
		this->ui.Settings_ao_format,
		this->ui.actionSetExport_AO,
		this->ui.Label_AO,
		this->ui.scrollArea_AO
		};

	this->UIScaleFactor = 1.0;

	//CONNECTIONS

	for (int i = 0; i < this->NumMaps; i++) {
		QObject::connect(this->MapSlots[i].ExportAction, &QAction::triggered, this, &fNablaGUI::CheckAnyExportSelected);

		if (this->MapSlots[i].ExportDepth->currentIndex() == 0)  //set initial conditions
		{
			qobject_cast<QStandardItemModel*>(this->MapSlots[i].ExportFormat->model())->item(1)->setEnabled(false);
		}

		QObject::connect(this->MapSlots[i].ExportDepth, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this, i](int idx) {
			bool mode_32bit = (idx == 0);
			qobject_cast<QStandardItemModel*>(this->MapSlots[i].ExportFormat->model())->item(1)->setEnabled(!mode_32bit);
			if (mode_32bit)
			{
				this->MapSlots[i].ExportFormat->setCurrentIndex(0);
			}
		});

		QObject::connect(this->MapSlots[i].DisplayScrollArea->horizontalScrollBar(), &QScrollBar::valueChanged, this, [this, i](int value) {
			for (int j = 0; j < this->NumMaps; j++) {
				if (i != j)
				{
					this->MapSlots[j].DisplayScrollArea->horizontalScrollBar()->setValue(value);
				}
			}
		});

		QObject::connect(this->MapSlots[i].DisplayScrollArea->verticalScrollBar(), &QScrollBar::valueChanged, this, [this, i](int value) {
			for (int j = 0; j < this->NumMaps; j++) {
				if (i != j)
				{
					this->MapSlots[j].DisplayScrollArea->verticalScrollBar()->setValue(value);
				}
			}
		});
	}

	QObject::connect(this->ui.swizzle_x, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &fNablaGUI::on_swizzle_updated);
	QObject::connect(this->ui.swizzle_y, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &fNablaGUI::on_swizzle_updated);

	QObject::connect(this->ui.actionExport_All, &QAction::triggered, this, [this]() {
		this->Export(true); 
	});
	QObject::connect(this->ui.actionExport_Selected, &QAction::triggered, this, [this]() {
		this->Export(false);
	});

	QObject::connect(this->ui.actionZoom_In, &QAction::triggered, this, [this]() {
		this->Zoom(1.25);
	});
	QObject::connect(this->ui.actionZoom_Out, &QAction::triggered, this, [this]() {
		this->Zoom(0.8);
	});
	QObject::connect(this->ui.actionFit_Window, &QAction::triggered, this, [this]() {
		this->Zoom(1.0, true);
	});

	QObject::connect(this->ui.actionLoad_Displacement, &QAction::triggered, this, [this]() {
		this->LoadMap(fNablaEngine::DISPLACEMENT);
	});
	QObject::connect(this->ui.actionLoad_TSNormal, &QAction::triggered, this, [this]() {
		this->LoadMap(fNablaEngine::NORMAL);
	});
	QObject::connect(this->ui.actionLoad_Curvature, &QAction::triggered, this, [this]() {
		this->LoadMap(fNablaEngine::CURVATURE);
	});

	QObject::connect(this->ui.actionExit, &QAction::triggered, this, [this]() {
		QCoreApplication::exit(0);
	});

	this->ui.actionSetExportGroup->setExclusive(false);

	//suffix validation
	this->SuffixValidator = new QRegExpValidator(QRegExp("[a-zA-Z0-9_-]{2,25}"), this);
	this->ui.Settings_displacement_suffix->setValidator(SuffixValidator);
	this->ui.Settings_normal_suffix->setValidator(SuffixValidator);
	this->ui.Settings_curvature_suffix->setValidator(SuffixValidator);
	this->ui.Settings_ao_suffix->setValidator(SuffixValidator);

	//initialize settings
	this->settings = new QSettings();

	//read settings from ini or defaults
	this->resize(this->settings->value("GUI/size", QSize(1350, 900)).toSize());
	this->move(this->settings->value("GUI/pos", QPoint(100, 50)).toPoint());

	this->ui.Settings_depth->setValue(this->settings->value("Global/depth", 0.25).toDouble());
	this->ui.Settings_high_pass->setValue(this->settings->value("Global/window", 1.0).toDouble());
	this->ui.Settings_displacement_mode->setCurrentIndex(this->settings->value("Displacement/colormap", 0).toInt());
	this->ui.Settings_curvature_sharpness->setValue(this->settings->value("Curvature/sharpness", 0.35).toDouble());
	this->ui.Settings_curvature_mode->setCurrentIndex(this->settings->value("Curvature/mode", 0).toInt());
	this->ui.Settings_ao_samples->setValue(this->settings->value("AO/samples", 16.0).toDouble());
	this->ui.Settings_ao_distance->setValue(this->settings->value("AO/distance", 0.35).toDouble());
	this->ui.Settings_ao_power->setValue(this->settings->value("AO/power", 0.45).toDouble());
	this->ui.swizzle_x->setCurrentIndex(this->settings->value("Normal/swizzle_x", 0).toInt());
	this->ui.swizzle_y->setCurrentIndex(this->settings->value("Normal/swizzle_y", 0).toInt());

	this->ui.Settings_displacement_format->setCurrentIndex(this->settings->value("Displacement/format", 0).toInt());
	this->ui.Settings_displacement_depth->setCurrentIndex(this->settings->value("Displacement/bitdepth", 0).toInt());
	this->ui.Settings_displacement_suffix->setText(this->settings->value("Displacement/suffix", QStringLiteral("_displacement")).toString());

	this->ui.Settings_normal_format->setCurrentIndex(this->settings->value("Normal/format", 1).toInt());
	this->ui.Settings_normal_depth->setCurrentIndex(this->settings->value("Normal/bitdepth", 1).toInt());
	this->ui.Settings_normal_suffix->setText(this->settings->value("Normal/suffix", QStringLiteral("_normal")).toString());

	this->ui.Settings_curvature_format->setCurrentIndex(this->settings->value("Curvature/format", 1).toInt());
	this->ui.Settings_curvature_depth->setCurrentIndex(this->settings->value("Curvature/bitdepth", 1).toInt());
	this->ui.Settings_curvature_suffix->setText(this->settings->value("Curvature/suffix", QStringLiteral("_curvature")).toString());

	this->ui.Settings_ao_format->setCurrentIndex(this->settings->value("AO/format", 1).toInt());
	this->ui.Settings_ao_depth->setCurrentIndex(this->settings->value("AO/bitdepth", 1).toInt());
	this->ui.Settings_ao_suffix->setText(this->settings->value("AO/suffix", QStringLiteral("_ambient_occlusion")).toString());
}

//ProgressBar Macros
#define START_PROGRESSBAR(NUM_MILESTONES)		\
	int _NUM_MILESTONES = NUM_MILESTONES;		\
	int _MILESTONE_COUNTER = 0;					\
	this->ui.ProgressBar->setValue(0);			\


#define MILESTONE(STATUS)																\
	this->ui.Status->setText(QStringLiteral(STATUS));									\
	_MILESTONE_COUNTER += 1;															\
	this->ui.ProgressBar->setValue(int(100.0 * _MILESTONE_COUNTER/_NUM_MILESTONES));	\


void fNablaGUI::closeEvent(QCloseEvent* event) {
	this->settings->setValue("GUI/size", this->size());
	this->settings->setValue("GUI/pos", this->pos());
}

void fNablaGUI::on_actionSaveSettings_clicked() {
	this->settings->setValue("Global/depth", this->ui.Settings_depth->Value());
	this->settings->setValue("Global/window", this->ui.Settings_high_pass->Value());
	this->settings->setValue("Displacement/colormap", this->ui.Settings_displacement_mode->currentIndex());
	this->settings->setValue("Curvature/sharpness", this->ui.Settings_curvature_sharpness->Value());
	this->settings->setValue("Curvature/mode", this->ui.Settings_curvature_mode->currentIndex());
	this->settings->setValue("AO/samples", this->ui.Settings_ao_samples->Value());
	this->settings->setValue("AO/distance", this->ui.Settings_ao_distance->Value());
	this->settings->setValue("AO/power", this->ui.Settings_ao_power->Value());
	this->settings->setValue("Normal/swizzle_x", this->ui.swizzle_x->currentIndex());
	this->settings->setValue("Normal/swizzle_y", this->ui.swizzle_y->currentIndex());

	this->settings->setValue("Displacement/format", this->ui.Settings_displacement_format->currentIndex());
	this->settings->setValue("Displacement/bitdepth", this->ui.Settings_displacement_depth->currentIndex());
	this->settings->setValue("Displacement/suffix", this->ui.Settings_displacement_suffix->text());

	this->settings->setValue("Normal/format", this->ui.Settings_normal_format->currentIndex());
	this->settings->setValue("Normal/bitdepth", this->ui.Settings_normal_depth->currentIndex());
	this->settings->setValue("Normal/suffix", this->ui.Settings_normal_suffix->text());

	this->settings->setValue("Curvature/format", this->ui.Settings_curvature_format->currentIndex());
	this->settings->setValue("Curvature/bitdepth", this->ui.Settings_curvature_depth->currentIndex());
	this->settings->setValue("Curvature/suffix", this->ui.Settings_curvature_suffix->text());

	this->settings->setValue("AO/format", this->ui.Settings_ao_format->currentIndex());
	this->settings->setValue("AO/bitdepth", this->ui.Settings_ao_depth->currentIndex());
	this->settings->setValue("AO/suffix", this->ui.Settings_ao_suffix->text());
}


void fNablaGUI::CheckAnyExportSelected() {
	bool anyChecked = false;
	for (int i = 0; i < this->NumMaps; i++) {
		anyChecked = (anyChecked || this->MapSlots[i].ExportAction->isChecked());
	}
	this->ui.actionExport_Selected->setEnabled(anyChecked && (this->input_map_type != -1));
}

//DISPLAY
void fNablaGUI::UpdateDisplay(int i) {
	cv::Mat mat_8bit = this->MapSlots[i].Map->Export(CV_8U);
	QPixmap pix;
	if (mat_8bit.channels() == 3) {
		pix = QPixmap::fromImage(QImage((unsigned char*)mat_8bit.data, mat_8bit.cols, mat_8bit.rows, mat_8bit.step, QImage::Format_RGB888).rgbSwapped());
	}
	else {
		pix = QPixmap::fromImage(QImage((unsigned char*)mat_8bit.data, mat_8bit.cols, mat_8bit.rows, mat_8bit.step, QImage::Format_Grayscale8));
	}
	if (this->UIScaleFactor != 1.0) {
		pix = pix.scaled(QSize(mat_8bit.cols * this->UIScaleFactor, mat_8bit.rows * this->UIScaleFactor), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	}
	this->MapSlots[i].DisplayLabel->setPixmap(pix);
	this->MapSlots[i].DisplayLabel->adjustSize();
}

//SETTINGS EVENTS

void fNablaGUI::on_Settings_high_pass_valueChanged(double value) {
	dynamic_cast<fNablaEngine::DisplacementMap*>(this->MapSlots[fNablaEngine::DISPLACEMENT].Map)->integration_window = exp(-16.0 * value);
	if (this->input_map_type != -1) {
		this->ProcessInput();
		for (int i = 0; i < this->NumMaps; i++) {
			this->UpdateDisplay(i);
		}
	}
}
void fNablaGUI::on_Settings_depth_valueChanged(double value) {
	for (int i = 0; i < this->NumMaps; i++) {
		this->MapSlots[i].Map->scale = value; //broadcast depth setting to all maps
	}
	if (this->input_map_type != -1) {
		this->ProcessInput();
		for (int i = 0; i < this->NumMaps; i++) {
			this->UpdateDisplay(i);
		}
	}
}
void fNablaGUI::on_Settings_displacement_mode_currentIndexChanged(int index) {
	dynamic_cast<fNablaEngine::DisplacementMap*>(this->MapSlots[fNablaEngine::DISPLACEMENT].Map)->mode = index;
	if (this->input_map_type != -1) {
		this->UpdateDisplay(fNablaEngine::DISPLACEMENT);
	}
}
void fNablaGUI::on_Settings_curvature_sharpness_valueChanged(double value) {
	dynamic_cast<fNablaEngine::CurvatureMap*>(this->MapSlots[fNablaEngine::CURVATURE].Map)->curvature_sharpness = value * value;
	if (this->input_map_type != -1) {
		this->ProcessInput();
		for (int i = 0; i < this->NumMaps; i++) {
			this->UpdateDisplay(i);
		}
	}
}
void fNablaGUI::on_Settings_curvature_mode_currentIndexChanged(int index) {
	dynamic_cast<fNablaEngine::CurvatureMap*>(this->MapSlots[fNablaEngine::CURVATURE].Map)->mode = index;
	if (this->input_map_type != -1) {
		this->UpdateDisplay(fNablaEngine::CURVATURE);
	}
}
void fNablaGUI::on_Settings_ao_power_valueChanged(double value) {
	dynamic_cast<fNablaEngine::AmbientOcclusionMap*>(this->MapSlots[fNablaEngine::AO].Map)->ao_power = value * 5.0;
	if (this->input_map_type != -1) {
		START_PROGRESSBAR(2)
		MILESTONE("Recalculating AO")
		this->UpdateDisplay(fNablaEngine::AO); //just update since power is postprocess
		MILESTONE("Done!")
	}
}
void fNablaGUI::on_Settings_ao_distance_valueChanged(double value) {
	dynamic_cast<fNablaEngine::AmbientOcclusionMap*>(this->MapSlots[fNablaEngine::AO].Map)->ao_distance = value;
	if (this->input_map_type != -1) {
		START_PROGRESSBAR(2)
		MILESTONE("Recalculating AO")
			dynamic_cast<fNablaEngine::AmbientOcclusionMap*>(this->MapSlots[fNablaEngine::AO].Map)->Compute(this->MapSlots[fNablaEngine::DISPLACEMENT].Map, this->MapSlots[fNablaEngine::NORMAL].Map);
		this->UpdateDisplay(fNablaEngine::AO);
		MILESTONE("Done!")
	}
}
void fNablaGUI::on_Settings_ao_samples_valueChanged(double value) {
	dynamic_cast<fNablaEngine::AmbientOcclusionMap*>(this->MapSlots[fNablaEngine::AO].Map)->ao_samples = int(value);
	if (this->input_map_type != -1) {
		START_PROGRESSBAR(2)
		MILESTONE("Recalculating AO")
		dynamic_cast<fNablaEngine::AmbientOcclusionMap*>(this->MapSlots[fNablaEngine::AO].Map)->Compute(this->MapSlots[fNablaEngine::DISPLACEMENT].Map, this->MapSlots[fNablaEngine::NORMAL].Map);
		this->UpdateDisplay(fNablaEngine::AO);
		MILESTONE("Done!")
	}
}
void fNablaGUI::on_swizzle_updated() {
	dynamic_cast<fNablaEngine::NormalMap*>(this->MapSlots[fNablaEngine::NORMAL].Map)->swizzle_xy_coordinates = cv::Scalar(pow(-1.0, this->ui.swizzle_x->currentIndex()), pow(-1.0, this->ui.swizzle_y->currentIndex()));
	if (this->input_map_type != -1) {
		this->ProcessInput();
		for (int i = 0; i < this->NumMaps; i++) {
			this->UpdateDisplay(i);
		}
	}
}
void fNablaGUI::on_WorkingResolution_currentIndexChanged(int index) {
	if (this->input_map_type != -1) {
		this->ProcessInput();
		for (int i = 0; i < this->NumMaps; i++) {
			this->UpdateDisplay(i);
		}
		this->ui.actionFit_Window->trigger();
	}
}

//CLEAR AND EXIT
void fNablaGUI::on_actionClear_triggered() {
	this->input_map_type = -1;
	this->ui.ProgressBar->setValue(0);
	this->ui.Status->setText(QStringLiteral(""));
	for (int i = 0; i < this->NumMaps; i++) {
		this->MapSlots[i].Map->Mat.release();
		this->MapSlots[i].Map->Spectrum.release();
		this->MapSlots[i].DisplayLabel->setPixmap(this->DefaultImage);
	}
	this->SetLoadedState(false);
}

void fNablaGUI::SetLoadedState(bool loaded) {
	this->ui.actionClear->setEnabled(loaded);
	this->ui.actionExport_All->setEnabled(loaded);
	this->CheckAnyExportSelected();
	this->ui.actionFit_Window->setEnabled(loaded);
	this->ui.actionZoom_In->setEnabled(loaded);
	this->ui.actionZoom_Out->setEnabled(loaded);
}

void fNablaGUI::Zoom(float factor, bool fit) {
	double OldScale = this->UIScaleFactor;

	if (fit)
	{
		this->UIScaleFactor = std::min((float)this->MapSlots[0].DisplayScrollArea->height() / (float)this->MapSlots[0].Map->Mat.rows, (float)this->MapSlots[0].DisplayScrollArea->width() / (float)this->MapSlots[0].Map->Mat.cols);
	}
	else {
		this->UIScaleFactor *= factor;
	}

	QScrollArea* RefScrollArea = this->MapSlots[0].DisplayScrollArea;

	QPointF Pivot = RefScrollArea->pos() + QPointF(RefScrollArea->size().width() / 2.0, RefScrollArea->size().height() / 2.0);
	QPointF ScrollbarPos = QPointF(RefScrollArea->horizontalScrollBar()->value(), RefScrollArea->verticalScrollBar()->value());
	QPointF DeltaToPos = Pivot / OldScale - RefScrollArea->pos() / OldScale;
	QPointF Delta = DeltaToPos * this->UIScaleFactor - DeltaToPos * OldScale;

	for (int i = 0; i < this->NumMaps; i++) {
		this->UpdateDisplay(i);
	}

	RefScrollArea->horizontalScrollBar()->setValue(ScrollbarPos.x() + Delta.x());
	RefScrollArea->verticalScrollBar()->setValue(ScrollbarPos.y() + Delta.y());

	this->ui.actionZoom_In->setEnabled(this->UIScaleFactor < 10.0); //disable zoom in at 1000%
	this->ui.actionZoom_Out->setEnabled(this->UIScaleFactor > 0.1); //disable zoom out at 10%
}

//LOADING
void fNablaGUI::LoadMap(int i) {
	//LOADING
	QString fileName = QFileDialog::getOpenFileName(this,
		QStringLiteral("Load ") + QString::fromStdString(this->MapSlots[i].Map->Name), "",
		QStringLiteral("Image files (*.png *.tiff *.tif *.pbm);;All Files (*)"));
	if (!fileName.isEmpty()) {
		this->ui.Status->setText(QStringLiteral("Loading"));
		if (this->input_map_type != -1)
		{
			this->ui.actionClear->trigger();
		}
		this->input_map_type = i;
		this->input_image = cv::imread(fileName.toStdString(), this->MapSlots[this->input_map_type].Map->ReadFlags);
		//--------------------------
		//PROCESSING
		this->ProcessInput();
		//-------------------
		//SET LOADED STATE
		this->ui.actionFit_Window->trigger();
		this->SetLoadedState(true);
	}
}

//PROCESSSING

void fNablaGUI::ProcessInput(bool override_work_res) {
	START_PROGRESSBAR(4)

	MILESTONE("Allocating memory")

	double scale_factor = 1.0;

	if (!(override_work_res) && !(this->ui.WorkingResolution->currentIndex() == 0)) {
		scale_factor = exp(-log(2.0) * double(this->ui.WorkingResolution->currentIndex()));
	}

	this->MapSlots[this->input_map_type].Map->Import(this->input_image, scale_factor);
	const cv::Size shape = this->MapSlots[this->input_map_type].Map->Mat.size();

	for (int i = 0; i < this->NumMaps; i++) {
		if (i != this->input_map_type) {
			this->MapSlots[i].Map->Mat = cv::Mat(shape, this->MapSlots[i].Map->Type);
		}
	}

	cv::Mat* spectrums[3] = {
		this->MapSlots[fNablaEngine::DISPLACEMENT].Map->AllocateSpectrum(),
		this->MapSlots[fNablaEngine::NORMAL].Map->AllocateSpectrum(),
		this->MapSlots[fNablaEngine::CURVATURE].Map->AllocateSpectrum(),
	};

	MILESTONE("Processing outputs")

	int compute_plan = 1 << (this->input_map_type + fNablaEngine::NUM_OUTPUTS) | fNablaEngine::OUTPUT_MASK; //all outputs

	this->MapSlots[this->input_map_type].Map->Normalize();
	this->MapSlots[this->input_map_type].Map->CalculateSpectrum();

	fNablaEngine::ComputeSpectrums(
		spectrums,
		shape,
		compute_plan,
		dynamic_cast<fNablaEngine::DisplacementMap*>(this->MapSlots[fNablaEngine::DISPLACEMENT].Map)->integration_window,
		dynamic_cast<fNablaEngine::CurvatureMap*>(this->MapSlots[fNablaEngine::CURVATURE].Map)->curvature_sharpness,
		scale_factor
	);

	for (int i = 0; i < 3; i++) {
		this->MapSlots[i].Map->ReconstructFromSpectrum();
		this->MapSlots[i].Map->Normalize();
	}
	MILESTONE("Calculating ambient occlusion");

	dynamic_cast<fNablaEngine::AmbientOcclusionMap*>(this->MapSlots[fNablaEngine::AO].Map)->Compute(this->MapSlots[fNablaEngine::DISPLACEMENT].Map, this->MapSlots[fNablaEngine::NORMAL].Map);

	//-------------------
	MILESTONE("Done!");
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
		if (this->ui.WorkingResolution->currentIndex() != 0)
		{
			this->ProcessInput(true); //process at full res
		}
		for (int i = 0; i < this->NumMaps; i++) {
			if ((ExportAll || this->MapSlots[i].ExportAction->isChecked())) {
				if (this->MapSlots[i].ExportSuffix->hasAcceptableInput())
				{
					QString output = fileInfo.absolutePath() + "/" + fileInfo.baseName() + this->MapSlots[i].ExportSuffix->text() + "." + this->MapSlots[i].ExportFormat->currentText();
					int displacement_colormap = dynamic_cast<fNablaEngine::DisplacementMap*>(this->MapSlots[fNablaEngine::DISPLACEMENT].Map)->mode;
					dynamic_cast<fNablaEngine::DisplacementMap*>(this->MapSlots[fNablaEngine::DISPLACEMENT].Map)->mode = 0;
					cv::Mat save_img = this->MapSlots[i].Map->Export(int(-2.5 * (double)this->MapSlots[i].ExportDepth->currentIndex() + 5.0)); //mapping of my index to the corresponding CV_Depth's
					dynamic_cast<fNablaEngine::DisplacementMap*>(this->MapSlots[fNablaEngine::DISPLACEMENT].Map)->mode = displacement_colormap;
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

void fNablaGUI::on_actionAbout_fNabla_triggered() {
	QMessageBox::about(this, "About fNabla", "<b>fNabla</b><br><br>Version 1.0<br>fNabla is a tool for conversion between various mesh maps<br>Copyright (C) 2020 Borja Franco Garcia");
}