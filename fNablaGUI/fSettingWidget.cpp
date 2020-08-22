#include "fSettingWidget.h"

SettingWidget::SettingWidget(QWidget* parent)
	: QWidget(parent)
{
	this->setMinimumSize(250, 40);

	this->internal_layout = new QHBoxLayout(this);
	this->internal_layout->setSpacing(6);
	this->internal_layout->setContentsMargins(11, 11, 11, 11);

	this->var_label = new QLabel(this);
	this->var_label->setAlignment(Qt::AlignLeading | Qt::AlignLeft | Qt::AlignVCenter);
	this->var_label->setWordWrap(false);
	this->var_label->setMargin(0);
	this->internal_layout->addWidget(this->var_label);

	this->var_slider = new QSlider(this);
	this->var_slider->setOrientation(Qt::Horizontal);
	this->var_slider->setFocusPolicy(Qt::ClickFocus);
	this->var_slider->setTracking(false);
	this->var_slider->setMinimum(0);
	this->var_slider->setSingleStep(1);
	this->internal_layout->addWidget(this->var_slider);

	this->var_doublespinbox = new QDoubleSpinBox(this);
	this->var_doublespinbox->setDecimals(4);
	this->var_doublespinbox->setKeyboardTracking(false);
	this->internal_layout->addWidget(this->var_doublespinbox);

	this->var_spinbox = new QSpinBox(this);
	this->var_spinbox->setVisible(false);
	this->var_spinbox->setKeyboardTracking(false);
	this->internal_layout->addWidget(this->var_spinbox);

	//set default values

	this->var_doublespinbox->setValue(this->m_value);
	this->var_spinbox->setValue(int(this->m_value));
	this->UpdateRanges();

	//connect signals and slots
	connect(this->var_slider, SIGNAL(valueChanged(int)), this, SLOT(UpdateValueFromSlider()));
	connect(this->var_spinbox, SIGNAL(editingFinished()), this, SLOT(UpdateValueFromSpinbox()));
	connect(this->var_doublespinbox, SIGNAL(editingFinished()), this, SLOT(UpdateValueFromDoubleSpinbox()));
}

void SettingWidget::UpdateRanges() {
	this->var_slider->setMaximum(int(this->m_single_steps));

	this->var_slider->blockSignals(true);
	this->var_slider->setValue(int(((this->m_value - this->m_min) / (this->m_max - this->m_min)) * this->m_single_steps));
	this->var_slider->blockSignals(false);

	this->var_doublespinbox->setMinimum(this->m_min);
	this->var_doublespinbox->setMaximum(this->m_max);
	this->var_doublespinbox->setSingleStep((this->m_max - this->m_min) / this->m_single_steps);

	this->var_spinbox->setMinimum(int(this->m_min));
	this->var_spinbox->setMaximum(int(this->m_max));
	if (this->var_doublespinbox->singleStep() < 1.0) {
		this->var_spinbox->setSingleStep(1);
	}
	else {
		this->var_spinbox->setSingleStep(int(this->var_doublespinbox->singleStep()));
	}
}
void SettingWidget::UpdateValueFromSlider() {
	double slider_percent = double(qobject_cast<QSlider*>(sender())->value()) / this->m_single_steps;
	double value_lerp = this->m_min + slider_percent * (this->m_max - this->m_min);
	if (this->isDiscrete()) {
		value_lerp = double(int(value_lerp));
	}
	this->setValue(value_lerp);
}
void SettingWidget::UpdateValueFromSpinbox() {
	this->setValue(double(qobject_cast<QSpinBox*>(sender())->value()));
}
void SettingWidget::UpdateValueFromDoubleSpinbox() {
	this->setValue(qobject_cast<QDoubleSpinBox*>(sender())->value());
}

const QString& SettingWidget::Name() {
	return this->var_label->text();
}
void SettingWidget::setName(const QString& newName) {
	this->var_label->setText(newName);
}
bool SettingWidget::isDiscrete() {
	return this->var_spinbox->isVisible();
}
void SettingWidget::setDiscrete(bool new_mode) {
	this->var_spinbox->setVisible(new_mode);
	this->var_doublespinbox->setVisible(!new_mode);
}
double SettingWidget::Minimum() {
	return this->m_min;
}
void SettingWidget::setMinimum(double new_value) {
	this->m_min = new_value;
	this->UpdateRanges();
}
double SettingWidget::Maximum() {
	return this->m_max;
}
void SettingWidget::setMaximum(double new_value) {
	this->m_max = new_value;
	this->UpdateRanges();
}
double SettingWidget::Value() {
	return this->m_value;
}
void SettingWidget::setValue(double new_value) {
	if (new_value != this->m_value) {
		this->m_value = new_value;

		this->var_doublespinbox->blockSignals(true);
		this->var_doublespinbox->setValue(this->m_value);
		this->var_doublespinbox->blockSignals(false);

		this->var_spinbox->blockSignals(true);
		this->var_spinbox->setValue(int(this->m_value));
		this->var_spinbox->blockSignals(false);

		this->var_slider->blockSignals(true);
		this->var_slider->setValue(int(((this->m_value - this->m_min) / (this->m_max - this->m_min)) * this->m_single_steps));
		this->var_slider->blockSignals(false);

		emit this->valueChanged(this->m_value);
	}
}
double SettingWidget::SingleSteps() {
	return this->m_single_steps;
}
void SettingWidget::setSingleSteps(double new_value) {
	this->m_single_steps = new_value;
	this->UpdateRanges();
}
int SettingWidget::Decimals() {
	return this->var_doublespinbox->decimals();
}
void SettingWidget::setDecimals(int new_value) {
	this->var_doublespinbox->setDecimals(new_value);
}