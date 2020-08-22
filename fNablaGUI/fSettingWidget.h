#pragma once

#include <QtWidgets/QWidget>
#include <QtWidgets/QLabel>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QSlider>
#include <QtWidgets/QHBoxLayout>

#ifdef EXPORT_PLUGIN
#define API Q_DECL_EXPORT
#else
#define API
#endif

class API SettingWidget : public QWidget
{
	Q_OBJECT

	//components
	QHBoxLayout* internal_layout;
	QLabel* var_label;
	QSlider* var_slider;
	QDoubleSpinBox* var_doublespinbox;
	QSpinBox* var_spinbox;

	//member variables
	double m_min = 0.0;
	double m_max = 1.0;
	double m_single_steps = 100.0;
	double m_value = 0.5;

	//properties
	Q_PROPERTY(int decimals READ Decimals WRITE setDecimals)
	Q_PROPERTY(double singleSteps READ SingleSteps WRITE setSingleSteps)
	Q_PROPERTY(double value READ Value WRITE setValue)
	Q_PROPERTY(double maximum READ Maximum WRITE setMaximum)
	Q_PROPERTY(double minimum READ Minimum WRITE setMinimum)
	Q_PROPERTY(QString name READ Name WRITE setName)
	Q_PROPERTY(bool discrete READ isDiscrete WRITE setDiscrete)

private slots:
	void UpdateRanges();
	void UpdateValueFromSlider();
	void UpdateValueFromSpinbox();
	void UpdateValueFromDoubleSpinbox();

public:
	SettingWidget(QWidget* parent = Q_NULLPTR);

public slots:

	const QString& Name();
	void setName(const QString& newName);

	bool isDiscrete();
	void setDiscrete(bool new_value);

	double Minimum();
	void setMinimum(double new_value);

	double Maximum();
	void setMaximum(double new_value);

	double SingleSteps();
	void setSingleSteps(double new_value);

	int Decimals();
	void setDecimals(int new_value);

	double Value();
	void setValue(double new_value);


signals:
	void valueChanged(double value);
};