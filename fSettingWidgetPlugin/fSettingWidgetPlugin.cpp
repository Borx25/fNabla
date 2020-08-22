#include "fSettingWidgetPlugin.h"

fSettingWidgetPlugin::fSettingWidgetPlugin(QObject *parent)
    : QObject(parent)
{
    initialized = false;
}

void fSettingWidgetPlugin::initialize(QDesignerFormEditorInterface * /*core*/)
{
    if (initialized)
        return;

    initialized = true;
}

bool fSettingWidgetPlugin::isInitialized() const
{
    return initialized;
}

QWidget *fSettingWidgetPlugin::createWidget(QWidget *parent)
{
    return new SettingWidget(parent);
}

QString fSettingWidgetPlugin::name() const
{
    return "SettingWidget";
}

QString fSettingWidgetPlugin::group() const
{
    return "My Plugins";
}

QIcon fSettingWidgetPlugin::icon() const
{
    return QIcon();
}

QString fSettingWidgetPlugin::toolTip() const
{
    return QString("Widget for value control");
}

QString fSettingWidgetPlugin::whatsThis() const
{
    return QString("Contains a label and links a QSlider with a QSpinBox or QDoubleSpinBox");
}

bool fSettingWidgetPlugin::isContainer() const
{
    return false;
}

QString fSettingWidgetPlugin::domXml() const
{
    return "<widget class=\"SettingWidget\" name=\"SettingWidget\">\n"
        " <property name=\"geometry\">\n"
        "  <rect>\n"
        "   <x>0</x>\n"
        "   <y>0</y>\n"
        "   <width>250</width>\n"
        "   <height>40</height>\n"
        "  </rect>\n"
        " </property>\n"
        "</widget>\n";
}

QString fSettingWidgetPlugin::includeFile() const
{
    return "fSettingWidget.h";
}
