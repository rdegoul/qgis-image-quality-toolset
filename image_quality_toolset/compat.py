# -------------------------------------------------------------------------
# Copyright (C) 2026 Telespazio
# -------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------
"""
Enums whose location differs between QGIS 3 (Qt5) and QGIS 4 (Qt6).

QGIS 4 removes the unscoped enum aliases (QVariant.String, QgsWkbTypes.Polygon,
QgsProcessingParameterNumber.Double...) that QGIS 3 still accepts.
"""

from qgis.core import Qgis, QgsProcessingParameterFile, QgsProcessingParameterNumber, QgsWkbTypes

# Field types: QMetaType.Type since QGIS 3.38, QVariant.Type before
if Qgis.QGIS_VERSION_INT >= 33800:
    from qgis.PyQt.QtCore import QMetaType

    FIELD_LONGLONG = QMetaType.Type.LongLong
    FIELD_INT = QMetaType.Type.Int
    FIELD_DOUBLE = QMetaType.Type.Double
    FIELD_STRING = QMetaType.Type.QString
    FIELD_DATETIME = QMetaType.Type.QDateTime
else:
    from qgis.PyQt.QtCore import QVariant

    FIELD_LONGLONG = QVariant.LongLong
    FIELD_INT = QVariant.Int
    FIELD_DOUBLE = QVariant.Double
    FIELD_STRING = QVariant.String
    FIELD_DATETIME = QVariant.DateTime

# Geometry type: Qgis.WkbType since QGIS 3.30
WKB_POLYGON = Qgis.WkbType.Polygon if hasattr(Qgis, "WkbType") else QgsWkbTypes.Polygon

# Processing parameter enums: moved to the Qgis namespace in QGIS 3.36
if hasattr(Qgis, "ProcessingNumberParameterType"):
    NUMBER_INTEGER = Qgis.ProcessingNumberParameterType.Integer
    NUMBER_DOUBLE = Qgis.ProcessingNumberParameterType.Double
else:
    NUMBER_INTEGER = QgsProcessingParameterNumber.Integer
    NUMBER_DOUBLE = QgsProcessingParameterNumber.Double

if hasattr(Qgis, "ProcessingFileParameterBehavior"):
    FILE_FOLDER = Qgis.ProcessingFileParameterBehavior.Folder
else:
    FILE_FOLDER = QgsProcessingParameterFile.Folder
