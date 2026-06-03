# -------------------------------------------------------------------------
# Copyright (C) 2025 Telespazio
# -------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

from qgis.core import QgsProcessingParameterNumber, QgsProcessingParameterEnum
from .base_mtf_estimator_algorithm import BaseMTFEstimatorAlgorithm
from .mtf_knife_edge import MtfKnifeEdge
from ..tools.raster_tools import roi_extraction


class MTFEstimatorAlgorithmKnifeEdge(BaseMTFEstimatorAlgorithm):
    """MTF Estimator using MtfKnifeEdge."""

    ALGORITHM_NAME = 'MTFEstimatorKnifeEdge'
    DISPLAY_NAME = 'Knife Edge Method MTF'
    GROUP_NAME = 'MTF'
    GROUP_ID = 'mtf'

    SCALE = 'SCALE'
    OFFSET = 'OFFSET'
    PX_MARGIN = 'PX_MARGIN'
    EDGE_DIRECTION = 'EDGE_DIRECTION'
    ESF_MODEL = 'ESF_MODEL'
    SAMPLING = 'SAMPLING'
    INPUT_ROTATION = 'INPUT_ROTATION'

    EDGE_DIRECTION_OPTIONS = ['Along Track', 'Across Track']
    EDGE_DIRECTION_VALUES = ['AL', 'CT']

    ESF_MODEL_OPTIONS = ['sigmoid', 'esf_tanh', 'esf_fermi', 'esf_gauss_exp_param', 'esf_erf', 'esf_loess', 'esf_to_eq_space_polynomial']
    ESF_MODEL_VALUES = ['sigmoid', 'esf_tanh', 'esf_fermi', 'esf_gauss_exp_param', 'esf_erf', 'esf_loess', 'esf_to_eq_space_polynomial']

    def createInstance(self):
        return MTFEstimatorAlgorithmKnifeEdge(self.result_widget)
    
    def shortHelpString(self):
        return self.tr('The Knife Edge Method is an optical test technique used to determine the Modulation Transfer Function (MTF) of an imaging system. Instead of directly measuring sinusoidal patterns, the method captures the image of a sharp knife‑edge target. From this captured image, the Edge Spread Function (ESF) is extracted by profiling the intensity transition across the edge. By differentiating the ESF, the Line Spread Function (LSF) is obtained, and taking the Fourier transform of the LSF yields the MTF.')

    def create_mtf(self, vlayer, memraster, band_n, scale, offset, px_margin, edge_direction, esf_model, sampling, feedback, debug_dir=None, input_angle=None):
        return MtfKnifeEdge(vlayer, memraster, band_n, scale, offset, px_margin, edge_direction, esf_model, sampling, input_angle=input_angle, feedback=feedback, debug=False, debug_dir=debug_dir)

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SCALE,
                self.tr("Scale"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.01
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OFFSET,
                self.tr("Offset"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PX_MARGIN,
                self.tr("Pixel margin"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.EDGE_DIRECTION,
                self.tr("Edge direction"),
                options=self.EDGE_DIRECTION_OPTIONS,
                defaultValue=0
            )
        )
        esf_model_param = QgsProcessingParameterEnum(
            self.ESF_MODEL,
            self.tr("ESF model"),
            options=self.ESF_MODEL_OPTIONS,
            defaultValue=0,
            optional=True
        )
        self.addParameter(esf_model_param)
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SAMPLING,
                self.tr("Sampling (oversampling factor)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.2,
                minValue=0.0,
                maxValue=1.0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.INPUT_ROTATION,
                self.tr("Input rotation angle (degrees)"),
                type=QgsProcessingParameterNumber.Double,
                optional=True
            )
        )
        super().initAlgorithm(config=config)
        
    def processAlgorithm(self, parameters, context, feedback):
        scale = self.parameterAsDouble(parameters, self.SCALE, context)
        offset = self.parameterAsDouble(parameters, self.OFFSET, context)
        px_margin = self.parameterAsInt(parameters, self.PX_MARGIN, context)
        edge_direction_index = self.parameterAsEnum(parameters, self.EDGE_DIRECTION, context)
        edge_direction = self.EDGE_DIRECTION_VALUES[edge_direction_index]
        esf_model_index = self.parameterAsEnum(parameters, self.ESF_MODEL, context)
        esf_model = self.ESF_MODEL_VALUES[esf_model_index]
        sampling = self.parameterAsDouble(parameters, self.SAMPLING, context)
        input_angle = self.parameterAsDouble(parameters, self.INPUT_ROTATION, context) if parameters.get(self.INPUT_ROTATION) is not None else None
        raster_layer = self.parameterAsRasterLayer(parameters, self.RASTER, context)
        band_n = self.parameterAsInt(parameters, self.BAND, context)
        vlayer = self.parameterAsVectorLayer(parameters, self.ROI, context)

        output_directory = self.parameterAsFile(parameters, self.OUTPUT_DIRECTORY, context)

        memraster = roi_extraction(raster_layer, band_n, vlayer, context, feedback)

        mtf = self.create_mtf(vlayer, memraster, band_n, scale, offset, px_margin, edge_direction, esf_model, sampling, feedback, output_directory, input_angle)

        self.process_results(mtf, parameters, context, feedback)

        return {}