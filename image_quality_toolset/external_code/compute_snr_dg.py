# -------------------------------------------------------------------------
# Copyright (C) 2025 Telespazio
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

import os, sys
from osgeo import gdal, osr, ogr
import glob
from matplotlib import pyplot as plt

from os.path import abspath, dirname

project_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(project_dir)

import QI_visu.edap_snr as snr
import libs.md.md_dgfly as md

'''
    10/12/2025 EDAP Phase2 - Dragon Fly

    03/05/2023 EDAP Phase2

    Workaround for displaying graphics

     import matplotlib
     matplotlib.use('qt5agg')
     from matplotlib import pyplot as plt
'''

global OUTDIR
OUTDIR = '/home/romain/work/edap_process/output'

#Location of input images:
global dataset_dir
dataset_dir = r'C:\D\DATA\20251028-DragonFly\Livraison_28062025\Radiometrie\IN\20250124_libya_4'

global lmin_lmax_dic
lmin_lmax_dic = {
            'BLUE': {'bandId': 1, 'l_min': 70, 'l_max': 110},
            'GREEN': {'bandId': 1, 'l_min': 75, 'l_max': 110},
            'RED': {'bandId': 1, 'l_min': 90, 'l_max': 120},
            'NIR1': {'bandId': 1, 'l_min': 60, 'l_max': 110},
            'NIR2': {'bandId': 1, 'l_min': 60, 'l_max': 130},
            'RE1': {'bandId': 1, 'l_min': 70, 'l_max': 130},
            'RE2': {'bandId': 1, 'l_min': 70, 'l_max': 130},
            'RE3': {'bandId': 1, 'l_min': 70, 'l_max': 130},
            'PAN': {'bandId': 1, 'l_min': 0, 'l_max': 100},
                }

global roi_definition  # Row coordinates (lx), column (ly) and width, all in pixels

def do_snr(product, metadata,
           window_size, sobel_threshold,
           roi_id,
           roi_definition):

    res_file = os.path.join(OUTDIR, 'stat.txt')

    st = md.dgfly(os.path.join(dataset_dir,
                               product,
                               '0',
                               'DATA',
                               metadata))

    if not os.path.exists(res_file):
        with open(res_file, 'a+') as f:
            f.write('id roi band lmin lamx snr snr_ref_radiance snr_jacie snr_jacie_ref_radiance ref_toa \n')

    processing_dic = {'BGRN': {'1': 'BLUE','2': 'GREEN','3': 'RED', '4': 'NIR1'},
                      'REN': {'1': 'RE1','2': 'RE2','3': 'RE3', '4': 'NIR2'},
                      'PAN': {'1': 'PAN'}}
    processing_dic = {'BGRN': {'1': 'BLUE','2': 'GREEN','3': 'RED', '4': 'NIR1'}}
    #processing_dic = {'REN': {'1': 'RE1','2': 'RE2','3': 'RE3', '4': 'NIR2'}}

    snr_object_list = []
    # LOOP ON ALL DG IMAGES (B,R,G,N1, N2, RE?, PAN)

    for rec in processing_dic.items():

        image_f = st.metadata_file.replace('.geojson', '_'+rec[0]+'.tif')
        sds = gdal.Open(image_f,
                        gdal.GA_ReadOnly)
        for i in rec[1].items():

            print('process image band {}'.format(i[1]))

            # Get Parameters
            l_min = lmin_lmax_dic[i[1]]['l_min']
            l_max = lmin_lmax_dic[i[1]]['l_max']
            # Open image for SNR
            im_array = (sds.GetRasterBand(int(i[0]))).ReadAsArray()
            # Read 'Dn to Radiance' scaling factor
            g = st.To_rad_coef_dic[i[1]]

            # Convert to Radiance
            im_array_rad = g * im_array

            # Clip Image based on ROI
            corner_lx = roi_definition[roi_id]['corner_lx']
            corner_ly = roi_definition[roi_id]['corner_ly']
            width = roi_definition[roi_id]['width']

            im_array_rad_test = im_array_rad[corner_lx:corner_lx
                                                       + width,
                                corner_ly:corner_ly + width]

            band_label = 'band_' + (i[1])

            # Create SNR Objet:
            snr_dg = snr.SNR(im_array_rad_test, 0,
                                 window_size= window_size,
                                 snr_precision=1,
                                 L_min=l_min,
                                 L_max=l_max,
                                 band_label=band_label)


            snr_dg.second_method(sob_th=sobel_threshold)

            snr_object_list.append(snr_dg)

            #Export resultsd to test file
            ch = '{:s} {:s} {:s} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} \n' \
                .format(os.path.basename(product),
                        roi_id,
                        snr_dg.band_label,
                        snr_dg.L_min,
                        snr_dg.L_max,
                        snr_dg.snr_formulation_1,
                        snr_dg.snr_formulation_1_reference_radiance,
                        snr_dg.jacie_snr_value,
                        snr_dg.jacie_ref_radiance,
                        0.0)

            with open(res_file, 'a+') as f:
                f.write(ch)

            # Graphical output
            radical = os.path.basename(product).replace('.tif','')
            #ch = 'winsize_'+ str(window_size)
            f_name_1 = '{}_BD{}_snr_w{}_sbth{}.png'.format(radical,band_label,
                                                         str(window_size), str(sobel_threshold))
            f_name_2 = '{}_BD{}_image_w{}_sbth{}.png'.format(radical,band_label,
                                                             str(window_size), str(sobel_threshold))
            f_name_3 = '{}_BD{}_histo_w{}_sbth{}.png'.format(radical,band_label,
                                                             str(window_size), str(sobel_threshold))

            snr_image_res = os.path.join(OUTDIR, f_name_1)
            # snr_dg.display_input_images( f_name_2,f_name_3,
            #                              label='Demo')
            str1 = '\n'.join(['Dragon Fly {}'.format(str(band_label)),
                             ' ROI {}'.format(roi_id),
                             ' Sobel Th {:.2f}'.format(sobel_threshold),
                             ' Win size {:.2f}'.format(window_size)])

            snr_dg.create_snr_graphics(snr_image_res,
                                       title_label = str1,
                                       SHOWFIG = False)
            snr_dg.display_sobel_image()




        # snr.do_show_snr_image(snr_object_list,
        #                       doSave=True,
        #                       doShow=False,
        #                       rad=rad,
        #                       out=outdir)


if __name__ == "__main__":

    # sobel_threshold : Threshold on sobel results;
    sobel_threshold = 3

    # Window size to compute SNR (JACIE)
    window_size = 7

    # Location of window depends on location ... Define one ROI per resolution.
    roi_definition = {'ROI1': {'corner_lx': 2600, 'corner_ly': 2600, 'width': 800}, #BGRN
                      'ROI2': {'corner_lx': 1300, 'corner_ly': 1300, 'width': 400}, #REN
                      'ROI3': {'corner_lx': 5200, 'corner_ly': 5200, 'width': 1600}  #PAN
                      }
    # Select ROI1 for this processing
    roi_id = 'ROI1'  # 'ROI1 ou ROI2'

    # Selected Product:
    product = r'EOSSAT-1_L1C_20250124T081026_20250124T081029_34RGT_R1C1_0'
    md_file = r'EOSSAT-1_HR-250_20250124T081026_20250124T081029_L1C_R1C1.geojson'
    do_snr(product,md_file,
           window_size,
           sobel_threshold,
           roi_id,
           roi_definition)


