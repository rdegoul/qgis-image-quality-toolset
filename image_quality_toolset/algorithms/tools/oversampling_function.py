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

#! /usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt


def rotate_mat(im, angle,
               oversample=0.25, margin=1,
               debug=False, debug_dir=None):
    '''
    Rotate and oversample an image matrix for MTF analysis.

    Parameters:
    -----------
    im : ndarray
        Input image matrix
    angle : float
        Rotation angle in degrees
    oversample : float
        Oversampling factor (default 0.25)
    margin : int
        Margin in pixels for edge detection (default 1)
    debug : bool
        Enable debug mode to generate visualizations
    debug_dir : str
        Directory to save debug figures
    '''

    step = 1.0
    min_range = int(np.nanmin(im[im > 0]))
    max_range = int(int(np.nanmax(im)))

    angle_rad = np.double(angle)*np.pi/180.0

    # Define the size of output bounding box :
    ANbLig = int(im.shape[0] / np.double(oversample)) + 1
    ANbCol = int(im.shape[1] / np.double(oversample)) + 1
    A = np.zeros((ANbLig, ANbCol), np.float64)

    # Center coordinates - original image (h,w) :
    h = (im.shape[0] - 1)/2.0 + 1
    w = (im.shape[1] - 1)/2.0 + 1

    # Center position for rotation of each ligne
    # Consider pixel position in straight edge as projection of pixel in the slant edge
    # with angle : - angle_rad :

    center_pos = rotation_center_position(im, - angle_rad)

    # Position du point d'inflexion (infl_pos)
    infl_pos = edge_subpixel_location(im, oversample, margin=margin)
    l = np.shape(im)[0]
    x = np.arange(1, l + 1, 1)

    # Parcourir les lignes de l image d entree :
    # (Coordonnees image numpy comme a 0)
    for i in range(0, im.shape[0], 1) :
        #print('Ligne: Iteration   ===> ' , i + 1 , ' / ', im.shape[0])
        #li : selection de la ligne :
        li = (im[i])
        #Definir position du centre, à partir duquel s'applique la rotation
        #center_pos : Position theorique du contour
        j0_scl = center_pos[i][0]
        j_scl = infl_pos[i][0]

        # x_delta: Correction factor (a appliquer ou pas)
        # difference entre position point d inflexion (spline) et position ideal
        # x_delta doit etre faible
        x_delta = j0_scl - j_scl

        #Boucle sur le colonne image
        for j in range(0, im.shape[1], 1):
            v = float(li[j])
            #print(j, v)
            #(xo,yo) repere orthonorme sur la ligne centree en j_scl (Lr)
            xo = j_scl
            yo = 0
            #A1(i,j) exprime dans (Lr) => (x_a1, y_a1)
            x_a1 = (j+1) - xo  # j+1 because j start at 0
            y_a1 = 0
            #A1_p : la projection orthogonal de A1  sur la droite definie
            # à angle_rad de (Lr) et passant par O (xo,yo)
            #A1_p est le milieu de la corde [A1 A2]
            #ou A2 l'image de A1 par la rotation d'angle de 2*angle rad

            x_a2, y_a2 = rotationtheta(-2.0*angle_rad, x_a1, y_a1)
            x_i = (x_a2 + x_a1) / 2.0
            y_i = (y_a2 + y_a1) / 2.0

            #ïtch : Translation pour corriger les variations ligne a ligne du point d'inflexion:
            x_delta2, y_delta2 = rotationtheta(-2.0*angle_rad, x_delta, 0)
            x_delta_i = (x_delta + x_delta2) / 2.0
            y_delta_i = (y_delta2 + 0.0) / 2.0

            #(i_ovf,j_ovf) Coordonnees fractionnaire matrix surechantillonnée :
            pitch_correction = True
            if pitch_correction:
                i_ovf = i + 1 - y_i - y_delta_i
                j_ovf = w + x_i - x_delta_i
            else :
                i_ovf = i + 1 - y_i
                j_ovf = w + x_i

            '''
            i_ovf = i + 1 - y_i
            j_ovf = w + x_i
            '''
            cast_i_ovf = np.round(i_ovf / oversample)*oversample
            cast_j_ovf = np.round(j_ovf / oversample)*oversample

            # Transformation pour indice dans la matrice np, Valeur commence a 0
            i_f = np.around(cast_i_ovf/oversample - 1.0)
            j_f = np.around(cast_j_ovf/oversample - 1.0)
            if i_f >= 0 and i_f < ANbLig:
                if j_f >= 0 and j_f < ANbCol:
                    A[int(i_f)][int(j_f)] = v
    return A, x, infl_pos, center_pos, im

def rotationtheta(theta, i, j):
    # rotation theta dans le repere centre
    # theta radians: angle de rotation
    # i, j position (x, y)
    theta = np.double(theta)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
    u = np.array([[i], [j]])
    v = np.dot(R, u)
    return [(v[0][0]), (v[1][0])]

def rotation_center_position(im_array, angle_rad):
    '''
    Return the sub pixel location in the line of the rotation center;
    point location predicted by rotation angle (angle_rad)
    Input : im_array (2d array)
            angle de rotation
    '''

    #sortie out :
    out = np.zeros((im_array.shape[0], 1), np.float64)
    # Center coordinates in original image (h,w) :
    h = (im_array.shape[0] - 1 )/2.0 + 1
    w = (im_array.shape[1] - 1 )/2.0 + 1

    #sign = - angle_rad / np.abs(angle_rad)
    sign = 1.0
    for i in range(0, im_array.shape[0], 1) :
        d = (im_array.shape[0] - i) - h
        out[i] = w + sign * d * np.tan(angle_rad)

    return out

def edge_subpixel_location(im_array,
                           oversample,
                           margin=5):
    '''
    Return the sub pixel of the inflextion point location
    in an Array of length image height

    Parameters:
    -----------
    im_array : ndarray
        Input image array (2D)
    oversample : float
        Oversampling factor
    margin : int
        Margin in pixels to exclude from edge detection
    debug : bool
        Enable debug mode to generate visualizations
    debug_dir : str
        Directory to save debug figures
    '''

    #sortie out :
    out = np.zeros((im_array.shape[0], 1), np.float64)

    # Store debug data for a few sample rows
    debug_samples = []
    sample_rows = [0, im_array.shape[0] // 4, im_array.shape[0] // 2,
                   3 * im_array.shape[0] // 4, im_array.shape[0] - 1]

    # Parcourir les lignes de l image d entree :
    for i in range(0, im_array.shape[0], 1):
        #li : selection de la ligne :
        li_w = (im_array[i])[
               margin:im_array.shape[1]-margin]
        nan_filter = np.isnan(li_w)
        nan_margin = int(np.argmax(~nan_filter))
        li_w = li_w[np.logical_not(nan_filter)]
        x = np.arange(1, len(li_w)+1, 1)
        try:
            f1 = spline_interpolation(x, li_w)
            xx = np.arange(1, len(li_w), oversample*0.1)
            #from scipy.interpolate import splev splev pas utile
            #yy = splev(xx,f1)
            yy = f1(xx)

            #Convolution (Differentiation)
            v = np.array([-1, 0, 1])
            dd = np.convolve(yy, v, 'same')
            inflexion_point = 1 + np.argmax(np.abs(dd[1:-1]))

            out[i] = xx[inflexion_point] + margin + nan_margin
        except ValueError as e:
            continue
    return out


def spline_interpolation(x, y):
    from scipy.interpolate import interp1d
    #from scipy.interpolate import splrep #Find the B-spline representation of a 1-D curve.
    nan_mask = np.isnan(y)
    x_valid = x[np.logical_not(nan_mask)]
    y_valid = y[np.logical_not(nan_mask)]
    if len(y_valid) < 4:
        raise ValueError(
            f"Cannot build cubic spline: need at least 4 values "
            f"(got {len(y_valid)} valid points."
        )
    # , kind='linear', fill_value='extrapolate')
    # Use CUBIC instead of SLINEAR (SLINEAR BAD RESULTS).)
    f2 = interp1d(x_valid, y_valid, kind='cubic')
    #f2 = splrep(np.array(x), np.array(y))
    return f2


