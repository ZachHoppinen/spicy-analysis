#!/usr/bin/env python3
"""
:This file:

    `vtk2gslib.py`

:Purpose:

    Convert a VTK structured grid into a GSLIB file.

:Usage:

    From a terminal, you can run the command:

        python3 vtk2gslib.py <file.vtk> <file.gslib>

    with the two arguments:
   
        <file.vtk>
            Name of the input VTK file. (Provide the complete path if
            the file is not in the directory where you are running the
            script).
        <file.gslib>
            Name of the output GSLIB file.

:Version:

    0.1 , 2015-05-12 :

        * First version

:Authors:

    Alessandro Comunian

.. notes::

    This script requires Python3.X and the numerical libraries "numpy"
    and "pandas" installed. 

.. limitations::

    See the details in the corresponding functions
"""

import numpy as np
import pandas as pd

#
# Functions
#

def vtk2numpy(file_name, dtype=None, verbose=False):
    '''
    Read a VTK ``STRUCTURED_GRID`` file with one scalar and return it
    into a numpy array.
    
    Parameters:
        in_file: string
            The name of the VTK structured grid input file.
        dtype: numpy.dtype.name, optional
            A type which should be used to "cast" the readed data and
            and the output type of the numpy array. If the default
            value is not specified, then the value extracted from the
            VTK file is used.
        verbose: bool, optional
            A flag to decide to print read information or not. The
            default is *False*. If *True*, then some verbose info is
            printed out.

    Returns:
        This function returns the tuple (data, grid), where:

        data: numpy array
            Numpy array containing the scalars read from the VTK.
        grid : dictionary
            This is a python dictionary containing information about 
            the grid (size, spacing, origin, number of points...)
            
    .. note::
        * Only structured grid files with only one scalar can be
          handled.
        * The type of the data which is recognized in the VTK format
          for the moment is only "float" or "int".

    .. warning::
        * `POINT_DATA` and `CELL_DATA` are both accepted, but you
          should take care yourself about the kind of data you are
          dealing with.
        * The VTK file should be formatted with all the data in one
          column, otherwise the pandas reader will not be able to read
          it correctly.
    '''
    try:
        with open(file_name, 'rb') as in_file:
            # A dictionary to contain info about the grid
            grid = {}
            if verbose :
                print(('\n    Reading file: "{0}"'.format(file_name)))
            # Skip the first 3 lines
            line = in_file.readline()
            line = in_file.readline()
            line = in_file.readline()
            # This line should contain the description of the DATASET
            line = in_file.readline()
            line_split = line.split()
            if line_split[1] != b"STRUCTURED_POINTS":
                print("    Error: 'vtk2numpy' can read only ", end="")
                print("    `STRUCTURED_POINTS` datasets.")
            # The order of the following keywords can be generic...
            for i in range(4):
                line = in_file.readline()
                line_split = line.split()
                if line_split[0] == b'DIMENSIONS':
                    grid['nx']= int(line_split[1])
                    grid['ny']= int(line_split[2])
                    grid['nz']= int(line_split[3])
                elif line_split[0] == b'ORIGIN':
                    grid['ox'] = float(line_split[1])
                    grid['oy'] = float(line_split[2])
                    grid['oz'] = float(line_split[3])
                elif line_split[0] == b'SPACING':
                    grid['dx'] = float(line_split[1])
                    grid['dy'] = float(line_split[2])
                    grid['dz'] = float(line_split[3])
                elif line_split[0] == b'POINT_DATA':
                    grid['type'] = 'points'
                    grid['points']  = int(line_split[1])
                elif line_split[0] == b'CELL_DATA':
                    grid['type'] = 'cells'
                    grid['cells'] = int(line_split[1])
                else:
                    print("    Error in 'vtk2numpy'. Check you keywords.")

            # Read the data type
            line = in_file.readline()
            line_split = line.split()
            data_type = line_split[2]
            if not dtype:
                # If dtype is provided as agrument, it overwrites the
                # dtype of the dataset.
                if data_type in (b'int',):
                    dtype = np.int32
                elif data_type  in (b'float',):
                    dtype = np.float64
                else:
                    dtype = None

            # Skip the line containing the information about the
            # LOOKUP_TABLE
            line = in_file.readline()

            # Read the data as a CSV file
            data_tmp = pd.read_csv(in_file, dtype=dtype,
                                   engine='c',header=None).values
            data = np.reshape(data_tmp, (grid['nx'], grid['ny'], grid['nz']),
                              order='F')

        return data, grid

    except IOError:
        print(('    Error reading file "{0}"'.format(file_name)))
        print('    Check if the file exists...')


def numpy2gslib(data, file_name, grid=None, varname="data"):
    '''
    Convert a numpy array into a GSLIB ASCII file.

    Parameters:
        data : numpy array
            The numpy array to be saved as GSLIB file.
        file_name : string
            The name of the GSLIB file where to save the data.
        grid : python dictionary, optional
            If available, should provide the information about grid
            grid origin ('ox', 'oy' and 'oz') and the spacing ('dx',
            'dy' and 'dz').  Otherwise default values are used
        varname : string, optional
            The name of the variable to be stored in the GSLIB file.

    .. note::
        * Works only with 1D, 2D or 3D numpy arrays.
        * Only one variable per file.
    '''

    try:
        nx, ny, nz = data.shape
    except ValueError:
        print('    Warning (numpy2gslib): input data considered as 2D.')
        nx, ny = data.shape
        nz = 1

    with open(file_name, 'wb') as file_obj:
        # Write the 1st line, which can be a comment, or contain some
        # useful information about the grid.
        if grid is None:
            file_obj.write(('%d %d %d 1.0 1.0 1.0 0.0 0.0 0.0\n1\n'
                            % (nx, ny, nz)
                            ).encode('utf-8'))
        else:
            file_obj.write(('%d %d %d %f %f %f %f %f %f 1\n1\n'% 
                            (nx, ny, nz,
                             grid['dx'], grid['dy'], grid['dz'], 
                             grid['ox'], grid['oy'], grid['oz'])
                            ).encode('utf-8'))

        # Write the name of the variable
        file_obj.write(('%s\n'%(varname)).encode('utf-8'))

        # Write the data
        if data.dtype in ['float']:
            np.savetxt(file_obj, np.reshape(data, nx*ny*nz, order='F'),
                       fmt='%12.4e')
        elif data.dtype in ['int', 'int32']:
            np.savetxt(file_obj, np.reshape(data, nx*ny*nz, order='F'),
                       fmt='%d')
        else:
            print("    ERROR: Wrong input dtype (only 'int' and 'float'"
                  " are OK)")
            print("           Value provided: ", data.dtype)
 

#
# Calling the functions...
#
if __name__ == '__main__':

    import sys
    import os

    # This is the name of the input VTK
    file_vtk = sys.argv[1]

    # Name of the output GSLIB file
    file_gslib = sys.argv[2]
    
    # Read the input VTK file
    print('    Reading file "{0}"'.format(file_vtk))
    data, grid = vtk2numpy(file_vtk)

    print('    Writing file "{0}"'.format(file_gslib))
    # Save the file as GSLIB
    numpy2gslib(data, file_gslib, grid)



