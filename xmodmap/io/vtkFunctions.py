# Libraries to Import
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import os
import glob
import scipy.interpolate as spi
import scipy as sp
# these methods from scipy will be used for displaying some images
from scipy.linalg import eigh
from scipy.stats import norm

# Tools for Generating Paths
from numpy import linalg as LA
from itertools import combinations
from scipy.special import comb
from scipy import stats

# Tools for caching
from os import path

# Tools for saving images
import scipy.misc
from scipy import io

# Tools for Reading in File
import nibabel as nib
from nibabel import processing
from nibabel import funcs

import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util.numpy_support import vtk_to_numpy


def crossProduct(a,b):
    c = np.zeros((3,1))
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c

def computeSurfaceIntegral(fValues,v,f):
    '''
    Compute the integral of fValues (given per vertices) by computing average 
    value of v for each f (faces) and multiplying by surface area
    compute surface area with cross product 
    '''
    if (np.min(f) > 0):
        fnew = (f-1).astype(int)
    else:
        fnew = (f).astype(int) # assume vertices are 0 indexed
    tot = 0
    for face in f:
        v1 = v[face[0]]
        v2 = v[face[1]]
        v3 = v[face[2]]
        area = 0.5*np.sum(crossProduct(v2-v1,v3-v1)**2)
        avg = (fValues[face[0]] + fValues[face[1]] + fValues[face[2]])/3.0
        tot += avg*area
    return tot

def computeSurfaceIntegralPerV(fValues,v,f):
    vAreas = np.zeros((len(v),1))
    for face in f:
        v1 = v[face[0]]
        v2 = v[face[1]]
        v3 = v[face[2]]
        area = 0.5*np.sum(crossProduct(v2-v1,v3-v1)**2)
        vAreas[face[0]] += area/3.0
        vAreas[face[1]] += area/3.0
        vAreas[face[2]] += area/3.0
    return np.sum(np.squeeze(vAreas)*np.squeeze(fValues))
    
def convertVTKtoMAT(vtkFile,matFile):
    '''
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtkFile)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    polydata = reader.GetOutput()
    num_faces = polydata.GetPolys().GetNumberOfCells()
    num_pts_in_faces = polydata.GetPolys().GetData().GetNumberOfTuples()
    numpy_points = dsa.WrapDataObject(polydata).Points
    numpy_polygons = dsa.WrapDataObject(polydata).Polygons
    '''
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtkFile)
    reader.Update()

    polydata = reader.GetOutput()

    points = polydata.GetPoints()
    array = points.GetData()
    numpy_points = vtk_to_numpy(array)
    print(numpy_points.shape)
    
    cells = polydata.GetPolys()
    nCells = cells.GetNumberOfCells()
    print(nCells)
    if (nCells == 0):
        dictToSave = dict()
        dictToSave['vertices'] = numpy_points
        sp.io.savemat(matFile,dictToSave)
        return numpy_points,None
    
    array = cells.GetData()
    # This holds true if all polys are of the same kind, e.g. triangles.
    assert(array.GetNumberOfValues()%nCells==0)
    nCols = array.GetNumberOfValues()//nCells
    numpy_cells = vtk_to_numpy(array)
    numpy_polygons = numpy_cells.reshape((-1,nCols))
    
    dictToSave = dict()
    dictToSave['vertices'] = numpy_points
    dictToSave['faces'] = numpy_polygons[:,1:]
    
    sp.io.savemat(matFile,dictToSave)
    return numpy_points,numpy_polygons

def getThickness(temp,evol,savename):
    # get and save displacements
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(evol)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    polydata = reader.GetOutput()
    disp = polydata.GetPointData().GetArray(3) # assume displacement is fourth array # 4 if brain 4
    
    reader2 = vtk.vtkPolyDataReader()
    reader2.SetFileName(temp)
    reader2.ReadAllScalarsOn()
    reader2.ReadAllVectorsOn()
    reader2.Update()
    polydata2 = reader2.GetOutput()
    
    polydata2.GetPointData().AddArray(disp)
    writer = vtk.vtkPolyDataWriter()
    #writer = vtk.vtkBYUWriter()
    writer.SetFileName(savename)
    writer.SetInputData(polydata2)
    writer.Update()
    writer.Write()
    # save mat as well
    dictToSave = dict()
    dictToSave['displacement'] = vtk_to_numpy(disp)
    #sp.io.savemat(os.path.split(savename)[0]+'/erc_thickness_0803.mat',dictToSave)
    snMat = savename.replace(".vtk","_thickness.mat")
    sp.io.savemat(snMat,dictToSave)
    return
   
def combineParts(filenames,savename):
    reader = vtk.vtkPolyDataReader()
    append = vtk.vtkAppendPolyData()

    for file in filenames:
        reader.SetFileName(file)
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.Update()
        polydata = vtk.vtkPolyData()
        polydata.ShallowCopy(reader.GetOutput())
        append.AddInputData(polydata)

    append.Update()    

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(savename)
    writer.SetInputData(append.GetOutput())
    writer.Update()
    writer.Write()
    return

def writeVTK(YXZ,features,featureNames,savename,polyData=None,fields=None,fieldNames=None):
    '''
    Write YXZ coordinates (assume numpts x 3 as X,Y,Z in vtk file)
    polyData should be in format 3 vertex, vertex, vertex (0 based numbering)
    '''
    f_out_data = []

    # Version 3.0 header
    f_out_data.append("# vtk DataFile Version 3.0\n")
    f_out_data.append("Surface Data\n")
    f_out_data.append("ASCII\n")
    f_out_data.append("DATASET POLYDATA\n")
    f_out_data.append("\n")

    num_pts = YXZ.shape[0]
    f_out_data.append("POINTS %d float\n" % num_pts)
    for pt in range(num_pts):
        f_out_data.append("%f %f %f\n" % (YXZ[pt,1], YXZ[pt,0], YXZ[pt,2])) # x,y,z
    
    if (polyData is not None):
        r = polyData.shape[0]
        c = polyData.shape[1]
        if (c > 3):
            f_out_data.append("POLYGONS %d %d\n" % (r,c*r))
        else:
            f_out_data.append("LINES %d %d\n" % (r,c*r))
        for i in range(r):
            if (c == 4):
                f_out_data.append("%d %d %d %d\n" % (polyData[i,0], polyData[i,1], polyData[i,2], polyData[i,3]))
            elif (c == 3):
                f_out_data.append("%d %d %d\n" % (polyData[i,0], polyData[i,1], polyData[i,2]))
    f_out_data.append("POINT_DATA %d\n" % num_pts)
    fInd = 0;
    for f in featureNames:
        f_out_data.append("SCALARS " + f + " float 1\n")
        f_out_data.append("LOOKUP_TABLE default\n")
        fCurr = features[fInd]
        for pt in range(num_pts):
            f_out_data.append("%.9f\n" % fCurr[pt])
        fInd = fInd + 1
    if fields is not None:
        for f in range(len(fields)):
            f_out_data.append("VECTORS " + fieldNames[f] + " float\n")
            fieldCurr = fields[f]
            #for pt in range(num_pts):
                

    # Write output data array to file
    with open(savename, "w") as f_out:
        f_out.writelines(f_out_data)
    return

def writeVTKCells(YXZ,features,featureNames,savename,polyData=None):
    '''
    Write YXZ coordinates (assume numpts x 3 as X,Y,Z in vtk file)
    polyData should be in format 3 vertex, vertex, vertex (0 based numbering)
    '''
    f_out_data = []

    # Version 3.0 header
    f_out_data.append("# vtk DataFile Version 3.0\n")
    f_out_data.append("Surface Data\n")
    f_out_data.append("ASCII\n")
    f_out_data.append("DATASET POLYDATA\n")
    f_out_data.append("\n")

    num_pts = YXZ.shape[0]
    f_out_data.append("POINTS %d float\n" % num_pts)
    for pt in range(num_pts):
        f_out_data.append("%f %f %f\n" % (YXZ[pt,1], YXZ[pt,0], YXZ[pt,2])) # x,y,z
    
    if (polyData is not None):
        r = polyData.shape[0]
        c = polyData.shape[1]
        f_out_data.append("POLYGONS %d %d\n" % (r,c*r))
        for i in range(r):
            f_out_data.append("%d %d %d %d\n" % (polyData[i,0], polyData[i,1], polyData[i,2], polyData[i,3]))
    f_out_data.append("CELL_DATA %d\n" % r)
    fInd = 0;
    for f in featureNames:
        f_out_data.append("SCALARS " + f + " float 1\n")
        f_out_data.append("LOOKUP_TABLE default\n")
        fCurr = features[fInd]
        print("shape of features")
        print(fCurr.shape)
        for pt in range(r):
            f_out_data.append("%f\n" % fCurr[pt])
        fInd = fInd + 1

    # Write output data array to file
    with open(savename, "w") as f_out:
        f_out.writelines(f_out_data)
    return

def writeVarifoldVTK(YXZ,features,featureNames,savename,polyData=None):
    '''
    Write YXZ coordinates (assume numpts x 3 as X,Y,Z in vtk file)
    polyData should be in format 3 vertex, vertex, vertex (0 based numbering)
    '''
    f_out_data = []

    # Version 3.0 header
    f_out_data.append("# vtk DataFile Version 3.0\n")
    f_out_data.append("Surface Data\n")
    f_out_data.append("ASCII\n")
    f_out_data.append("DATASET POLYDATA\n")
    f_out_data.append("\n")

    num_pts = YXZ.shape[0]
    f_out_data.append("POINTS %d float\n" % num_pts)
    for pt in range(num_pts):
        f_out_data.append("%f %f %f\n" % (YXZ[pt,1], YXZ[pt,0], YXZ[pt,2])) # x,y,z
    
    if (polyData is not None):
        r = polyData.shape[0]
        c = polyData.shape[1]
        f_out_data.append("POLYGONS %d %d\n" % (r,c*r))
        for i in range(r):
            f_out_data.append("%d %d %d %d\n" % (polyData[i,0], polyData[i,1], polyData[i,2], polyData[i,3]))
    f_out_data.append("POINT_DATA %d\n" % num_pts)
    fInd = 0;
    for f in featureNames:
        f_out_data.append("SCALARS " + f + " float 1\n")
        f_out_data.append("LOOKUP_TABLE default\n")
        fCurr = features[fInd]
        print("shape of features")
        print(fCurr.shape)
        for pt in range(num_pts):
            f_out_data.append("%f\n" % fCurr[pt])
        fInd = fInd + 1

    # Write output data array to file
    with open(savename, "w") as f_out:
        f_out.writelines(f_out_data)
    return

def matToVTK(matFile,savename,vString='V',fString='F',featNames=['Krimer_Labels'],featIdent=['label'],takeMax=True):
    params = sp.io.loadmat(matFile, appendmat=False, struct_as_record=False)
    v = np.asarray(params[vString])
    YXZ = np.copy(v)
    if (vString != 'YXZ'):
        YXZ[:,0] = v[:,1]
        YXZ[:,1] = v[:,0]

    if (fString != 'polys'):
        f = np.asarray(params[fString])-1 # vtk should be 0 based 
    else:
        f = np.asarray(params[fString])
    feats = []
    for fi in featIdent:
        if (takeMax):
            feats.append(np.argmax(np.asarray(params[fi]),axis=-1)+1)
        else:
            feats.append(np.asarray(params[fi]))
        
    polydata = np.zeros((f.shape[0],4))
    polydata[:,0] = 3
    polydata[:,1:] = f
    
    writeVTK(YXZ,feats,featNames,savename,polydata)
    return

def moveToMai(matFile,vtkFile,savename,inverse=-1,shift=[0,0,0]):
    '''
    Assume matFile to move byu (so is XYZ coordinates, same as in VTK)
    '''
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtkFile)
    reader.Update()

    polydata = reader.GetOutput()

    points = polydata.GetPoints()
    array = points.GetData()
    coords = vtk_to_numpy(array)
    params = sp.io.loadmat(matFile, appendmat=False, struct_as_record=False)
    Ainv = np.asarray(params['A'])
    if (inverse > 0):
        Ainv = np.linalg.inv(Ainv)
        print("doing inverse")
        
    coordsRet = np.zeros_like(coords)
    coordsRet[...,0] = Ainv[0,0]*coords[...,0] + Ainv[0,1]*coords[...,1] + Ainv[0,2]*coords[...,2] + Ainv[0,3]
    coordsRet[...,1] = Ainv[1,0]*coords[...,0] + Ainv[1,1]*coords[...,1] + Ainv[1,2]*coords[...,2] + Ainv[1,3]
    coordsRet[...,2] = Ainv[2,0]*coords[...,0] + Ainv[2,1]*coords[...,1] + Ainv[2,2]*coords[...,2] + Ainv[2,3]
    
    if (inverse > 0):
        tmp = np.zeros_like(coordsRet)
        tmp[...,0] = coordsRet[...,1]
        tmp[...,1] = coordsRet[...,0]
        tmp[...,2] = coordsRet[...,2]
        coordsRet = tmp
    coordsRet[...,0] += shift[0]
    coordsRet[...,1] += shift[1]
    coordsRet[...,2] += shift[2]
    
    print("ranges")
    print(str(np.min(coordsRet[...,0])) + ", " + str(np.min(coordsRet[...,1])) + ", " + str(np.min(coordsRet[...,2])))
    print(str(np.max(coordsRet[...,0])) + ", " + str(np.max(coordsRet[...,1])) + ", " + str(np.max(coordsRet[...,2])))
    
    Points = vtk.vtkPoints()
    Points.SetData(numpy_to_vtk(coordsRet))
    Points.SetNumberOfPoints(coordsRet.shape[0])
    polydata.SetPoints(Points)
    writer = vtk.vtkPolyDataWriter()
    #writer = vtk.vtkBYUWriter()
    writer.SetFileName(savename)
    writer.SetInputData(polydata)
    writer.Update()
    writer.Write()
    return


def flipVTK(vtkFile):
    '''
    Flips over first axis by default 
    '''
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtkFile)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()

    polydata = reader.GetOutput()

    points = polydata.GetPoints()
    array = points.GetData()
    coords = vtk_to_numpy(array)

    coords[:,0] = -coords[:,0]
    
    Points = vtk.vtkPoints()
    Points.SetData(numpy_to_vtk(coords))
    Points.SetNumberOfPoints(coords.shape[0])
    polydata.SetPoints(Points)
    writer = vtk.vtkPolyDataWriter()
    #writer = vtk.vtkBYUWriter()
    writer.SetFileName(vtkFile.replace(".vtk","_flipped.vtk"))
    writer.SetInputData(polydata)
    writer.Update()
    writer.Write()
    return

def removeZeros(vtkFile,featNumber=0,thresh=0):
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtkFile)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()

    polydata = reader.GetOutput()
    disp = polydata.GetPointData().GetArray(featNumber)
    i = disp > thresh
    points = polydata.getPoints()
    newpoints = points[i,...]
    polydata.setPoints(newpoints)
    
    writer = vtk.vtkPolyDataWriter()
    #writer = vtk.vtkBYUWriter()
    writer.SetFileName(vtkFile.replace(".vtk","_thresh.vtk"))
    writer.SetInputData(polydata)
    writer.Update()
    writer.Write()
    
    return
 
