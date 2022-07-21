#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:27:00 2021

@author: kollar2
"""

import numpy
import pylab
import sys
import time


KollarLabClassPath = r'/Users/kollar2/Documents/KollarLab/MainClasses/GraphCodes3D'
if not KollarLabClassPath in sys.path:
    sys.path.append(KollarLabClassPath)


   
# from GeneralLayoutGenerator import GeneralLayout
# from GeneralLayoutGenerator import TreeResonators

from EuclideanLayoutGenerator3D import UnitCell3D
from EuclideanLayoutGenerator3D import EuclideanLayout3D

from PauliCode3D import PauliCode3D


cubeCell = UnitCell3D('cubic')



# cubeLGCell = cubeCell.line_graph_cell()
resonators = numpy.zeros((15,6))
dell = 0.5
resonators[0,:] = [dell,0, 0,   0, dell,0]
resonators[1,:] = [0,dell,0,   -dell, 0,0]
resonators[2,:] = [-dell, 0,0,   0, -dell,0]
resonators[3,:] = [0, -dell,0,   dell, 0,0]

resonators[4,:] = [dell,0, 0,   0, 0,dell]
resonators[5,:] = [0,dell,0,   0, 0,dell]
resonators[6,:] = [-dell, 0,0,   0, 0,dell]
resonators[7,:] = [0, -dell,0,   0, 0,dell]

resonators[8,:] = [dell,0, 0,   0, 0,-dell]
resonators[9,:] = [0,dell,0,   0, 0,-dell]
resonators[10,:] = [-dell, 0,0,   0, 0,-dell]
resonators[11,:] = [0, -dell,0,   0, 0,-dell]

resonators[12,:] = [dell,0, 0,   -dell,0, 0]
resonators[13,:] = [0,dell,0,   0,-dell,0]
resonators[14,:] = [0,0,dell,   0,0, -dell]


cubeLGCell = UnitCell3D('line_graph_of_cube', resonators = resonators, 
                        a1 = [1,0,0], a2 = [0,1,0], a3 = [0,0,1],
                        maxDegree = 12)

hackCell = UnitCell3D('line_graph_of_cube', resonators = resonators, 
                        a1 = [1,0,0], a2 = [0,1,0], a3 = [0,0,1],
                        maxDegree = 12)



code3D = PauliCode3D(cubeLGCell, name = 'cubic', xSize = 2, ySize = 2, zSize = 2,
                  bosonization_type = 'edge',
                  fiducial = False,
                  verbose = True)




for size in [10,100,1000]:
    print('matrix size:' + str(size))
    t0 = time.time()
    npmat = numpy.zeros((size,size))
    nmodmat = code3D.convert_np_to_nmod(npmat)
    t1 = time.time()
    print(numpy.round(t1-t0,5))
    
    null = code3D.find_commutant(nmodmat)
    t2 = time.time()
    print(numpy.round(t2-t1,5))
    
    
    # npmat2 = code3D.convert_nmod_to_np(nmodmat)
    # t2 = time.time()
    # print(numpy.round(t2-t1,5))


