#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:49:51 2021

@author: kollar2
"""

import numpy
import pylab
import sys


KollarLabClassPath = r'/Users/kollar2/Documents/KollarLab/MainClasses/GraphCodes3D'
if not KollarLabClassPath in sys.path:
    sys.path.append(KollarLabClassPath)


   
# from GeneralLayoutGenerator import GeneralLayout
# from GeneralLayoutGenerator import TreeResonators

from EuclideanLayoutGenerator3D import UnitCell3D

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



# splitGrapheneCell = grapheneCell.split_cell()
# kapheneCell = splitGrapheneCell.line_graph_cell()


# code3D = PauliCode3D(cubeCell, name = 'cubic', xSize = 4, ySize = 4, 
#                   bosonization_type = 'edge',
#                   fiducial = False,
#                   verbose = False)

code3D = PauliCode3D(cubeLGCell, name = 'cubic', xSize = 4, ySize = 4, 
                  bosonization_type = 'edge',
                  fiducial = False,
                  verbose = True)

# code3D = PauliCode3D(cubeCell, name = 'cubic', xSize = 4, ySize = 4, 
#                   bosonization_type = 'edge',
#                   fiducial = True,
#                   verbose = True)

# code = PauliCode(kapheneCell, name = 'kaphene', xSize = 4, ySize = 4, 
#                   bosonization_type = 'edge',
#                   fiducial = True,
#                   vertex_assignment = [],
#                   verbose = True)



code3D.check_unit_cell_labels()
# code3D.check_vertex_assignment()

############
#manually make a fiducial Hamiltonian

from GeneralLayoutGenerator3D import shift_resonators
from GeneralLayoutGenerator3D import rotate_resonators
from GeneralLayoutGenerator3D import draw_resonators


code3D.fiducialH = code3D.make_fiducial_H(check_plot = code3D.verbose)
fignum = 44

thetas = [-numpy.pi/10, -numpy.pi/10,-numpy.pi/7]
phis = [-numpy.pi/10, -1.9*numpy.pi/8, -numpy.pi/10]

res0 = numpy.copy(code3D.unitcell.resonators)
for vind in range(0, code3D.verticesPerCell):
    print(vind)
    fig = pylab.figure(fignum+vind)
    pylab.clf()
    

    ax = pylab.subplot(1, 1,1)
    code3D.plot_single_term(code3D.fiducialH, vind, fignum = -1, 
                            theta = thetas[vind], phi = phis[vind],
          numberSites = True,
          spotSize = 400,
          axis = ax)
    
    if vind == 0:
        shiftVec = -code3D.unitcell.a1
    elif vind == 1:
        shiftVec = -code3D.unitcell.a2
    else:
        shiftVec = -code3D.unitcell.a3
    res1 = shift_resonators(res0, shiftVec[0], shiftVec[1], shiftVec[2])
    draw_resonators(res1, ax, theta = thetas[vind], phi = phis[vind])    
    
    # pylab.title(str(vind))
    
    pylab.suptitle('Hamiltonian term: ' + str(vind))
        
    pylab.tight_layout()
    pylab.tight_layout()
    pylab.show()


# theta = 0.1
# phi = 0.1
# coordMat = numpy.zeros((len(code3D.cellXs), 3))
# coordMat[:,0] = code3D.cellXs
# coordMat[:,1] = code3D.cellYs
# coordMat[:,2] = code3D.cellZs
# plotPoints1 = code3D.unitcell.rotate_coordinates(coordMat, theta, phi)
# plotPoints2 = code3D.unitcell.rotate_coordinates(code3D.unitcell.SDcoords, theta, phi)



code3D.fiducialDimers = code3D.make_fiducial_dimers(check_plot = code3D.verbose)

# vind = 0
# linkedVertInds = numpy.where(code3D.unitcell.rootLinks[:,0] == vind)[0]
# linkedVerts = numpy.unique(code3D.unitcell.rootLinks[linkedVertInds,1])

# linkedQubitInds = numpy.where(code3D.cellIncidence[:,0] == vind)[0]
# linkedQubits = numpy.unique(code3D.cellIncidence[linkedQubitInds,1])

# tind = 0
# print(tind)
# print('    ')
# for key in code3D.fiducialDimers.keys():
#     print(key)
#     temp = numpy.where( code3D.fiducialDimers[key][:,tind] != 'I')[0]
#     if len(temp>0):
#         for ind in temp:
#             print(str(ind) + ' : ' + code3D.fiducialDimers[key][ind,tind])
#     else:
#         print('  ')







# code3D.H0 = code3D.fiducialH
                
# code3D.H_torus = code3D.unpack_to_torus(code3D.H0)





# 

cubeCell = UnitCell3D('cubic', a1 = [1,0,0], a2 = [0,1,0], a3 = [0,0,1])
torusSize = 3
# torusSize = 5

testCode = PauliCode3D(cubeCell, name = '3DBaconShor', 
                         xSize = torusSize, 
                         ySize = torusSize, 
                         zSize = torusSize, 
                         bosonization_type = 'vertex',
                         fiducial = False,
                         vertex_assignment = [],
                         verbose = False)


















