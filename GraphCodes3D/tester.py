#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 15:00:51 2021

@author: kollar2
"""

import numpy
from GraphCodes3D.unit_cell_3D import UnitCell3D


#get_coords (basically identification of identical points in 3D)

resonators = numpy.zeros((2,6))

resonators[0,:] = [0,0,0, 1,0,0]
resonators[1,:] = [1.0001,0,0, 1,1,0]

# temp1 = resonators[0,3:]

# temp2 = resonators[1,0:3]

# print(temp1 ==temp2)


# temp3 = tuple(temp1)
# temp4 = tuple(temp2)
# print(temp3 == temp4)



def get_coords(resonators, roundDepth = 3):
    '''
    take in a set of resonators and calculate the set of end points.
    
    Will round all coordinates the the specified number of decimals.
    
    Should remove all redundancies.
    '''
    temp1 = resonators[:,0:3]
    temp2 = resonators[:,3:]
    coords_overcomplete = numpy.concatenate((temp1, temp2))
    
    coords = numpy.unique(numpy.round(coords_overcomplete, roundDepth), axis = 0)
    
    return coords
    
temp = get_coords(resonators)
    




#####
#rotations for projection view
#####


def rotate_resonators2D(resonators, theta):
    '''
    take matrix of resonators and rotate them by angle theta (in radians)
    
    returns modified resonators 
    '''
    
    newResonators = numpy.zeros(resonators.shape)
    
    newResonators[:,0] = resonators[:,0]*numpy.cos(theta) - resonators[:,1]*numpy.sin(theta)
    newResonators[:,1] = resonators[:,0]*numpy.sin(theta) + resonators[:,1]*numpy.cos(theta)
    
    newResonators[:,2] = resonators[:,2]*numpy.cos(theta) - resonators[:,3]*numpy.sin(theta)
    newResonators[:,3] = resonators[:,2]*numpy.sin(theta) + resonators[:,3]*numpy.cos(theta)
    
    return newResonators



def rotate_resonators(resonators, theta, phi):
    '''rotate the coordinates of resonators into a projection view  
    
    theta is the angle to the z axis.
    
    phis is the aximuthal angle to the x axis.
    
    '''
    
    
    initial_points1 = resonators[:,0:3]
    initial_points2 = resonators[:,3:]
    # return initial_points1, initial_points2
    
    #first rotate by the angle with respect to the z axis
    new_points1 = numpy.zeros(initial_points1.shape)
    new_points2 = numpy.zeros(initial_points2.shape)
    
    new_points1[:,0] = initial_points1[:,0]*numpy.cos(theta) - initial_points1[:,2]*numpy.sin(theta)
    new_points1[:,1] = initial_points1[:,1]
    new_points1[:,2] = initial_points1[:,0]*numpy.sin(theta) + initial_points1[:,2]*numpy.cos(theta)
    
    new_points2[:,0] = initial_points2[:,0]*numpy.cos(theta) - initial_points2[:,2]*numpy.sin(theta)
    new_points2[:,1] = initial_points2[:,1]
    new_points2[:,2] = initial_points2[:,0]*numpy.sin(theta) + initial_points2[:,2]*numpy.cos(theta)
    
    #store the rotate coordinates
    initial_points1 = numpy.copy(new_points1)
    initial_points2 = numpy.copy(new_points2)
    
    
    
    #now do the phi rotation
    new_points1 = numpy.zeros(initial_points1.shape)
    new_points2 = numpy.zeros(initial_points2.shape)
    
    new_points1[:,0] = initial_points1[:,0]*numpy.cos(phi) - initial_points1[:,1]*numpy.sin(phi)
    new_points1[:,1] = initial_points1[:,0]*numpy.sin(phi) + initial_points1[:,1]*numpy.cos(phi)
    new_points1[:,2] = initial_points1[:,2]
    
    new_points2[:,0] = initial_points2[:,0]*numpy.cos(phi) - initial_points2[:,1]*numpy.sin(phi)
    new_points2[:,1] = initial_points2[:,0]*numpy.sin(phi) + initial_points2[:,1]*numpy.cos(phi)
    new_points2[:,2] = initial_points2[:,2]
    
    newResonators = numpy.concatenate((new_points1, new_points2), axis = 1)

    return newResonators


temp = rotate_resonators(resonators, 0, numpy.pi/4) 
   
# print(numpy.round(resonators,3))
# print('   ')
# print(numpy.round(temp,3)) 
    
    
    
    
def rotate_coordinates(coords, theta, phi):
    '''rotate a set of points into projection view
    
    theta is the angle from the z axis
    
    phi is the azimuthal angle from the x axis
    
    expects the coordinates to come a row vectors [:,0:3] = [x,y,z]

    '''
    
    
    
    #do the theta rotation
    new_points = numpy.zeros(coords.shape)
    new_points[:,0] = coords[:,0]*numpy.cos(theta) - coords[:,2]*numpy.sin(theta)
    new_points[:,1] = coords[:,1]
    new_points[:,2] = coords[:,0]*numpy.sin(theta) + coords[:,2]*numpy.cos(theta)
    
    #store the new values
    coords = numpy.copy(new_points)
    
    #now do the phi rotation
    new_points = numpy.zeros(coords.shape)
    
    new_points[:,0] = coords[:,0]*numpy.cos(phi) - coords[:,1]*numpy.sin(phi)
    new_points[:,1] = coords[:,0]*numpy.sin(phi) + coords[:,1]*numpy.cos(phi)
    new_points[:,2] = coords[:,2]
    
    return new_points
    
    
# coords_0 = get_coords(resonators)    
    
# newCoords = rotate_coordinates(coords_0, 0,numpy.pi/4)
    
    
# print(coords_0)
# print('   ')
# print(newCoords)
    
    
    
    
    
    
######
#root graph redundancies
#####  

roundDepth = 3
def check_redundnacy(site, svec_all, shift1, shift2, shift3):
    '''
    check_redundnacy _summary_

    :param site: _description_
    :type site: _type_
    :param svec_all: _description_
    :type svec_all: _type_
    :param shift1: _description_
    :type shift1: _type_
    :param shift2: _description_
    :type shift2: _type_
    :param shift3: _description_
    :type shift3: _type_
    :return: _description_
    :rtype: _type_
    '''    
    # vec1 = numpy.round(self.a1[0] + 1j*self.a1[1], roundDepth)
    # vec2 = numpy.round(self.a2[0] + 1j*self.a2[1], roundDepth)
    
    shiftVec = shift1*a1 + shift2*a2 + shift3*a3
    
    shiftedCoords = numpy.zeros(svec_all.shape)
    shiftedCoords[:,0] = svec_all[:,0] + shiftVec[0]
    shiftedCoords[:,1] = svec_all[:,1] + shiftVec[1]
    shiftedCoords[:,2] = svec_all[:,2] + shiftVec[2]

    check  = numpy.round(site - shiftedCoords, roundDepth)

    redundancies = numpy.where((check  == (0, 0,0)).all(axis=1))[0] 
    return redundancies
    
  

coords_0 = get_coords(resonators)     
a1 = numpy.asarray([1,0,0])
a2 = numpy.asarray([0,1,0])
a3 = numpy.asarray([0,0,1])

shift1 = -1
shift2 = 0
shift3 = 0
shiftVec = shift1*a1 + shift2*a2 + shift3*a3

svec_all = numpy.copy(coords_0)
shiftedCoords = numpy.zeros(svec_all.shape)
shiftedCoords[:,0] = svec_all[:,0] + shiftVec[0]
shiftedCoords[:,1] = svec_all[:,1] + shiftVec[1]
shiftedCoords[:,2] = svec_all[:,2] + shiftVec[2]

site = coords_0[0,:]
check  = site - shiftedCoords

# temp = check_redundnacy(coords_0[0,:], coords_0, 1, 0, 0)


# temp = numpy.where((check  == (0, 0,0)).all(axis=1))[0]






#####
#auto generation of SD links
#####

cubeCell = UnitCell3D('cubic', a1 = [1,0,0], a2 = [0,1,0], a3 = [0,0,1])

'''
function to autogenerate the links between two sets of resonators
deltaA1 and deltaA2 specify how many lattice vectors the two cells are seperated by
in the first (~x) and second  (~y) lattice directions

deltaA3 is the same for the new z direction

could be twice the same set, or it could be two different unit cells.

will return a matrix of all the links [start, target, deltaA1, deltaA2, start_polarity, end_polarity]

'''
deltaA1 = -1
deltaA2 = 0
deltaA3 = 0

ress1 = cubeCell.resonators
len1 = ress1.shape[0]

#find the new unit cell
xmask = numpy.zeros((cubeCell.numSites,6))
ymask = numpy.zeros((cubeCell.numSites,6))
zmask = numpy.zeros((cubeCell.numSites,6))
xmask[:,0] = 1
xmask[:,3] = 1

ymask[:,1] = 1
ymask[:,4] = 1

zmask[:,2] = 1
zmask[:,5] = 1

xOffset = deltaA1*cubeCell.a1[0] + deltaA2*cubeCell.a2[0] + deltaA3*cubeCell.a3[0]
yOffset = deltaA1*cubeCell.a1[1] + deltaA2*cubeCell.a2[1] + deltaA3*cubeCell.a3[1]
zOffset = deltaA1*cubeCell.a1[2] + deltaA2*cubeCell.a2[2] + deltaA3*cubeCell.a3[2]
ress2 = ress1 + xOffset*xmask + yOffset*ymask + zOffset*zmask

#place to store the links
# linkMat = numpy.zeros((len1*4+len1*4,6))  #I'm pretty sure this only works for low degree
maxLineCoordination = (cubeCell.maxDegree-1)*2
linkMat = numpy.zeros((len1*maxLineCoordination+len1*maxLineCoordination,7))

#find the links

#round the coordinates to prevent stupid mistakes in finding the connections
plusEnds = numpy.round(ress2[:,0:3],3)
minusEnds = numpy.round(ress2[:,3:],3)

extraLinkInd = 0
# for resInd in range(0,ress1.shape[0]):
for resInd in [1]:
    res = numpy.round(ress1[resInd,:],3)
    x1 = res[0]
    y1 = res[1]
    z1 = res[2]
    
    x0 = res[3]
    y0 = res[4]
    z0 = res[5]

    plusPlus = numpy.where((plusEnds == (x1, y1,z1)).all(axis=1))[0]
    minusMinus = numpy.where((minusEnds == (x0, y0,z0)).all(axis=1))[0]
    
    plusMinus = numpy.where((minusEnds == (x1, y1,z1)).all(axis=1))[0] #plus end of new res, minus end of old
    minusPlus = numpy.where((plusEnds == (x0, y0,z0)).all(axis=1))[0]
    
    for ind in plusPlus:
        if ind == resInd:
            #self link
            pass
        else:
            linkMat[extraLinkInd,:] = [resInd, ind, deltaA1, deltaA2, deltaA3, 1,1]
            extraLinkInd = extraLinkInd+1
            
    for ind in minusMinus:
        if ind == resInd:
            #self link
            pass
        else:
            linkMat[extraLinkInd,:] = [resInd, ind, deltaA1, deltaA2, deltaA3,  0,0]
            extraLinkInd = extraLinkInd+1
            
    for ind in plusMinus:
        linkMat[extraLinkInd,:] = [resInd, ind, deltaA1, deltaA2, deltaA3,  1,0]
        extraLinkInd = extraLinkInd+1
        
    for ind in minusPlus:
        linkMat[extraLinkInd,:] = [ resInd, ind, deltaA1, deltaA2, deltaA3,  0,1]
        extraLinkInd = extraLinkInd+1

#clean the skipped links away 
linkMat = linkMat[~numpy.all(linkMat == 0, axis=1)]  


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


cubeLGCell_manual = UnitCell3D('line_graph_of_cube', resonators = resonators, 
                        a1 = [1,0,0], a2 = [0,1,0], a3 = [0,0,1],
                        maxDegree = 12)


# testCell = cubeLGCell_manual
testCell = cubeCell

mind= 0
testRes = 1
for ind in range(0, testCell.SDlinks.shape[0]):
    r1, r2, dx, dy, dz = testCell.SDlinks[ind,:]
    
    if r1==testRes:
    
        print('   ')
        print('   ')
        print('link number:' + str(mind))
        print(testCell.resonators[int(r1),:])
        print(dx)
        print(dy)
        print(dz)
        print(testCell.resonators[int(r2),:])
        
        mind = mind+1
        
print('   ')
print(str(mind))

Hmat = testCell.generate_Bloch_matrix(0,0,0,  modeType = 'FW')

print('    ')
print(str(numpy.sum(Hmat[testRes,:])))
print('   ')
        
# for ind in range(0, testCell.SDlinks.shape[0]):
#     r1, r2, dx, dy, dz = testCell.SDlinks[ind,:]
    
#     print('   ')
#     print('   ')
#     print('link number:' + str(ind))
#     print(testCell.resonators[int(r1),:])
#     print(dx)
#     print(dy)
#     print(dz)
#     print(testCell.resonators[int(r2),:])
    
#     if ind == 2:
#         break






########
#checking the ordering of SD points versus resonators
#####
from .EuclideanLayoutGenerator3D import EuclideanLayout3D
testLattice = EuclideanLayout3D(initialCell = testCell, xcells = 3, ycells = 3, zcells = 3)

for ind in range(0, 10):
    print('    ')
    print(str(ind))
    print('SD point: ' + str(testLattice.SDx[ind]) + ' , ' + str(testLattice.SDy[ind])+ ' , ' + str(testLattice.SDz[ind]))
    ress = testLattice.resonators[ind,:]
    x_av = (ress[0] + ress[3])/2
    y_av = (ress[1] + ress[4])/2
    z_av = (ress[2] + ress[5])/2
    print('resonator mid point: ' + str(x_av) + ' , ' + str(y_av) + ' , ' + str(z_av))









 
    
    