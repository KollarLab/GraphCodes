#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:55:14 2020

@author: kollar2


File to contain newly developed CDS (Cvetkovics-Doob-Sachs) configuration object.

different unit cells are selected by the number form the table at the back of that book.

Current version allows 1D and 2D covers with up to 2 links.

"""

import re
import scipy
import pylab
import numpy


import pickle
import sys
import os.path
import matplotlib.gridspec as gridspec

KollarLabClassPath = r'/Users/kollar2/Documents/KollarLab/MainClasses/'
if not KollarLabClassPath in sys.path:
    sys.path.append(KollarLabClassPath)

#hyperbolicFolderPath = r'/Users/kollar2/Documents/HouckLab/HyperbolicPlanning/'
#euclideanFolderPath = r'/Users/kollar2/Documents/HouckLab/EuclideanLatticePlanning/'
#generalFolderPath = r'/Users/kollar2/Documents/HouckLab/GeneralLayoutCode/'
#if not hyperbolicFolderPath in sys.path:
#    sys.path.append(hyperbolicFolderPath)
#if not euclideanFolderPath in sys.path:
#    sys.path.append(euclideanFolderPath)
#if not generalFolderPath in sys.path:
#    sys.path.append(generalFolderPath)
    
    
#FunctionFolderPath = r'/home/pranav/PhotonicLattices'
DataPickleFolderPath = r'/volumes/ourphoton/Alicia/Layouts/HyperbolicPickles'
#if not FunctionFolderPath in sys.path:
#    sys.path.append(FunctionFolderPath)
   
from GeneralLayoutGenerator import GeneralLayout
from GeneralLayoutGenerator import TreeResonators

from EuclideanLayoutGenerator2 import UnitCell
from EuclideanLayoutGenerator2 import EuclideanLayout

from LayoutGenerator5 import PlanarLayout


from GeneralLayoutGenerator import split_resonators
from GeneralLayoutGenerator import rotate_resonators
from GeneralLayoutGenerator import generate_line_graph
from GeneralLayoutGenerator import shift_resonators
#from GeneralLayoutGenerator import decorate_layout




#%% defaults
    
#############
#defaults
##########
    
    
bigCdefault = 110
smallCdefault = 30

layoutLineColor = 'mediumblue'
layoutCapColor = 'goldenrod'

FWlinkAlpha = 0.7
FWsiteAlpha = 0.6

HWlinkAlpha = 0.8
HWsiteAlpha = 0.6

FWlinkColor = 'dodgerblue'
FWsiteColor = 'lightsteelblue'
FWsiteEdgeColor = 'mediumblue'

HWlinkColor = 'lightsteelblue'
HWminusLinkColor = 'b'
HWsiteColor = 'lightgrey'
HWsiteEdgeColor = 'midnightblue'

stateColor1 = 'gold'
stateEdgeColor1 = 'darkgoldenrod'
stateEdgeWidth1 = 1.25

stateColor2 = 'firebrick'
stateEdgeColor2 = 'maroon'
stateEdgeWidth2 = 1







class CDSconfig(object):
    def __init__(self, cellNum,  coverLinks = '', coverPolarity = '', coverDim = '2d', lattice_type= 'square', mag = 2.5, baseTheta = 0):
        '''start from a number in the table is CKS book and make a basic unit cell, then cover it 
        in 2D.'''

        self.cellNum = cellNum
        self.mag = mag
        self.lattice_type = lattice_type
        self.coverDim = coverDim
        
        self.baseTheta = baseTheta #allow some rotation to help the graphs look nice.


        if self.lattice_type == 'square':
            self.a1 = numpy.asarray([self.mag,0])
#            self.a1 = numpy.asarray([self.mag,0.6]) #!!!!!hack!!!!!
            self.a2 = numpy.asarray([0.0,self.mag])
        else:
            raise ValueError('other types not yet supported. Should be: square')
    #    if lattice_type == 'triangular':
    #        angle = numpy.pi/3
    #        mag = 2.5
    #        a1 = numpy.asarray([mag,0])
    #        a2 = numpy.asarray([numpy.cos(angle)*mag,numpy.sin(angle)*mag])


        if coverLinks == '':
            self.coverLinks = [0,2]
        else:
            self.coverLinks = coverLinks
            
        if coverPolarity == '':
            self.coverPolarity = [0,0]
        else:
            self.coverPolarity = coverPolarity


        if self.cellNum == 26:
            self.numNodes = 10
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([ [0,5], [1,6], [4,9], [1,9], [3,7], [2,0], [0,8]   ])
            self.removals = numpy.asarray([    [0,1], [0,9] ])
            
        elif self.cellNum == 7:
            self.numNodes = 8
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([ [0,3], [7,4], [1,6], [2,5]  ])
            self.removals = numpy.asarray([   ])
            
        elif self.cellNum == 3:
            self.numNodes = 6
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([ [0,3],[1,5],[2,4] ])
            self.removals = numpy.asarray([   ])
            
        elif self.cellNum == 82:
            self.numNodes = 12
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([ [0,3], [11,8], [2,9], [1,6], [10,5], [4,7] ])
            self.removals = numpy.asarray([   ])
            
        elif self.cellNum == 35:
            self.numNodes = 12
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([ [0,6],[1,3], [11,9], [5,7], [4,10],[2,8] ])
            self.removals = numpy.asarray([   ])
            
        elif self.cellNum == 6:
            self.numNodes = 8
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([ [0,4], [1,6], [7,2], [3,5] ])
            self.removals = numpy.asarray([   ])
            
        elif self.cellNum == 105:
            self.numNodes = 12
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([ [0,3], [1,10], [2,5], [4,7], [6,9], [8,11] ])
            self.removals = numpy.asarray([   ])
            
        elif self.cellNum == 1:
            self.numNodes = 4
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([ [0,2] , [1,3]    ])
            self.removals = numpy.asarray([   ])
            
        elif self.cellNum == 52:
            self.numNodes = 12
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([ [0,6], [1,11] , [2,4] , [10, 8], [5,7], [3,9]  ])
            self.removals = numpy.asarray([   ])
            
        elif self.cellNum == 2:
            self.numNodes = 6
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([ [0,3], [1,4], [2,5]  ])
            self.removals = numpy.asarray([   ])   
        
        elif self.cellNum == 1:
            self.numNodes = 4
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([   [0,2], [1,3] ])
            self.removals = numpy.asarray([   ]) 
            
        elif self.cellNum == 5:
            self.numNodes = 8
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([  [2,6], [0,4], [1,3], [7,5] ])
            self.removals = numpy.asarray([   ])
            
        elif self.cellNum == 452:
            self.numNodes = 18
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([  [0,9], [3,12], [6,15], [1,17], [2,4], [5,7], [8,10], [11,13], [14,16] ])
            self.removals = numpy.asarray([   ])
            
        elif self.cellNum == 8:
            self.numNodes = 8
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([  [0,2], [1,7], [3,5], [4,6] ])
            self.removals = numpy.asarray([   ]) 
            
        elif self.cellNum == 16:
            self.numNodes = 10
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([  [0,5], [1,3], [2,6], [4,8], [7,9] ])
            self.removals = numpy.asarray([   ]) 
            
        elif self.cellNum == 110:
            self.numNodes = 12
            self.numBonds = int(3*self.numNodes/2)
            
            self.cellResonators = numpy.zeros((self.numBonds,4))
            self.baseRing = numpy.zeros((self.numNodes,4))
            
            self.chords = numpy.asarray([  [0,6], [1,11], [2,4], [3,8], [5,10], [7,11], [9,0]  ])
            self.removals = numpy.asarray([ [0,11]   ]) 
            
        else:
            raise ValueError('this number is not supported yet')
            
          
        self.folderName ='CDS_' + lattice_type + 'Primitive_cell_' + str(cellNum)
    
        #########
        #fill in the basic ring
#        x0 = 0.
#        y0 = 1.
        x0 = numpy.sin(self.baseTheta)
        y0 = numpy.cos(self.baseTheta)
        self.theta = 2*numpy.pi/(self.numNodes)
        for bind in range(0, self.numNodes):
            x1 = numpy.cos(self.theta)*x0 + numpy.sin(self.theta)*y0
            y1 = -numpy.sin(self.theta)*x0 + numpy.cos(self.theta)*y0
            
            self.baseRing[bind,:] = [x0,y0, x1, y1]
            
            x0 = x1
            y0 = y1
            
        #flag the necessary azimuthals for removal
        self.skips = []
        for bind in range(0, self.removals.shape[0]):
            mind= numpy.min(self.removals[bind,:])
            maxind = numpy.max(self.removals[bind,:])
            if maxind == self.numNodes-1:
                startind = maxind
            else:
                startind = mind
                
            self.skips.append(startind)
            
        #gather the remaining azimuthals  
        bind = 0      
        for rind in range(0, self.numNodes):
            if rind in self.skips:
                #this one should not be included. Skip
                pass
            else:
                self.cellResonators[bind,:] = self.baseRing[rind,:]
                bind = bind+1
            
        #add the chords
        for rind in range(0, self.chords.shape[0]):
            startind = self.chords[rind,0]
            stopind  = self.chords[rind,1]
            
            x0 = self.baseRing[startind,0]
            y0 = self.baseRing[startind,1]
            
            x1 = self.baseRing[stopind,0]
            y1 = self.baseRing[stopind,1]
            
            self.cellResonators[bind,:] = [x0, y0, x1, y1]
            bind = bind+1
               
        
        #starting unit cell graph
        self.baseGraph = GeneralLayout(resonators = self.cellResonators, name = str(cellNum), resonatorsOnly = True)
    
        #fill in 2D
        self.update_cover(self.coverLinks, self.coverPolarity, self.coverDim)
    
    def update_cover(self, coverLinks, coverPolarity, coverDim = '2d'):
        self.coverLinks = coverLinks
        self.coverPolarity = coverPolarity
        self.coverDim = coverDim
    
        #go to 2D
    
        #make the lattice unit cell
        self.latticeCellResonators = numpy.copy(self.cellResonators)
        xlink = self.coverLinks[0]
        xpol =self.coverPolarity[0]
        ylink = self.coverLinks[1]
        ypol =self.coverPolarity[1]
        if self.coverDim == '2d':
            if xpol == 0:
                vec1 = numpy.asarray([self.a1[0], self.a1[1], 0,0])
            else:
                vec1 = numpy.asarray([0,0, self.a1[0], self.a1[1]])
            if ypol == 0:
                vec2 = numpy.asarray([self.a2[0], self.a2[1], 0,0])
            else:
                vec2 = numpy.asarray([0,0, self.a2[0], self.a2[1]])
    #        tempVec = numpy.asarray([self.a2[0], self.a2[1], 0,0]) + numpy.asarray([self.a1[0], self.a1[1], 0,0])
            
        elif self.coverDim == 'xx':
            #tempVec = numpy.asarray([self.a1[0], self.a1[1], 0,0])
#            vec1 = tempVec #!!!!!!!!!hack!!!!!!!!!!
#            vec2 = tempVec#!!!!!!!!!hack!!!!!!!!!!
            list0 = [self.a1[0], self.a1[1]]
            if xpol == 0:
                tempVec = numpy.asarray(list0 +[0,0])
                vec1 = tempVec
            else:
                tempVec = numpy.asarray([0,0]+list0)
                vec1 = tempVec
            if ypol == 0:
                tempVec = numpy.asarray(list0 +[0,0])
                vec2 = tempVec
            else:
                tempVec = numpy.asarray([0,0]+list0)
                vec2 = tempVec
                
        elif self.coverDim == 'x-x':
#            tempVec = numpy.asarray([self.a1[0], self.a1[1], 0,0])
#            vec1 = tempVec #!!!!!!!!!hack!!!!!!!!!!
#            vec2 = -tempVec#!!!!!!!!!hack!!!!!!!!!!
            list0 = [self.a1[0], self.a1[1]]
            list1 = [-self.a1[0], -self.a1[1]]
            if xpol == 0:
                tempVec = numpy.asarray(list0 +[0,0])
                vec1 = tempVec
            else:
                tempVec = numpy.asarray([0,0]+list0)
                vec1 = tempVec
            if ypol == 0:
                tempVec = numpy.asarray(list1 +[0,0])
                vec2 = tempVec
            else:
                tempVec = numpy.asarray([0,0]+list1)
                vec2 = tempVec
        elif self.coverDim == 'x0':
            if xpol ==0:
                tempVec = numpy.asarray([self.a1[0], self.a1[1], 0,0])
            else:
                tempVec = numpy.asarray([0,0,self.a1[0], self.a1[1]])
            vec1 = tempVec #!!!!!!!!!hack!!!!!!!!!!
            vec2 = 0*tempVec#!!!!!!!!!hack!!!!!!!!!!
        else:
            raise ValueError('invalid cover type. should be : 2d, xx, x-x')
            
        self.latticeCellResonators[xlink,:] = self.latticeCellResonators[xlink,:] +vec1
        self.latticeCellResonators[ylink,:] = self.latticeCellResonators[ylink,:] +vec2
        
#        tempInd = 6
##        self.latticeCellResonators[tempInd,:] = self.latticeCellResonators[tempInd,:] -vec1#!!!!!!!!!hack!!!!!!
#        self.latticeCellResonators[tempInd,:] = self.latticeCellResonators[tempInd,:] +vec2
##        
        self.latticeCellGraph = GeneralLayout(resonators = self.latticeCellResonators, name = str(self.cellNum) + '_mod', resonatorsOnly = True)
        self.latticeCell =  UnitCell(lattice_type = str(self.cellNum) + '_mod',side = 1, resonators = self.latticeCellResonators, a1 = self.a1, a2 = self.a2)
        
        #make the Euclidean lattice
        self.lattice =  EuclideanLayout(xcells = 3, ycells = 3, modeType = 'FW', resonatorsOnly=False, initialCell = self.latticeCell)

        return
    
    def compute_bands(self, numSurfPoints = 300, modeType = 'FW'):
        '''
        
        This function is carved out of the middle of the compute_DOS function from DOScompiler.py
        
        It only computes all of the points on the bands at a given mesh size.
        
        modified to CDSconfig class AK 1-20-20.
        
        probably should move to unit cell class eventually because it is more general, but shoved in here for now.
        
        '''
        
        divisions1 = numpy.linspace(0, 1, numSurfPoints)
        divisions2 = numpy.linspace(0, 1, numSurfPoints)
        
        
        Xgrid, Ygrid = numpy.meshgrid(divisions1, divisions2)
        
        #lattice vectors
        a1 = numpy.zeros(3)
        a1[0:2] = self.latticeCell.a1
        
        a2 = numpy.zeros(3)
        a2[0:2] = self.latticeCell.a2
        
        a3 = numpy.zeros(3)
        a3[2] = 1
        
        #reciprocal lattice vectors
        denom = numpy.dot(a1, numpy.cross(a2,a3))
        b1 = 2*numpy.pi * numpy.cross(a2,a3) /denom
        b2 = 2*numpy.pi * numpy.cross(a3,a1) /denom
        b3 = 2*numpy.pi * numpy.cross(a1,a2) /denom
        
        #compute the grid in k space
        for n in range(0, numSurfPoints):
            for m in range(0, numSurfPoints):
                ind1 = divisions1[n]
                ind2 = divisions2[m]
                
                vec = ind1*b1 + ind2*b2
                
                Xgrid[n,m] = vec[0]
                Ygrid[n,m] = vec[1]
                
        #allocate space for the bands
        bands = numpy.zeros((self.latticeCell.numSites, len(divisions1), len(divisions2)))
        
        #####compute surfaces
        for xind in range(0,len(divisions1)):
            for yind in range(0,len(divisions2)):
                xval = Xgrid[xind, yind]
                yval = Ygrid[xind, yind]
                kx,yk, Es = self.latticeCell.compute_band_structure(xval, yval, xval, yval, numsteps = 1, modeType = modeType)
                bands[:, xind, yind] = numpy.transpose(Es)
        return bands

    def compute_X_bands(self, numSurfPoints = 300, modeType = 'FW', split_layout = False):
        '''
        
        modified 1D version of compute_bands to do 1D cases efficiently.
        I think it will compute the x direction only, but it's sketchy
        
        modified to CDSconfig class AK 1-20-20.
        probably should move to unit cell class eventually because it is more general, but shoved in here for now.
        
        '''
        
        divisions1 = numpy.linspace(0, 1, numSurfPoints)
        
        
        Xgrid = numpy.linspace(0, 1, numSurfPoints)
    #    Ygrid = numpy.linspace(0, 1, numSurfPoints)
        
        #lattice vectors
        a1 = numpy.zeros(3)
        a1[0:2] = self.latticeCell.a1
        
        a2 = numpy.zeros(3)
        a2[0:2] = self.latticeCell.a2
        
        a3 = numpy.zeros(3)
        a3[2] = 1
        
        #reciprocal lattice vectors
        denom = numpy.dot(a1, numpy.cross(a2,a3))
        b1 = 2*numpy.pi * numpy.cross(a2,a3) /denom
        b2 = 2*numpy.pi * numpy.cross(a3,a1) /denom
        b3 = 2*numpy.pi * numpy.cross(a1,a2) /denom
        
        #compute the grid in k space
        for n in range(0, numSurfPoints):
                ind1 = divisions1[n]
                
                vec = ind1*b1
                
                Xgrid[n] = vec[0]
                
        #allocate space for the bands
        if split_layout:
            bands = numpy.zeros((2*self.latticeCell.numSites, len(divisions1)))
        else:
            bands = numpy.zeros((self.latticeCell.numSites, len(divisions1)))
        
        #####compute surfaces
        for xind in range(0,len(divisions1)):
                xval = Xgrid[xind]
                kx,yk, Es = self.latticeCell.compute_band_structure(xval, 0, xval, 0, numsteps = 1, modeType = modeType)
                bands[:, xind] = numpy.transpose(Es)
        return bands
    
    
    def compute_DOS(self, numSurfPoints = 300, modeType = 'FW', freq_res = 0.04, detectFlatBands = True):    
        '''
        computes the DOS of a config. modified from DOScompiler.py AK 1-20-20
        
        DANGEROUS. ONLY BINS BETWEEN HARD CODED LIMITS. DOES NOT ADAPT TO THE CURRENT LATTICE.
        
        probably should move to unit cell class eventually because it is more general, but shoved in here for now.
        '''
    
        
        bands = self.compute_bands(numSurfPoints = numSurfPoints, modeType = modeType)
    
        #bin and comput DOS
        freq_range = 4.0+ freq_res/2
        freqs = scipy.arange(-2-freq_res/2, 4+ freq_res/2,  freq_res) + freq_res/2.
        freq_bins = scipy.arange(-2-freq_res/2, 4+ 1.5*freq_res/2, freq_res)
        
    #    print bands.shape
    #    print freq_bins
        [DOS, bins_out] = numpy.histogram(bands, freq_bins)
        bins_centers = (bins_out[0:-1] + bins_out[1:])/2
        binWidth = bins_out[1] - bins_out[0]
        
        #normalize DOS
        DOS = 1.*DOS/bands.size
        
        #autodetect flat bands
        if detectFlatBands:
            FBs = numpy.zeros(len(DOS))
            FBEs = numpy.where(DOS > 0.05)[0]
            
            DOS[FBEs] = 0
            
            FBs[FBEs] = 1.
            
            return DOS, FBs, bins_centers, binWidth
        else:
            return DOS, bins_centers, binWidth
    
    
    
    
    
#%%
if __name__=="__main__":  
    
    def try_lattice(CDS, coverLinks = '', coverPolarity= '', surfRes = 50):
    
        # update the connectivity of the cell
        if coverLinks != '':
            if coverPolarity !='':
                CDS.update_cover(coverLinks, coverPolarity)
                
        
        # Display the graphs
    
        fig1 = pylab.figure(1)
        pylab.clf()
        ax = pylab.subplot(2,3,1)
        CDS.baseGraph.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
        CDS.baseGraph.draw_resonator_end_points(ax, color = layoutCapColor, edgecolor = 'k',  marker = 'o' , size = smallCdefault, zorder = 5)
        #baseCell.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 1)
        #baseCell.draw_SDlinks(ax,  color = FWlinkColor, linewidth = 2)
        #testCell.draw_SD_points(ax,color =  FWsiteColor, size = smallCdefault, edgecolor = FWsiteEdgeColor, zorder = 5)
        #baseCell.draw_sites(ax, zorder = 5, size = 15)
        ax.set_aspect('equal')
        ax.axis('off')
        pylab.title(CDS.baseGraph.name)
        
        ax = pylab.subplot(2,3,2)
        CDS.latticeCellGraph.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.0)
        CDS.latticeCellGraph.draw_resonator_end_points(ax, color = layoutCapColor, edgecolor = 'k',  marker = 'o' , size = smallCdefault, zorder = 5)
        ax.set_aspect('equal')
        ax.axis('off')
        pylab.title(CDS.latticeCellGraph.name)
        
        
        ax = pylab.subplot(2,3,3)
        CDS.lattice.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 1.5)
        CDS.lattice.draw_resonator_end_points(ax, color = layoutCapColor, edgecolor = 'k',  marker = 'o' , size = smallCdefault, zorder = 5)
        ax.set_aspect('equal')
        ax.axis('off')
        pylab.title('small lattice')
        
        
        pylab.tight_layout()
        pylab.show()
        
        
        # compute band structures and densities of states
    
        numSteps = 200
        #ks = numpy.linspace(-numpy.pi, numpy.pi, numSteps)
        
        ksize = numpy.pi*2
        
        kxs, kys, cutx = CDS.latticeCell.compute_band_structure(-ksize, 0, ksize, 0, numsteps = numSteps, modeType = 'FW', returnStates = False)
        kxs, kys, cuty = CDS.latticeCell.compute_band_structure(0,-ksize, 0, ksize, numsteps = numSteps, modeType = 'FW', returnStates = False)
        kxs, kys, cutxy = CDS.latticeCell.compute_band_structure(-ksize,-ksize, ksize, ksize, numsteps = numSteps, modeType = 'FW', returnStates = False)
    #    kxs, kys, cutother = CDS.latticeCell.compute_band_structure(-ksize,-ksize*0.5, ksize, ksize*0.5, numsteps = numSteps, modeType = 'FW', returnStates = False)
        
        numSurfPoints = surfRes
        DOS, FBs, binfreqs, res = CDS.compute_DOS(numSurfPoints = numSurfPoints, modeType = 'FW', freq_res = 0.08, detectFlatBands = True)
        
        
        #pylab.figure(2)
        #pylab.clf()
        
        ax = pylab.subplot(2,3,4)
        CDS.latticeCell.plot_band_cut(ax, cutx)
        pylab.title('Band structure of L(X)')
        pylab.ylabel('Energy (|t|)')
        pylab.xlabel('$k_x$ (AU)')
        #pylab.xlabel('$k_x$ ($\pi$/a)')
        #pylab.xticks([0, cutx.shape[1]/2, cutx.shape[1]], [-2,0,2], rotation='horizontal')
        #pylab.ylim([-2.5, 6.5])
    #    
    #    ax = pylab.subplot(2,3,4)
    #    CDS.latticeCell.plot_band_cut(ax, cuty)
    #    pylab.title('Band structure of L(X)')
    #    pylab.ylabel('Energy (|t|)')
    #    pylab.xlabel('$k_y$ (AU)')
    #    #pylab.xlabel('$k_y$ ($\pi$/a)')
    #    #pylab.xticks([0, cutx.shape[1]/2, cutx.shape[1]], [-2,0,2], rotation='horizontal')
    #    #pylab.ylim([-2.5, 6.5])
        
        ax = pylab.subplot(2,3,5)
        CDS.latticeCell.plot_band_cut(ax, cutxy)
        pylab.title('Band structure of L(X)')
        pylab.ylabel('Energy (|t|)')
        pylab.xlabel('$k_45$ (AU)')
        
    #    ax = pylab.subplot(2,3,5)
    #    CDS.latticeCell.plot_band_cut(ax, cutother)
    #    pylab.title('Band structure of L(X)')
    #    pylab.ylabel('Energy (|t|)')
    #    pylab.xlabel('$k_{45}$ (AU)')
        
        
        ax = pylab.subplot(2,3,6)
        #ax.fill_between(binfreqs, 0, DOS, color = 'deepskyblue', zorder = 2)
        pylab.bar(binfreqs, 1.*DOS, width = res, color =  'dodgerblue', label = str('fullDOS'), alpha = 1, align = 'center')
        pylab.bar(binfreqs, 1*FBs, width = res, color =  'mediumblue', label = str('fullDOS'), alpha = 1, align = 'center', zorder = 4)
        ax.set_ylim([0,0.03])
        ax.set_aspect(150)
        pylab.title('L of ' + CDS.latticeCell.type)
        pylab.xlabel('Energy (|t|)')
        pylab.ylabel('DOS')
        pylab.xticks([-2,-1,0,1,2,3,4], [-2,-1,0,1,2,3,4], rotation='horizontal')
        
        
        titleStr = str(CDS.cellNum) + ': '+ lattice_type
        pylab.suptitle(titleStr)
        
        fig1.set_size_inches(9.1, 7.4)
        pylab.tight_layout()
        pylab.show()
    
        #
        ###########
        

        
    

    cellNum = 26
    cellNum = 7
    cellNum = 3
    cellNum = 82
    cellNum = 35
    cellNum = 6
    cellNum = 105
    cellNum = 1
    cellNum = 52
    cellNum = 452
    cellNum = 8
    cellNum = 16
    cellNum = 110
    lattice_type = 'square'
    #lattice_type = 'triangular' #this one doesn't work


    #make a default instance
    CDS = CDSconfig(cellNum, lattice_type = lattice_type)
    
    try_lattice(CDS)
    
#    fig11 = pylab.figure(11)
#    pylab.clf()
#    ax = pylab.subplot(1,1,1)
#    CDS.baseGraph.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
#    CDS.baseGraph.draw_resonator_end_points(ax, color = layoutCapColor, edgecolor = 'k',  marker = 'o' , size = smallCdefault, zorder = 5)
#    #baseCell.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 1)
#    #baseCell.draw_SDlinks(ax,  color = FWlinkColor, linewidth = 2)
#    #testCell.draw_SD_points(ax,color =  FWsiteColor, size = smallCdefault, edgecolor = FWsiteEdgeColor, zorder = 5)
#    #baseCell.draw_sites(ax, zorder = 5, size = 15)
#    ax.set_aspect('equal')
#    ax.axis('off')
#    pylab.title(CDS.baseGraph.name)
#    pylab.show()





    