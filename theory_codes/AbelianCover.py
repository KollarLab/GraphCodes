#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 18:06:48 2023

@author: kollar2
"""

import re
import scipy
import pylab
import numpy
import copy

import pickle
import sys
import os.path
import matplotlib.gridspec as gridspec
    
#FunctionFolderPath = r'/home/pranav/PhotonicLattices'
# DataPickleFolderPath = r'/volumes/ourphoton/Alicia/Layouts/HyperbolicPickles'
#if not FunctionFolderPath in sys.path:
#    sys.path.append(FunctionFolderPath)
   
from GraphCodes.GeneralLayoutGenerator import GeneralLayout
from GraphCodes.TreeResonators import TreeResonators

from GraphCodes.UnitCell import UnitCell
from GraphCodes.EuclideanLayoutGenerator2 import EuclideanLayout

from GraphCodes.LayoutGenerator5 import PlanarLayout


from GraphCodes.resonator_utility import split_resonators
from GraphCodes.resonator_utility import rotate_resonators
from GraphCodes.resonator_utility import generate_line_graph
from GraphCodes.resonator_utility import shift_resonators

from theory_codes.CDSconfig import CDSconfig



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








#%%


# class EuclideanLayout3D(GeneralLayout3D):
#     def __init__(self, xcells = 4, 
#                        ycells = 4, 
#                        zcells = 4,
#                        lattice_type = 'Huse', 
#                        side = 1, 
#                        file_path = '', 
#                        modeType = 'FW', 
#                        resonatorsOnly=False,
#                        Hamiltonian = False,
#                        initialCell = ''):


class AbelianCover(CDSconfig):
    def __init__(self, baseGraph,  
                 coverLinks = '', 
                 coverPolarity = '', 
                 coverDim = '2d', 
                 lattice_type= 'square',
                 unpeel_method = 'duplicate',
                 mag = 2.5):
        '''start from a number in the table is CKS book and make a basic unit cell, then cover it 
        in 2D.'''

        self.baseGraph = baseGraph
        self.cellResonators = self.baseGraph.resonators
        
        self.mag = mag
        self.lattice_type = lattice_type
        self.coverDim = coverDim
        
        if unpeel_method == 'duplicate':
            self.unpeel_method = 'duplicate' #copy and edge and unpeel
        elif unpeel_method == 'replace':
            self.unpeel_method = 'replace' #cut an edge and replace with the peeled version
        else:
            raise ValueError('currently unpeel_method can only be "duplicate" or "replace" ')

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

        self.folderName =self.baseGraph.name + '_' + lattice_type
        
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
        
        if self.unpeel_method == 'duplicate':
            edge1 = copy.deepcopy(self.latticeCellResonators[xlink,:])
            edge2 = copy.deepcopy(self.latticeCellResonators[ylink,:])
            
            if self.coverDim =='x0':
                edges_to_reinstate = numpy.zeros((1,4))
                edges_to_reinstate[0,:] = edge1
            else:
                edges_to_reinstate = numpy.zeros((2,4))
                edges_to_reinstate[0,:] = edge1
                edges_to_reinstate[1,:] = edge2
                    
        self.latticeCellResonators[xlink,:] = self.latticeCellResonators[xlink,:] +vec1
        self.latticeCellResonators[ylink,:] = self.latticeCellResonators[ylink,:] +vec2
        
        cellMaxEdges = 3
        
        if self.unpeel_method == 'duplicate':
            self.latticeCellResonators = numpy.concatenate((self.latticeCellResonators, edges_to_reinstate))
            cellMaxEdges = 5
        
#        tempInd = 6
##        self.latticeCellResonators[tempInd,:] = self.latticeCellResonators[tempInd,:] -vec1#!!!!!!!!!hack!!!!!!
#        self.latticeCellResonators[tempInd,:] = self.latticeCellResonators[tempInd,:] +vec2
##        
        self.latticeCellGraph = GeneralLayout(resonators = self.latticeCellResonators, 
                                              name = self.baseGraph.name + '_mod', 
                                              resonatorsOnly = True,
                                              maxDegree = cellMaxEdges)
        
        self.latticeCell =  UnitCell(lattice_type = self.baseGraph.name + '_mod',
                                     side = 1, 
                                     resonators = self.latticeCellResonators, 
                                     a1 = self.a1, 
                                     a2 = self.a2,
                                     maxDegree = cellMaxEdges)
        
        #make the Euclidean lattice
        self.lattice =  EuclideanLayout(xcells = 3, ycells = 3, 
                                        modeType = 'FW', 
                                        resonatorsOnly=False, 
                                        initialCell = self.latticeCell)
        
        
        #generate the root graphs
        #calling these functions adds root graph properties to the objects
        #self.baseGraph,self.latticeCellGraph 
        self.baseGraph.generate_root_graph()
        self.latticeCellGraph.generate_root_graph()
        self.lattice.generate_root_graph()
        
        self.latticeCell.find_root_cell()
        # self.latticeCellRootBlochMat = self.latticeCell.generate_root_Bloch_matrix()
        

        return
    
    #####inherrited from CDSconfig
    # def compute_bands(self, numSurfPoints = 300, modeType = 'FW'):
    #     '''
        
    #     This function is carved out of the middle of the compute_DOS function from DOScompiler.py
        
    #     It only computes all of the points on the bands at a given mesh size.
        
    #     modified to CDSconfig class AK 1-20-20.
        
    #     probably should move to unit cell class eventually because it is more general, but shoved in here for now.
        
    #     '''
    
    # def compute_X_bands(self, numSurfPoints = 300, modeType = 'FW', split_layout = False):
    #     '''
        
    #     modified 1D version of compute_bands to do 1D cases efficiently.
    #     I think it will compute the x direction only, but it's sketchy
        
    #     modified to CDSconfig class AK 1-20-20.
    #     probably should move to unit cell class eventually because it is more general, but shoved in here for now.
        
    #     '''
    
    def compute_DOS(self, numSurfPoints = 300, modeType = 'FW', freq_res = 0.04, detectFlatBands = True):    
        '''
        previous version with CDSconfig was: 
            computes the DOS of a config. modified from DOScompiler.py AK 1-20-20
            
            DANGEROUS. ONLY BINS BETWEEN HARD CODED LIMITS. DOES NOT ADAPT TO THE CURRENT LATTICE.
            
            probably should move to unit cell class eventually because it is more general, but shoved in here for now.
        
        new version for AbelianCover:
            6-7-23 AK is tring to fix the limits now that max degree is a thing
        '''
    
        
        bands = self.compute_bands(numSurfPoints = numSurfPoints, modeType = modeType)
    
        #bin and comput DOS
        # freq_range = 4.0+ freq_res/2
        freqs = numpy.arange(-2-freq_res/2, self.latticeCell.maxDegree + freq_res/2,  freq_res) + freq_res/2.
        freq_bins = numpy.arange(-2-freq_res/2, self.latticeCell.maxDegree+ 1.5*freq_res/2, freq_res)
        
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
    
    def try_lattice(cover, coverLinks = '', coverPolarity= '', surfRes = 50):
    
        # update the connectivity of the cell
        if coverLinks != '':
            if coverPolarity !='':
                cover.update_cover(coverLinks, coverPolarity)
                
        
        # Display the graphs
    
        fig1 = pylab.figure(1)
        pylab.clf()
        ax = pylab.subplot(2,3,1)
        cover.baseGraph.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
        cover.baseGraph.draw_resonator_end_points(ax, color = layoutCapColor, edgecolor = 'k',  marker = 'o' , size = smallCdefault, zorder = 5)
        #baseCell.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 1)
        #baseCell.draw_SDlinks(ax,  color = FWlinkColor, linewidth = 2)
        #testCell.draw_SD_points(ax,color =  FWsiteColor, size = smallCdefault, edgecolor = FWsiteEdgeColor, zorder = 5)
        #baseCell.draw_sites(ax, zorder = 5, size = 15)
        ax.set_aspect('equal')
        ax.axis('off')
        pylab.title(cover.baseGraph.name)
        
        ax = pylab.subplot(2,3,2)
        cover.latticeCellGraph.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.0)
        cover.latticeCellGraph.draw_resonator_end_points(ax, color = layoutCapColor, edgecolor = 'k',  marker = 'o' , size = smallCdefault, zorder = 5)
        ax.set_aspect('equal')
        ax.axis('off')
        pylab.title(cover.latticeCellGraph.name)
        
        
        ax = pylab.subplot(2,3,3)
        cover.lattice.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 1.5)
        cover.lattice.draw_resonator_end_points(ax, color = layoutCapColor, edgecolor = 'k',  marker = 'o' , size = smallCdefault, zorder = 5)
        ax.set_aspect('equal')
        ax.axis('off')
        pylab.title('small lattice')
        
        
        pylab.tight_layout()
        pylab.show()
        
        
        # compute band structures and densities of states
    
        numSteps = 200
        #ks = numpy.linspace(-numpy.pi, numpy.pi, numSteps)
        
        ksize = numpy.pi*2
        
        kxs, kys, cutx = cover.latticeCell.compute_band_structure(-ksize, 0, ksize, 0, numsteps = numSteps, modeType = 'FW', returnStates = False)
        kxs, kys, cuty = cover.latticeCell.compute_band_structure(0,-ksize, 0, ksize, numsteps = numSteps, modeType = 'FW', returnStates = False)
        kxs, kys, cutxy = cover.latticeCell.compute_band_structure(-ksize,-ksize, ksize, ksize, numsteps = numSteps, modeType = 'FW', returnStates = False)
    #    kxs, kys, cutother = CDS.latticeCell.compute_band_structure(-ksize,-ksize*0.5, ksize, ksize*0.5, numsteps = numSteps, modeType = 'FW', returnStates = False)
        
        numSurfPoints = surfRes
        DOS, FBs, binfreqs, res = cover.compute_DOS(numSurfPoints = numSurfPoints, modeType = 'FW', freq_res = 0.08, detectFlatBands = True)
        
        
        #pylab.figure(2)
        #pylab.clf()
        
        ax = pylab.subplot(2,3,4)
        cover.latticeCell.plot_band_cut(ax, cutx)
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
        cover.latticeCell.plot_band_cut(ax, cutxy)
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
        pylab.title('L of ' + cover.latticeCell.type)
        pylab.xlabel('Energy (|t|)')
        pylab.ylabel('DOS')
        pylab.xticks([-2,-1,0,1,2,3,4,5,6], [-2,-1,0,1,2,3,4,5,6], rotation='horizontal')
        
        
        titleStr = cover.baseGraph.name
        pylab.suptitle(titleStr)
        
        fig1.set_size_inches(9.1, 7.4)
        pylab.tight_layout()
        pylab.show()
    
        #
        ###########
        

        
    
    resonators = numpy.zeros((6,4))
    ind = 0
    resonators[ind,:] = [0,0,0,1]
    ind = 1
    resonators[ind,:] = [0,0,1,0]
    ind = 2
    resonators[ind,:] = [1,0,1,1]
    ind = 3
    resonators[ind,:] = [0,1,1,1]
    ind = 4
    resonators[ind,:] = [0,0,1,1]
    ind = 5
    resonators[ind,:] = [1,0,0,1]
    
    initialGraph = GeneralLayout(resonators = resonators, name = 'kite', resonatorsOnly = True)
    


    #make a default instance
    cover = AbelianCover(initialGraph, coverDim = 'x0')
    
    try_lattice(cover)
    
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
        

