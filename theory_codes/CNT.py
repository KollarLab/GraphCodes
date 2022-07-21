#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 20:11:27 2020

@author: kollar2

class for making carbon nanotubes, since I seem to be doing this a lot.

"""

import re
import scipy
import pylab
import numpy


import pickle
import sys
import os.path
import matplotlib.gridspec as gridspec

import fractions

KollarLabClassPath = r'/Users/kollar2/Documents/KollarLab/MainClasses/'
if not KollarLabClassPath in sys.path:
    sys.path.append(KollarLabClassPath)


   
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



class CNT(object):
    def __init__(self, n,m, InitDiagnosticPlots = False):
        '''start from n and m, carve up a graphene flake, and make a unit cell corresponding to a carbon nanotube.
        
        Warning. Doesn;t work well if m >> n
        
        Make sure to check the diangostic plots. 
        
        Cutting sometimes fails.
        
        Another warning. Finding root cells in failing sometimes. It's a rounding problem.
        Something like it gets the coordinates and rounds them.
        Then it rotates and maybe rounds again, and then it doesn't quite line up with 
        the lattice vetor, which hasn't been rounded.
        and I don't have roundDepth arguments enough places to fix it
        
        This seems to have been fixed by using numpy.isclose(x,y) rather than round (x) == round(y)
        in the find_root_cell_function of the unit cell
        
        But we should continue to be wary.
        
        '''
        
        self.n = n
        self.m = m
        
        self.name = 'nanotube_' + str(n) + '_' + str(m)
        
        self.InitDiagnosticPlots = InitDiagnosticPlots
        
        self.roundDepth = 3
        
        #graphene properties
        self.K = numpy.asarray([0, 4*numpy.pi/3])
        self.Kprime = numpy.asarray([0, -4*numpy.pi/3])
        self.b1 = 2*numpy.pi * numpy.asarray([1/numpy.sqrt(3), 1])
        self.b2 = 2*numpy.pi * numpy.asarray([1/numpy.sqrt(3), -1])
        
        #general nanotube properties (don't need to make the graph for these)
        self.c_hat = self.find_c_hat()
        self.D = self.find_D()  
        self.c_perp_hat= self.find_c_perp_hat()
        
        self.find_dirac_approach(valley = 1)
                
        
        a= n
        b= m
        
        Ccoeffs = [a,b]
        
        #set up the lattice vectors
        grapheneCell = UnitCell('kagome')
        mya1 = grapheneCell.a1
        mya2 = grapheneCell.a2
        
        A1 = numpy.copy(mya1)
        A2 = mya1- mya2
        
        
        #set up the nanotube and find the size of the unit cell
        self.Cvec = Ccoeffs[0]*A1 + Ccoeffs[1]*A2 #vector around nanotube diameter
        
        self.theta = numpy.arctan2(self.Cvec[1], self.Cvec[0])
        self.That = numpy.asarray([numpy.cos(self.theta+numpy.pi/2), numpy.sin(self.theta+numpy.pi/2) ]) 
        #direction along the nanotube
        
        
        #from much algebra, we can find the lattice vector along the nanotube
        multiple = self.lcm(4*a + 2*b, 2*a + 4 *b)
        k = -multiple/(2*a + 4 *b)
        j = multiple/(4*a + 2 *b)
        
        self.Tvec = j * A1 + k *A2 #the lattice vector along the nanotube
        
      
        #make  a flake to carve the unit cell out of
        collumns = a+b+1
        extra = numpy.abs( numpy.abs(k) - numpy.abs(j))+1
        xSize= collumns + extra
        ySize = numpy.abs( k) + numpy.abs(b) + 1
        grapheneFlake_0 = EuclideanLayout(int(xSize),int(ySize), 'kagome', resonatorsOnly = True)
    
        
        #shiftVec = -(extra-1)*mya1 + -(b)*mya2  - 1* numpy.asarray([ 1/numpy.sqrt(3) ,0])
    #    shiftVec = -(extra-1)*mya1 + -(b)*mya2  - 0* numpy.asarray([ 1/numpy.sqrt(3) ,0]) #+ numpy.asarray([0, 0.1]) #fudge to keep sights off unit cell boundary
        
        if b >=0:
            shiftVec = -(extra-1)*mya1 + -(b)*mya2  - 0* numpy.asarray([ 1/numpy.sqrt(3) ,0])
        else:
            shiftVec = -(extra-1)*mya1 + numpy.abs(0)*mya2  - 0* numpy.asarray([ 1/numpy.sqrt(3) ,0])
        
        resonators = grapheneFlake_0.resonators
        #resonators = grapheneFlake_1.resonators
        resonators = shift_resonators(resonators, shiftVec[0], shiftVec[1])
        self.grapheneFlake = GeneralLayout(resonators, name = 'grapheneFlake', roundDepth  = self.roundDepth) #I think this all the rotating, rounding is causing a problem
        
        
        if self.InitDiagnosticPlots:
            #show the original flake and important vectors
        
            pylab.figure(101)
            pylab.clf()
            ax = pylab.subplot(1,1,1)
            
            
            self.grapheneFlake.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
            self.grapheneFlake.draw_resonator_end_points(ax, color = layoutCapColor, edgecolor = 'k',  marker = 'o' , size = smallCdefault, zorder = 5)
            
            #pylab.plot([0,mya1[0]], [0, mya1[1]], linewidth = 2, color = 'firebrick')
            #pylab.plot([0,mya2[0]], [0, mya2[1]], linewidth = 2, color = 'maroon')
            
            pylab.plot([0,A1[0]], [0, A1[1]], linewidth = 2, color = 'darkgoldenrod')
            pylab.plot([0,A2[0]], [0, A2[1]], linewidth = 2, color = 'darkorange')
            
            
            pylab.plot([0,self.Cvec[0]], [0, self.Cvec[1]], linewidth = 2.5, color = 'firebrick')
            pylab.plot([0,self.Tvec[0]], [0, self.Tvec[1]], linewidth = 2.5, color = 'firebrick')
            
            pylab.plot([self.Tvec[0],self.Cvec[0]+self.Tvec[0]], [self.Tvec[1], self.Cvec[1]+self.Tvec[1]], linewidth = 2.5, color = 'firebrick')
            pylab.plot([self.Cvec[0],self.Tvec[0]+self.Cvec[0]], [self.Cvec[1], self.Tvec[1]+self.Cvec[1]], linewidth = 2.5, color = 'firebrick')
            
            ax.set_aspect('equal')
            ax.axis('off')
            pylab.title('sorting out the tube and unit cell size')
            
            pylab.tight_layout()
            pylab.show()
            
        
    
        # rotate everything to cardinal axes
        self.Cmag = numpy.linalg.norm(self.Cvec)
        self.Tmag = numpy.linalg.norm(self.Tvec)
        
        self.cellAngle = numpy.arctan2(self.Cvec[0], self.Cvec[1])
        
        resonators2 = numpy.copy(self.grapheneFlake.resonators)
        resonators2 = rotate_resonators(resonators2, self.cellAngle)
        
        self.tubeFlake = GeneralLayout(resonators2, name = 'tubeFlake', roundDepth = self.roundDepth, resonatorsOnly = True) 
        #this flake is now set up so that the unit cell is easy to find
        
        
        if self.InitDiagnosticPlots:
            pylab.figure(102)
            pylab.clf()
            ax = pylab.subplot(1,1,1)
            
            
            self.tubeFlake.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
            self.tubeFlake.draw_resonator_end_points(ax, color = layoutCapColor, edgecolor = 'k',  marker = 'o' , size = smallCdefault, zorder = 5)
            
            
            pylab.plot([0,0], [0, self.Cmag], linewidth = 2.5, color = 'firebrick')
            pylab.plot([-self.Tmag,0], [0, 0], linewidth = 2.5, color = 'firebrick')
            
            pylab.plot([-self.Tmag,-self.Tmag], [0, self.Cmag], linewidth = 2.5, color = 'firebrick')
            pylab.plot([-self.Tmag,0], [self.Cmag, self.Cmag], linewidth = 2.5, color = 'firebrick')
        
        
        #find the resonators of the unit cell
        self.newResonators = numpy.zeros(self.tubeFlake.resonators.shape)
        nind = 0
        
        for rind in range(0, self.tubeFlake.resonators.shape[0]):
            res = self.tubeFlake.resonators[rind,:]
            x0, y0, x1, y1= res
            
            #check if this is internal, external, or going across the edge of the unit cell.
            firstEndIn = False
            if (numpy.round(y0,self.roundDepth) < self.Cmag) and (numpy.round(y0,self.roundDepth) >= 0):
                if (numpy.round(x0,self.roundDepth) > -self.Tmag) and (numpy.round(x0,self.roundDepth) <= 0):
                    firstEndIn = True #change from default
                    
            secondEndIn = False
            if (numpy.round(y1,self.roundDepth) < self.Cmag) and (numpy.round(y1,self.roundDepth) >= 0):
                if (numpy.round(x1,self.roundDepth) > -self.Tmag) and (numpy.round(x1,self.roundDepth) <= 0):
                    secondEndIn = True #change from default
                
            if (firstEndIn and secondEndIn):
                colorStr = 'darkorange'
                self.newResonators[nind,:] = [x0,y0, x1, y1] #this is an internal resonator. Keep it
                nind = nind +1
            else:
                if (firstEndIn or secondEndIn):
                    #one end of this guy is inside, and one is not
                    colorStr = 'deepskyblue'
                    
                    #I only want to keep resonators going out the top side
                    #and to fix redundancy, I don't want to fix the bottom
                    flag1 = numpy.round(y0,self.roundDepth) >= self.Cmag
                    flag2 = numpy.round(y1,self.roundDepth) >= self.Cmag
                    
                    if (flag1) or (flag2):
                        #keep this one, but modify it
                        if flag1:
                            shiftVec = numpy.asarray([0, -self.Cmag, 0,0])
                        elif flag2:
                            shiftVec = numpy.asarray([0, 0, 0,-self.Cmag])
                            
                        temp = numpy.asarray([x0,y0,x1,y1])
                        self.newResonators[nind,:] = temp + shiftVec
                        nind = nind+1
                        
                    #to make the tube work out I also need to keep 
                    #stragglers on one side
                    if (numpy.round(x0,self.roundDepth) > 0) or (numpy.round(x1,self.roundDepth) > 0):
                        #keep this one
                        self.newResonators[nind,:] = numpy.asarray([x0,y0,x1,y1])
                        nind = nind+1
                    
                else:
                    #external resonator. Pass
                    colorStr = 'gray'
               
            if self.InitDiagnosticPlots :
                pylab.plot([x0,x1], [y0, y1], linewidth = 2.5, color = colorStr, zorder = 11)
        
        
        if self.InitDiagnosticPlots :
            ax.set_aspect('equal')
            ax.axis('off')
            pylab.title('sorting unitcell from rest of flake')
            pylab.tight_layout()
            pylab.show()
        
        
        
        #trim the unfilled rows from newREsonators
        self.newResonators = self.newResonators[~numpy.all(self.newResonators == 0, axis=1)]  
        
        #general layout caontaining just the resonators of a single unit cell of the 
        #carbon nanotube
        self.singleTubeCellGraph = GeneralLayout(self.newResonators, 'resonators of a unit cell', resonatorsOnly = True )
        
        #make a unitcell object of the cabon nanotube
        name = str(a) + '_' + str(b) + '_nanotube'
        self.tubeCell = UnitCell(name, resonators = self.newResonators,  a1 = numpy.asarray([self.Tmag, 0]), a2 = numpy.asarray([0, 2*self.Cmag]))
        self.tubeCell.find_root_cell(roundDepth = self.roundDepth)
        
        
        if self.InitDiagnosticPlots:
            pylab.figure(103)
            pylab.clf()
            ax = pylab.subplot(1,1,1)
            
            
            self.singleTubeCellGraph.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
            self.singleTubeCellGraph.draw_resonator_end_points(ax, color = layoutCapColor, edgecolor = 'k',  marker = 'o' , size = smallCdefault, zorder = 5)
            
            
            pylab.plot([0,0], [0, self.Cmag], linewidth = 2.5, color = 'firebrick')
            pylab.plot([-self.Tmag,0], [0, 0], linewidth = 2.5, color = 'firebrick')
            
            pylab.plot([-self.Tmag,-self.Tmag], [0, self.Cmag], linewidth = 2.5, color = 'firebrick')
            pylab.plot([-self.Tmag,0], [self.Cmag, self.Cmag], linewidth = 2.5, color = 'firebrick')
            
            
            ax.set_aspect('equal')
            ax.axis('off')
            pylab.title('single unit cell (hopefully)')
            
            pylab.tight_layout()
            pylab.show()
        
    
        
            #compute unit cell band structures as a test
            plotLattice = EuclideanLayout(xcells = 2, ycells = 1, modeType = 'FW', resonatorsOnly=False, initialCell = self.tubeCell)
            
            
            fig = pylab.figure(105)
            pylab.clf()
            gs = gridspec.GridSpec(1,5)
            ax = fig.add_subplot(gs[0, 0:4])
            plotLattice.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 1.5)
            plotLattice.draw_resonator_end_points(ax, color = layoutCapColor, edgecolor = 'k',  marker = 'o' , size = smallCdefault, zorder = 5)
            ax.set_aspect('equal')
            ax.axis('off')
            
            
            numSteps = 200
            ksize = 2*numpy.pi/numpy.linalg.norm(self.tubeCell.a1)
            kxs, kys, cutx = self.tubeCell.compute_band_structure(-ksize, 0, ksize, 0, numsteps = numSteps, modeType = 'FW', returnStates = False)
            
            ax = fig.add_subplot(gs[0, 4])
            
            
            #tubeCell.plot_band_cut(ax, cutx)
            self.tubeCell.plot_band_cut(ax, cutx, linewidth = 2.5)
            pylab.ylabel('Energy (|t|)')
            pylab.xlabel('$k_x$ ($\pi$/a)')
            pylab.xticks([0, cutx.shape[1]/2, cutx.shape[1]], [-2,0,2], rotation='horizontal')
            
            pylab.suptitle('checking cell band structure: ' + name)
            
            pylab.tight_layout()
            pylab.show()
    
        return
            


    def find_dirac_approach(self,valley =1):
        '''find closest approach to the Dirac point for
        the current nanotube
        Dpoint = 1 is the K point.
        2 is the K prime point'''
        
        if valley == 1:
            Dpoint = self.K
        else:
            Dpoint = self.Kprime
        
        Mat = numpy.zeros((2,2))
        # Mat[:,0] = 2*numpy.pi * self.c_hat/self.D  #warning, D is the circumference, not the diameter
        Mat[:,0] = 2 * self.c_hat/self.D
        Mat[:,1] = self.c_perp_hat
        
        #decompose the K point in the c_perp and 2pi/D *chat basis
        temp = numpy.linalg.solve(Mat, Dpoint)
        alpha = temp[0]
        beta = temp[1]
        
        #find the closest approach to the Dirac point
        round1 = alpha - numpy.floor(alpha)
        round2 = numpy.ceil(alpha) - alpha
        if round1 < round2:
            #alpha is farether from ceil(alpha) than from floor(alpha)
            #so to get from Dirac to the point of closest approach, I need to go backwards
            signum = -1
        else:
            signum = 1
        
        minRound = numpy.min([round1, round2])
        # minPerp =  signum*minRound*2*numpy.pi * self.c_hat/self.D #warning, D is the circumference, not the diameter
        minPerp =  signum*minRound*2 * self.c_hat/self.D 
        minDist = numpy.linalg.norm(minPerp)
        
        absolutePos = Dpoint + minPerp
        
        self.closestPointToDirac = absolutePos
        self.minVecFromDirac = minPerp
        self.minDistFromDirac = minDist
        self.gap = self.graphene_energy(self.closestPointToDirac)
    
        return  
        
    
    
    #graphene helpers
    def find_D(self):
        #return numpy.sqrt(n**2 +m**2 + n*m/numpy.pi)  #ERRORRRRRR!!!! was multiplying by pi before
        #####ACTUALLY!!!!!!! there should be NO factor of pi at all!!!!!!!!!!
        
        ###### this is the circumpherence
        # numpy.sqrt(1.*self.n**2 +1.*self.m**2 + 1.*self.n*self.m)
        
        ###### this is the diameter
        # numpy.sqrt(1.*self.n**2 +1.*self.m**2 + 1.*self.n*self.m)/numpy.pi
        
        return numpy.sqrt(1.*self.n**2 +1.*self.m**2 + 1.*self.n*self.m)/numpy.pi


    def find_theta(self):
        '''find the chiral angle '''
        return numpy.arctan2(numpy.sqrt(3)*self.m , 2*self.n +self.m)
    
    def find_c_hat(self):
        ''' find the chiral vector'''
        vec= numpy.zeros(2)
        vec[0] = numpy.cos(numpy.pi/6 - self.find_theta())
        vec[1] = numpy.sin(numpy.pi/6 - self.find_theta())
        return vec
    
    def find_c_perp_hat(self):
        '''find the unit vector perpendicular to the chiral vector ''' 
    #    #!!!!!! the old form is of this was wrong. Found it on 6/10/20
    #    vec= numpy.zeros(2)
    #    vec[0] = numpy.cos(numpy.pi/3 + theta(n,m))
    #    vec[1] = numpy.sin(numpy.pi/3 + theta(n,m))
        vec= numpy.zeros(2)
        vec[0] = numpy.cos(-numpy.pi/3 - self.find_theta())
        vec[1] = numpy.sin(-numpy.pi/3 - self.find_theta())
        return vec
    
    def graphene_energy(self,vec):
        '''find energy in graphene band structure of a given kx,ky '''
        kx = vec[0]
        ky = vec[1]
        term1 = 1
        term2 = 4*numpy.cos(numpy.sqrt(3) *kx / 2)*numpy.cos(ky /2)
        term3 = 4 * (numpy.cos(ky/2 )   )**2
        val = numpy.sqrt(term1 + term2 +term3)
        return val
    
    def graphene_energy_reduced(self,thetaVec):
        ''' Find energy inthe graphene band structure of a given thetax, thetay '''
        thetax = thetaVec[0]
        thetay = thetaVec[1]
        term1 = 1
        term2 = 4*numpy.cos(thetax)*numpy.cos(thetay)
        term3 = 4 * (numpy.cos(thetay )   )**2
        val = numpy.sqrt(term1 + term2 +term3)
        return val
    
    
    def lcm(self, x, y):
        '''least common multiple '''
        return numpy.abs(x * y) // fractions.gcd(x, y)
    



if __name__=="__main__":  
    
#    showGapvTube = True
    showGapvTube = False
    if showGapvTube:

        
        nMax = 5
        ns = scipy.arange(0,nMax,1.)
        ms = scipy.arange(0,nMax, 1.)
        
        tempNs = scipy.arange(0,nMax+1,1.) - 0.5
        tempMs = scipy.arange(0,nMax+1,1.) - 0.5
        nGrid, mGrid = numpy.meshgrid(tempNs,tempMs)
        
        DistMatK = numpy.zeros((len(ns), len(ms)))
        #DistMatKprime = numpy.zeros((len(ns), len(ms)))
        GapMat = numpy.zeros((len(ns), len(ms)))
        
        for nind in range(0, len(ns)):
            for mind in range(0, len(ms)):
                n = ns[nind]
                m = ms[mind]
                
                if n+m <2:
                    pass
                else:

                    testCell = CNT(n,m)
                    DistMatK[nind, mind] = testCell.minDistFromDirac
                    GapMat[nind, mind] = testCell.gap
            
        
        pylab.figure(1)
        pylab.clf()
        #pylab.pcolor(DistMat, cmap = 'jet')
        ax = pylab.subplot(1,2,1)
        pylab.pcolormesh(nGrid, mGrid, DistMatK, cmap = 'jet')
        pylab.colorbar()
        pylab.title('smallest distance from K Point')
        
        
        ax = pylab.subplot(1,2,2)
        pylab.pcolormesh(nGrid, mGrid, GapMat, cmap = 'jet')
        pylab.colorbar()
        pylab.title('energy gaps')

        
        pylab.tight_layout()
        pylab.show()


    ntest = 6
    mtest = 5
    testCNT = CNT(ntest,mtest, InitDiagnosticPlots = False)
    print( str(testCNT.gap))
    
    pylab.figure(2)
    pylab.clf()
    ax = pylab.subplot(1,2,1)
    
    
    
    testCNT.tubeCell.draw_resonators(ax)
    testCNT.tubeCell.draw_resonator_end_points(ax)
    testCNT.tubeCell.draw_SDlinks(ax)
    
    xs = testCNT.tubeCell.rootCoords[:,0]
    ys = testCNT.tubeCell.rootCoords[:,1]
    pylab.scatter(xs, ys)
    
    ax.axis('off')
    
    ax = pylab.subplot(1,2,2)
    numSteps = 100
    ksize = 2*numpy.pi/numpy.linalg.norm(testCNT.tubeCell.a1)
#    kxs, kys, cutx = testCNT.tubeCell.compute_band_structure(-ksize, 0, ksize, 0, numsteps = numSteps, modeType = 'FW', returnStates = False)
    kxs, kys, cutx = testCNT.tubeCell.compute_root_band_structure(-ksize, 0, ksize, 0, numsteps = numSteps, modeType = 'FW', returnStates = False)

    
    
    #tubeCell.plot_band_cut(ax, cutx)
    testCNT.tubeCell.plot_band_cut(ax, cutx, linewidth = 2.5)
    pylab.ylabel('Energy (|t|)')
    pylab.xlabel('$k_x$ ($\pi$/a)')
    pylab.xticks([0, cutx.shape[1]/2, cutx.shape[1]], [-2,0,2], rotation='horizontal')
    
    pylab.suptitle('checking cell : ' + testCNT.name)
    
    
    pylab.tight_layout()
    pylab.show()


