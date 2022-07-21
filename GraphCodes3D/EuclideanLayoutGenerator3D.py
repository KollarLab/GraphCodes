#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:22:52 2018

@author: kollar2


This class was created from the 2D Eaclidean Layout code starting on 11-21-21.


3D version of Euclidean lattice no longer has the extra resonators feature that
I wrote into the 2D version to try to make clean open boundaries. That was a 
device-motivated feature, and the 3D code is more for theory. So, I didn't
want to go through the hassle of porting it up to the higher dimension.



UnitCell3D Class
    Object to conveniently hold and define a single unit cell. Will store the number
    of site, where they are, what the links are between them and neighboring unit cells,
    and which sites are needed to close an incomplete unit cell
    
    Supported Types:
        Cubic
    
    Methods:
        ########
        #generating the cell
        ########
        _generate_cubic_cell
        _generate_arbitrary_Cell
        clean_semiduals
        
        ########
        #drawing the cell
        ########
        rotate_resonators
        rotate_coords
        draw_resonators
        draw_resonator_end_points
        draw_sites
        draw_SDlinks
        
        ########
        #auto construction functions for SD links
        ########
        _auto_generate_SDlinks
        _auto_generate_cell_SDlinks
        
        ########
        #Bloch theory function
        ########
        generate_Bloch_matrix
        compute_band_structure
        plot_band_cut
        plot_bloch_wave
        removed: plot_bloch_wave_end_state
        
        ########
        #making new cells
        ########
        split_cell
        line_graph_cell #for now this only works for coordination numbers 3 or smaller
        #4 and up require more link matrix space to be allocated.
        
        ##########
        #methods for calculating things about the root graph
        #########
        find_root_cell #determine and store the unit cell of the root graph
        generate_root_Bloch_matrix #generate a Bloch matrix for the root graph
        compute_root_band_structure
        plot_root_bloch_wave
     
    Sample syntax:
        #####
        #creating unit cell
        #####
        from EuclideanLayoutGenerator import UnitCell
        #built-in cell
        testCell = UnitCell3D(lattice_type = 'Huse', side = 1)
        
        #custom cell
        testCell = UnitCell(lattice_type = 'name', side = 1, resonators = resonatorMat, a1 = vec1, a2 = vec2, a3 = vec3)
        


EuclideanLayout3D Class
    Chose your UnitCell type, wave type, and number of unit cells and make a lattice
     
     Methods:
        ###########
        #automated construction, saving, loading
        ##########
        populate (autoruns at creation time)
        save
        load
        
        ########
        #functions to generate the resonator lattice
        #######
        generateLattice
        _fix_edge_resonators (already stores some SD properties of fixed edge)
         
        #######
        #resonator lattice get /view functions
        #######
        get_xs
        get_ys
        rotate_resonators
        rotate_coords
        draw_resonator_lattice
        draw_resonator_end_points
        get_all_resonators
        get_coords
        get_cell_offset
        get_cell_location
        get_section_cut
        
        ########
        #functions to generate effective JC-Hubbard lattice (semiduals)
        ######## 
        generate_semiduals
        #### _fix_SDedge
        
        #######
        #get and view functions for the JC-Hubbard (semi-dual lattice)
        #######
        draw_SD_points
        draw_SDlinks
        get_semidual_points (semi-defunct)
        
        ######
        #Hamiltonian related methods
        ######
        generate_Hamiltonian
        get_eigs
        
        ##########
        #methods for calculating/looking at states and interactions
        #########
        get_SDindex (removed for now. Needs to be reimplemented in sensible fashion)
        build_local_state
        ###V_int
        ###V_int_map
        plot_layout_state
        ####plot_map_state
        ####get_end_state_plot_points
        ####plot_end_layout_state
        
        ##########
        #methods for calculating things about the root graph
        #########
        generate_root_Hamiltonian
        plot_root_state
        
        
    Sample syntax:
        #####
        #loading precalculated layout
        #####
        from EuclideanLayoutGenerator3D import EuclideanLayout3D
        testLattice = EuclideanLayout3D(file_path = 'Huse_4x4_FW.pkl')
        
        #####
        #making new layout
        #####
        from EuclideanLayoutGenerator3D import EuclideanLayout3D
        #from built-in cell
        testLattice = EuclideanLayout3D(xcells = 4, ycells = 4, zcells = 4,lattice_type = 'Huse', side = 1, file_path = '', modeType = 'FW')
        
        #from custom cell
        testCell = UnitCell3D(lattice_type = 'name', side = 1, resonators = resonatorMat, a1 = vec1, a2 = vec2, a3 = vec3)
        testLattice = EuclideanLayout3D(xcells = 4, ycells = 4, zcells = 4, modeType = 'FW', resonatorsOnly=False, initialCell = testCell)
        
        #####
        #saving computed layout
        #####
        testLattice.save( name = 'filename.pkl') #filename can be a full path, but must have .pkl extension
        



"""



import re
import scipy
import pylab
import numpy
import time

import pickle
import datetime
import os
import sys

import scipy.linalg


class UnitCell3D(object):
    def __init__(self, lattice_type, 
                       side = 1, 
                       resonators = '', 
                       a1 = numpy.asarray([1,0,0]), 
                       a2 = numpy.asarray([0,1,0]),
                       a3 = numpy.asarray([0,0,1]),
                       maxDegree = 6):
        '''
        optional resonator and a1, a2, a3 reciprocal lattice vector input arguments will only be used 
        if making a cell of non-built-in type using _generate_arbitrary_cell
        
        maxDegree argument is to make sure that enough space is allocated for 
        line graph (semidual) stuff
        
        '''
        
        self.side = side*1.0
        
        
        
        #distinguish between built-in types and custom
            
        if lattice_type == 'cubic':
            self.type = lattice_type
            self._generate_cubic_cell(self.side)  
            
        else:
            #arbitrary lattice type
            self.type = lattice_type
            self.maxDegree = maxDegree
            self._generate_arbitrary_cell(resonators, a1, a2)
            
            
    ########
    #generator functions for unit cells
    ######## 
    def _generate_cubic_cell(self, side = 1):
        '''
        generate cubic-lattice-type unit cell
        '''
        self.maxDegree = 6
        
        #set up the sites of the semidual
        self.numSites = 3
        xs = numpy.zeros(self.numSites)
        ys = numpy.zeros(self.numSites)
        
        #set up the lattice vectors
        self.a1 = self.side*numpy.asarray([1,0,0])
        self.a2 = self.side*numpy.asarray([0,1,0])
        self.a3 = self.side*numpy.asarray([0,0,1])
        dx = self.a1[0]/2

        
        #set up the positions of the sites of the effective lattice. ! look to newer functions for auto way to do these
        # xs = numpy.asarray([dx, -dx, 0, 0, 0, 0])
        # ys = numpy.asarray([0, 0, dx, -dx, 0, 0])
        # zs = numpy.asarray([0, 0, 0 ,0, dx, -dx])
        xs = numpy.asarray([dx, 0, 0])
        ys = numpy.asarray([0, dx, 0])
        zs = numpy.asarray([0, 0, dx])
        self.SDx = xs
        self.SDy = ys
        self.SDz = zs
        
        self.SDcoords = numpy.zeros((len(self.SDx),3))
        self.SDcoords[:,0] = self.SDx
        self.SDcoords[:,1] = self.SDy
        self.SDcoords[:,2] = self.SDz
        
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = numpy.zeros((self.numSites,6)) #pairs of resonator end points for each resonator
        self.coords = numpy.zeros((self.numSites,2)) #set of all resonator start points
        
        a = self.side
        #xo,yo,z0, x1,y1z1
        #define them so their orientation matches the chosen one. First entry is plus end, second is minus
        # self.resonators[0,:] = [0,0,0, a,0,0]
        # self.resonators[1,:] = [0,0,0, -a,0,0]
        # self.resonators[2,:] = [0,0,0, 0,a,0]
        # self.resonators[3,:] = [0,0,0, 0,-a,0]
        # self.resonators[4,:] = [0,0,0, 0,0,a]
        # self.resonators[5,:] = [0,0,0, 0,0,-a]
        self.resonators[0,:] = [0,0,0, a,0,0]
        self.resonators[1,:] = [0,0,0, 0,a,0]
        self.resonators[2,:] = [0,0,0, 0,0,a]
        
        self.coords = self.get_coords(self.resonators)
        
        
        self.clean_semiduals(shiftFactor = 0.6) #remove any accidentally overlapping SD points
        
        #####auto population of the SD links
        self._auto_generate_SDlinks()
        
        
        
        #make note of which resonator you need in order to close the unit cell
        closure = {}
        for indx in [-1,0,1]:
            for indy in [-1,0,1]:
                for indz in [-1,0,1]:
                    closure[(indx,indy,indz)] =numpy.asarray([])
        self.closure = closure
        
        
        return
       
    # def _generate_kagome_cell(self, side = 1):
    #     '''
    #     generate kagome-type unit cell
    #     '''
    #     self.maxDegree = 3
        
    #     #set up the sites
    #     self.numSites = 3
    #     xs = numpy.zeros(self.numSites)
    #     ys = numpy.zeros(self.numSites)
        
    #     #set up the lattice vectors
    #     self.a1 = numpy.asarray([self.side*numpy.sqrt(3)/2, self.side/2])
    #     self.a2 = numpy.asarray([0, self.side])
    #     dy = self.a1[1]/2
    #     dx = self.a1[0]/2
    #     xcorr = self.side/numpy.sqrt(3)/2/2
        
    #     #set up the positions of the sites of the effective lattice. ! look to newer functions for auto way to do these
    #     xs = numpy.asarray([-dx, -dx, 0])
    #     ys = numpy.asarray([dy, -dy, -2*dy])
    #     self.SDx = xs
    #     self.SDy = ys
        
    #     #set up the poisitions of all the resonators  and their end points
    #     self.resonators = numpy.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
    #     self.coords = numpy.zeros((self.numSites,2)) #set of all resonator start points
        
    #     a = self.side/numpy.sqrt(3)
    #     b = self.a1[0]-a
    #     #xo,yo,x1,y1
    #     #define them so their orientation matches the chosen one. First entry is plus end, second is minus
    #     self.resonators[0,:] = [-a/2, 2*dy, -b-a/2,  0]
    #     self.resonators[1,:] = [-a/2-b, 0, -a/2,  -2*dy]
    #     self.resonators[2,:] = [a/2, -2*dy, -a/2,  -2*dy]
        
    #     self.coords = self.get_coords(self.resonators)
        
        
    #     #####auto population of the SD links
    #     self._auto_generate_SDlinks()
        
        
        
    #     #make note of which resonator you need in order to close the unit cell
    #     closure = {}
        
    #     #a1 direction (x)
    #     closure[(1,0)] =numpy.asarray([1])
    #     #-a1 direction (-x)
    #     closure[(-1,0)] =numpy.asarray([])
        
    #     #a2 direction (y)
    #     closure[(0,1)] =numpy.asarray([2])
    #     #-a2 direction (y)
    #     closure[(0,-1)] =numpy.asarray([])
        
    #      #a1,a2 direction (x,y)
    #     closure[(1,1)] =numpy.asarray([])
    #     #-a1,a2 direction (-x,y)
    #     closure[(-1,1)] =numpy.asarray([])
    #     #a1,-a2 direction (x,-y)
    #     closure[(1,-1)] =numpy.asarray([0])
    #     #-a1,-a2 direction (-x,-y)
    #     closure[(-1,-1)] =numpy.asarray([])
    #     self.closure = closure
        
        
    #     return
    
    
    
#     def _generate_square_cell(self, side = 1):
#         '''
#         generate sqare lattice unit cell
#         '''
#         self.maxDegree = 4
        
#         #set up the sites
#         self.numSites = 2
# #        self.numSites = 4
#         xs = numpy.zeros(self.numSites)
#         ys = numpy.zeros(self.numSites)
        
#         #set up the lattice vectors
#         self.a1 = numpy.asarray([self.side, 0])
#         self.a2 = numpy.asarray([0, self.side])
#         dy = self.a1[1]/2
#         dx = self.a1[0]/2
#         xcorr = self.side/numpy.sqrt(3)/2/2
        
        
#         #set up the poisitions of all the resonators  and their end points
#         self.resonators = numpy.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
#         self.coords = numpy.zeros((self.numSites,2)) #set of all resonator start points
        
#         a = self.side
#         #xo,yo,x1,y1
#         #define them so their orientation matches the chosen one. First entry is plus end, second is minus
#         self.resonators[0,:] = [0, 0, 0, a]
#         self.resonators[1,:] = [a, 0, 0, 0]
        
# #        self.resonators[0,:] = [0, 0, 0, a/2.]
# #        self.resonators[1,:] = [0, 0, a/2., 0]
# #        self.resonators[2,:] = [-a/2., 0, 0, 0]
# #        self.resonators[3,:] = [0, -a/2., 0, 0]
        
#         self.coords = self.get_coords(self.resonators)
        
#         #set up the positions of the sites of the effective lattice
#         xs = numpy.zeros(self.numSites)
#         ys = numpy.zeros(self.numSites)
#         for rind in range(0, self.resonators.shape[0]):
#             res = self.resonators[rind,:]
#             xs[rind] = (res[0] + res[2])/2
#             ys[rind] = (res[1] + res[3])/2
#         self.SDx = xs
#         self.SDy = ys
        
        
#         #####auto population of the SD links
#         self._auto_generate_SDlinks()
        
        
        
#         #make note of which resonator you need in order to close the unit cell
#         closure = {}
        
#         #a1 direction (x)
#         closure[(1,0)] =numpy.asarray([])
#         #-a1 direction (-x)
#         closure[(-1,0)] =numpy.asarray([])
        
#         #a2 direction (y)
#         closure[(0,1)] =numpy.asarray([])
#         #-a2 direction (y)
#         closure[(0,-1)] =numpy.asarray([])
        
#          #a1,a2 direction (x,y)
#         closure[(1,1)] =numpy.asarray([])
#         #-a1,a2 direction (-x,y)
#         closure[(-1,1)] =numpy.asarray([])
#         #a1,-a2 direction (x,-y)
#         closure[(1,-1)] =numpy.asarray([])
#         #-a1,-a2 direction (-x,-y)
#         closure[(-1,-1)] =numpy.asarray([])
#         self.closure = closure
        
        
#         return
    
    
    def _generate_arbitrary_cell(self, resonators, a1 = [1,0,0], a2 = [0,1,0], a3 = [0,0,1]):
        '''
        generate arbitrary unit cell
        
        it needs to take in a set of resonators
        and possibly reciprocal lattice vectors
        
        it will multiply everything by self.side, so make sure resonators agrees with a1, and a2
        '''
        if resonators == '':
            raise ValueError('not a built-in unit cell type and no resonators given')
        else:
#            print resonators.shape
            if resonators.shape[1] != 6:
                raise ValueError('provided resonators are not the right shape')
        
        
        #set up the lattice vectors
        self.a1 = numpy.asarray(a1)
        self.a2 = numpy.asarray(a2)   
        self.a3 = numpy.asarray(a3) 
        
        if self.a1.shape != (3,):
            raise ValueError('first lattice vector has invalid shape')
            
        if self.a2.shape != (3,):
            raise ValueError('second lattice vector has invalid shape')
            
        if self.a3.shape != (3,):
            raise ValueError('third lattice vector has invalid shape')
        
        #set up the sites
        self.numSites = resonators.shape[0]
        
        
        
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = numpy.zeros((self.numSites,6)) #pairs of resonator end points for each resonator
        self.resonators=resonators*self.side
        
        self.coords = self.get_coords(self.resonators)
        
        #set up the positions of the sites of the effective lattice
        x0 = self.resonators[:,0]
        y0 = self.resonators[:,1]
        z0 = self.resonators[:,2]
        x1 = self.resonators[:,3]
        y1 = self.resonators[:,4]
        z1 = self.resonators[:,5]
        self.SDx = (x0+x1)/2.
        self.SDy = (y0+y1)/2.
        self.SDz = (z0+z1)/2.
        
        self.SDcoords = numpy.zeros((len(self.SDx),3))
        self.SDcoords[:,0] = self.SDx
        self.SDcoords[:,1] = self.SDy
        self.SDcoords[:,2] = self.SDz
        
        self.clean_semiduals(shiftFactor = 0.6) #remove any accidentally overlapping SD points
        
        #####auto population of the SD links
        self._auto_generate_SDlinks()
        
        
        #make note of which resonator you need in order to close the unit cell
        closure = {}
        for indx in [-1,0,1]:
            for indy in [-1,0,1]:
                for indz in [-1,0,1]:
                    closure[(indx,indy,indz)] =numpy.asarray([])
        self.closure = closure
        
        
        return
    
    def clean_semiduals(self, shiftFactor = 0.6):
        '''function to remove accidentally overlapping lattice sites 
        by shifting them along their resonators
        
        '''
        
        for sind in range(0, len(self.SDx)):
            x0 = self.SDx[sind]
            y0 = self.SDy[sind]
            z0 = self.SDz[sind]
            
            
            redundancies = numpy.where((self.SDcoords  == (x0, y0,z0)).all(axis=1))[0]
            
            #now I need to move the conflicting points
            if len(redundancies) > 1:
                #every site matches to itself, so I only
                #want to shift things if it hits more than one thing
                for find in redundancies:
                    x0,y0, z0, x1, y1, z1 = self.resonators[find]
                    
                    newX = x0*shiftFactor + x1*(1-shiftFactor)
                    newY = y0*shiftFactor + y1*(1-shiftFactor)
                    newZ = z0*shiftFactor + z1*(1-shiftFactor)
                    
                    self.SDcoords[find,:] = [newX, newY, newZ]
                    self.SDx[find] = newX
                    self.SDy[find] = newY
                    self.SDz[find] = newZ
        
        return
        
        
        
        
    
    def get_coords(self, resonators, roundDepth = 3):
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
    

    #######
    #draw functions
    #######  
    
    def rotate_resonators(self, resonators, theta, phi):
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
        
        newResonators = numpy.round(newResonators, 3)
    
        return newResonators
        
    def rotate_coordinates(self, coords, theta, phi):
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
        
        new_points = numpy.round(new_points, 3)
        
        return new_points
    
    def draw_resonators(self, ax, theta = numpy.pi/10, phi = numpy.pi/10, 
                        color = 'g', 
                        alpha = 1 , 
                        linewidth = 0.5, 
                        zorder = 1):
        '''
        draw each resonator as a line
        '''
        plotRes = self.rotate_resonators(self.resonators, theta, phi)
        for res in range(0,plotRes.shape[0] ):
            [x0, y0,z0, x1, y1,z1]  = plotRes[res,:]
            #plot the x, z projection
            ax.plot([x0, x1],[z0, z1] , color = color, alpha = alpha, linewidth = linewidth, zorder = zorder)
        return
    
    def draw_resonator_end_points(self, ax, theta = numpy.pi/10, phi = numpy.pi/10,
                                  color = 'g', 
                                  edgecolor = 'k',  
                                  marker = 'o' , 
                                  size = 10, 
                                  zorder = 1):
        '''will double draw some points'''
        plotRes = self.rotate_resonators(self.resonators, theta, phi)
        
        x0s = plotRes[:,0]
        y0s = plotRes[:,1]
        z0s = plotRes[:,2]
        
        x1s = plotRes[:,3]
        y1s = plotRes[:,4]
        z1s = plotRes[:,5]
        
        
        #plot only the x, z projection
        pylab.sca(ax)
        pylab.scatter(x0s, z0s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        pylab.scatter(x1s, z1s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        return
      
    def draw_sites(self, ax, theta = numpy.pi/10, phi = numpy.pi/10,
                   color = 'g', 
                   edgecolor = 'k',  
                   marker = 'o' , 
                   size = 10, 
                   zorder=1):
        '''
        draw sites of the semidual (effective lattice)
        '''
        
        plotMat = self.rotate_coordinates(self.SDcoords, theta, phi)
        
        xs = plotMat[:,0]
        ys = plotMat[:,1]
        zs = plotMat[:,2]
        
        #plot the x-z projection
        pylab.sca(ax)
        pylab.scatter(xs, zs ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        ax.set_aspect('equal')
        return
    
    def draw_SDlinks(self, ax, theta = numpy.pi/10, phi = numpy.pi/10,
                     color = 'firebrick', 
                     linewidth = 0.5, 
                     HW = False, 
                     minus_color = 'goldenrod', 
                     zorder = 1, alpha = 1):
        '''
        draw all the links of the semidual lattice
        
        if extra is True it will draw only the edge sites required to fix the edge of the tiling
        
        set HW to true if you want the links color coded by sign
        minus_color sets the sign of the negative links
        '''
        
        #prepare to rotate the lattice vectors for projection view
        aMat = numpy.zeros((3,3))
        aMat[0,:] = self.a1
        aMat[1,:] = self.a2
        aMat[2,:] = self.a3
        
        plotMat = self.rotate_coordinates(self.SDcoords, theta, phi)
        sdxs = plotMat[:,0]
        sdys = plotMat[:,1]
        sdzs = plotMat[:,2]
        
        plotAs = self.rotate_coordinates(aMat, theta, phi)
        pa1 = plotAs[0,:]
        pa2 = plotAs[1,:]
        pa3 = plotAs[2,:]

        links = self.SDHWlinks[:]
        
        for link in range(0, links.shape[0]):
            [startSite, endSite, deltaA1, deltaA2, deltaA3]  = links[link,0:5]
            startSite = int(startSite)
            endSite = int(endSite)
            
            [x0,y0,z0] = [sdxs[startSite], sdys[startSite],sdzs[startSite]]
            [x1,y1,z1] = numpy.asarray([sdxs[endSite], sdys[endSite], sdzs[endSite]]) + deltaA1*pa1 + deltaA2*pa2 + deltaA3*pa3
            
            #plot the x-z projection
            if HW:
                ends = links[link,4:6]
                if ends[0]==ends[1]:
                    #++ or --, use normal t
                    ax.plot([x0, x1],[z0, z1] , color = color, linewidth = linewidth, zorder = zorder, alpha = alpha)
                else:
                    #+- or -+, use inverted t
                    ax.plot([x0, x1],[z0, z1] , color = minus_color, linewidth = linewidth, zorder = zorder, alpha = alpha)
            else :
                ax.plot([x0, x1],[z0, z1] , color = color, linewidth = linewidth, zorder = zorder, alpha = alpha)
                
        return
    
    # def _get_orientation_plot_points(self,scaleFactor = 0.5):
    #     '''
    #     find end coordinate locations part way along each resonator so that
    #     they can be used to plot the field at both ends of the resonator.
        
    #     Scale factor says how far appart the two points will be: +- sclaeFactor.2 of the total length
        
    #     returns the polt points as collumn matrix
    #     '''
    #     if scaleFactor> 1:
    #         raise ValueError('scale factor too big')
            
            
    #     size = len(self.SDx)
    #     plot_points = numpy.zeros((size*2, 2))
        
    #     resonators = self.resonators
    #     for ind in range(0, size):
    #         [x0, y0, x1, y1]  = resonators[ind, :]
    #         xmean = (x0+x1)/2
    #         ymean = (y0+y1)/2
            
    #         xdiff = x1-x0
    #         ydiff = y1-y0
            
    #         px0 = xmean - xdiff*scaleFactor/2
    #         py0 = ymean - ydiff*scaleFactor/2
            
    #         px1 = xmean + xdiff*scaleFactor/2
    #         py1 = ymean + ydiff*scaleFactor/2
            
            
    #         plot_points[2*ind,:] = [px0,py0]
    #         plot_points[2*ind+1,:] = [px1,py1]
    #         ind = ind+1
            
    #     return plot_points
    
#     def draw_site_orientations(self,ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'jet_r', scaleFactor = 0.5, mSizes = 60, zorder = 1):
#         Amps = numpy.ones(len(self.SDx))
#         Probs = numpy.abs(Amps)**2
#         mSizes = Probs * len(Probs)*30
#         mColors = Amps
       
#         mSizes = 60
        
#         #build full state with value on both ends of the resonators 
#         mColors_end = numpy.zeros(len(Amps)*2)
#         mColors_end[0::2] = mColors

#         #put opposite sign on other side
#         mColors_end[1::2] = -mColors
# #        mColors_end[1::2] = 5
        
#         cm = pylab.cm.get_cmap(cmap)
        
#         #get coordinates for the two ends of the resonator
#         plotPoints = self._get_orientation_plot_points(scaleFactor = scaleFactor)
#         xs = plotPoints[:,0]
#         ys = plotPoints[:,1]
        
#         pylab.sca(ax)
# #        pylab.scatter(xs, ys,c =  mColors_end, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -1, vmax = 1, zorder = zorder)
#         pylab.scatter(xs, ys,c =  mColors_end, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -1.5, vmax = 2.0, zorder = zorder)
#         if colorbar:
#             cbar = pylab.colorbar(fraction=0.046, pad=0.04)
#             cbar.set_label('phase (pi radians)', rotation=270)
              
#         if plot_links:
#             self.draw_SDlinks(ax,linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
#         pylab.title(title, fontsize=8)
#         ax.xaxis.set_visible(False)
#         ax.yaxis.set_visible(False)
#         ax.set_aspect('equal')
        
#         return mColors_end
    
    #####
    #auto construction functions for SD links
    ######
    def _auto_generate_SDlinks(self):
        '''
        start from all the resonators of a unit cell auto generate the full link matrix,
        including neighboring cells
        '''
        # xmask = numpy.zeros((self.numSites,6))
        # ymask = numpy.zeros((self.numSites,6))
        # zmask = numpy.zeros((self.numSites,6))
        
        # xmask[:,0] = 1
        # xmask[:,3] = 1
        
        # ymask[:,1] = 1
        # ymask[:,4] = 1
        
        # zmask[:,2] = 1
        # zmask[:,5] = 1
        
        # if self.type[0:2] == '74':
        #     self.SDHWlinks = numpy.zeros((self.numSites*4+4,6))
        # elif self.type == 'square':
        #     self.SDHWlinks = numpy.zeros((self.numSites*6,6))
        # elif self.type[0:4] == 'kite':
        #     self.SDHWlinks = numpy.zeros((self.numSites*10,6))
        # else:
        #     # self.SDHWlinks = numpy.zeros((self.numSites*4,6))
        #     # self.SDHWlinks = numpy.zeros((self.numSites*8,6)) #temporary hack to allow some line graph games
            
        # #now fixing to use the max degree given
        maxLineCoordination = (self.maxDegree-1)*2
        self.SDHWlinks = numpy.zeros((self.numSites*maxLineCoordination, 7))
        
        lind = 0
        # for da1 in range(-1,2):
        #     for da2 in range(-1,2):
        #         for da3 in range(-1,2):
        #####NOTE: 3D version seems to need to check a larger neighborhood!!!!
        for da1 in range(-2,3):
            for da2 in range(-2,3):
                for da3 in range(-2,3):
                    links = self._auto_generate_cell_SDlinks(da1, da2,da3)
                    newLinks = links.shape[0]
                    self.SDHWlinks[lind:lind+newLinks,:] = links
                    lind = lind + newLinks
        
        #remove blank links (needed for some types of arbitrary cells)
        self.SDHWlinks = self.SDHWlinks[~numpy.all(self.SDHWlinks == 0, axis=1)] 
        
        #also store the old link format
        oldlinks = self.SDHWlinks[:,0:5]
        self.SDlinks = oldlinks 
        
        return
            
    def _auto_generate_cell_SDlinks(self, deltaA1, deltaA2, deltaA3):
        '''
        function to autogenerate the links between two sets of resonators
        deltaA1 and deltaA2 specify how many lattice vectors the two cells are seperated by
        in the first (~x) and second  (~y) lattice directions
        
        deltaA3 is the same for the new z direction
        
        could be twice the same set, or it could be two different unit cells.
        
        will return a matrix of all the links [start, target, deltaA1, deltaA2, start_polarity, end_polarity]
        
        '''
        ress1 = self.resonators
        len1 = ress1.shape[0]
        
        #find the new unit cell
        xmask = numpy.zeros((self.numSites,6))
        ymask = numpy.zeros((self.numSites,6))
        zmask = numpy.zeros((self.numSites,6))
        xmask[:,0] = 1
        xmask[:,3] = 1
        
        ymask[:,1] = 1
        ymask[:,4] = 1
        
        zmask[:,2] = 1
        zmask[:,5] = 1
        
        xOffset = deltaA1*self.a1[0] + deltaA2*self.a2[0] + deltaA3*self.a3[0]
        yOffset = deltaA1*self.a1[1] + deltaA2*self.a2[1] + deltaA3*self.a3[1]
        zOffset = deltaA1*self.a1[2] + deltaA2*self.a2[2] + deltaA3*self.a3[2]
        ress2 = ress1 + xOffset*xmask + yOffset*ymask + zOffset*zmask

        #place to store the links
        # linkMat = numpy.zeros((len1*4+len1*4,6))  #I'm pretty sure this only works for low degree
        maxLineCoordination = (self.maxDegree-1)*2
        linkMat = numpy.zeros((len1*maxLineCoordination+len1*maxLineCoordination,7))
        
        #find the links
        
        #round the coordinates to prevent stupid mistakes in finding the connections
        roundDepth = 3
        plusEnds = numpy.round(ress2[:,0:3],roundDepth)
        minusEnds = numpy.round(ress2[:,3:],roundDepth)
        
        extraLinkInd = 0
        for resInd in range(0,ress1.shape[0]):
            res = numpy.round(ress1[resInd,:],roundDepth)
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
        
        return linkMat
    
    ######
    #Bloch theory calculation functions
    ######
    def generate_Bloch_matrix(self, kx, ky,kz, modeType = 'FW', t = 1, phase = 0):
        BlochMat = numpy.zeros((self.numSites, self.numSites))*(0 + 0j)
        
        for lind in range(0, self.SDHWlinks.shape[0]):
            link = self.SDHWlinks[lind,:]
            startInd = int(link[0]) #within the unit cell
            targetInd = int(link[1])
            deltaA1 = int(link[2])
            deltaA2   = int(link[3])
            deltaA3   = int(link[4])
            startPol = int(link[5])
            targetPol = int(link[6])
            
            polarity = startPol^targetPol #xor of the two ends. Will be one when the two ends are different
            if phase == 0: #all the standard FW HW cases
                if modeType == 'HW':
                    signum =(-1.)**(polarity)
                elif modeType == 'FW':
                    signum = 1.
                else:
                    raise ValueError('Incorrect mode type. Must be FW or HW.')
            else: #artificially break TR symmetry
                if modeType == 'HW':
                    signum =(-1.)**(polarity)
                    if signum < 0:
                        if startInd > targetInd:
                            phaseFactor = numpy.exp(1j *phase) #e^i phi in one corner
                        elif startInd < targetInd:
                            phaseFactor = numpy.exp(-1j *phase) #e^-i phi in one corner, so it's Hermitian
                        else:
                            phaseFactor = 1
                            
                        signum = signum*phaseFactor
                        
                elif modeType == 'FW':
                    signum = 1.
                else:
                    raise ValueError('Incorrect mode type. Must be FW or HW.')
            
            #corrdiates of origin site
            x0 = self.SDx[startInd]
            y0 = self.SDy[startInd]
            z0 = self.SDz[startInd]
            
            #coordinates of target site
            x1 = self.SDx[targetInd] + deltaA1*self.a1[0] + deltaA2*self.a2[0] + deltaA3*self.a3[0]
            y1 = self.SDy[targetInd] + deltaA1*self.a1[1] + deltaA2*self.a2[1] + deltaA3*self.a3[1]
            z1 = self.SDz[targetInd] + deltaA1*self.a1[2] + deltaA2*self.a2[2] + deltaA3*self.a3[2]
            
            deltaX = x1-x0
            deltaY = y1-y0
            deltaZ = z1-z0
            
            phaseFactor = numpy.exp(1j*kx*deltaX)*numpy.exp(1j*ky*deltaY)*numpy.exp(1j*kz*deltaZ)
            BlochMat[startInd, targetInd] = BlochMat[startInd, targetInd]+ t*phaseFactor*signum
        return BlochMat
    
    def compute_band_structure(self, kx_0, ky_0,kz_0,
                               kx_1, ky_1,kz_1,
                               numsteps = 100, 
                               modeType = 'FW', 
                               returnStates = False, phase  = 0):
        '''
        from scipy.linalg.eigh:
        The normalized selected eigenvector corresponding to the eigenvalue w[i] is the column v[:,i].
        
        This returns same format with two additional kx, ky indices
        '''
        
        kxs = numpy.linspace(kx_0, kx_1,numsteps)
        kys = numpy.linspace(ky_0, ky_1,numsteps)
        kzs = numpy.linspace(kz_0, kz_1,numsteps)
        
        bandCut = numpy.zeros((self.numSites, numsteps))
        
        stateCut = numpy.zeros((self.numSites, self.numSites, numsteps)).astype('complex')
        
        for ind in range(0, numsteps):
            kvec = [kxs[ind],kys[ind], kzs[ind]]
            
            H = self.generate_Bloch_matrix(kvec[0], kvec[1],kvec[2], modeType = modeType, phase  = phase)
        
            #Psis = numpy.zeros((self.numSites, self.numSites)).astype('complex')
            Es, Psis = scipy.linalg.eigh(H)
            
            bandCut[:,ind] = Es
            stateCut[:,:,ind] = Psis
        if returnStates:
            return kxs, kys,kzs, bandCut, stateCut
        else:
            return kxs, kys,kzs, bandCut
    
    def plot_band_cut(self, ax, bandCut,
                      colorlist = '', 
                      zorder = 1, 
                      dots = False, 
                      linewidth = 2.5):
        if colorlist == '':
            colorlist = ['firebrick', 'dodgerblue', 'blueviolet', 'mediumblue', 'goldenrod', 'cornflowerblue']
        
        pylab.sca(ax)
        
        for ind in range(0,bandCut.shape[0]):
            colorInd = numpy.mod(ind, len(colorlist))
            if dots:
                pylab.plot(bandCut[ind,:], color = colorlist[colorInd] , marker = '.', markersize = '5', linestyle = '', zorder = zorder)
            else:
                pylab.plot(bandCut[ind,:], color = colorlist[colorInd] , linewidth = linewidth, linestyle = '-', zorder = zorder)
#            pylab.plot(bandCut[ind,:], '.')
        pylab.title('some momentum cut')
        pylab.ylabel('Energy')
        pylab.xlabel('k_something')
    
    def plot_bloch_wave(self, state_vect, ax,
                        theta = numpy.pi/10, phi = numpy.pi/10,
                        title = 'state weight', 
                        colorbar = False, 
                        plot_links = False, 
                        cmap = 'Wistia', 
                        zorder = 1):
        '''
        plot a state (wavefunction) on the graph of semidual points
        
        Only really works for full-wave solutions
        '''
        Amps = state_vect
        Probs = numpy.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = numpy.angle(Amps)/numpy.pi
        
        #move the branch cut to -0.5
        outOfRange = numpy.where(mColors< -0.5)[0]
        mColors[outOfRange] = mColors[outOfRange] + 2
        
        
        cm = pylab.cm.get_cmap(cmap)
        
        plotMat = self.rotate_coordinates(self.SDcoords, theta, phi)
        sdxs = plotMat[:,0]
        sdys = plotMat[:,1]
        sdzs = plotMat[:,2]
        
        #plot the x-z projection
        pylab.sca(ax)
        pylab.scatter(sdxs, sdzs,c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            print('making colorbar')
            cbar = pylab.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax, theta = theta, phi = phi, 
                              linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        pylab.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return
    
#     def plot_bloch_wave_end_state(self, state_vect, ax, modeType, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', scaleFactor = 0.5, zorder = 1):
#         '''
#         plot a state (wavefunction) on the graph of semidual points, but with a 
#         value plotted for each end of the resonator
        
#         If you just want a single value for the resonator use plot_layout_state
        
#         Takes states defined on only one end of each resonator. Will autogenerate 
#         the value on other end based on mode type.
        
        
#         SOMETHING may be hinky with the range and flipping the sign
        
#         '''
#         Amps = state_vect
#         Probs = numpy.abs(Amps)**2
#         mSizes = Probs * len(Probs)*30
#         mColors = numpy.angle(Amps)/numpy.pi
        
#         #build full state with value on both ends of the resonators
#         mSizes_end = numpy.zeros(len(Amps)*2)
#         mSizes_end[0::2] = mSizes
#         mSizes_end[1::2] = mSizes
        
#         mColors_end = numpy.zeros(len(Amps)*2)
#         mColors_end[0::2] = mColors
#         if modeType == 'FW':
#             mColors_end[1::2] = mColors
#         elif modeType == 'HW':
#             #put opposite phase on other side
#             oppositeCols = mColors + 1
#             #rectify the phases back to between -0.5 and 1.5 pi radians
#             overflow = numpy.where(oppositeCols > 1.5)[0]
#             newCols = oppositeCols
#             newCols[overflow] = oppositeCols[overflow] - 2
            
#             mColors_end[1::2] = newCols
#         else:
#             raise ValueError('You screwed around with the mode type. It must be FW or HW.')
        
#         cm = pylab.cm.get_cmap(cmap)
        
#         #get coordinates for the two ends of the resonator
#         plotPoints = self._get_orientation_plot_points(scaleFactor = scaleFactor)
#         xs = plotPoints[:,0]
#         ys = plotPoints[:,1]
        
#         pylab.sca(ax)
#         pylab.scatter(xs, ys,c =  mColors_end, s = mSizes_end, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
#         if colorbar:
#             print('making colorbar')
#             cbar = pylab.colorbar(fraction=0.046, pad=0.04)
#             cbar.set_label('phase (pi radians)', rotation=270)
              
#         if plot_links:
#             self.draw_SDlinks(ax,linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
#         pylab.title(title, fontsize=8)
#         ax.xaxis.set_visible(False)
#         ax.yaxis.set_visible(False)
#         ax.set_aspect('equal')
        
# #        return plotPoints
#         return mColors
    
    def split_cell(self, splitIn = 2, name = 'TBD'):
        resMat = self.resonators
        
        oldNum = resMat.shape[0]
    
        if type(splitIn) != int:
            raise ValueError('need an integer split')
        newNum = oldNum*splitIn
        
        newResonators = numpy.zeros((newNum,4))
        
        for rind in range(0, oldNum):
            oldRes = resMat[rind,:]
            xstart = oldRes[0]
            ystart = oldRes[1]
            zstart = oldRes[2]
            xend = oldRes[3]
            yend = oldRes[4]
            zend = oldRes[5]
            
            xs = numpy.linspace(xstart, xend, splitIn+1)
            ys = numpy.linspace(ystart, yend, splitIn+1)
            zs = numpy.linspace(zstart, zend, splitIn+1)
            for sind in range(0, splitIn):
                newResonators[splitIn*rind + sind,:] = [xs[sind], ys[sind], zs[sind], xs[sind+1], ys[sind+1], zs[sind+1]]
                
        newCell = UnitCell3D(name, 
                           resonators = newResonators, 
                           a1 = self.a1, 
                           a2 = self.a2, 
                           a3 = self.a3, 
                           maxDegree = self.maxDegree)
        return newCell
    
    def line_graph_cell(self, name = 'TBD', resonatorsOnly = False):
        newResonators = numpy.zeros((self.SDHWlinks.shape[0], 6))
        
        for lind in range(0, self.SDHWlinks.shape[0]):
            link = self.SDHWlinks[lind,:]
            startInd = int(link[0]) #within the unit cell
            targetInd = int(link[1])
            
            deltaA1 = int(link[2])
            deltaA2   = int(link[3])
            deltaA3   = int(link[4])
            
            # startPol = int(link[5])
            # targetPol = int(link[6])
            
            flag = 0
            if deltaA1 > 0:
                flag = 1
            elif (deltaA1 == 0) and (deltaA2 >0):
                flag = 1
            elif (deltaA1 == 0) and (deltaA2 ==0) and (deltaA3 >=0):
                flag = 1
            if (flag):
                if (deltaA1,deltaA2,deltaA3) == (0,0,0) and  startInd > targetInd:
                    pass
                    #don't want to double count going the other way within the cell
                    #links to neighboring cells won't get double counted in this same way
                else:
                    #corrdiates of origin site
                    x0 = self.SDx[startInd]
                    y0 = self.SDy[startInd]
                    z0 = self.SDz[startInd]
                    
                    #coordinates of target site
                    x1 = self.SDx[targetInd] + deltaA1*self.a1[0] + deltaA2*self.a2[0] + deltaA3*self.a3[0]
                    y1 = self.SDy[targetInd] + deltaA1*self.a1[1] + deltaA2*self.a2[1] + deltaA3*self.a3[1]
                    z1 = self.SDz[targetInd] + deltaA1*self.a1[2] + deltaA2*self.a2[2] + deltaA3*self.a3[2]
                    
                    res = numpy.asarray([x0, y0,z0, x1, y1, z1])
                    newResonators[lind, :] = res
                    
        
        #clean out balnk rows that were for redundant resonators
        newResonators = newResonators[~numpy.all(newResonators == 0, axis=1)]  

        if resonatorsOnly:
            return newResonators
        else:
            newCell = UnitCell3D(name, 
                               resonators = newResonators, 
                               a1 = self.a1, 
                               a2 = self.a2,
                               a3 = self.a3,
                               maxDegree = (self.maxDegree-1)*2)
            return newCell
        
    def find_root_cell(self, roundDepth = 3):
        '''determine the unit cell for the root graph of the layout. The ends of the resonators is too big a set.
        So, it has to be narrowed down and the redundancies lumped together.
        
        The problem is that resonators link the unit cells together, so coords contains
        some cites from neighboring unit cells
        
        makes a list of the indices of the vertices the consititutes just a single unit cell
        also makes a dictionary showing which vertices are actually redundant.
        
        '''
        allCoords = numpy.round(self.coords[:,:], roundDepth)
        # svec_all = allCoords[:,0] + 1j*allCoords[:,1]
        svec_all = allCoords
        
        
        
        def check_redundnacy(site, svec_all, shift1, shift2, shift3):
            # vec1 = numpy.round(self.a1[0] + 1j*self.a1[1], roundDepth)
            # vec2 = numpy.round(self.a2[0] + 1j*self.a2[1], roundDepth)
            
            shiftVec = shift1*self.a1 + shift2*self.a2 + shift3*self.a3
            
            shiftedCoords = numpy.zeros(svec_all.shape)
            shiftedCoords[:,0] = svec_all[:,0] + shiftVec[0]
            shiftedCoords[:,1] = svec_all[:,1] + shiftVec[1]
            shiftedCoords[:,2] = svec_all[:,2] + shiftVec[2]

            check  = numpy.round(site - shiftedCoords, roundDepth)

            redundancies = numpy.where((check  == (0, 0,0)).all(axis=1))[0] 
            return redundancies
            
        
        #determine coordinate equivalences
        redundancyDict = {}
        for cind in range(0, allCoords.shape[0]):
            site = svec_all[cind,:] #the site to compare
            
            redundancyDict[cind] = []
            for shift1 in (-1,0,1):
                for shift2 in (-1,0,1):
                    for shift3 in (-1,0,1):
                        redundancies = check_redundnacy(site, svec_all, shift1, shift2,shift3)
                        
                        if len(redundancies) > 0:
                            if not (shift1 ==0 and shift2 == 0 and shift3 == 0):
                                #found an actual redundancy
                                redundancyDict[cind] = numpy.concatenate((redundancyDict[cind], redundancies))
            
            
        #find the minimum cell
        minCellInds = [0.]
        for cind in range(1, allCoords.shape[0]):
            equivalentInds = redundancyDict[cind] #all the site that are the same as the one we are looking at
            if len(equivalentInds)>0:
                for cind2 in range(0, len(equivalentInds)):
                    currInd = equivalentInds[cind2]
                    if currInd in minCellInds:
                        break
                    if cind2 == len(equivalentInds)-1:
                        #no matches found for the site cind
                        minCellInds = numpy.concatenate((minCellInds, [cind]))
            else:
                #no redundant sites
                minCellInds = numpy.concatenate((minCellInds, [cind]))
                
        minCellInds = numpy.asarray(minCellInds, dtype = 'int')
#        minCellInds = minCellInds.astype('int')
        
        #store the results
        self.rootCellInds = minCellInds
        self.rootVertexRedundnacy = redundancyDict
        self.numRootSites = len(minCellInds)
        self.rootCoords = self.coords[self.rootCellInds,:]
        
        #compile a matrix of root links
        self._auto_generate_root_links(roundDepth = roundDepth)
        return 
    
    def _auto_generate_root_links(self, roundDepth = 3):
        '''
        start from all the resonators of a unit cell auto generate the full link matrix
        for the root graph
        including neighboring cells
        
        needs to be called from/after ind_root cell
        
        Will generate a root link matrices with rows of the form
        [ind1, ind2, xdriection, ydirection]
        describing the link between cite indexed by ind1 to cite indexed by ind 2
        in direction given by xdirection = {-1,0,1} and y direction  = {-1,0,2}
        
        ind1 and ind 2 will be positions in rootCellInds
        
        '''
        allCoords = numpy.round(self.coords[:,:], roundDepth)
        # svec_all = allCoords[:,0] + 1j*allCoords[:,1]
        svec_all = allCoords
        
        #get the coordinates of the minimum unit cell
        coords = numpy.round(self.rootCoords, roundDepth)
        # svec = numpy.zeros((coords.shape[0]))*(1 + 1j)
        # svec[:] = coords[:,0] + 1j*coords[:,1]
        svec = coords
        
        #store away the resonators, which tell me about all possible links
        resonators = numpy.round(self.resonators, roundDepth)
        # zmat = numpy.zeros((resonators.shape[0],2))*(1 + 1j)
        # zmat[:,0] = resonators[:,0] + 1j*resonators[:,1]
        # zmat[:,1] = resonators[:,2] + 1j*resonators[:,3]
        zmat = resonators

        self.rootLinks = numpy.zeros((self.resonators.shape[0]*2,5))
        
        def check_cell_relation(site, svec, shift1, shift2, shift3):
            ''' check if a given point in a copy of the unit cell translated by
            shift1*a1 +shift2*a2'''
            shiftVec = shift1*self.a1 + shift2*self.a2 + shift3*self.a3
            
            shiftedCoords = numpy.zeros(svec.shape)
            shiftedCoords[:,0] = svec[:,0] + shiftVec[0]
            shiftedCoords[:,1] = svec[:,1] + shiftVec[1]
            shiftedCoords[:,2] = svec[:,2] + shiftVec[2]
            
            check  = numpy.round(site - shiftedCoords, roundDepth)
            
            matches = numpy.where((check  == (0, 0,0)).all(axis=1))[0]
            # matches = numpy.where(numpy.isclose(site,shiftedCoords, atol = 2*10**(-roundDepth)))[0] #rounding is causing issues. Hopefully this is better
            if len(matches)>0:
                return True
            else:
                return False
            
        def find_cell_relation(site, svec):
            ''' find out which translate of the unit cell a cite is in'''
            #added a broader search range because I ran into trouble with 
            #stuff like this in line graph generation
            for da1 in range(-2,3):
                for da2 in range(-2,3):
                    for da3 in range(-2,3):
                        match = check_cell_relation(site, svec, da1 , da2, da3)
                        if match:
                            return da1, da2, da3
            
            #raise a an error if no match found
            raise ValueError('not match found')


        lind = 0
        #convert the resonator matrix to links, basically fold it back to the unit cell
        for rind in range(0, resonators.shape[0]):
            #default to an internal link
            xpol = 0
            ypol = 0
            
            source = zmat[rind,0:3]
            target = zmat[rind,3:]
            
            check  = numpy.round(source - svec_all, roundDepth)
            sourceInd = numpy.where((check  == (0, 0,0)).all(axis=1))[0][0]
            # sourceInd = numpy.where(numpy.round(source,roundDepth) == numpy.round(svec_all,roundDepth))[0][0]
            check  = numpy.round(target - svec_all, roundDepth)
            targetInd = numpy.where((check  == (0, 0,0)).all(axis=1))[0][0]
            # targetInd = numpy.where(numpy.round(target,roundDepth) == numpy.round(svec_all,roundDepth))[0][0]
    
    
            #figure out which types of points in the unit cell we are talking about
            #will call these variables the source class and target class
            if sourceInd in self.rootCellInds:
                internalSource = True
                #this guy is in the basic unit cell
                sourceClass = sourceInd
            else:
                internalSource = False
                for cind in self.rootCellInds:
                    if sourceInd in self.rootVertexRedundnacy[cind]:
                        sourceClass = cind
                        
                      
            if targetInd in self.rootCellInds:
                internalTarget = True 
                #this guy is in the basic unit cell
                targetClass = targetInd
            else:
                internalTarget = False
                for cind in self.rootCellInds:
                    if targetInd in self.rootVertexRedundnacy[cind]:
                        targetClass = cind
            
            #convert from self.rootCellInds which tells which entries of the
            #total coords form a unit cell to
            #indices that label the entires for the matrix of the root graph
            sourceMatInd = numpy.where(sourceClass == self.rootCellInds)[0][0]
            targetMatInd = numpy.where(targetClass == self.rootCellInds)[0][0]
            
            
            #determine which translates of the unit cell are linked by the resonators            
            pos0X, pos0Y, pos0Z = find_cell_relation(source, svec)
            pos1X, pos1Y, pos1Z = find_cell_relation(target, svec)
            
            xPol = pos1X - pos0X
            yPol = pos1Y - pos0Y
            zPol = pos1Z - pos0Z
            
            self.rootLinks[lind,:] = [sourceMatInd, targetMatInd, xPol, yPol, zPol]
            self.rootLinks[lind+1,:] = [targetMatInd, sourceMatInd, -xPol, -yPol, -zPol]
            lind = lind+2

        
#        #remove blank links (needed for some types of arbitrary cells)
#        self.rootlinks = self.rootlinks[~numpy.all(self.rootlinkss == 0, axis=1)] 
        
        return
    
    def generate_root_Bloch_matrix(self, kx, ky,kz, t = 1):
        ''' 
        generates a Bloch matrix for the root graph of the layout for a given kx and ky
        
        needs the root cell and its links to be found first
        
        
        11-23-21: rewrote this guy to directly use the root links.
        
        '''
        BlochMat = numpy.zeros((self.numRootSites, self.numRootSites))*(0 + 0j)
        
        for lind in range(0, self.rootLinks.shape[0]):
            link = self.rootLinks[lind,:]
            startInd = int(link[0]) #within the unit cell
            targetInd = int(link[1])
            deltaA1 = int(link[2])
            deltaA2   = int(link[3])
            deltaA3   = int(link[4])
            
            #corrdiates of origin site
            x0 = self.rootCoords[startInd,0]
            y0 = self.rootCoords[startInd,1]
            z0 = self.rootCoords[startInd,2]
            
            #coordinates of target site
            x1 = self.rootCoords[targetInd,0] + deltaA1*self.a1[0] + deltaA2*self.a2[0] + deltaA3*self.a3[0]
            y1 = self.rootCoords[targetInd,1] + deltaA1*self.a1[1] + deltaA2*self.a2[1] + deltaA3*self.a3[1]
            z1 = self.rootCoords[targetInd,2] + deltaA1*self.a1[2] + deltaA2*self.a2[2] + deltaA3*self.a3[2]
            
            deltaX = x1-x0
            deltaY = y1-y0
            deltaZ = z1-z0
            
            phaseFactor = numpy.exp(1j*kx*deltaX)*numpy.exp(1j*ky*deltaY)*numpy.exp(1j*kz*deltaZ)
            BlochMat[startInd, targetInd] = BlochMat[startInd, targetInd]+ t*phaseFactor
        return BlochMat

    
    def compute_root_band_structure(self, kx_0, ky_0, kz_0,
                                    kx_1, ky_1,kz_1,
                                    numsteps = 100, modeType = 'FW', returnStates = False):
            '''
            computes a cut through the band structure of the root graph of the layout
            
            from scipy.linalg.eigh:
            The normalized selected eigenvector corresponding to the eigenvalue w[i] is the column v[:,i].
            
            This returns same format with two additional kx, ky indices
            '''
            
            kxs = numpy.linspace(kx_0, kx_1,numsteps)
            kys = numpy.linspace(ky_0, ky_1,numsteps)
            kzs = numpy.linspace(kz_0, kz_1,numsteps)
            
            #check if the root cell has already been found
            #if it is there, do nothing, otherwise make it.
            try:
                self.rootCellInds[0]
            except:
                self.find_root_cell()
            minCellInds = self.rootCellInds
#            redundancyDict = self.rootVertexRedundnacy
                
            numLayoutSites = len(minCellInds)
            
            bandCut = numpy.zeros((numLayoutSites, numsteps))
            
            stateCut = numpy.zeros((numLayoutSites, numLayoutSites, numsteps)).astype('complex')
            
            for ind in range(0, numsteps):
                kvec = [kxs[ind],kys[ind],kzs[ind]]
                
                H = self.generate_root_Bloch_matrix(kvec[0], kvec[1], kvec[2])
            
                #Psis = numpy.zeros((self.numSites, self.numSites)).astype('complex')
                Es, Psis = scipy.linalg.eigh(H)
                
                bandCut[:,ind] = Es
                stateCut[:,:,ind] = Psis
            if returnStates:
                return kxs, kys,kzs, bandCut, stateCut
            else:
                return kxs, kys,kzs, bandCut
    
    def plot_root_bloch_wave(self, state_vect, ax,
                             theta = numpy.pi/10, phi = numpy.pi/10,
                             title = 'state weight', 
                             colorbar = False, 
                             plot_links = False, 
                             cmap = 'Wistia', 
                             zorder = 1):
        '''
        plot a state (wavefunction) on the root graph of the layout
        
        Only really works for full-wave solutions
        '''
        Amps = state_vect
        Probs = numpy.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = numpy.angle(Amps)/numpy.pi
        
        #move the branch cut to -0.5
        outOfRange = numpy.where(mColors< -0.5)[0]
        mColors[outOfRange] = mColors[outOfRange] + 2
        
        cm = pylab.cm.get_cmap(cmap)
        
        plotMat = self.rotate_coordinates(self.rootCoords, theta, phi)
        xs = plotMat[:,0]
        ys = plotMat[:,1]
        zs = plotMat[:,2]
        
        #plot the x-z projection
        pylab.sca(ax)
        pylab.scatter(xs, zs,c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            print( 'making colorbar')
            cbar = pylab.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            # self.draw_SDlinks(ax, linewidth = 0.5, color = 'firebrick', zorder = zorder)
            self.draw_resonators(ax, linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        pylab.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return        
        

class EuclideanLayout3D(object):
    def __init__(self, xcells = 4, 
                       ycells = 4, 
                       zcells = 4,
                       lattice_type = 'Huse', 
                       side = 1, 
                       file_path = '', 
                       modeType = 'FW', 
                       resonatorsOnly=False,
                       Hamiltonian = False,
                       initialCell = ''):
        '''
        
        '''
        
        
        if file_path != '':
            self.load(file_path)
        else:
            #create plank planar layout object with the bare bones that you can build on
            self.xcells = xcells
            self.ycells = ycells
            self.zcells = zcells
            self.side = side*1.0
            
            

            self.lattice_type = lattice_type
            
            if type(initialCell) == UnitCell3D:
                #use the unit cell object provided
                self.unitcell = initialCell
            else:
                #use a built in unit cell specified by keyword
                #starting unit cell
                self.unitcell = UnitCell3D(self.lattice_type, self.side)

            self.maxDegree = self.unitcell.maxDegree            

            if not ((modeType == 'FW') or (modeType  == 'HW')):
                raise ValueError('Invalid mode type. Must be FW or HW')
            self.modeType = modeType
            
            self.populate(resonatorsOnly = resonatorsOnly, Hamiltonian  = Hamiltonian)
            
            
    ###########
    #automated construction, saving, loading
    ##########
    def populate(self, resonatorsOnly=False, Hamiltonian = True, save = False, save_name = ''):
        '''
        fully populate the structure up to itteration = MaxItter
        
        if Hamiltonian = False will not generate H
        save is obvious
        '''
         
        #make the resonator lattice
        self.generate_lattice(self.xcells, self.ycells, self.zcells)
        
        if not resonatorsOnly:
            #make the JC-Hubbard lattice
            self.generate_semiduals()
            
            if Hamiltonian:
                self.generate_Hamiltonian()
            
            if save:
                self.save(save_name)
            
        return
    
    def save(self, name = ''):
        '''
        save structure to a pickle file
        
        if name is blank, will use dafualt name
        '''
        if self.modeType == 'HW':
            waveStr = 'HW'
        else:
            waveStr = ''
            
        if name == '':
            name = str(self.lattice_type) + '_' + str(self.xcells) + 'x ' + str(self.ycells) + '_' + 'x' + str(self.ycells) + '_' + waveStr + '.pkl'
        
        savedict = self.__dict__
        pickle.dump(savedict, open(name, 'wb'))
        return
    
    def load(self, file_path):
        '''
        laod structure from pickle file
        '''
        pickledict = pickle.load(open(file_path, "rb" ) )
        
        for key in pickledict.keys():
            setattr(self, key, pickledict[key])
           
        #handle the case of old picle files that do not have a mode type property  
        #they are all calculated for the full wave
        if not 'modeType' in self.__dict__.keys():
            print('Old pickle file. Pre FW-HW.')
            self.modeType = 'FW'
        return    
            
    #######
    #resonator lattice get /view functions
    #######       
    def get_xs(self):
        '''
        return x coordinates of all the resonator end points
        '''
        return self.coords[:,0]
    
    def get_ys(self):
        '''
        return y coordinates of all the resonator end points
        '''
        return self.coords[:,1]  
    
    def get_zs(self):
        '''
        return z coordinates of all the resonator end points
        '''
        return self.coords[:,2] 
    
    def get_all_resonators(self, maxItter = -1):
        '''
        function to get all resonators as a pair of end points
        
        each resontator returned as a row with four entries.
        (orientation is important to TB calculations)
        x0,y0,x1,y1
        
        '''
        return self.resonators
    
    def get_coords(self, resonators, roundDepth = 3):
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
    
    def get_cell_offset(self,indx, indy, indz):
        '''return the x, y, and z coordinates of a vector required to translate to cell
        labeled by indices indx, indy, indz'''
        xOffset = indx*self.unitcell.a1[0] + indy*self.unitcell.a2[0] + indz*self.unitcell.a3[0]
        yOffset = indx*self.unitcell.a1[1] + indy*self.unitcell.a2[1] + indz*self.unitcell.a3[1]
        zOffset = indx*self.unitcell.a1[2] + indy*self.unitcell.a2[2] + indz*self.unitcell.a3[2]
        
        return xOffset, yOffset, zOffset
    
    def get_cell_location(self, indx, indy, indz):
        '''get the absolute index of the unit cell given x, y ,z indices 
        
        x axis first
        then y
        then z
        
        '''
        if (indx < 0) or (indx >= self.xcells):
            raise ValueError('invalid cell location.')
        if (indy < 0) or (indy >= self.ycells):
            raise ValueError('invalid cell location.')
        if (indz < 0) or (indz >= self.zcells):
            raise ValueError('invalid cell location.')
        
        loc = indx + self.xcells*indy + self.xcells*self.ycells*indz
        
        return loc
    
    def get_section_cut(self, xrange = -1, yrange=-1, zrange=-1, returnCells = False):
        '''function to get the locations of all the unit cells corresponding
        to a section cut.
        
        if any of the ranges are -1, it will take the full system size by default.
        
        Otherwise it will find the unit cells in the window
        [x0, x1]
        [y0,y1]
        [z0,z1]
        
        optional argument return Cells determine whether it returns an array
        of the unit cell indices or whether it returns the array values
        
        (will probably have to do a bit or regiggering here to eventually handle 
         vertex bosonizations, but I will worry about that later.)
        
        '''
        if xrange == -1:
            xrange = [0, int(self.xcells)]
        if yrange == -1:
            yrange = [0, int(self.ycells)]
        if zrange == -1:
            zrange = [0, int(self.zcells)]
            
        cellList = numpy.zeros(self.xcells*self.ycells*self.zcells)
        
        if (xrange[0] < 0) or (xrange[1] > self.xcells):
            raise ValueError('invalid cut range.')
        if (yrange[0] < 0) or (yrange[1] > self.ycells):
            raise ValueError('invalid cut range.')
        if (zrange[0] < 0) or (zrange[1] > self.zcells):
            raise ValueError('invalid cut range.')
            
        # print(xrange)
        # print(yrange)
        # print(zrange)
        
        mind = 0
        for xind in range(xrange[0], xrange[1]):
            for yind in range(yrange[0], yrange[1]):
                for zind in range(zrange[0], zrange[1]):
                    cellList[mind] = self.get_cell_location(xind, yind, zind)
                    mind = mind+1
        #trim the zero entries out of the end of the list
        cellList = cellList[0:mind]
        
        #now make a list of all the raw resonator indices that correspond to these cells
        resList = numpy.zeros(len(cellList)*self.unitcell.numSites, dtype = 'int')
        mind = 0
        for cind in range(0, len(cellList)):
            for sind in range(0, self.unitcell.numSites):
                cell = cellList[cind]
                resList[mind] = cell*self.unitcell.numSites + sind
                mind = mind+1
                
        if returnCells:
            return cellList
        else:
            return resList
            
        
    
    #####
    #draw functions
    ######
    def rotate_resonators(self, resonators, theta, phi):
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
        
        newResonators = numpy.round(newResonators, 3)
    
        return newResonators
        
    def rotate_coordinates(self, coords, theta, phi):
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
        
        new_points = numpy.round(new_points, 3)
        
        return new_points
    
    

    def draw_resonator_lattice(self, ax, 
                               theta = numpy.pi/10,
                               phi = numpy.pi/10,
                               color = 'g', 
                               alpha = 1 , 
                               linewidth = 0.5, 
                               zorder = 1,
                               xrange = -1,
                               yrange = -1,
                               zrange = -1):

        #trim out only the right resonators
        resList = self.get_section_cut(xrange=xrange, yrange=yrange, zrange=zrange, returnCells = False)
        resonators = self.resonators[resList,:]
        
        #rotate into potions
        plotRes = self.rotate_resonators(resonators, theta, phi)
            
        #plot the x-z projection
        for res in range(0,plotRes.shape[0] ):
            [x0, y0, z0, x1, y1,z1]  = plotRes[res,:]
            ax.plot([x0, x1],[z0, z1] , color = color, alpha = alpha, linewidth = linewidth, zorder = zorder)
        return
      
    def draw_resonator_end_points(self, ax,
                                  theta = numpy.pi/10,phi = numpy.pi/10,
                                  xrange = -1,
                                  yrange = -1,
                                  zrange = -1,
                                  color = 'g', edgecolor = 'k',  marker = 'o' , size = 10, zorder = 1):
        '''will double draw some points'''
        
        #trim out only the right resonators
        resList = self.get_section_cut(xrange=xrange, yrange=yrange, zrange=zrange, returnCells = False)
        resonators = self.resonators[resList,:]
        
        #rotate into potions
        plotRes= self.rotate_resonators(resonators, theta, phi)
            
        #plot the x-z projection
        x0s = plotRes[:,0]
        z0s = plotRes[:,2]
        
        x1s = plotRes[:,3]
        z1s = plotRes[:,5]
        
        pylab.sca(ax)
        pylab.scatter(x0s, z0s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        pylab.scatter(x1s, z1s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        return   

    
    
    
    ########
    #functions to generate the resonator lattice
    #######
    def generate_lattice(self, xsize = -1, ysize = -1, zsize = -1):
        '''
        Hopefully will become a general function to fill out the lattice. Has some issues
        with the edges right now
        
        it is important not to reverse the order of the endpoints of the lattice. These indicate
        the orinetation of the site, and will be needed to fill in the extra links to fix the edge of the lattice
        in HW mode
        '''
        if xsize <0:
            xsize = self.xcells
        if ysize <0:
            ysize = self.ycells
        if zsize <0:
            zsize = self.zcells
            
        #make sure that the object has the right size recorded
        self.xcells = xsize
        self.ycells = ysize
        self.zcells = zsize
        
        self.resonators = numpy.zeros((int(xsize*ysize*zsize*self.unitcell.numSites), 6))
    
        xmask = numpy.zeros((self.unitcell.numSites,6))
        ymask = numpy.zeros((self.unitcell.numSites,6))
        zmask = numpy.zeros((self.unitcell.numSites,6))
        
        xmask[:,0] = 1
        xmask[:,3] = 1
        
        ymask[:,1] = 1
        ymask[:,4] = 1
        
        zmask[:,2] = 1
        zmask[:,5] = 1
        
        # ind = 0
        for indx in range(0,int(xsize)):
            for indy in range(0,int(ysize)):
                for indz in range(0, int(zsize)):
                    # xOffset = indx*self.unitcell.a1[0] + indy*self.unitcell.a2[0] + indz*self.unitcell.a3[0]
                    # yOffset = indx*self.unitcell.a1[1] + indy*self.unitcell.a2[1] + indz*self.unitcell.a3[1]
                    # zOffset = indx*self.unitcell.a1[2] + indy*self.unitcell.a2[2] + indz*self.unitcell.a3[2]
                    ind = self.unitcell.numSites*self.get_cell_location(indx, indy, indz) 
                    xOffset, yOffset, zOffset = self.get_cell_offset(indx, indy, indz)
                    xOffset = numpy.round(xOffset,3)
                    yOffset = numpy.round(yOffset,3)
                    zOffset = numpy.round(zOffset,3)
                    self.resonators[ind:ind+self.unitcell.numSites, :] = self.unitcell.resonators + xOffset*xmask + yOffset*ymask + zOffset*zmask
                    # ind = ind + self.unitcell.numSites
                    
        self.coords = self.get_coords(self.resonators)        
        
        return

        
        

    ########
    #functions to generate effective JC-Hubbard lattice
    ########
    def generate_semiduals(self):
        '''
        main workhorse function to generate the JC-Hubbard lattice.
        This is the one you shold call. All the others are workhorses that it uses.
        
        Will loop through the existing and create attributes for the 
        JC-Hubbard lattice (here jokingly called semi-dual) and fill them
        '''
        xsize = self.xcells
        ysize = self.ycells
        zsize = self.zcells
        
        self.SDcoords = numpy.zeros((xsize*ysize*zsize*self.unitcell.numSites,3))
        self.SDx = numpy.zeros(xsize*ysize*zsize*self.unitcell.numSites)
        self.SDy = numpy.zeros(xsize*ysize*zsize*self.unitcell.numSites)
        self.SDz = numpy.zeros(xsize*ysize*zsize*self.unitcell.numSites)
        

            
        #now fixing to use the max degree given
        maxLineCoordination = (self.maxDegree-1)*2
        self.SDHWlinks = numpy.zeros((xsize*ysize*zsize*self.unitcell.numSites*maxLineCoordination, 4))
        
        
        #set up for getting the positions of the semidual points
        xmask = numpy.zeros((self.unitcell.numSites,6))
        ymask = numpy.zeros((self.unitcell.numSites,6))
        zmask = numpy.zeros((self.unitcell.numSites,6))
        
        xmask[:,0] = 1
        xmask[:,3] = 1
        
        ymask[:,1] = 1
        ymask[:,4] = 1
        
        zmask[:,2] = 1
        zmask[:,5] = 1

        #links will be done by site index, which will include the unit cell number
        latticelinkInd = 0
        # ind = 0
        for indx in range(0,xsize):
            for indy in range(0,ysize):
                for indz in range(0, zsize):
                    currCell = [indx, indy, indz]
                    
                    # xOffset = indx*self.unitcell.a1[0] + indy*self.unitcell.a2[0]
                    # yOffset = indx*self.unitcell.a1[1] + indy*self.unitcell.a2[1]
                    xOffset ,yOffset , zOffset = self.get_cell_offset(indx, indy, indz)
                    #have to make sure that the oder in which things appear in SD points matches the description order for the cells
                    ind = self.unitcell.numSites*self.get_cell_location(currCell[0], currCell[1], currCell[2]) 
                    self.SDx[ind:ind+self.unitcell.numSites] = self.unitcell.SDx + xOffset
                    self.SDy[ind:ind+self.unitcell.numSites] = self.unitcell.SDy + yOffset
                    self.SDz[ind:ind+self.unitcell.numSites] = self.unitcell.SDz + zOffset
                    
                    # ind = ind + self.unitcell.numSites
                    
                    for link in range(0, self.unitcell.SDlinks.shape[0]):
                        [startSite, targetSite, deltaA1, deltaA2, deltaA3, startEnd, targetEnd]  = self.unitcell.SDHWlinks[link,:]
                        targetCell = [indx + deltaA1, indy + deltaA2, indz + deltaA3]
    #                    print [startSite, targetSite, deltaA1, deltaA2, startEnd, targetEnd]
    #                    print currCell
    #                    print targetCell
                        if (targetCell[0]<0) or (targetCell[1]<0 or targetCell[2]<0) or (targetCell[0]>xsize-1) or (targetCell[1]>ysize-1) or (targetCell[2]>zsize-1):
                            #this cell is outside of the simulation. Leave it
    #                        print 'passing by'
                            pass
                        else:
                            # startInd = startSite + currCell[0]*self.unitcell.numSites*ysize + currCell[1]*self.unitcell.numSites
                            startInd = startSite + self.unitcell.numSites*self.get_cell_location(currCell[0], currCell[1], currCell[2])
                            
                            # targetInd = targetSite + targetCell[0]*self.unitcell.numSites*ysize + targetCell[1]*self.unitcell.numSites
                            targetInd = targetSite + self.unitcell.numSites*self.get_cell_location(targetCell[0], targetCell[1], targetCell[2])
                            
                            self.SDHWlinks[latticelinkInd,:] = [startInd, targetInd, startEnd, targetEnd]
    #                        print [startInd, targetInd, startEnd, targetEnd]
                            latticelinkInd = latticelinkInd +1  
    #                    print '   '
        
        # #fix the edge
        # self._fix_SDedge()
        
        #clean the skipped links away 
        self.SDHWlinks = self.SDHWlinks[~numpy.all(self.SDHWlinks == 0, axis=1)]  
        
        #make the truncated SD links
        self.SDlinks = self.SDHWlinks[:,0:2]
        
        #make the full SD coords matrix
        self.SDcoords[:,0] = self.SDx
        self.SDcoords[:,1] = self.SDy
        self.SDcoords[:,2] = self.SDz
            
        return
    
    def draw_SD_points(self, ax, 
                       theta = numpy.pi/10,phi = numpy.pi/10,
                       xrange = -1,
                       yrange = -1,
                       zrange = -1,
                       color = 'g', edgecolor = 'k',  marker = 'o' , size = 10,  zorder = 1, alpha = 1):
        '''
        draw the locations of all the semidual sites
        
        if extra is True it will draw only the edge sites required to fix the edge of the tiling
        '''
        #trim out only the right resonators
        resList = self.get_section_cut(xrange=xrange, yrange=yrange, zrange=zrange, returnCells = False)
        SDpoints= self.SDcoords[resList,:]
        
        #rotate into potions
        plotPoints= self.rotate_coordinates(SDpoints, theta, phi)
            
        #plot the x-z projection
        xs = plotPoints[:,0]
        zs = plotPoints[:,2]
        
        pylab.sca(ax)
        pylab.scatter(xs, zs ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder, alpha = alpha)
        
        return

    def draw_SDlinks(self, ax,
                     theta = numpy.pi/10,phi = numpy.pi/10,
                       xrange = -1,
                       yrange = -1,
                       zrange = -1,
                       color = 'firebrick', linewidth = 0.5, extra = False, minus_links = False, minus_color = 'goldenrod', zorder = 1, alpha = 1):
        '''
        draw all the links of the semidual lattice
        
        if extra is True it will draw only the edge sites required to fix the edge of the tiling
        
        set minus_links to true if you want the links color coded by sign
        minus_color sets the sign of the negative links
        '''
        
        #trim out only the right resonators
        resList = self.get_section_cut(xrange=xrange, yrange=yrange, zrange=zrange, returnCells = False)
        # SDpoints= self.SDcoords[resList,:]
        
        #rotate into potions
        # plotPoints= self.rotate_coordinates(SDpoints, theta, phi)
        plotPoints= self.rotate_coordinates(self.SDcoords, theta, phi)

        xs = plotPoints[:,0]
        zs = plotPoints[:,2]
        links = self.SDHWlinks[:]
        
        for link in range(0, links.shape[0]):
            [startInd, endInd]  = links[link,0:2]
            startInd = int(startInd)
            endInd = int(endInd)
            
            if (startInd in resList) or (endInd in resList):
            
                [x0,z0] = [xs[startInd], zs[startInd]]
                [x1,z1] = [xs[endInd], zs[endInd]]
                
                if  minus_links == True and self.modeType == 'HW':
                    ends = links[link,2:4]
                    if ends[0]==ends[1]:
                        #++ or --, use normal t
                        ax.plot([x0, x1],[z0, z1] , color = color, linewidth = linewidth, zorder = zorder, alpha = alpha)
                    else:
                        #+- or -+, use inverted t
                        ax.plot([x0, x1],[z0, z1] , color = minus_color, linewidth = linewidth, zorder = zorder, alpha = alpha)
                else :
                    ax.plot([x0, x1],[z0, z1] , color = color, linewidth = linewidth, zorder = zorder, alpha = alpha)
                    
        return

    def get_semidual_points(self):
        '''
        get all the semidual points in a given itteration.
        
        Mostly vestigial for compatibility
        '''
        return[self.SDx, self.SDy, self.SDz]

    
    ######
    #Hamiltonian related methods
    ######
    def generate_Hamiltonian(self, t = 1, internalBond = 1000):
        '''
        create the effective tight-binding Hamiltonian
        
        Also calculated as stores eigenvectors and eigenvalues for that H
        
        
        Will use FW or HW TB coefficients depending on self.modeType
        '''
        

        self.t = t
        self.internalBond = 1000*self.t
        
        totalSize = len(self.SDx)
            
        self.H = numpy.zeros((totalSize, totalSize))
        self.H_HW = numpy.zeros((totalSize*2, totalSize*2)) #vestigial
        
        #loop over the links and fill the Hamiltonian
        for link in range(0, self.SDlinks.shape[0]):
            [sourceInd, targetInd] = self.SDlinks[link, :]
            [sourceEnd, targetEnd] = self.SDHWlinks[link, 2:]
            source = int(sourceInd)
            target = int(targetInd)
            sourceEnd = int(sourceEnd)
            targetEnd = int(targetEnd)
            
            
            if self.modeType == 'FW':
                self.H[source, target] = self.t
            elif self.modeType == 'HW':
                polarity = sourceEnd^targetEnd #xor of the two ends. Will be one when the two ends are different
                signum =(-1.)**(polarity) #will be zero when  two ends are same, and minus 1 otherwise
                self.H[source, target] = self.t * signum
            else:
                raise ValueError('You screwed around with the mode type. It must be FW or HW.')
            self.H_HW[2*source + sourceEnd, 2*target+targetEnd] = 2*self.t
                
        #fix the bonds between the two ends of the same site
        for site in range(0, totalSize):
            self.H_HW[2*site, 2*site+1] = self.internalBond
            self.H_HW[2*site+1, 2*site] = self.internalBond
                
        self.Es, self.Psis = scipy.linalg.eigh(self.H)
        self.Eorder = numpy.argsort(self.Es)
        
        return
    
    def get_eigs(self):
        '''
        returns eigenvectors and eigenvalues
        '''
        return [self.Es, self.Psis, self.Eorder]
    
#    def get_SDindex(self,num, itt, az = True):
#        '''
#        get the index location of a semidual point. 
#        
#        Point spcified by
#        something TBD
#        
#        (useful for making localized states at specific sites)
#        '''
#        
#        return currInd

    def build_local_state(self, site):
        '''
        build a single site state at any location on the lattice.
        
        site is the absolute index coordinate of the lattice site
        (use get_SDindex to obtain this in a halfway sensible fashion)
        '''
        if site >= len(self.SDx):
            raise ValueError('lattice doesnt have this many sites')
            
        state = numpy.zeros(len(self.SDx))*(0+0j)
        
        state[site] = 1.
        
        return state
    
    # def V_int(self, ind1, ind2, states):
    #     '''
    #     Calculate total interaction enegery of two particles at lattice sites
    #     indexed by index 1 and index 2
        
    #     states is the set of eigenvectors that you want to include e.g. [0,1,2,3]
    #     '''
    #     psis_1 = self.Psis[ind1,states]
    #     psis_2 = self.Psis[ind2,states]
        
    #     return numpy.dot(numpy.conj(psis_2), psis_1)

    # def V_int_map(self, source_ind, states = []):
    #     '''
    #     calculate a map of the interaction energy for a given location of the first
    #     qubit.
    #     Lattice sites specified by index in semidual points array
        
    #     must also specify which igenstates to include. default = all
    #     '''
    #     if states == []:
    #         states = scipy.arange(0, len(self.Es),1)
        
    #     int_vals = numpy.zeros(len(self.SDx))
    #     for ind2 in range(0, len(self.SDx)):
    #         int_vals[ind2] = self.V_int(source_ind, ind2,states)
        
    #     return int_vals
    
    def plot_layout_state(self, state_vect, ax, 
                          theta = numpy.pi/10,
                          phi = numpy.pi/10,
                          xrange = -1,
                          yrange = -1,
                          zrange= -1,
                          title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', zorder = 1):
        '''
        plot a state (wavefunction) on the graph of semidual points
        '''
        #trim out only the right resonators
        resList = self.get_section_cut(xrange, yrange, zrange, returnCells = False)
        SDpoints = self.SDcoords[resList, :]
        plotPoints = self.rotate_coords(SDpoints, theta, phi)
        sdx = plotPoints[:,0]
        sdz = plotPoints[:,2]
        
        Amps = state_vect
        Probs = numpy.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = numpy.angle(Amps)/numpy.pi
        
        cm = pylab.cm.get_cmap(cmap)
        
        #trim the intended plot points to the desired range
        mSizes = mSizes[resList]
        mColors = mColors[resList]
        
        #plot the x-z projection
        pylab.sca(ax)
        pylab.scatter(sdx, sdz,c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            cbar = pylab.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax, 
                              theta = theta, phi = phi,
                              xrange = xrange,
                              yrange = yrange,
                              zrange = zrange,
                              linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        pylab.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return

#     def plot_map_state(self, map_vect, ax, title = 'ineraction weight', colorbar = False, plot_links = False, cmap = 'winter', autoscale = False, scaleFactor = 0.5, zorder = 1):
#         '''plot an interaction map on the graph
#         '''
    
    # def get_end_state_plot_points(self,scaleFactor = 0.5):
    #     '''
    #     find end coordinate locations part way along each resonator so that
    #     they can be used to plot the field at both ends of the resonator.
    #     (Will retun all values up to specified itteration. Default is the whole thing)
        
    #     Scale factor says how far appart the two points will be: +- sclaeFactor.2 of the total length
        
    #     returns the polt points as collumn matrix
    #     '''
   
    
#     def plot_end_layout_state(self, state_vect, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', scaleFactor = 0.5, zorder = 1):
#         '''
#         plot a state (wavefunction) on the graph of semidual points, but with a 
#         value plotted for each end of the resonator
        
#         If you just want a single value for the resonator use plot_layout_state
        
#         Takes states defined on only one end of each resonator. Will autogenerate 
#         the value on other end based on mode type.
        
#         '''

    
    # def generate_root_Hamiltonian(self, roundDepth = 3, t = 1, verbose = False, sparse = False, flags = 5):
    #     '''
    #     custom function so I can get vertex dict without having to run the full populate of general layout
    #     and thereby having to also diagonalize the effective Hamiltonian.
        
    #     Will process the resonator matrix to get the layout Hamiltonian.
        
    #     Will return a regular matrix of sparse  = false, and a sparse matrix data type if sparse  = true
        
    #     Does not need to SD Hamiltonian made first.
        
        
    #     '''
    #     resonators = self.get_all_resonators()
    #     resonators = numpy.round(resonators, roundDepth)
        
    #     maxRootDegree = int(numpy.ceil(self.maxDegree/2 -1))
        
    #     numVerts = self.coords.shape[0]
    #     if sparse:
    #         rowVec = numpy.zeros(numVerts*maxRootDegree +flags)
    #         colVec = numpy.zeros(numVerts*maxRootDegree +flags)
    #         Hvec = numpy.zeros(numVerts*maxRootDegree +flags)
    #     else:
    #         Hmat = numpy.zeros((numVerts, numVerts))
            
            
    #     # def check_redundnacy(site, svec_all, shift1, shift2, shift3):
    #     #     # vec1 = numpy.round(self.a1[0] + 1j*self.a1[1], roundDepth)
    #     #     # vec2 = numpy.round(self.a2[0] + 1j*self.a2[1], roundDepth)
            
    #     #     shiftVec = shift1*self.a1 + shift2*self.a2 + shift3*self.a3
            
    #     #     shiftedCoords = numpy.zeros(svec_all.shape)
    #     #     shiftedCoords[:,0] = svec_all[:,0] + shiftVec[0]
    #     #     shiftedCoords[:,1] = svec_all[:,1] + shiftVec[1]
    #     #     shiftedCoords[:,2] = svec_all[:,2] + shiftVec[2]

    #     #     check  = numpy.round(site - shiftedCoords, roundDepth)

    #     #     redundancies = numpy.where((check  == (0, 0,0)).all(axis=1))[0] 
    #     #     return redundancies
        
    #     coords_complex = numpy.round(self.coords[:,0] + 1j*self.coords[:,1], roundDepth)
        
    #     currInd = 0
    #     for rind in range(0, resonators.shape[0]):
    #         resPos = resonators[rind,:]
    #         startPos = numpy.round(resPos[0],roundDepth)+ 1j*numpy.round(resPos[1],roundDepth)
    #         stopPos = numpy.round(resPos[2],roundDepth)+ 1j*numpy.round(resPos[3],roundDepth)
            
    #         startInd = numpy.where(startPos == coords_complex)[0][0]
    #         stopInd = numpy.where(stopPos == coords_complex)[0][0]
    
    #         if sparse:
    #             rowVec[currInd] = startInd
    #             colVec[currInd] = stopInd
    #             Hvec[currInd] = t #will end up adding t towhatever this entry was before.
    #             currInd = currInd +1
                
    #             rowVec[currInd] = stopInd
    #             colVec[currInd] = startInd
    #             Hvec[currInd] = t #will end up adding t towhatever this entry was before.
    #             currInd = currInd +1
                
    #         else:
    #             Hmat[startInd, stopInd] = Hmat[startInd, stopInd] + t
    #             Hmat[stopInd, startInd] = Hmat[stopInd, startInd] + t
        
    #     #finish making the sparse matrix if we are in sparse matrix mode.
    #     if sparse:
    #         #pad the end of the matrix with values so that I can see if one of those is the missing one
    #         for ind in range(0, flags):
    #             rowVec[currInd] = numVerts+ind
    #             colVec[currInd] = numVerts+ind
    #             Hvec[currInd] = -7.5 #will end up adding t towhatever this entry was before.
    #             currInd = currInd +1
    
    #         Hmat = coo_matrix((Hvec,(rowVec,colVec)), shape = (numVerts+flags,numVerts+flags), dtype = 'd')
    #         Hmat.eliminate_zeros() #removed the unused spots since this is not a regular graph
            
    #     if verbose:
    #         temp = numpy.sum(Hmat)/numVerts
    #         print( 'average degree = ' + str(temp))
        
    #     self.rootHamiltonian = Hmat
        
    #     if not sparse:
    #         self.rootEs, self.rootPsis = numpy.linalg.eigh(self.rootHamiltonian)
            
    #     return
    
    # def plot_root_state(self, state_vect, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', zorder = 1):
    #     '''
    #     plot a state (wavefunction) on the root graph of original vertices
    #     '''
    #     Amps = state_vect
    #     Probs = numpy.abs(Amps)**2
    #     mSizes = Probs * len(Probs)*30
    #     mColors = numpy.angle(Amps)/numpy.pi
        
    #     cm = pylab.cm.get_cmap(cmap)
        
    #     pylab.sca(ax)
    #     pylab.scatter(self.coords[:,0], self.coords[:,1],c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
    #     if colorbar:
    #         cbar = pylab.colorbar(fraction=0.046, pad=0.04)
    #         cbar.set_label('phase (pi radians)', rotation=270)
              
    #     if plot_links:
    #         self.draw_SDlinks(ax, linewidth = 0.5, color = 'firebrick')
        
    #     pylab.title(title, fontsize=8)
    #     ax.xaxis.set_visible(False)
    #     ax.yaxis.set_visible(False)
    #     ax.set_aspect('equal')
    #     return    
    
    
    
    
    
    
    
    
    

if __name__=="__main__": 
    
    #####
    #testing the unit cell functions
    #####
    
    #creat the cell
    cubeCell = UnitCell3D('cubic', a1 = [1,0,0], a2 = [0,1,0], a3 = [0,0,1])
    
    
    #draw the cell
    theta = -numpy.pi/5
    phi = -(3*numpy.pi/4)
    # theta = 0
    # phi = 0.05
    # theta = 0
    # phi = numpy.pi/2
    pylab.figure(1)
    pylab.clf()
    ax = pylab.subplot(1,2,1)
    
    cubeCell.draw_resonators(ax, theta, phi)
    cubeCell.draw_resonator_end_points(ax, theta, phi)
    
    ax.set_aspect('equal')
    
    
    ax = pylab.subplot(1,2,2)
    cubeCell.draw_resonators(ax, theta, phi)
    cubeCell.draw_resonator_end_points(ax, theta, phi)
    cubeCell.draw_SDlinks(ax, theta, phi)
    cubeCell.draw_sites(ax, theta, phi, marker = 'x', size = 50)
    
    ax.set_aspect('equal')
    pylab.suptitle('cubic lattice')
    
    pylab.show()
    
    
    manualLineGraph = True
    if not manualLineGraph:
        #####
        #make a line graph cell, the automatic way
        ####
        cubeLGCell = cubeCell.line_graph_cell()

        #draw the cell
        theta = -numpy.pi/5
        phi = -(3*numpy.pi/4)
        # theta = 0
        # phi = 0.05
        # theta = 0
        # phi = numpy.pi/2
        pylab.figure(2)
        pylab.clf()
        ax = pylab.subplot(1,2,1)
        cubeLGCell.draw_resonators(ax, theta, phi, color = 'mediumblue')
        cubeLGCell.draw_resonator_end_points(ax, theta, phi, color = 'darkgoldenrod')
        ax.set_aspect('equal')
        
        ax = pylab.subplot(1,2,2)
        cubeLGCell.draw_resonators(ax, theta, phi)
        cubeLGCell.draw_resonator_end_points(ax, theta, phi)
        cubeLGCell.draw_SDlinks(ax, theta, phi)
        cubeLGCell.draw_sites(ax, theta, phi, marker = 'x', size = 50)
        ax.set_aspect('equal')
        pylab.suptitle('line graph of cubic lattice')
        pylab.show()
    
    
    if manualLineGraph:
        #####
        #make a line graph cell, manually
        ####
        
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
        
        #draw the cell
        theta = -numpy.pi/10
        phi = -numpy.pi/10
        
        # theta = numpy.pi/4
        # phi = numpy.pi/25
        
        # theta = 0
        # phi = 0.05
        
        # theta = 0
        # phi = numpy.pi/2
        
        pylab.figure(2)
        pylab.clf()
        ax = pylab.subplot(1,2,1)
        cubeLGCell.draw_resonators(ax, theta, phi, color = 'mediumblue', linewidth = 1)
        cubeLGCell.draw_resonator_end_points(ax, theta, phi, color = 'darkgoldenrod', size = 50)
        ax.set_aspect('equal')
        
        ax = pylab.subplot(1,2,2)
        cubeLGCell.draw_resonators(ax, theta, phi, color = 'mediumblue', linewidth = 1)
        cubeLGCell.draw_resonator_end_points(ax, theta, phi, color = 'darkgoldenrod', size = 50)
        # cubeLGCell.draw_SDlinks(ax, theta, phi)
        cubeLGCell.draw_sites(ax, theta, phi, marker = 'x', size = 50)
        ax.set_aspect('equal')
        pylab.suptitle('line graph of cubic lattice')
        pylab.show()
    
    
    
    
    
    ####
    #next, lets check a Bloch-wave calculation
    ####
    
    
    #####
    #testing bloch theory
    ####
    
    # testCell = cubeCell
    testCell = cubeLGCell
    modeType = 'FW'

    Hmat = testCell.generate_Bloch_matrix(0,0,0,  modeType = modeType)
    pylab.figure(3)
    pylab.clf()
    ax = pylab.subplot(1,2,1)
    pylab.imshow(numpy.abs(Hmat))
    pylab.title('|H|')

    ax = pylab.subplot(1,2,2)
    pylab.imshow(numpy.real(Hmat - numpy.transpose(numpy.conj(Hmat))))
    pylab.title('H - Hdagger')

    pylab.show()



    
    kx_x, ky_y, kz_x, cutx = testCell.compute_band_structure(-numpy.pi, 0,0, 
                                                       numpy.pi, 0, 0,
                                                       numsteps = 100, modeType = modeType)
    kx_x, ky_y, kz_y, cuty = testCell.compute_band_structure(0,numpy.pi, 0,
                                                             0,-numpy.pi,0,
                                                       numsteps = 100, modeType = modeType)
    kx_x, ky_y, kz_z, cutxz = testCell.compute_band_structure(-numpy.pi,0,-numpy.pi, 
                                                       numpy.pi, 0,numpy.pi,
                                                       numsteps = 100, modeType = modeType)
    fig2 = pylab.figure(4)
    pylab.clf()
    ax = pylab.subplot(1,3,1)
    testCell.plot_band_cut(ax, cutx)
    pylab.title('xcut')

    ax = pylab.subplot(1,3,2)
    testCell.plot_band_cut(ax, cuty)
    pylab.title('ycut')
    
    ax = pylab.subplot(1,3,3)
    testCell.plot_band_cut(ax, cutxz)
    pylab.title('xzcut')

    titleStr = testCell.type + ', modeType: ' + modeType + ' (Made with UnitCell class)' 
    pylab.suptitle(titleStr)

    pylab.show()
    
    
    
    
    
    
    
    #######
    #now a Bloch-wave calculation on the root graphs (and a check that we are getting those right)
    ######
    
    # testCell = cubeCell
    testCell = cubeLGCell
    testCell.find_root_cell()
    modeType = 'FW'

    Hmat = testCell.generate_root_Bloch_matrix(0,0,0)
    pylab.figure(5)
    pylab.clf()
    ax = pylab.subplot(1,2,1)
    pylab.imshow(numpy.abs(Hmat))
    pylab.title('|H|')

    ax = pylab.subplot(1,2,2)
    pylab.imshow(numpy.real(Hmat - numpy.transpose(numpy.conj(Hmat))))
    pylab.title('H - Hdagger')

    pylab.show()



    
    kx_x, ky_y, kz_x, cutx = testCell.compute_root_band_structure(-numpy.pi, 0,0, 
                                                       numpy.pi, 0, 0,
                                                       numsteps = 100)
    kx_x, ky_y, kz_y, cuty = testCell.compute_root_band_structure(0,numpy.pi, 0,
                                                             0,-numpy.pi,0,
                                                       numsteps = 100)
    kx_x, ky_y, kz_z, cutxyz = testCell.compute_root_band_structure(-numpy.pi,-numpy.pi,-numpy.pi, 
                                                       numpy.pi, numpy.pi,numpy.pi,
                                                       numsteps = 100)
    fig2 = pylab.figure(6)
    pylab.clf()
    ax = pylab.subplot(1,3,1)
    testCell.plot_band_cut(ax, cutx)
    pylab.title('xcut')

    ax = pylab.subplot(1,3,2)
    testCell.plot_band_cut(ax, cuty)
    pylab.title('ycut')
    
    ax = pylab.subplot(1,3,3)
    testCell.plot_band_cut(ax, cutxyz)
    pylab.title('xyzcut')

    titleStr = testCell.type + ', modeType: ' + modeType + ' (Made with UnitCell class)' 
    pylab.suptitle(titleStr)

    pylab.show()
    
    
    
    
    
    #####
    #now to try making and displaying an Euclidean lattice
    #######################
    
    testCell = cubeCell
    size = 3
    # testLattice = EuclideanLayout3D(initialCell = cubeCell, xcells = size, ycells = size, zcells = size)
    testLattice = EuclideanLayout3D(initialCell = cubeLGCell, 
                                    xcells = size, ycells = size, zcells = size)
    
    
    theta = -1.5*numpy.pi/10
    phi = -1.5*numpy.pi/10
    # theta = -numpy.pi/10
    # phi = -numpy.pi/10
    # theta = 0
    # phi = 0
    
    pylab.figure(7)
    pylab.clf()
    ax = pylab.subplot(1,5,1)
    testLattice.draw_resonator_lattice(ax, theta, phi, color = 'mediumblue', linewidth = 1)
    testLattice.draw_resonator_end_points(ax, theta, phi, color ='goldenrod', size = 25)
    ax.set_aspect('equal')
    
    
    ax = pylab.subplot(1,5,2)
    testLattice.draw_resonator_lattice(ax, theta, phi,color = 'mediumblue', linewidth = 1)
    testLattice.draw_resonator_end_points(ax, theta, phi, color ='goldenrod', size = 25)
    testLattice.draw_SDlinks(ax, theta, phi)
    testLattice.draw_SD_points(ax, theta, phi, marker = 'x', size = 50)
    ax.set_aspect('equal')
    
    ax = pylab.subplot(1,5,3)
    testLattice.draw_SDlinks(ax, theta, phi)
    testLattice.draw_SD_points(ax, theta, phi, marker = 'o', size = 20)
    ax.set_aspect('equal')
    
    
    ax = pylab.subplot(1,5,4)
    testLattice.draw_resonator_lattice(ax, theta, phi,color = 'mediumblue', linewidth = 1,
                             xrange= [0,3],
                             yrange = [0,1],
                             zrange = [0,3])
    testLattice.draw_resonator_end_points(ax, theta, phi,color ='goldenrod',
                             xrange= [0,3],
                             yrange = [0,1],
                             zrange = [0,3],
                             marker = 'o', size = 25)
    ax.set_aspect('equal')
    
    
    ax = pylab.subplot(1,5,5)
    testLattice.draw_SDlinks(ax, theta, phi,
                             xrange= [0,3],
                             yrange = [0,1],
                             zrange = [0,3])
    testLattice.draw_SD_points(ax, theta, phi,
                               xrange= [0,3],
                             yrange = [0,1],
                             zrange = [0,3],
                               marker = 'o', size = 20)
    ax.set_aspect('equal')
    
    pylab.suptitle('Full Euclidean lattice')
    pylab.tight_layout()
    pylab.show()
    
    
    
    
    
    
    
    
    
# #    Cell = True
#     Cell = False

#     Lattice = True
# #    Lattice = False    


#     ####Cell mode sub options
#     K0States = False  #display or not 


#     ####Lattice mode sub options
#     LatticeHamiltonian = False #display or not 
#     LatticeInteractionMap = False #display or not 
    
    
    
    
#     ##################################
#     ##################################
#     ##################################
#     ##################################
#     ###########
#     #lattice testing  and examples
#     ##################################
#     ##################################
#     ##################################
#     ##################################
#     if Cell:
#         testCell = UnitCell('Huse')
        
#         #pylab.rcParams.update({'font.size': 14})
#         #pylab.rcParams.update({'font.size': 8})
        
#         modeType = 'FW'
#         #modeType = 'HW'
        
# #        testCell = UnitCell('Huse')
# #        testCell = UnitCell('74Huse')
# #        testCell = UnitCell('84Huse')
# #        testCell = UnitCell('123Huse')
# #        testCell = UnitCell('kagome')
        
# #        testCell = UnitCell('Huse2_1')
#         #testCell = UnitCell('Huse2_2')
#         #testCell = UnitCell('Huse3_1')
#         #testCell = UnitCell('Huse3_3')
        
#         #testCell = UnitCell('84Huse2_1')
        
#         #testCell = UnitCell('PeterChain')
#         #testCell = UnitCell('PaterChain_tail')
        
#         ######
#         #test the unit cell
#         #######
#         pylab.figure(1)
#         pylab.clf()
#         ax = pylab.subplot(1,2,1)
#         testCell.draw_sites(ax)
#         pylab.title('Sites of Huse Cell')
        
#         ax = pylab.subplot(1,2,2)
#         testCell.draw_sites(ax,color = 'goldenrod', edgecolor = 'k',  marker = 'o' , size = 20)
#         testCell.draw_resonators(ax, color = 'cornflowerblue', linewidth = 1)
#         testCell.draw_SDlinks(ax, color = 'firebrick', linewidth = 1)
#         testCell.draw_resonator_end_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
#         pylab.title('Links of Unit Cell')
#         pylab.show()
        
        
#         ######
#         #show the orientations
#         ######
#         #alternate version
#         fig = pylab.figure(2)
#         pylab.clf()
#         ax = pylab.subplot(1,1,1)
#         testCell.draw_resonators(ax, color = 'cornflowerblue', linewidth = 1)
#         testCell.draw_resonator_end_points(ax, color = 'indigo', edgecolor = 'indigo',  marker = '+' , size = 20)
#         testCell.draw_site_orientations(ax, title = 'unit cell convention', colorbar = False, plot_links = False, cmap = 'jet', scaleFactor = 0.5)
#         testCell.draw_SDlinks(ax, linewidth = 1.5, HW = True , minus_color = 'goldenrod')
#         pylab.title('site orientations : ' + testCell.type)
#         #ax.set_aspect('auto')
#         ax.set_aspect('equal')
#         #    fig.savefig('HW.png', dpi = 200)
        
#         pylab.show()
        

        
#         #####
#         #testing bloch theory
#         ####
        
#         Hmat = testCell.generate_Bloch_matrix(0,0,  modeType = modeType)
#         pylab.figure(3)
#         pylab.clf()
#         ax = pylab.subplot(1,2,1)
#         pylab.imshow(numpy.abs(Hmat))
#         pylab.title('|H|')
        
#         ax = pylab.subplot(1,2,2)
#         pylab.imshow(numpy.real(Hmat - numpy.transpose(numpy.conj(Hmat))))
#         pylab.title('H - Hdagger')
        
#         pylab.show()
        
        
        
#         #kx_x, ky_y, cutx = testCell.compute_band_structure(-2*numpy.pi, 0, 2*numpy.pi, 0, numsteps = 100, modeType = modeType)
#         #kx_y, ky_y, cuty = testCell.compute_band_structure(0, -8./3*numpy.pi, 0, 8./3*numpy.pi, numsteps = 100, modeType = modeType)
#         kx_x, ky_y, cutx = testCell.compute_band_structure(-2*numpy.pi, 0, 2*numpy.pi, 0, numsteps = 100, modeType = modeType)
#         kx_y, ky_y, cuty = testCell.compute_band_structure(0, -2.5*numpy.pi, 0, 2.5*numpy.pi, numsteps = 100, modeType = modeType)
        
#         fig2 = pylab.figure(4)
#         pylab.clf()
#         ax = pylab.subplot(1,2,1)
#         testCell.plot_band_cut(ax, cutx)
#         pylab.title('xcut')
        
#         ax = pylab.subplot(1,2,2)
#         testCell.plot_band_cut(ax, cuty)
#         pylab.title('ycut')
        
#         titleStr = testCell.type + ', modeType: ' + modeType + ' (Made with UnitCell class)' 
#         pylab.suptitle(titleStr)
        
#         pylab.show()
        

        
#         #####
#         #look at above gap state at k= 0
#         #####
#         if K0States:
#             Es, Psis = scipy.linalg.eigh(Hmat)
            
#             stateInd = 0
#             aboveGap = Psis[:,stateInd]
#             print(Es[stateInd])
#             print(aboveGap)
            
#             pylab.figure(5)
#             pylab.clf()
            
#             ax = pylab.subplot(1,1,1)
#             #testCell.draw_sites(ax,color = 'goldenrod', edgecolor = 'k',  marker = 'o' , size = 20)
#             testCell.draw_SDlinks(ax, color = 'firebrick', linewidth = 1)
#             testCell.draw_resonators(ax, color = 'cornflowerblue', linewidth = 1)
#             testCell.draw_resonator_end_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
#             #testCell.plot_bloch_wave(aboveGap*2, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia')
#             temp = testCell.plot_bloch_wave_end_state(aboveGap*2, ax,modeType = modeType,  title = modeType + '_' + str(stateInd), colorbar = False, plot_links = False, cmap = 'Wistia')
#             ax.set_aspect('equal')
#             pylab.show()
            
            
#             ####try to plot all the unit cell wave functions. Doesn't work very well. You can't see anything
#             #pylab.figure(6)
#             #pylab.clf()
#             #for ind in range(0, testCell.numSites):
#             #    ax = pylab.subplot(1,testCell.numSites,ind+1)
#             #    testCell.draw_SDlinks(ax, color = 'firebrick', linewidth = 1)
#             #    testCell.draw_resonators(ax, color = 'cornflowerblue', linewidth = 1)
#             #    testCell.draw_resonator_end_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
#             ##    testCell.plot_bloch_wave(Psis[:,ind], ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia')
#             #    testCell.plot_bloch_wave_end_state(Psis[:,ind], ax,modeType = modeType,  title = str(ind), colorbar = False, plot_links = False, cmap = 'Wistia')
#             #    ax.set_aspect('equal')
#             #pylab.show()
#         else:
#             pylab.figure(5)
#             pylab.clf()
            
#             pylab.figure(6)
#             pylab.clf()
        
    
    
    
    
    
#     ##################################
#     ##################################
#     ##################################
#     ##################################
#     ###########
#     #lattice testing  and examples
#     ##################################
#     ##################################
#     ##################################
#     ##################################
#     if Lattice:
#     #    testLattice = EuclideanLayout(4,3,lattice_type = 'Huse', modeType = 'FW')
#     #    testLattice = EuclideanLayout(2,1,lattice_type = 'Huse', modeType = 'FW')
        
#     #    testLattice = EuclideanLayout(4,3,lattice_type = 'Huse', modeType = 'HW')
#         testLattice = EuclideanLayout(4,2,lattice_type = 'Huse', modeType = 'HW')
#     #    testLattice = EuclideanLayout(2,2,lattice_type = 'Huse', modeType = 'HW')
#     #    testLattice = EuclideanLayout(1,1,lattice_type = 'Huse', modeType = 'HW')
    
    
#         # testLattice = EuclideanLayout(4,3,lattice_type = 'Huse', modeType = 'FW', side = 500)
        
#         # testLattice = EuclideanLayout(4,4,lattice_type = 'kagome', modeType = 'FW')
        
# #        testLattice = EuclideanLayout(2,3,lattice_type = 'Huse2_1', modeType = 'FW')
    
# #        testLattice = EuclideanLayout(1,1,lattice_type = '84Huse2_1', modeType = 'FW')
        
#     #    testLattice = EuclideanLayout(2,1,lattice_type = '84Huse', modeType = 'FW')
# #        testLattice = EuclideanLayout(4,3,lattice_type = '74Huse', modeType = 'FW')
# #        testLattice = EuclideanLayout(4,3,lattice_type = '123Huse', modeType = 'FW')

#         # testLattice = EuclideanLayout(3,3,lattice_type = 'square', modeType = 'FW')
    
#         ######
#         #test the unit cell
#         #######
#         pylab.figure(1)
#         pylab.clf()
    
#         ######
#         #test the generate functions
#         #######
#     #    testLattice.generate_lattice()
#     #    testLattice.generate_semiduals()
#     #    testLattice.generate_Hamiltonian()
    
#         debugMode = False
#     #    debugMode = True
        
#         ######
#         #test the lattice and SD lattice constructions
#         #######
#         pylab.figure(2)
#         pylab.clf()
#         ax = pylab.subplot(1,2,1)
#         testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 1)
#         testLattice.draw_resonator_end_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
        
#         if debugMode:
#             testLattice.draw_resonator_lattice(ax, color = 'indigo', linewidth = 1, extras = True)
#             [x0, y0, x1, y1]  = testLattice.extraResonators[0,:]
#     #        ax.plot([x0, x1],[y0, y1] , color = 'firebrick', alpha = 1, linewidth = 1)
#             [x0, y0, x1, y1]  = testLattice.resonators[6,:]
#     #        ax.plot([x0, x1],[y0, y1] , color = 'indigo', alpha = 1, linewidth = 1)
        
#         pylab.title('Resonators of Huse Lattice')
        
#         ax = pylab.subplot(1,2,2)
#         testLattice.draw_SD_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
#         testLattice.draw_SDlinks(ax, color = 'firebrick', linewidth = 1)
        
#         if debugMode:
#             testLattice.draw_SD_points(ax, color = 'indigo', edgecolor = 'k',  marker = 'o' , size = 20, extra = True)
#             testLattice.draw_SDlinks(ax, color = 'cornflowerblue', linewidth = 1, extra = True)
#         #    pylab.scatter(testLattice.extraSDx,testLattice.extraSDy ,c =  'indigo', s = 25, marker ='o', edgecolors = 'k')
#         pylab.title('Links of the Huse Lattice')
#         pylab.show()
        
        
#         ######
#         #test the Hamiltonian
#         #######
#         eigNum = 168
#         eigNum = 167
#         eigNum = 0
#         if LatticeHamiltonian:
#             pylab.figure(3)
#             pylab.clf()
#             ax = pylab.subplot(1,2,1)
#             pylab.imshow(testLattice.H,cmap = 'winter')
#             pylab.title('Hamiltonian')
            
#             ax = pylab.subplot(1,2,2)
#             pylab.imshow(testLattice.H - numpy.transpose(testLattice.H),cmap = 'winter')
#             pylab.title('H - Htranspose')
#             pylab.show()
            
    
            
#             xs = scipy.arange(0,len(testLattice.Es),1)
#             eigAmps = testLattice.Psis[:,testLattice.Eorder[eigNum]]
            
#             pylab.figure(4)
#             pylab.clf()
#             ax1 = pylab.subplot(1,2,1)
#             pylab.plot(testLattice.Es, 'b.')
#             pylab.plot(xs[eigNum],testLattice.Es[testLattice.Eorder[eigNum]], color = 'firebrick' , marker = '.', markersize = '10' )
#             pylab.title('eigen spectrum')
#             pylab.ylabel('Energy (t)')
#             pylab.xlabel('eigenvalue number')
            
#             ax2 = pylab.subplot(1,2,2)
#             titleStr = 'eigenvector weight : ' + str(eigNum)
#             testLattice.plot_layout_state(eigAmps, ax2, title = titleStr, colorbar = True, plot_links = True, cmap = 'Wistia')
            
#             pylab.show()
#         else:
#             pylab.figure(3)
#             pylab.clf()
            
#             pylab.figure(4)
#             pylab.clf()
        
        
#         ######
#         #test the layout plotters (center dot)
#         #######
        
#         pylab.figure(5)
#         pylab.clf()
#         stateInd = eigNum
#         state1 = testLattice.Psis[:,stateInd]
#         if testLattice.xcells < 4 and testLattice.ycells <3:
#             state2 = testLattice.build_local_state(7)
#         else:
# #            state2 = testLattice.build_local_state(47)
#             state2 = testLattice.build_local_state(4)
        
        
#         ax = pylab.subplot(1,2,1)
#         testLattice.plot_layout_state(state1, ax, title = 'eigenstate', colorbar = False, plot_links = True, cmap = 'Wistia')
        
#         ax = pylab.subplot(1,2,2)
#         testLattice.plot_layout_state(state2/10, ax, title = 'local state', colorbar = False, plot_links = True, cmap = 'Wistia')
        
#         pylab.show()
        
        
#         ######
#         #test the interaction funtions
#         #######
#         if LatticeInteractionMap:
#             #    interactionStates = scipy.arange(0,len(testLattice.Es),1)
#             if testLattice.xcells < 4 and testLattice.ycells <3:
#                 interactionStates = scipy.arange(0,4,1)
#                 site1 = 1
#                 site2 = 5
#             else:
#                 interactionStates = scipy.arange(0,47,1)
#                 site1 = 10
#                 site2 = 54
            
            
            
#             V0 = testLattice.V_int(site1, site1, interactionStates)
#             VV = testLattice.V_int(site1, site2, interactionStates)
#             print(V0)
#             print(VV)
            
#             Vmap0 = testLattice.V_int_map(site2, interactionStates)
#             Vmap1 = testLattice.V_int_map(site2, interactionStates[0:4])
            
#             pylab.figure(6)
#             pylab.clf()
#             ax = pylab.subplot(1,2,1)
#             testLattice.plot_map_state(Vmap0, ax, title = 'ineraction weight: all FB states, hopefully', colorbar = True, plot_links = True, cmap = 'winter', autoscale = False)
#             pylab.scatter([testLattice.SDx[site2]], [testLattice.SDy[site2]], c =  'gold', s = 150, edgecolors = 'k')
            
#             ax = pylab.subplot(1,2,2)
#             testLattice.plot_map_state(Vmap1, ax, title = 'ineraction weight: first 4', colorbar = True, plot_links = True, cmap = 'winter', autoscale = False)
#             pylab.scatter([testLattice.SDx[site2]], [testLattice.SDy[site2]], c =  'gold', s = 150, edgecolors = 'k')
            
#             pylab.show()
#         else:
#             pylab.figure(6)
#             pylab.clf()
        
        
#         ######
#         #test visualization functions for shwing both ends of the resonators
#         #######
#         state_uniform = numpy.ones(len(testLattice.SDx))/numpy.sqrt(len(testLattice.SDx))
        
#         pylab.figure(7)
#         pylab.clf()
#         ax = pylab.subplot(1,2,1)
#     #    testLattice.plot_layout_state(state1, ax, title = 'eigenstate', colorbar = False, plot_links = True, cmap = 'Wistia')
#         testLattice.plot_layout_state(state_uniform, ax, title = 'eigenstate', colorbar = False, plot_links = True, cmap = 'Wistia')
        
#         ax = pylab.subplot(1,2,2)
#         endplot_points = testLattice.get_end_state_plot_points()
#     #    testLattice.plot_end_layout_state(state1, ax, title = 'end weights', colorbar = False, plot_links = True, cmap = 'Wistia', scaleFactor = 0.5)
#         testLattice.plot_end_layout_state(state_uniform, ax, title = 'end weights', colorbar = False, plot_links = True, cmap = 'Wistia', scaleFactor = 0.5)
        
#         pylab.show()
        
        
        
#     #    #####
#     #    #checking conventions
#     #    #####
#     #    
#     #    pylab.figure(17)
#     #    pylab.clf()
#     #    ax = pylab.subplot(1,2,1)
#     #    testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 1)
#     #    testLattice.draw_resonator_end_points(ax, color = 'indigo', edgecolor = 'indigo',  marker = '+' , size = 20)
#     ##    testLattice.plot_end_layout_state(state_uniform, ax, title = 'end weights', colorbar = False, plot_links = False, cmap = 'Wistia', scaleFactor = 0.5)
#     #    testLattice.plot_end_layout_state(state_uniform*1.4, ax, title = 'unit cell convention', colorbar = False, plot_links = False, cmap = 'jet', scaleFactor = 0.5)
#     #    testLattice.draw_SDlinks(ax, linewidth = 1, extra = False, minus_links = True, minus_color = 'goldenrod')
#     #    pylab.title('site orientations')
#     #    
#     #    ax = pylab.subplot(1,2,2)
#     #    pylab.imshow(testLattice.H,cmap = 'winter')
#     #    pylab.title('Hamiltonian')
#     #    pylab.show()
#     #    
#     #    pylab.figure(19)
#     #    pylab.clf()
#     #    ax = pylab.subplot(1,1,1)
#     #    testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 1)
#     #    testLattice.draw_resonator_end_points(ax, color = 'indigo', edgecolor = 'indigo',  marker = '+' , size = 20)
#     ##    testLattice.plot_end_layout_state(state_uniform, ax, title = 'end weights', colorbar = False, plot_links = False, cmap = 'Wistia', scaleFactor = 0.5)
#     #    testLattice.plot_end_layout_state(state_uniform*1.4, ax, title = 'unit cell convention', colorbar = False, plot_links = False, cmap = 'jet', scaleFactor = 0.5)
#     #    testLattice.draw_SDlinks(ax, linewidth = 1, extra = False, minus_links = True, minus_color = 'goldenrod')
#     #    pylab.title('site orientations')
#     #    ax.set_aspect('auto')
#     #    pylab.show()
        
#         #alternate version
#         fig = pylab.figure(19)
#         pylab.clf()
#         ax = pylab.subplot(1,1,1)
#         testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 1)
#         testLattice.draw_resonator_end_points(ax, color = 'indigo', edgecolor = 'indigo',  marker = '+' , size = 20)
#         testLattice.plot_end_layout_state(state_uniform, ax, title = 'unit cell convention', colorbar = False, plot_links = False, cmap = 'jet', scaleFactor = 0.5)
#         testLattice.draw_SDlinks(ax, linewidth = 1.5, extra = False, minus_links = True, minus_color = 'goldenrod')
#         pylab.title('site orientations')
# #        ax.set_aspect('auto')
#         ax.set_aspect('equal')
#     #    fig.savefig('HW.png', dpi = 200)
#         pylab.show()
    
#         #show lattice and medial
#         fig = pylab.figure(20)
#         pylab.clf()
#         ax = pylab.subplot(1,1,1)
#     #    testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 2)
#         testLattice.draw_resonator_lattice(ax, color = 'firebrick', linewidth = 2)
#         testLattice.draw_SDlinks(ax, linewidth = 2, extra = False, minus_links = False, color = 'goldenrod')
#         pylab.title('site orientations')
#         ax.set_aspect('auto')
# #        ax.set_aspect('equal')
#         ax.axis('off')
#     #    fig.savefig('HL.png', dpi = 200)
#         pylab.show()
    
#     #    #show just the medial
#     #    fig = pylab.figure(21)
#     #    pylab.clf()
#     #    ax = pylab.subplot(1,1,1)
#     #    testLattice.draw_SDlinks(ax, linewidth = 1.5, extra = False, minus_links = False, color = 'mediumblue')
#     ##    ax.set_aspect('auto')
#     #    ax.set_aspect('equal')
#     #    ax.axis('off')
#     ##    fig.savefig('Kagome.png', dpi = 200)
#     #    pylab.show()
        
        
        
#     #        #show lattice and medial
#     #    fig = pylab.figure(21)
#     #    pylab.clf()
#     #    ax = pylab.subplot(1,2,1)
#     #    testLattice.draw_resonator_lattice(ax, color = 'firebrick', linewidth = 2)
#     #    testLattice.draw_SDlinks(ax, linewidth = 2, extra = False, minus_links = False, color = 'goldenrod')
#     #    pylab.title('original resonators')
#     ##    ax.set_aspect('auto')
#     #    ax.set_aspect('equal')
#     #    ax.axis('off')
#     #    
#     #    ax = pylab.subplot(1,2,2)
#     #    testLattice.draw_SD_points(ax, color = 'dodgerblue', edgecolor = 'k',  marker = 'o' , size = 10)
#     #    pylab.title('SD sites')
#     #    ax.set_aspect('equal')
#     #    ax.axis('off')
#     #    
#     ##    fig.savefig('HL.png', dpi = 200)
        
        
#     #    
    













