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


EuclideanLayout3D Class
    Chose your UnitCell type, wave type, and number of unit cells and make a lattice
     
     Methods:
        
        *automated construction, saving, loading*
            * populate (autoruns at creation time)
            * save
        *functions to generate the resonator lattice*
            * generateLattice
            * _fix_edge_resonators (already stores some SD properties of fixed edge)
        *resonator lattice get /view functions*
            * get_xs
            * get_ys
            * rotate_resonators
            * rotate_coords
            * draw_resonator_lattice
            * draw_resonator_end_points
            * get_coords
            * get_cell_offset
            * get_cell_location
            * get_section_cut
        *functions to generate effective JC-Hubbard lattice (semiduals)*
            * generate_semiduals
            * _fix_SDedge
        
        *get and view functions for the JC-Hubbard (semi-dual lattice)*
            * draw_SD_points
            * draw_SDlinks
        
        *Hamiltonian related methods*
        
        *methods for calculating/looking at states and interactions*
            * plot_layout_state
        
        *methods for calculating things about the root graph*
            * generate_root_Hamiltonian
        
        
    Sample syntax:
        *loading precalculated layout*
        from EuclideanLayoutGenerator3D import EuclideanLayout3D
        testLattice = EuclideanLayout3D(file_path = 'Huse_4x4_FW.pkl')
        
        
        *making new layout*
        from EuclideanLayoutGenerator3D import EuclideanLayout3D
        #from built-in cell
        testLattice = EuclideanLayout3D(xcells = 4, ycells = 4, zcells = 4,lattice_type = 'Huse', side = 1, file_path = '', modeType = 'FW')
        
        from custom cell
        testCell = UnitCell3D(lattice_type = 'name', side = 1, resonators = resonatorMat, a1 = vec1, a2 = vec2, a3 = vec3)
        testLattice = EuclideanLayout3D(xcells = 4, ycells = 4, zcells = 4, modeType = 'FW', resonatorsOnly=False, initialCell = testCell)
        
        
        *saving computed layout*
        testLattice.save( name = 'filename.pkl') #filename can be a full path, but must have .pkl extension
"""



import matplotlib.pyplot as plt
import numpy as np

import pickle

from .GeneralLayoutGenerator3D import GeneralLayout3D
from .unit_cell_3D import UnitCell3D
       

class EuclideanLayout3D(GeneralLayout3D):
    '''
    EuclideanLayout3D _summary_

    :param GeneralLayout3D: _description_
    :type GeneralLayout3D: _type_
    '''    
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
        __init__ _summary_

        :param xcells: _description_, defaults to 4
        :type xcells: int, optional
        :param ycells: _description_, defaults to 4
        :type ycells: int, optional
        :param zcells: _description_, defaults to 4
        :type zcells: int, optional
        :param lattice_type: _description_, defaults to 'Huse'
        :type lattice_type: str, optional
        :param side: _description_, defaults to 1
        :type side: int, optional
        :param file_path: _description_, defaults to ''
        :type file_path: str, optional
        :param modeType: _description_, defaults to 'FW'
        :type modeType: str, optional
        :param resonatorsOnly: _description_, defaults to False
        :type resonatorsOnly: bool, optional
        :param Hamiltonian: _description_, defaults to False
        :type Hamiltonian: bool, optional
        :param initialCell: _description_, defaults to ''
        :type initialCell: str, optional
        :raises ValueError: _description_
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
            
            
    
    #automated construction, saving, loading
    
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
    
    #resonator lattice get /view functions
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
    
    def get_coords(self, resonators, roundDepth = 3):
        '''
        take in a set of resonators and calculate the set of end points.
        
        Will round all coordinates the the specified number of decimals.
        
        Should remove all redundancies.
        '''
        
        temp1 = resonators[:,0:3]
        temp2 = resonators[:,3:]
        coords_overcomplete = np.concatenate((temp1, temp2))
        
        coords = np.unique(np.round(coords_overcomplete, roundDepth), axis = 0)
        
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
            
        cellList = np.zeros(self.xcells*self.ycells*self.zcells)
        
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
        resList = np.zeros(len(cellList)*self.unitcell.numSites, dtype = 'int')
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
            
        
    
    #draw functions
    def rotate_resonators(self, resonators, theta, phi):
        '''rotate the coordinates of resonators into a projection view  
        
        theta is the angle to the z axis.
        
        phis is the aximuthal angle to the x axis.
        
        '''
                   
        initial_points1 = resonators[:,0:3]
        initial_points2 = resonators[:,3:]
        # return initial_points1, initial_points2
        
        #first rotate by the angle with respect to the z axis
        new_points1 = np.zeros(initial_points1.shape)
        new_points2 = np.zeros(initial_points2.shape)
        
        new_points1[:,0] = initial_points1[:,0]*np.cos(theta) - initial_points1[:,2]*np.sin(theta)
        new_points1[:,1] = initial_points1[:,1]
        new_points1[:,2] = initial_points1[:,0]*np.sin(theta) + initial_points1[:,2]*np.cos(theta)
        
        new_points2[:,0] = initial_points2[:,0]*np.cos(theta) - initial_points2[:,2]*np.sin(theta)
        new_points2[:,1] = initial_points2[:,1]
        new_points2[:,2] = initial_points2[:,0]*np.sin(theta) + initial_points2[:,2]*np.cos(theta)
        
        #store the rotate coordinates
        initial_points1 = np.copy(new_points1)
        initial_points2 = np.copy(new_points2)
        
        
        
        #now do the phi rotation
        new_points1 = np.zeros(initial_points1.shape)
        new_points2 = np.zeros(initial_points2.shape)
        
        new_points1[:,0] = initial_points1[:,0]*np.cos(phi) - initial_points1[:,1]*np.sin(phi)
        new_points1[:,1] = initial_points1[:,0]*np.sin(phi) + initial_points1[:,1]*np.cos(phi)
        new_points1[:,2] = initial_points1[:,2]
        
        new_points2[:,0] = initial_points2[:,0]*np.cos(phi) - initial_points2[:,1]*np.sin(phi)
        new_points2[:,1] = initial_points2[:,0]*np.sin(phi) + initial_points2[:,1]*np.cos(phi)
        new_points2[:,2] = initial_points2[:,2]
        
        newResonators = np.concatenate((new_points1, new_points2), axis = 1)
        
        newResonators = np.round(newResonators, 3)
    
        return newResonators
        
    def rotate_coordinates(self, coords, theta, phi):
        '''rotate a set of points into projection view
        
        theta is the angle from the z axis
        
        phi is the azimuthal angle from the x axis
        
        expects the coordinates to come a row vectors [:,0:3] = [x,y,z]
    
        '''

        
        #do the theta rotation
        new_points = np.zeros(coords.shape)
        new_points[:,0] = coords[:,0]*np.cos(theta) - coords[:,2]*np.sin(theta)
        new_points[:,1] = coords[:,1]
        new_points[:,2] = coords[:,0]*np.sin(theta) + coords[:,2]*np.cos(theta)
        
        #store the new values
        coords = np.copy(new_points)
        
        #now do the phi rotation
        new_points = np.zeros(coords.shape)
        
        new_points[:,0] = coords[:,0]*np.cos(phi) - coords[:,1]*np.sin(phi)
        new_points[:,1] = coords[:,0]*np.sin(phi) + coords[:,1]*np.cos(phi)
        new_points[:,2] = coords[:,2]
        
        new_points = np.round(new_points, 3)
        
        return new_points
    
    

    def draw_resonator_lattice(self, ax, 
                               theta = np.pi/10,
                               phi = np.pi/10,
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
                                  theta = np.pi/10,phi = np.pi/10,
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
        
        plt.sca(ax)
        plt.scatter(x0s, z0s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        plt.scatter(x1s, z1s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        return   

    
    
    
    #functions to generate the resonator lattice
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
        
        self.resonators = np.zeros((int(xsize*ysize*zsize*self.unitcell.numSites), 6))
    
        xmask = np.zeros((self.unitcell.numSites,6))
        ymask = np.zeros((self.unitcell.numSites,6))
        zmask = np.zeros((self.unitcell.numSites,6))
        
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
                    xOffset = np.round(xOffset,3)
                    yOffset = np.round(yOffset,3)
                    zOffset = np.round(zOffset,3)
                    self.resonators[ind:ind+self.unitcell.numSites, :] = self.unitcell.resonators + xOffset*xmask + yOffset*ymask + zOffset*zmask
                    # ind = ind + self.unitcell.numSites
                    
        self.coords = self.get_coords(self.resonators)        
        
        return

        
        

    #functions to generate effective JC-Hubbard lattice
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
        
        self.SDcoords = np.zeros((xsize*ysize*zsize*self.unitcell.numSites,3))
        self.SDx = np.zeros(xsize*ysize*zsize*self.unitcell.numSites)
        self.SDy = np.zeros(xsize*ysize*zsize*self.unitcell.numSites)
        self.SDz = np.zeros(xsize*ysize*zsize*self.unitcell.numSites)
        

            
        #now fixing to use the max degree given
        maxLineCoordination = (self.maxDegree-1)*2
        self.SDHWlinks = np.zeros((xsize*ysize*zsize*self.unitcell.numSites*maxLineCoordination, 4))
        
        
        #set up for getting the positions of the semidual points
        xmask = np.zeros((self.unitcell.numSites,6))
        ymask = np.zeros((self.unitcell.numSites,6))
        zmask = np.zeros((self.unitcell.numSites,6))
        
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
        self.SDHWlinks = self.SDHWlinks[~np.all(self.SDHWlinks == 0, axis=1)]  
        
        #make the truncated SD links
        self.SDlinks = self.SDHWlinks[:,0:2]
        
        #make the full SD coords matrix
        self.SDcoords[:,0] = self.SDx
        self.SDcoords[:,1] = self.SDy
        self.SDcoords[:,2] = self.SDz
            
        return
    
    def draw_SD_points(self, ax, 
                       theta = np.pi/10,phi = np.pi/10,
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
        
        plt.sca(ax)
        plt.scatter(xs, zs ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder, alpha = alpha)
        
        return

    def draw_SDlinks(self, ax,
                     theta = np.pi/10,phi = np.pi/10,
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

    #Hamiltonian related methods
    
    def plot_layout_state(self, state_vect, ax, 
                          theta = np.pi/10,
                          phi = np.pi/10,
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
        Probs = np.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = np.angle(Amps)/np.pi
        
        cm = plt.cm.get_cmap(cmap)
        
        #trim the intended plot points to the desired range
        mSizes = mSizes[resList]
        mColors = mColors[resList]
        
        #plot the x-z projection
        plt.sca(ax)
        plt.scatter(sdx, sdz,c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax, 
                              theta = theta, phi = phi,
                              xrange = xrange,
                              yrange = yrange,
                              zrange = zrange,
                              linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        plt.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return


    
    # def generate_root_Hamiltonian(self, roundDepth = 3, t = 1, verbose = False, sparse = False, flags = 5):
    #     '''
    #     custom function so I can get vertex dict without having to run the full populate of general layout
    #     and thereby having to also diagonalize the effective Hamiltonian.
        
    #     Will process the resonator matrix to get the layout Hamiltonian.
        
    #     Will return a regular matrix of sparse  = false, and a sparse matrix data type if sparse  = true
        
    #     Does not need to SD Hamiltonian made first.
        
        
    #     '''
    #     resonators = self.get_all_resonators()
    #     resonators = np.round(resonators, roundDepth)
        
    #     maxRootDegree = int(np.ceil(self.maxDegree/2 -1))
        
    #     numVerts = self.coords.shape[0]
    #     if sparse:
    #         rowVec = np.zeros(numVerts*maxRootDegree +flags)
    #         colVec = np.zeros(numVerts*maxRootDegree +flags)
    #         Hvec = np.zeros(numVerts*maxRootDegree +flags)
    #     else:
    #         Hmat = np.zeros((numVerts, numVerts))
            
            
    #     # def check_redundnacy(site, svec_all, shift1, shift2, shift3):
    #     #     # vec1 = np.round(self.a1[0] + 1j*self.a1[1], roundDepth)
    #     #     # vec2 = np.round(self.a2[0] + 1j*self.a2[1], roundDepth)
            
    #     #     shiftVec = shift1*self.a1 + shift2*self.a2 + shift3*self.a3
            
    #     #     shiftedCoords = np.zeros(svec_all.shape)
    #     #     shiftedCoords[:,0] = svec_all[:,0] + shiftVec[0]
    #     #     shiftedCoords[:,1] = svec_all[:,1] + shiftVec[1]
    #     #     shiftedCoords[:,2] = svec_all[:,2] + shiftVec[2]

    #     #     check  = np.round(site - shiftedCoords, roundDepth)

    #     #     redundancies = np.where((check  == (0, 0,0)).all(axis=1))[0] 
    #     #     return redundancies
        
    #     coords_complex = np.round(self.coords[:,0] + 1j*self.coords[:,1], roundDepth)
        
    #     currInd = 0
    #     for rind in range(0, resonators.shape[0]):
    #         resPos = resonators[rind,:]
    #         startPos = np.round(resPos[0],roundDepth)+ 1j*np.round(resPos[1],roundDepth)
    #         stopPos = np.round(resPos[2],roundDepth)+ 1j*np.round(resPos[3],roundDepth)
            
    #         startInd = np.where(startPos == coords_complex)[0][0]
    #         stopInd = np.where(stopPos == coords_complex)[0][0]
    
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
    #         temp = np.sum(Hmat)/numVerts
    #         print( 'average degree = ' + str(temp))
        
    #     self.rootHamiltonian = Hmat
        
    #     if not sparse:
    #         self.rootEs, self.rootPsis = np.linalg.eigh(self.rootHamiltonian)
            
    #     return
    

if __name__=="__main__": 
    
    #####
    #testing the unit cell functions
    #####
    
    #creat the cell
    cubeCell = UnitCell3D('cubic', a1 = [1,0,0], a2 = [0,1,0], a3 = [0,0,1])
    
    
    #draw the cell
    theta = -np.pi/5
    phi = -(3*np.pi/4)
    # theta = 0
    # phi = 0.05
    # theta = 0
    # phi = np.pi/2
    plt.figure(1)
    plt.clf()
    ax = plt.subplot(1,2,1)
    
    cubeCell.draw_resonators(ax, theta, phi)
    cubeCell.draw_resonator_end_points(ax, theta, phi)
    
    ax.set_aspect('equal')
    
    
    ax = plt.subplot(1,2,2)
    cubeCell.draw_resonators(ax, theta, phi)
    cubeCell.draw_resonator_end_points(ax, theta, phi)
    cubeCell.draw_SDlinks(ax, theta, phi)
    cubeCell.draw_sites(ax, theta, phi, marker = 'x', size = 50)
    
    ax.set_aspect('equal')
    plt.suptitle('cubic lattice')
    
    plt.show()
    
    
    manualLineGraph = True
    if not manualLineGraph:
        #make a line graph cell, the automatic way
        cubeLGCell = cubeCell.line_graph_cell()

        #draw the cell
        theta = -np.pi/5
        phi = -(3*np.pi/4)
        # theta = 0
        # phi = 0.05
        # theta = 0
        # phi = np.pi/2
        plt.figure(2)
        plt.clf()
        ax = plt.subplot(1,2,1)
        cubeLGCell.draw_resonators(ax, theta, phi, color = 'mediumblue')
        cubeLGCell.draw_resonator_end_points(ax, theta, phi, color = 'darkgoldenrod')
        ax.set_aspect('equal')
        
        ax = plt.subplot(1,2,2)
        cubeLGCell.draw_resonators(ax, theta, phi)
        cubeLGCell.draw_resonator_end_points(ax, theta, phi)
        cubeLGCell.draw_SDlinks(ax, theta, phi)
        cubeLGCell.draw_sites(ax, theta, phi, marker = 'x', size = 50)
        ax.set_aspect('equal')
        plt.suptitle('line graph of cubic lattice')
        plt.show()
    
    
    if manualLineGraph:
        #make a line graph cell, manually
        
        resonators = np.zeros((15,6))
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
        theta = -np.pi/10
        phi = -np.pi/10
        
        # theta = np.pi/4
        # phi = np.pi/25
        
        # theta = 0
        # phi = 0.05
        
        # theta = 0
        # phi = np.pi/2
        
        plt.figure(2)
        plt.clf()
        ax = plt.subplot(1,2,1)
        cubeLGCell.draw_resonators(ax, theta, phi, color = 'mediumblue', linewidth = 1)
        cubeLGCell.draw_resonator_end_points(ax, theta, phi, color = 'darkgoldenrod', size = 50)
        ax.set_aspect('equal')
        
        ax = plt.subplot(1,2,2)
        cubeLGCell.draw_resonators(ax, theta, phi, color = 'mediumblue', linewidth = 1)
        cubeLGCell.draw_resonator_end_points(ax, theta, phi, color = 'darkgoldenrod', size = 50)
        # cubeLGCell.draw_SDlinks(ax, theta, phi)
        cubeLGCell.draw_sites(ax, theta, phi, marker = 'x', size = 50)
        ax.set_aspect('equal')
        plt.suptitle('line graph of cubic lattice')
        plt.show()
    
    
    
    
    
    #next, lets check a Bloch-wave calculation
    
    
    #testing bloch theory
    
    # testCell = cubeCell
    testCell = cubeLGCell
    modeType = 'FW'

    Hmat = testCell.generate_Bloch_matrix(0,0,0,  modeType = modeType)
    plt.figure(3)
    plt.clf()
    ax = plt.subplot(1,2,1)
    plt.imshow(np.abs(Hmat))
    plt.title('|H|')

    ax = plt.subplot(1,2,2)
    plt.imshow(np.real(Hmat - np.transpose(np.conj(Hmat))))
    plt.title('H - Hdagger')

    plt.show()



    
    kx_x, ky_y, kz_x, cutx = testCell.compute_band_structure(-np.pi, 0,0, 
                                                       np.pi, 0, 0,
                                                       numsteps = 100, modeType = modeType)
    kx_x, ky_y, kz_y, cuty = testCell.compute_band_structure(0,np.pi, 0,
                                                             0,-np.pi,0,
                                                       numsteps = 100, modeType = modeType)
    kx_x, ky_y, kz_z, cutxz = testCell.compute_band_structure(-np.pi,0,-np.pi, 
                                                       np.pi, 0,np.pi,
                                                       numsteps = 100, modeType = modeType)
    fig2 = plt.figure(4)
    plt.clf()
    ax = plt.subplot(1,3,1)
    testCell.plot_band_cut(ax, cutx)
    plt.title('xcut')

    ax = plt.subplot(1,3,2)
    testCell.plot_band_cut(ax, cuty)
    plt.title('ycut')
    
    ax = plt.subplot(1,3,3)
    testCell.plot_band_cut(ax, cutxz)
    plt.title('xzcut')

    titleStr = testCell.type + ', modeType: ' + modeType + ' (Made with UnitCell class)' 
    plt.suptitle(titleStr)

    plt.show()
    
    
    
    
    
    
    
    #now a Bloch-wave calculation on the root graphs (and a check that we are getting those right)
    
    # testCell = cubeCell
    testCell = cubeLGCell
    testCell.find_root_cell()
    modeType = 'FW'

    Hmat = testCell.generate_root_Bloch_matrix(0,0,0)
    plt.figure(5)
    plt.clf()
    ax = plt.subplot(1,2,1)
    plt.imshow(np.abs(Hmat))
    plt.title('|H|')

    ax = plt.subplot(1,2,2)
    plt.imshow(np.real(Hmat - np.transpose(np.conj(Hmat))))
    plt.title('H - Hdagger')

    plt.show()



    
    kx_x, ky_y, kz_x, cutx = testCell.compute_root_band_structure(-np.pi, 0,0, 
                                                       np.pi, 0, 0,
                                                       numsteps = 100)
    kx_x, ky_y, kz_y, cuty = testCell.compute_root_band_structure(0,np.pi, 0,
                                                             0,-np.pi,0,
                                                       numsteps = 100)
    kx_x, ky_y, kz_z, cutxyz = testCell.compute_root_band_structure(-np.pi,-np.pi,-np.pi, 
                                                       np.pi, np.pi,np.pi,
                                                       numsteps = 100)
    fig2 = plt.figure(6)
    plt.clf()
    ax = plt.subplot(1,3,1)
    testCell.plot_band_cut(ax, cutx)
    plt.title('xcut')

    ax = plt.subplot(1,3,2)
    testCell.plot_band_cut(ax, cuty)
    plt.title('ycut')
    
    ax = plt.subplot(1,3,3)
    testCell.plot_band_cut(ax, cutxyz)
    plt.title('xyzcut')

    titleStr = testCell.type + ', modeType: ' + modeType + ' (Made with UnitCell class)' 
    plt.suptitle(titleStr)

    plt.show()
    
    
    
    
    
    #now to try making and displaying an Euclidean lattice
    
    testCell = cubeCell
    size = 3
    # testLattice = EuclideanLayout3D(initialCell = cubeCell, xcells = size, ycells = size, zcells = size)
    testLattice = EuclideanLayout3D(initialCell = cubeLGCell, 
                                    xcells = size, ycells = size, zcells = size)
    
    
    theta = -1.5*np.pi/10
    phi = -1.5*np.pi/10
    # theta = -np.pi/10
    # phi = -np.pi/10
    # theta = 0
    # phi = 0
    
    plt.figure(7)
    plt.clf()
    ax = plt.subplot(1,5,1)
    testLattice.draw_resonator_lattice(ax, theta, phi, color = 'mediumblue', linewidth = 1)
    testLattice.draw_resonator_end_points(ax, theta, phi, color ='goldenrod', size = 25)
    ax.set_aspect('equal')
    
    
    ax = plt.subplot(1,5,2)
    testLattice.draw_resonator_lattice(ax, theta, phi,color = 'mediumblue', linewidth = 1)
    testLattice.draw_resonator_end_points(ax, theta, phi, color ='goldenrod', size = 25)
    testLattice.draw_SDlinks(ax, theta, phi)
    testLattice.draw_SD_points(ax, theta, phi, marker = 'x', size = 50)
    ax.set_aspect('equal')
    
    ax = plt.subplot(1,5,3)
    testLattice.draw_SDlinks(ax, theta, phi)
    testLattice.draw_SD_points(ax, theta, phi, marker = 'o', size = 20)
    ax.set_aspect('equal')
    
    
    ax = plt.subplot(1,5,4)
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
    
    
    ax = plt.subplot(1,5,5)
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
    
    plt.suptitle('Full Euclidean lattice')
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
    
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
        
#         #plt.rcParams.update({'font.size': 14})
#         #plt.rcParams.update({'font.size': 8})
        
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
#         plt.figure(1)
#         plt.clf()
#         ax = plt.subplot(1,2,1)
#         testCell.draw_sites(ax)
#         plt.title('Sites of Huse Cell')
        
#         ax = plt.subplot(1,2,2)
#         testCell.draw_sites(ax,color = 'goldenrod', edgecolor = 'k',  marker = 'o' , size = 20)
#         testCell.draw_resonators(ax, color = 'cornflowerblue', linewidth = 1)
#         testCell.draw_SDlinks(ax, color = 'firebrick', linewidth = 1)
#         testCell.draw_resonator_end_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
#         plt.title('Links of Unit Cell')
#         plt.show()
        
        
#         ######
#         #show the orientations
#         ######
#         #alternate version
#         fig = plt.figure(2)
#         plt.clf()
#         ax = plt.subplot(1,1,1)
#         testCell.draw_resonators(ax, color = 'cornflowerblue', linewidth = 1)
#         testCell.draw_resonator_end_points(ax, color = 'indigo', edgecolor = 'indigo',  marker = '+' , size = 20)
#         testCell.draw_site_orientations(ax, title = 'unit cell convention', colorbar = False, plot_links = False, cmap = 'jet', scaleFactor = 0.5)
#         testCell.draw_SDlinks(ax, linewidth = 1.5, HW = True , minus_color = 'goldenrod')
#         plt.title('site orientations : ' + testCell.type)
#         #ax.set_aspect('auto')
#         ax.set_aspect('equal')
#         #    fig.savefig('HW.png', dpi = 200)
        
#         plt.show()
        

        
#         #####
#         #testing bloch theory
#         ####
        
#         Hmat = testCell.generate_Bloch_matrix(0,0,  modeType = modeType)
#         plt.figure(3)
#         plt.clf()
#         ax = plt.subplot(1,2,1)
#         plt.imshow(np.abs(Hmat))
#         plt.title('|H|')
        
#         ax = plt.subplot(1,2,2)
#         plt.imshow(np.real(Hmat - np.transpose(np.conj(Hmat))))
#         plt.title('H - Hdagger')
        
#         plt.show()
        
        
        
#         #kx_x, ky_y, cutx = testCell.compute_band_structure(-2*np.pi, 0, 2*np.pi, 0, numsteps = 100, modeType = modeType)
#         #kx_y, ky_y, cuty = testCell.compute_band_structure(0, -8./3*np.pi, 0, 8./3*np.pi, numsteps = 100, modeType = modeType)
#         kx_x, ky_y, cutx = testCell.compute_band_structure(-2*np.pi, 0, 2*np.pi, 0, numsteps = 100, modeType = modeType)
#         kx_y, ky_y, cuty = testCell.compute_band_structure(0, -2.5*np.pi, 0, 2.5*np.pi, numsteps = 100, modeType = modeType)
        
#         fig2 = plt.figure(4)
#         plt.clf()
#         ax = plt.subplot(1,2,1)
#         testCell.plot_band_cut(ax, cutx)
#         plt.title('xcut')
        
#         ax = plt.subplot(1,2,2)
#         testCell.plot_band_cut(ax, cuty)
#         plt.title('ycut')
        
#         titleStr = testCell.type + ', modeType: ' + modeType + ' (Made with UnitCell class)' 
#         plt.suptitle(titleStr)
        
#         plt.show()
        

        
#         #####
#         #look at above gap state at k= 0
#         #####
#         if K0States:
#             Es, Psis = scipy.linalg.eigh(Hmat)
            
#             stateInd = 0
#             aboveGap = Psis[:,stateInd]
#             print(Es[stateInd])
#             print(aboveGap)
            
#             plt.figure(5)
#             plt.clf()
            
#             ax = plt.subplot(1,1,1)
#             #testCell.draw_sites(ax,color = 'goldenrod', edgecolor = 'k',  marker = 'o' , size = 20)
#             testCell.draw_SDlinks(ax, color = 'firebrick', linewidth = 1)
#             testCell.draw_resonators(ax, color = 'cornflowerblue', linewidth = 1)
#             testCell.draw_resonator_end_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
#             #testCell.plot_bloch_wave(aboveGap*2, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia')
#             temp = testCell.plot_bloch_wave_end_state(aboveGap*2, ax,modeType = modeType,  title = modeType + '_' + str(stateInd), colorbar = False, plot_links = False, cmap = 'Wistia')
#             ax.set_aspect('equal')
#             plt.show()
            
            
#             ####try to plot all the unit cell wave functions. Doesn't work very well. You can't see anything
#             #plt.figure(6)
#             #plt.clf()
#             #for ind in range(0, testCell.numSites):
#             #    ax = plt.subplot(1,testCell.numSites,ind+1)
#             #    testCell.draw_SDlinks(ax, color = 'firebrick', linewidth = 1)
#             #    testCell.draw_resonators(ax, color = 'cornflowerblue', linewidth = 1)
#             #    testCell.draw_resonator_end_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
#             ##    testCell.plot_bloch_wave(Psis[:,ind], ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia')
#             #    testCell.plot_bloch_wave_end_state(Psis[:,ind], ax,modeType = modeType,  title = str(ind), colorbar = False, plot_links = False, cmap = 'Wistia')
#             #    ax.set_aspect('equal')
#             #plt.show()
#         else:
#             plt.figure(5)
#             plt.clf()
            
#             plt.figure(6)
#             plt.clf()
        
    
    
    
    
    
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
#         plt.figure(1)
#         plt.clf()
    
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
#         plt.figure(2)
#         plt.clf()
#         ax = plt.subplot(1,2,1)
#         testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 1)
#         testLattice.draw_resonator_end_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
        
#         if debugMode:
#             testLattice.draw_resonator_lattice(ax, color = 'indigo', linewidth = 1, extras = True)
#             [x0, y0, x1, y1]  = testLattice.extraResonators[0,:]
#     #        ax.plot([x0, x1],[y0, y1] , color = 'firebrick', alpha = 1, linewidth = 1)
#             [x0, y0, x1, y1]  = testLattice.resonators[6,:]
#     #        ax.plot([x0, x1],[y0, y1] , color = 'indigo', alpha = 1, linewidth = 1)
        
#         plt.title('Resonators of Huse Lattice')
        
#         ax = plt.subplot(1,2,2)
#         testLattice.draw_SD_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
#         testLattice.draw_SDlinks(ax, color = 'firebrick', linewidth = 1)
        
#         if debugMode:
#             testLattice.draw_SD_points(ax, color = 'indigo', edgecolor = 'k',  marker = 'o' , size = 20, extra = True)
#             testLattice.draw_SDlinks(ax, color = 'cornflowerblue', linewidth = 1, extra = True)
#         #    plt.scatter(testLattice.extraSDx,testLattice.extraSDy ,c =  'indigo', s = 25, marker ='o', edgecolors = 'k')
#         plt.title('Links of the Huse Lattice')
#         plt.show()
        
        
#         ######
#         #test the Hamiltonian
#         #######
#         eigNum = 168
#         eigNum = 167
#         eigNum = 0
#         if LatticeHamiltonian:
#             plt.figure(3)
#             plt.clf()
#             ax = plt.subplot(1,2,1)
#             plt.imshow(testLattice.H,cmap = 'winter')
#             plt.title('Hamiltonian')
            
#             ax = plt.subplot(1,2,2)
#             plt.imshow(testLattice.H - np.transpose(testLattice.H),cmap = 'winter')
#             plt.title('H - Htranspose')
#             plt.show()
            
    
            
#             xs = scipy.arange(0,len(testLattice.Es),1)
#             eigAmps = testLattice.Psis[:,testLattice.Eorder[eigNum]]
            
#             plt.figure(4)
#             plt.clf()
#             ax1 = plt.subplot(1,2,1)
#             plt.plot(testLattice.Es, 'b.')
#             plt.plot(xs[eigNum],testLattice.Es[testLattice.Eorder[eigNum]], color = 'firebrick' , marker = '.', markersize = '10' )
#             plt.title('eigen spectrum')
#             plt.ylabel('Energy (t)')
#             plt.xlabel('eigenvalue number')
            
#             ax2 = plt.subplot(1,2,2)
#             titleStr = 'eigenvector weight : ' + str(eigNum)
#             testLattice.plot_layout_state(eigAmps, ax2, title = titleStr, colorbar = True, plot_links = True, cmap = 'Wistia')
            
#             plt.show()
#         else:
#             plt.figure(3)
#             plt.clf()
            
#             plt.figure(4)
#             plt.clf()
        
        
#         ######
#         #test the layout plotters (center dot)
#         #######
        
#         plt.figure(5)
#         plt.clf()
#         stateInd = eigNum
#         state1 = testLattice.Psis[:,stateInd]
#         if testLattice.xcells < 4 and testLattice.ycells <3:
#             state2 = testLattice.build_local_state(7)
#         else:
# #            state2 = testLattice.build_local_state(47)
#             state2 = testLattice.build_local_state(4)
        
        
#         ax = plt.subplot(1,2,1)
#         testLattice.plot_layout_state(state1, ax, title = 'eigenstate', colorbar = False, plot_links = True, cmap = 'Wistia')
        
#         ax = plt.subplot(1,2,2)
#         testLattice.plot_layout_state(state2/10, ax, title = 'local state', colorbar = False, plot_links = True, cmap = 'Wistia')
        
#         plt.show()
        
        
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
            
#             plt.figure(6)
#             plt.clf()
#             ax = plt.subplot(1,2,1)
#             testLattice.plot_map_state(Vmap0, ax, title = 'ineraction weight: all FB states, hopefully', colorbar = True, plot_links = True, cmap = 'winter', autoscale = False)
#             plt.scatter([testLattice.SDx[site2]], [testLattice.SDy[site2]], c =  'gold', s = 150, edgecolors = 'k')
            
#             ax = plt.subplot(1,2,2)
#             testLattice.plot_map_state(Vmap1, ax, title = 'ineraction weight: first 4', colorbar = True, plot_links = True, cmap = 'winter', autoscale = False)
#             plt.scatter([testLattice.SDx[site2]], [testLattice.SDy[site2]], c =  'gold', s = 150, edgecolors = 'k')
            
#             plt.show()
#         else:
#             plt.figure(6)
#             plt.clf()
        
        
#         ######
#         #test visualization functions for shwing both ends of the resonators
#         #######
#         state_uniform = np.ones(len(testLattice.SDx))/np.sqrt(len(testLattice.SDx))
        
#         plt.figure(7)
#         plt.clf()
#         ax = plt.subplot(1,2,1)
#     #    testLattice.plot_layout_state(state1, ax, title = 'eigenstate', colorbar = False, plot_links = True, cmap = 'Wistia')
#         testLattice.plot_layout_state(state_uniform, ax, title = 'eigenstate', colorbar = False, plot_links = True, cmap = 'Wistia')
        
#         ax = plt.subplot(1,2,2)
#         endplot_points = testLattice.get_end_state_plot_points()
#     #    testLattice.plot_end_layout_state(state1, ax, title = 'end weights', colorbar = False, plot_links = True, cmap = 'Wistia', scaleFactor = 0.5)
#         testLattice.plot_end_layout_state(state_uniform, ax, title = 'end weights', colorbar = False, plot_links = True, cmap = 'Wistia', scaleFactor = 0.5)
        
#         plt.show()
        
        
        
#     #    #####
#     #    #checking conventions
#     #    #####
#     #    
#     #    plt.figure(17)
#     #    plt.clf()
#     #    ax = plt.subplot(1,2,1)
#     #    testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 1)
#     #    testLattice.draw_resonator_end_points(ax, color = 'indigo', edgecolor = 'indigo',  marker = '+' , size = 20)
#     ##    testLattice.plot_end_layout_state(state_uniform, ax, title = 'end weights', colorbar = False, plot_links = False, cmap = 'Wistia', scaleFactor = 0.5)
#     #    testLattice.plot_end_layout_state(state_uniform*1.4, ax, title = 'unit cell convention', colorbar = False, plot_links = False, cmap = 'jet', scaleFactor = 0.5)
#     #    testLattice.draw_SDlinks(ax, linewidth = 1, extra = False, minus_links = True, minus_color = 'goldenrod')
#     #    plt.title('site orientations')
#     #    
#     #    ax = plt.subplot(1,2,2)
#     #    plt.imshow(testLattice.H,cmap = 'winter')
#     #    plt.title('Hamiltonian')
#     #    plt.show()
#     #    
#     #    plt.figure(19)
#     #    plt.clf()
#     #    ax = plt.subplot(1,1,1)
#     #    testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 1)
#     #    testLattice.draw_resonator_end_points(ax, color = 'indigo', edgecolor = 'indigo',  marker = '+' , size = 20)
#     ##    testLattice.plot_end_layout_state(state_uniform, ax, title = 'end weights', colorbar = False, plot_links = False, cmap = 'Wistia', scaleFactor = 0.5)
#     #    testLattice.plot_end_layout_state(state_uniform*1.4, ax, title = 'unit cell convention', colorbar = False, plot_links = False, cmap = 'jet', scaleFactor = 0.5)
#     #    testLattice.draw_SDlinks(ax, linewidth = 1, extra = False, minus_links = True, minus_color = 'goldenrod')
#     #    plt.title('site orientations')
#     #    ax.set_aspect('auto')
#     #    plt.show()
        
#         #alternate version
#         fig = plt.figure(19)
#         plt.clf()
#         ax = plt.subplot(1,1,1)
#         testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 1)
#         testLattice.draw_resonator_end_points(ax, color = 'indigo', edgecolor = 'indigo',  marker = '+' , size = 20)
#         testLattice.plot_end_layout_state(state_uniform, ax, title = 'unit cell convention', colorbar = False, plot_links = False, cmap = 'jet', scaleFactor = 0.5)
#         testLattice.draw_SDlinks(ax, linewidth = 1.5, extra = False, minus_links = True, minus_color = 'goldenrod')
#         plt.title('site orientations')
# #        ax.set_aspect('auto')
#         ax.set_aspect('equal')
#     #    fig.savefig('HW.png', dpi = 200)
#         plt.show()
    
#         #show lattice and medial
#         fig = plt.figure(20)
#         plt.clf()
#         ax = plt.subplot(1,1,1)
#     #    testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 2)
#         testLattice.draw_resonator_lattice(ax, color = 'firebrick', linewidth = 2)
#         testLattice.draw_SDlinks(ax, linewidth = 2, extra = False, minus_links = False, color = 'goldenrod')
#         plt.title('site orientations')
#         ax.set_aspect('auto')
# #        ax.set_aspect('equal')
#         ax.axis('off')
#     #    fig.savefig('HL.png', dpi = 200)
#         plt.show()
    
#     #    #show just the medial
#     #    fig = plt.figure(21)
#     #    plt.clf()
#     #    ax = plt.subplot(1,1,1)
#     #    testLattice.draw_SDlinks(ax, linewidth = 1.5, extra = False, minus_links = False, color = 'mediumblue')
#     ##    ax.set_aspect('auto')
#     #    ax.set_aspect('equal')
#     #    ax.axis('off')
#     ##    fig.savefig('Kagome.png', dpi = 200)
#     #    plt.show()
        
        
        
#     #        #show lattice and medial
#     #    fig = plt.figure(21)
#     #    plt.clf()
#     #    ax = plt.subplot(1,2,1)
#     #    testLattice.draw_resonator_lattice(ax, color = 'firebrick', linewidth = 2)
#     #    testLattice.draw_SDlinks(ax, linewidth = 2, extra = False, minus_links = False, color = 'goldenrod')
#     #    plt.title('original resonators')
#     ##    ax.set_aspect('auto')
#     #    ax.set_aspect('equal')
#     #    ax.axis('off')
#     #    
#     #    ax = plt.subplot(1,2,2)
#     #    testLattice.draw_SD_points(ax, color = 'dodgerblue', edgecolor = 'k',  marker = 'o' , size = 10)
#     #    plt.title('SD sites')
#     #    ax.set_aspect('equal')
#     #    ax.axis('off')
#     #    
#     ##    fig.savefig('HL.png', dpi = 200)
        
        
#     #    
    













