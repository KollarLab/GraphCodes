#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:22:52 2018

@author: kollar2


this class is modified from the 2D version on 11-21-21

Description for the old 2D version
GeneralLayout Class:
#####################
    input a set of resoantors (the full lattice/tree/etc) and calculate properties
    v0 - self.coords is wierd, and may not contain all the capacitor points
     
     Methods:
        *automated construction, saving, loading*
            * populate (autoruns at creation time)
            * save
            * load
        
        *functions to generate the resonator lattice*
            * NA
        
        *resonator lattice get /view functions*
            * get_xs
            * get_ys
            * draw_resonator_lattice
            * draw_resonator_end_points
            * get_all_resonators
            * get_coords
        
        *functions to generate effective JC-Hubbard lattice (semiduals)*
            * generate_semiduals
            * generate_vertex_dict
        
        *get and view functions for the JC-Hubbard (semi-dual lattice)*
            * draw_SD_points
            * draw_SDlinks
            * get_semidual_points (semi-defunct)
        
        *Hamiltonian related methods*
            * generate_Hamiltonian
            * get_eigs
        
        *methods for calculating/looking at states and interactions*
            * get_SDindex (removed for now. Needs to be reimplemented in sensible fashion)
            * build_local_state
            * V_int
            * V_int_map
            * plot_layout_state
            * plot_map_state
            * get_end_state_plot_points
            * plot_end_layout_state
        
        *methods for calculating things about the root graph*
            * generate_root_Hamiltonian
            * plot_root_state
        
        
    Sample syntax:
        *loading precalculated layout*
        from GeneralLayoutGenerator import GeneralLayout
        testLattice = GeneralLayout(file_path = 'name.pkl')
        
        *making new layout*
        from GeneralLayoutGenerator import GeneralLayout
        from EuclideanLayoutGenerator2 import UnitCell
        from LayoutGenerator5 import PlanarLayout
        from GeneralLayoutGenerator import TreeResonators
        
        #hyperbolic
        test1 = PlanarLayout(gon = 7, vertex = 3, side =1, radius_method = 'lin')
        test1.populate(2, resonatorsOnly=False)
        resonators = test1.get_all_resonators()
        #Euclidean
        test1 = EuclideanLayout(4,3,lattice_type = 'Huse', modeType = 'FW')
        resonators = test1.resonators
        #tree
        Tree = TreeResonators(degree = 3, iterations = 3, side = 1, file_path = '', modeType = 'FW')
        resonators = Tree.get_all_resonators()
        
        #generate full layout with SD simulation
        testLattice = GeneralLayout(resonators , modeType = 'FW', name =  'NameMe')
        
        *saving computed layout*
        testLattice.save( name = 'filename.pkl') #filename can be a full path, but must have .pkl extension
        




Resonator Processing Functions
        *resonator array processing functions*
            * split_resonators
            * generate_line_graph
            * max_degree TBD
            * shift_resonators
            * rotate_resonators
            * get_coordsrrrg
        
    Samples syntax:
        *split each resonator in two*
        from GeneralLayoutGenerator import split_resonators
        splitGraph = split_resonators(resonators)
        


"""



import scipy
import pylab
import numpy

import pickle

import scipy.linalg




class GeneralLayout3D(object):
    '''
    GeneralLayout3D _summary_

    :param object: _description_
    :type object: _type_
    '''    
    def __init__(self, resonators = [0,0,0,0], 
                       side = 1, 
                       file_path = '', 
                       modeType = 'FW', 
                       name = 'TBD', 
                       vertexDict = True, 
                       resonatorsOnly = False, 
                       roundDepth = 3,
                       maxDegree = 4):
        '''
        
        '''
        
        if file_path != '':
            self.load(file_path)
        else:
            if numpy.all(numpy.asarray(resonators) == numpy.asarray([0,0,0,0])):
                raise ValueError('need input resonators')
            
            self.name  =  name

            if not ((modeType == 'FW') or (modeType  == 'HW')):
                raise ValueError('Invalid mode type. Must be FW or HW')

            self.modeType = modeType
            
            self.roundDepth = roundDepth
            
            self.resonators = resonators
            self.coords = self.get_coords(self.resonators)
            
            self.maxDegree = maxDegree #to properly allocate spacefor links, it needs
            #to know the max degree. Degault is set to 4, so usually it will allocate plenty of space
            #for things with higher coordination, it will need to be told to allocate more

            if not resonatorsOnly:
                self.populate()
                
                if vertexDict:
                    self.generate_vertex_dict()
            
            
    ###########
    #automated construction, saving, loading
    ##########
    def populate(self, Hamiltonian = True, save = False, save_name = ''):
        '''
        fully populate the structure up to itteration = MaxItter
        
        if Hamiltonian = False will not generate H
        save is obvious
        '''
         
        # #make the resonator lattice
        #NA
        
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
            waveStr = '_HW'
        else:
            waveStr = ''
            
        if name == '':
            name = self.name + waveStr + '.pkl'
        
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
            print ('Old pickle file. Pre FW-HW.')
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
    
    def draw_resonator_lattice(self, ax, color = 'g', alpha = 1 , linewidth = 0.5, extras = False, zorder = 1):
        if extras == True:
            resonators = self.extraResonators
        else:
            resonators = self.resonators
            
        for res in range(0,resonators.shape[0] ):
            [x0, y0, x1, y1]  = resonators[res,:]
            ax.plot([x0, x1],[y0, y1] , color = color, alpha = alpha, linewidth = linewidth, zorder = zorder)
        return
            
    def draw_resonator_end_points(self, ax, color = 'g', edgecolor = 'k',  marker = 'o' , size = 10, zorder = 1, alpha = 1):
        '''will double draw some points'''
        x0s = self.resonators[:,0]
        y0s = self.resonators[:,1]
        
        x1s = self.resonators[:,2]
        y1s = self.resonators[:,3]
        
        pylab.sca(ax)
        pylab.scatter(x0s, y0s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder, alpha = alpha)
        pylab.scatter(x1s, y1s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder, alpha = alpha)
        return   

    def get_all_resonators(self, maxItter = -1):
        '''
        function to get all resonators as a pair of end points
        
        each resontator returned as a row with four entries.
        (orientation is important to TB calculations)
        x0,y0,x1,y1
        
        '''
        return self.resonators
    
    def get_coords(self, resonators):
        '''
        take in a set of resonators and calculate the set of end points.
        
        Will round all coordinates the the specified number of decimals.
        
        Should remove all redundancies.
        '''
        
        coords_overcomplete = numpy.zeros((resonators.shape[0]*2, 1)).astype('complex')
        coords_overcomplete =  numpy.concatenate((resonators[:,0], resonators[:,2])) + 1j * numpy.concatenate((resonators[:,1], resonators[:,3]))
        
        coords_complex = numpy.unique(numpy.round(coords_overcomplete, self.roundDepth))
    
        coords = numpy.zeros((coords_complex.shape[0],2))
        coords[:,0] = numpy.real(coords_complex)
        coords[:,1] = numpy.imag(coords_complex)
        
        return coords
    
    
    ########
    #functions to generate tlattice properties
    #######
    def generate_semiduals(self):
        '''
        function to autogenerate the links between a set of resonators and itself
        
        
        will return a matrix of all the links [start, target, start_polarity, end_polarity]
        
        
        '''


        ress1 = self.resonators
        len1 = ress1.shape[0]
        
        ress2 = self.resonators

        #place to store the links
        # linkMat = numpy.zeros((len1*4+len1*4,4)) #i think this is fudged to work for things with degree 4 or less
        maxLineCoordination = (self.maxDegree-1)*2
        linkMat = numpy.zeros((len1*maxLineCoordination,4)) # this should adjust automatically
        
        #find the links
        
        #round the coordinates to prevent stupid mistakes in finding the connections
        plusEnds = numpy.round(ress2[:,0:2], self.roundDepth)
        minusEnds = numpy.round(ress2[:,2:4],self.roundDepth)
        
        extraLinkInd = 0
        for resInd in range(0,ress1.shape[0]):
            res = numpy.round(ress1[resInd,:], self.roundDepth)
            x1 = res[0]
            y1 = res[1]
            x0 = res[2]
            y0 = res[3]

            plusPlus = numpy.where((plusEnds == (x1, y1)).all(axis=1))[0]
            minusMinus = numpy.where((minusEnds == (x0, y0)).all(axis=1))[0]
            
            plusMinus = numpy.where((minusEnds == (x1, y1)).all(axis=1))[0] #plus end of new res, minus end of old
            minusPlus = numpy.where((plusEnds == (x0, y0)).all(axis=1))[0]
            
            for ind in plusPlus:
                if ind == resInd:
                    #self link
                    pass
                else:
                    linkMat[extraLinkInd,:] = [resInd, ind, 1,1]
                    extraLinkInd = extraLinkInd+1
                    
            for ind in minusMinus:
                if ind == resInd:
                    #self link
                    pass
                else:
                    linkMat[extraLinkInd,:] = [resInd, ind, 0,0]
                    extraLinkInd = extraLinkInd+1
                    
            for ind in plusMinus:
                if ind == resInd: #this is a self loop edge
                    linkMat[extraLinkInd,:] = [resInd, ind,  1,0]
                    extraLinkInd = extraLinkInd+1
                elif ind in plusPlus: #don't double count if you hit a self loop edge 
                    pass 
                elif ind in minusMinus:
                    pass 
                else:
                    linkMat[extraLinkInd,:] = [resInd, ind,  1,0]
                    extraLinkInd = extraLinkInd+1
                
            for ind in minusPlus:
                if ind == resInd:#this is a self loop edge
                    linkMat[extraLinkInd,:] = [ resInd, ind,  0,1]
                    extraLinkInd = extraLinkInd+1
                elif ind in plusPlus: #don't double count if you hit a self loop edge 
                    pass 
                elif ind in minusMinus:
                    pass 
                else:
                    linkMat[extraLinkInd,:] = [ resInd, ind,  0,1]
                    extraLinkInd = extraLinkInd+1
        
        #clean the skipped links away 
        linkMat = linkMat[~numpy.all(linkMat == 0, axis=1)]  
        self.SDHWlinks = linkMat

        xs = numpy.zeros(self.resonators.shape[0])
        ys = numpy.zeros(self.resonators.shape[0])
        for rind in range(0, self.resonators.shape[0]):
            res = self.resonators[rind,:]
            xs[rind] = (res[0] + res[2])/2
            ys[rind] = (res[1] + res[3])/2
        self.SDx = xs
        self.SDy = ys
        self.SDlinks = self.SDHWlinks[:,0:2]
        
        
        return linkMat
    
    def generate_vertex_dict(self):
        plusEnds = numpy.round(self.resonators[:,0:2],self.roundDepth)
        minusEnds = numpy.round(self.resonators[:,2:4],self.roundDepth)
        
        self.vertexDict = {}
        
        #loop over the vertices.
        for vind in range(0, self.coords.shape[0]):
#            vertex = self.coords[vind, :]
            vertex = numpy.round(self.coords[vind, :],self.roundDepth)
            
            startMatch = numpy.where((plusEnds == (vertex[0], vertex[1])).all(axis=1))[0]
            endMatch = numpy.where((minusEnds == (vertex[0], vertex[1])).all(axis=1))[0]
            
            matchList = []
            for rind in startMatch:
                matchList.append(int(rind))
            for rind in endMatch:
                matchList.append(int(rind))
             
            #store the results
            self.vertexDict[vind] = numpy.asarray(matchList)
        
        return self.vertexDict

        
    
    def draw_SD_points(self, ax, color = 'g', edgecolor = 'k',  marker = 'o' , size = 10,  extra = False, zorder = 1, alpha = 1):
        '''
        draw the locations of all the semidual sites
        
        if extra is True it will draw only the edge sites required to fix the edge of the tiling
        '''
        if extra == True:
            xs = self.extraSDx
            ys = self.extraSDy
        else:
            xs = self.SDx
            ys = self.SDy
        
        pylab.sca(ax)
        pylab.scatter(xs, ys ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder, alpha = alpha)
        
        return

    def draw_SDlinks(self, ax, color = 'firebrick', linewidth = 0.5, extra = False, minus_links = False, minus_color = 'goldenrod', NaNs = True, alpha = 1, zorder = 1):
        '''
        draw all the links of the semidual lattice
        
        if extra is True it will draw only the edge sites required to fix the edge of the tiling
        
        set minus_links to true if you want the links color coded by sign
        minus_color sets the sign of the negative links
        '''
        if extra == True:
            xs = self.SDx
            ys = self.SDy
            links = self.extraSDHWlinks[:]
        else:
            xs = self.SDx
            ys = self.SDy
            links = self.SDHWlinks[:]
        
        if NaNs:
            if minus_links == True and self.modeType == 'HW':
                plotVecx_plus = numpy.asarray([])
                plotVecy_plus = numpy.asarray([])
                
                plotVecx_minus = numpy.asarray([])
                plotVecy_minus = numpy.asarray([])
                
                for link in range(0, links.shape[0]):
                    [startInd, endInd]  = links[link,0:2]
                    startInd = int(startInd)
                    endInd = int(endInd)
                    
                    ends = links[link,2:4]
                    
                    if ends[0]==ends[1]:
                        plotVecx_plus = numpy.concatenate((plotVecx_plus, [xs[startInd]], [xs[endInd]], [numpy.NaN]))
                        plotVecy_plus = numpy.concatenate((plotVecy_plus, [ys[startInd]], [ys[endInd]], [numpy.NaN]))
                    else:
                        plotVecx_minus = numpy.concatenate((plotVecx_minus, [xs[startInd]], [xs[endInd]], [numpy.NaN]))
                        plotVecy_minus = numpy.concatenate((plotVecy_minus, [ys[startInd]], [ys[endInd]], [numpy.NaN]))
                
                ax.plot(plotVecx_plus,plotVecy_plus, color = color, linewidth = linewidth, alpha = alpha, zorder = zorder)
                ax.plot(plotVecx_minus,plotVecy_minus , color = minus_color, linewidth = linewidth, alpha = alpha, zorder = zorder)
            else:
                plotVecx = numpy.zeros(links.shape[0]*3)
                plotVecy = numpy.zeros(links.shape[0]*3)
                
                for link in range(0, links.shape[0]):
                    [startInd, endInd]  = links[link,0:2]
                    startInd = int(startInd)
                    endInd = int(endInd)
                    
                    plotVecx[link*3:link*3 + 3] = [xs[startInd], xs[endInd], numpy.NaN]
                    plotVecy[link*3:link*3 + 3] = [ys[startInd], ys[endInd], numpy.NaN]
                
                ax.plot(plotVecx,plotVecy , color = color, linewidth = linewidth, alpha = alpha, zorder = zorder)
            
        else:
            for link in range(0, links.shape[0]):
                [startInd, endInd]  = links[link,0:2]
                startInd = int(startInd)
                endInd = int(endInd)
                
                [x0,y0] = [xs[startInd], ys[startInd]]
                [x1,y1] = [xs[endInd], ys[endInd]]
                
                if  minus_links == True and self.modeType == 'HW':
                    ends = links[link,2:4]
                    if ends[0]==ends[1]:
                        #++ or --, use normal t
                        ax.plot([x0, x1],[y0, y1] , color = color, linewidth = linewidth, alpha = alpha, zorder = zorder)
                    else:
                        #+- or -+, use inverted t
                        ax.plot([x0, x1],[y0, y1] , color = minus_color, linewidth = linewidth, alpha = alpha, zorder = zorder)
                else :
                    ax.plot([x0, x1],[y0, y1] , color = color, linewidth = linewidth, alpha = alpha, zorder = zorder)
                
        return

    def get_semidual_points(self):
        '''
        get all the semidual points in a given itteration.
        
        Mostly vestigial for compatibility
        '''
        return[self.SDx, self.SDy]

    
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
            raise ValueError ('lattice doesnt have this many sites')
            
        state = numpy.zeros(len(self.SDx))*(0+0j)
        
        state[site] = 1.
        
        return state
    
    def V_int(self, ind1, ind2, states):
        '''
        Calculate total interaction enegery of two particles at lattice sites
        indexed by index 1 and index 2
        
        states is the set of eigenvectors that you want to include e.g. [0,1,2,3]
        '''
        psis_1 = self.Psis[ind1,states]
        psis_2 = self.Psis[ind2,states]
        
        return numpy.dot(numpy.conj(psis_2), psis_1)

    def V_int_map(self, source_ind, states = []):
        '''
        calculate a map of the interaction energy for a given location of the first
        qubit.
        Lattice sites specified by index in semidual points array
        
        must also specify which igenstates to include. default = all
        '''
        if states == []:
            states = scipy.arange(0, len(self.Es),1)
        
        int_vals = numpy.zeros(len(self.SDx))
        for ind2 in range(0, len(self.SDx)):
            int_vals[ind2] = self.V_int(source_ind, ind2,states)
        
        return int_vals
    
    def plot_layout_state(self, state_vect, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', zorder = 1):
        '''
        plot a state (wavefunction) on the graph of semidual points
        '''
        Amps = state_vect
        Probs = numpy.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = numpy.angle(Amps)/numpy.pi
        
        cm = pylab.cm.get_cmap(cmap)
        
        pylab.sca(ax)
        pylab.scatter(self.SDx, self.SDy,c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            cbar = pylab.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax, linewidth = 0.5, color = 'firebrick')
        
        pylab.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return

    def plot_map_state(self, map_vect, ax, title = 'ineraction weight', colorbar = False, plot_links = False, cmap = 'winter', autoscale = False, scaleFactor = 0.5, zorder = 1):
        '''plot an interaction map on the graph
        '''
        Amps = map_vect
        
        mSizes = 100
        mColors = Amps
        
    #    cm = pylab.cm.get_cmap('seismic')
        cm = pylab.cm.get_cmap(cmap)
    #    cm = pylab.cm.get_cmap('RdBu')
        
        
        vals = numpy.sort(mColors)
        peak = vals[-1]
        second_biggest = vals[-2]
        
        if autoscale:
            vmax = peak
            if self.modeType == 'HW':
                vmin = -vmax
            else:
                vmin = vals[0]
        else:
            vmax = second_biggest
            vmin = -second_biggest
        
        if self.modeType == 'FW':
            pylab.sca(ax)
            pylab.scatter(self.SDx, self.SDy,c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmax = vmax, vmin = vmin, zorder = zorder)
        elif self.modeType == 'HW':
            #build full state with value on both ends of the resonators
            mColors_end = numpy.zeros(len(Amps)*2)
            mColors_end[0::2] = mColors
            mColors_end[1::2] = -mColors
            
            #get coordinates for the two ends of the resonator
            plotPoints = self.get_end_state_plot_points(scaleFactor = scaleFactor)
            xs = plotPoints[:,0]
            ys = plotPoints[:,1]
            
#            mColors_end = scipy.arange(1.,len(Amps)*2+1,1)/300
#            print mColors_end.shape
#            print Amps.shape
            
            #plot
            pylab.sca(ax)
            pylab.scatter(xs, ys,c =  mColors_end, s = mSizes/1.4, marker = 'o', edgecolors = 'k', cmap = cm, vmax = vmax, vmin = vmin, zorder = zorder)
        else:
            raise ValueError ('You screwed around with the mode type. It must be FW or HW.')
            
        
        if colorbar:
            cbar = pylab.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('interaction energy (AU)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax, linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        pylab.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return
    
    def get_end_state_plot_points(self,scaleFactor = 0.5):
        '''
        find end coordinate locations part way along each resonator so that
        they can be used to plot the field at both ends of the resonator.
        (Will retun all values up to specified itteration. Default is the whole thing)
        
        Scale factor says how far appart the two points will be: +- sclaeFactor.2 of the total length
        
        returns the polt points as collumn matrix
        '''
        if scaleFactor> 1:
            raise ValueError ('scale factor too big')
            
            
        size = len(self.SDx)
        plot_points = numpy.zeros((size*2, 2))
        
        resonators = self.get_all_resonators()
        for ind in range(0, size):
            [x0, y0, x1, y1]  = resonators[ind, :]
            xmean = (x0+x1)/2
            ymean = (y0+y1)/2
            
            xdiff = x1-x0
            ydiff = y1-y0
            
            px0 = xmean - xdiff*scaleFactor/2
            py0 = ymean - ydiff*scaleFactor/2
            
            px1 = xmean + xdiff*scaleFactor/2
            py1 = ymean + ydiff*scaleFactor/2
            
            
            plot_points[2*ind,:] = [px0,py0]
            plot_points[2*ind+1,:] = [px1,py1]
            ind = ind+1
            
        return plot_points
    
    def plot_end_layout_state(self, state_vect, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', scaleFactor = 0.5, zorder = 1):
        '''
        plot a state (wavefunction) on the graph of semidual points, but with a 
        value plotted for each end of the resonator
        
        If you just want a single value for the resonator use plot_layout_state
        
        Takes states defined on only one end of each resonator. Will autogenerate 
        the value on other end based on mode type.
        
        '''
        Amps = state_vect
        Probs = numpy.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = numpy.angle(Amps)/numpy.pi
        
        #build full state with value on both ends of the resonators
        mSizes_end = numpy.zeros(len(Amps)*2)
        mSizes_end[0::2] = mSizes
        mSizes_end[1::2] = mSizes
        
        mColors_end = numpy.zeros(len(Amps)*2)
        mColors_end[0::2] = mColors
        if self.modeType == 'FW':
            mColors_end[1::2] = mColors
        elif self.modeType == 'HW':
            #put opposite phase on other side
            oppositeCols = mColors + 1
            #rectify the phases back to between -0.5 and 1.5 pi radians
            overflow = numpy.where(oppositeCols > 1.5)[0]
            newCols = oppositeCols
            newCols[overflow] = oppositeCols[overflow] - 2
            
            mColors_end[1::2] = newCols
        else:
            raise ValueError ('You screwed around with the mode type. It must be FW or HW.')
        
        cm = pylab.cm.get_cmap(cmap)
        
        #get coordinates for the two ends of the resonator
        plotPoints = self.get_end_state_plot_points(scaleFactor = scaleFactor)
        xs = plotPoints[:,0]
        ys = plotPoints[:,1]
        
        pylab.sca(ax)
        pylab.scatter(xs, ys,c =  mColors_end, s = mSizes_end, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            cbar = pylab.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax,linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        pylab.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        
#        return plotPoints
        return mColors
    
    def generate_root_Hamiltonian(self, roundDepth = 3, t = 1, verbose = False, sparse = False, flags = 5):
        '''
        custom function so I can get vertex dict without having to run the full populate of general layout
        and thereby having to also diagonalize the effective Hamiltonian.
        
        Will process the resonator matrix to get the layout Hamiltonian.
        
        Will return a regular matrix of sparse  = false, and a sparse matrix data type if sparse  = true
        
        Does not need to SD Hamiltonian made first.
        
        
        '''
        resonators = self.get_all_resonators()
        resonators = numpy.round(resonators, roundDepth)
        
        numVerts = self.coords.shape[0]
        if sparse:
            rowVec = numpy.zeros(numVerts*4+flags)
            colVec = numpy.zeros(numVerts*4+flags)
            Hvec = numpy.zeros(numVerts*4+flags)
        else:
            Hmat = numpy.zeros((numVerts, numVerts))
        
        coords_complex = numpy.round(self.coords[:,0] + 1j*self.coords[:,1], roundDepth)
        
        currInd = 0
        for rind in range(0, resonators.shape[0]):
            resPos = resonators[rind,:]
            startPos = numpy.round(resPos[0],roundDepth)+ 1j*numpy.round(resPos[1],roundDepth)
            stopPos = numpy.round(resPos[2],roundDepth)+ 1j*numpy.round(resPos[3],roundDepth)
            
            startInd = numpy.where(startPos == coords_complex)[0][0]
            stopInd = numpy.where(stopPos == coords_complex)[0][0]
    
            if sparse:
                rowVec[currInd] = startInd
                colVec[currInd] = stopInd
                Hvec[currInd] = t #will end up adding t towhatever this entry was before.
                currInd = currInd +1
                
                rowVec[currInd] = stopInd
                colVec[currInd] = startInd
                Hvec[currInd] = t #will end up adding t towhatever this entry was before.
                currInd = currInd +1
                
            else:
                Hmat[startInd, stopInd] = Hmat[startInd, stopInd] + t
                Hmat[stopInd, startInd] = Hmat[stopInd, startInd] + t
        
        #finish making the sparse matrix if we are in sparse matrix mode.
        if sparse:
            #pad the end of the matrix with values so that I can see if one of those is the missing one
            for ind in range(0, flags):
                rowVec[currInd] = numVerts+ind
                colVec[currInd] = numVerts+ind
                Hvec[currInd] = -7.5 #will end up adding t towhatever this entry was before.
                currInd = currInd +1
    
            Hmat = coo_matrix((Hvec,(rowVec,colVec)), shape = (numVerts+flags,numVerts+flags), dtype = 'd')
            Hmat.eliminate_zeros() #removed the unused spots since this is not a regular graph
            
        if verbose:
            temp = numpy.sum(Hmat)/numVerts
            print( 'average degree = ' + str(temp))
        
        self.rootHamiltonian = Hmat
        
        if not sparse:
            self.rootEs, self.rootPsis = numpy.linalg.eigh(self.rootHamiltonian)
            
        return
    
    def plot_root_state(self, state_vect, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', zorder = 1):
        '''
        plot a state (wavefunction) on the root graph of original vertices
        '''
        Amps = state_vect
        Probs = numpy.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = numpy.angle(Amps)/numpy.pi
        
        cm = pylab.cm.get_cmap(cmap)
        
        pylab.sca(ax)
        pylab.scatter(self.coords[:,0], self.coords[:,1],c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            cbar = pylab.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax, linewidth = 0.5, color = 'firebrick')
        
        pylab.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return
  

if __name__=="__main__":      
    
    
    print('not doing anything yet')
    
#     #tree
#     Tree = TreeResonators(degree = 3, iterations = 4, side = 1, file_path = '', modeType = 'FW')
#     resonators = Tree.get_all_resonators()
# #    Tree2 = TreeResonators(file_path = '3regularTree_ 3_.pkl')
#     testLattice = GeneralLayout(resonators , modeType = Tree.modeType, name =  'TREEEEE')

#     # ######split tree
#     # Tree = TreeResonators(degree = 3, iterations = 4, side = 1, file_path = '', modeType = 'FW')
#     # resonators = Tree.get_all_resonators()
#     # splitGraph = split_resonators(resonators)
#     # resonators = splitGraph
#     # testLattice = GeneralLayout(resonators , modeType = Tree.modeType, name =  'McLaughlinTree')

# #    ######non-trivial tree
# #    Tree = TreeResonators(cell ='Peter', degree = 3, iterations = 3, side = 1, file_path = '', modeType = 'FW')
# #    resonators = Tree.get_all_resonators()
# #    testLattice = GeneralLayout(resonators , modeType = Tree.modeType, name =  'PeterTREEEEE')
#     ##testLattice = GeneralLayout(Tree.cellResonators , modeType = Tree.modeType, name =  'NameMe')
#     ##testLattice = GeneralLayout(rotate_resonators(Tree.cellResonators,numpy.pi/3) , modeType = Tree.modeType, name =  'NameMe')

    
# #    #generate full layout with SD simulation
# #    testLattice = GeneralLayout(resonators , modeType = Tree.modeType, name =  'NameMe')
    
#     showLattice = True
#     showHamiltonian = True
    
    
#     if showLattice:
    
# #        fig1 = pylab.figure(1)
# #        pylab.clf()
# #        ax = pylab.subplot(1,1,1)
# #        Tree.draw_resonator_lattice(ax, color = 'mediumblue', alpha = 1 , linewidth = 2.5)
# #        xs = Tree.coords[:,0]
# #        ys = Tree.coords[:,1]
# #        pylab.sca(ax)
# #        #pylab.scatter(xs, ys ,c =  'goldenrod', s = 20, marker = 'o', edgecolors = 'k', zorder = 5)
# #        pylab.scatter(xs, ys ,c =  'goldenrod', s = 30, marker = 'o', edgecolors = 'k', zorder = 5)
# #        #pylab.scatter(xs, ys ,c =  'goldenrod', s = 40, marker = 'o', edgecolors = 'k', zorder = 5)
# #        ax.set_aspect('equal')
# #        ax.axis('off')
# #        pylab.tight_layout()
# #        pylab.show()
# #        fig1.set_size_inches(5, 5)


# #        fig1 = pylab.figure(1)
# #        pylab.clf()
# #        ax = pylab.subplot(1,1,1)
# #        testLattice.draw_resonator_lattice(ax, color = 'mediumblue', alpha = 1 , linewidth = 1.5)
# #        testLattice.draw_SDlinks(ax, color = 'deepskyblue', linewidth = 2.5, minus_links = False, minus_color = 'goldenrod')
# #        pylab.scatter(testLattice.SDx, testLattice.SDy,c =  'goldenrod', marker = 'o', edgecolors = 'k', s = 5,  zorder=5)
# #        
# #        ax.set_aspect('equal')
# #        ax.axis('off')
# #        pylab.tight_layout()
# #        pylab.show()
# #        fig1.set_size_inches(5, 5)
        
#         fig1 = pylab.figure(1)
#         pylab.clf()
#         ax = pylab.subplot(1,1,1)
#         testLattice.draw_SDlinks(ax, color = 'deepskyblue', linewidth = 1.5, minus_links = True, minus_color = 'goldenrod')
#         testLattice.draw_resonator_lattice(ax, color = 'mediumblue', alpha = 1 , linewidth = 2.5)
#         xs = testLattice.coords[:,0]
#         ys = testLattice.coords[:,1]
#         pylab.sca(ax)
#         #pylab.scatter(xs, ys ,c =  'goldenrod', s = 20, marker = 'o', edgecolors = 'k', zorder = 5)
#         pylab.scatter(xs, ys ,c =  'goldenrod', s = 30, marker = 'o', edgecolors = 'k', zorder = 5)
#         #pylab.scatter(xs, ys ,c =  'goldenrod', s = 40, marker = 'o', edgecolors = 'k', zorder = 5)
#         ax.set_aspect('equal')
#         ax.axis('off')
#         pylab.tight_layout()
#         pylab.show()
#         pylab.title('generalized layout and effective model')
#         fig1.set_size_inches(5, 5)
#     else:
#         pylab.figure(1)
#         pylab.clf()
        
        
#     if showHamiltonian:
#         eigNum = 168
#         eigNum = 167
#         eigNum = 0
        
#         pylab.figure(2)
#         pylab.clf()
#         ax = pylab.subplot(1,2,1)
#         pylab.imshow(testLattice.H,cmap = 'winter')
#         pylab.title('Hamiltonian')
#         ax = pylab.subplot(1,2,2)
#         pylab.imshow(testLattice.H - numpy.transpose(testLattice.H),cmap = 'winter')
#         pylab.title('H - Htranspose')
#         pylab.show()
        

        
#         xs = scipy.arange(0,len(testLattice.Es),1)
#         eigAmps = testLattice.Psis[:,testLattice.Eorder[eigNum]]
        
#         pylab.figure(3)
#         pylab.clf()
#         ax1 = pylab.subplot(1,2,1)
#         pylab.plot(testLattice.Es, 'b.')
#         pylab.plot(xs[eigNum],testLattice.Es[testLattice.Eorder[eigNum]], color = 'firebrick' , marker = '.', markersize = '10' )
#         pylab.title('eigen spectrum')
#         pylab.ylabel('Energy (t)')
#         pylab.xlabel('eigenvalue number')
        
#         ax2 = pylab.subplot(1,2,2)
#         titleStr = 'eigenvector weight : ' + str(eigNum)
#         testLattice.plot_layout_state(eigAmps, ax2, title = titleStr, colorbar = True, plot_links = True, cmap = 'Wistia')
        
#         pylab.show()
#     else:
#         pylab.figure(2)
#         pylab.clf()
        
#         pylab.figure(3)
#         pylab.clf()
        
    
    

    
    
    
    
    











