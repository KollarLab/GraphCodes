import numpy as np
import matplotlib.pyplot as plt
import scipy 

'''
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
'''  

class UnitCell3D(object):
    def __init__(self, lattice_type, 
                       side = 1, 
                       resonators = '', 
                       a1 = np.asarray([1,0,0]), 
                       a2 = np.asarray([0,1,0]),
                       a3 = np.asarray([0,0,1]),
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
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        
        #set up the lattice vectors
        self.a1 = self.side*np.asarray([1,0,0])
        self.a2 = self.side*np.asarray([0,1,0])
        self.a3 = self.side*np.asarray([0,0,1])
        dx = self.a1[0]/2

        
        #set up the positions of the sites of the effective lattice. ! look to newer functions for auto way to do these
        # xs = np.asarray([dx, -dx, 0, 0, 0, 0])
        # ys = np.asarray([0, 0, dx, -dx, 0, 0])
        # zs = np.asarray([0, 0, 0 ,0, dx, -dx])
        xs = np.asarray([dx, 0, 0])
        ys = np.asarray([0, dx, 0])
        zs = np.asarray([0, 0, dx])
        self.SDx = xs
        self.SDy = ys
        self.SDz = zs
        
        self.SDcoords = np.zeros((len(self.SDx),3))
        self.SDcoords[:,0] = self.SDx
        self.SDcoords[:,1] = self.SDy
        self.SDcoords[:,2] = self.SDz
        
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = np.zeros((self.numSites,6)) #pairs of resonator end points for each resonator
        self.coords = np.zeros((self.numSites,2)) #set of all resonator start points
        
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
                    closure[(indx,indy,indz)] =np.asarray([])
        self.closure = closure
        
        
        return
       
    # def _generate_kagome_cell(self, side = 1):
    #     '''
    #     generate kagome-type unit cell
    #     '''
    #     self.maxDegree = 3
        
    #     #set up the sites
    #     self.numSites = 3
    #     xs = np.zeros(self.numSites)
    #     ys = np.zeros(self.numSites)
        
    #     #set up the lattice vectors
    #     self.a1 = np.asarray([self.side*np.sqrt(3)/2, self.side/2])
    #     self.a2 = np.asarray([0, self.side])
    #     dy = self.a1[1]/2
    #     dx = self.a1[0]/2
    #     xcorr = self.side/np.sqrt(3)/2/2
        
    #     #set up the positions of the sites of the effective lattice. ! look to newer functions for auto way to do these
    #     xs = np.asarray([-dx, -dx, 0])
    #     ys = np.asarray([dy, -dy, -2*dy])
    #     self.SDx = xs
    #     self.SDy = ys
        
    #     #set up the poisitions of all the resonators  and their end points
    #     self.resonators = np.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
    #     self.coords = np.zeros((self.numSites,2)) #set of all resonator start points
        
    #     a = self.side/np.sqrt(3)
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
    #     closure[(1,0)] =np.asarray([1])
    #     #-a1 direction (-x)
    #     closure[(-1,0)] =np.asarray([])
        
    #     #a2 direction (y)
    #     closure[(0,1)] =np.asarray([2])
    #     #-a2 direction (y)
    #     closure[(0,-1)] =np.asarray([])
        
    #      #a1,a2 direction (x,y)
    #     closure[(1,1)] =np.asarray([])
    #     #-a1,a2 direction (-x,y)
    #     closure[(-1,1)] =np.asarray([])
    #     #a1,-a2 direction (x,-y)
    #     closure[(1,-1)] =np.asarray([0])
    #     #-a1,-a2 direction (-x,-y)
    #     closure[(-1,-1)] =np.asarray([])
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
#         xs = np.zeros(self.numSites)
#         ys = np.zeros(self.numSites)
        
#         #set up the lattice vectors
#         self.a1 = np.asarray([self.side, 0])
#         self.a2 = np.asarray([0, self.side])
#         dy = self.a1[1]/2
#         dx = self.a1[0]/2
#         xcorr = self.side/np.sqrt(3)/2/2
        
        
#         #set up the poisitions of all the resonators  and their end points
#         self.resonators = np.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
#         self.coords = np.zeros((self.numSites,2)) #set of all resonator start points
        
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
#         xs = np.zeros(self.numSites)
#         ys = np.zeros(self.numSites)
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
#         closure[(1,0)] =np.asarray([])
#         #-a1 direction (-x)
#         closure[(-1,0)] =np.asarray([])
        
#         #a2 direction (y)
#         closure[(0,1)] =np.asarray([])
#         #-a2 direction (y)
#         closure[(0,-1)] =np.asarray([])
        
#          #a1,a2 direction (x,y)
#         closure[(1,1)] =np.asarray([])
#         #-a1,a2 direction (-x,y)
#         closure[(-1,1)] =np.asarray([])
#         #a1,-a2 direction (x,-y)
#         closure[(1,-1)] =np.asarray([])
#         #-a1,-a2 direction (-x,-y)
#         closure[(-1,-1)] =np.asarray([])
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
        self.a1 = np.asarray(a1)
        self.a2 = np.asarray(a2)   
        self.a3 = np.asarray(a3) 
        
        if self.a1.shape != (3,):
            raise ValueError('first lattice vector has invalid shape')
            
        if self.a2.shape != (3,):
            raise ValueError('second lattice vector has invalid shape')
            
        if self.a3.shape != (3,):
            raise ValueError('third lattice vector has invalid shape')
        
        #set up the sites
        self.numSites = resonators.shape[0]
        
        
        
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = np.zeros((self.numSites,6)) #pairs of resonator end points for each resonator
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
        
        self.SDcoords = np.zeros((len(self.SDx),3))
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
                    closure[(indx,indy,indz)] =np.asarray([])
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
            
            
            redundancies = np.where((self.SDcoords  == (x0, y0,z0)).all(axis=1))[0]
            
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
        coords_overcomplete = np.concatenate((temp1, temp2))
        
        coords = np.unique(np.round(coords_overcomplete, roundDepth), axis = 0)
        
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
    
    def draw_resonators(self, ax, theta = np.pi/10, phi = np.pi/10, 
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
    
    def draw_resonator_end_points(self, ax, theta = np.pi/10, phi = np.pi/10,
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
        plt.sca(ax)
        plt.scatter(x0s, z0s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        plt.scatter(x1s, z1s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        return
      
    def draw_sites(self, ax, theta = np.pi/10, phi = np.pi/10,
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
        plt.sca(ax)
        plt.scatter(xs, zs ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        ax.set_aspect('equal')
        return
    
    def draw_SDlinks(self, ax, theta = np.pi/10, phi = np.pi/10,
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
        aMat = np.zeros((3,3))
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
            [x1,y1,z1] = np.asarray([sdxs[endSite], sdys[endSite], sdzs[endSite]]) + deltaA1*pa1 + deltaA2*pa2 + deltaA3*pa3
            
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
    #     plot_points = np.zeros((size*2, 2))
        
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
#         Amps = np.ones(len(self.SDx))
#         Probs = np.abs(Amps)**2
#         mSizes = Probs * len(Probs)*30
#         mColors = Amps
       
#         mSizes = 60
        
#         #build full state with value on both ends of the resonators 
#         mColors_end = np.zeros(len(Amps)*2)
#         mColors_end[0::2] = mColors

#         #put opposite sign on other side
#         mColors_end[1::2] = -mColors
# #        mColors_end[1::2] = 5
        
#         cm = plt.cm.get_cmap(cmap)
        
#         #get coordinates for the two ends of the resonator
#         plotPoints = self._get_orientation_plot_points(scaleFactor = scaleFactor)
#         xs = plotPoints[:,0]
#         ys = plotPoints[:,1]
        
#         plt.sca(ax)
# #        plt.scatter(xs, ys,c =  mColors_end, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -1, vmax = 1, zorder = zorder)
#         plt.scatter(xs, ys,c =  mColors_end, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -1.5, vmax = 2.0, zorder = zorder)
#         if colorbar:
#             cbar = plt.colorbar(fraction=0.046, pad=0.04)
#             cbar.set_label('phase (pi radians)', rotation=270)
              
#         if plot_links:
#             self.draw_SDlinks(ax,linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
#         plt.title(title, fontsize=8)
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
        # xmask = np.zeros((self.numSites,6))
        # ymask = np.zeros((self.numSites,6))
        # zmask = np.zeros((self.numSites,6))
        
        # xmask[:,0] = 1
        # xmask[:,3] = 1
        
        # ymask[:,1] = 1
        # ymask[:,4] = 1
        
        # zmask[:,2] = 1
        # zmask[:,5] = 1
        
        # if self.type[0:2] == '74':
        #     self.SDHWlinks = np.zeros((self.numSites*4+4,6))
        # elif self.type == 'square':
        #     self.SDHWlinks = np.zeros((self.numSites*6,6))
        # elif self.type[0:4] == 'kite':
        #     self.SDHWlinks = np.zeros((self.numSites*10,6))
        # else:
        #     # self.SDHWlinks = np.zeros((self.numSites*4,6))
        #     # self.SDHWlinks = np.zeros((self.numSites*8,6)) #temporary hack to allow some line graph games
            
        # #now fixing to use the max degree given
        maxLineCoordination = (self.maxDegree-1)*2
        self.SDHWlinks = np.zeros((self.numSites*maxLineCoordination, 7))
        
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
        self.SDHWlinks = self.SDHWlinks[~np.all(self.SDHWlinks == 0, axis=1)] 
        
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
        xmask = np.zeros((self.numSites,6))
        ymask = np.zeros((self.numSites,6))
        zmask = np.zeros((self.numSites,6))
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
        # linkMat = np.zeros((len1*4+len1*4,6))  #I'm pretty sure this only works for low degree
        maxLineCoordination = (self.maxDegree-1)*2
        linkMat = np.zeros((len1*maxLineCoordination+len1*maxLineCoordination,7))
        
        #find the links
        
        #round the coordinates to prevent stupid mistakes in finding the connections
        roundDepth = 3
        plusEnds = np.round(ress2[:,0:3],roundDepth)
        minusEnds = np.round(ress2[:,3:],roundDepth)
        
        extraLinkInd = 0
        for resInd in range(0,ress1.shape[0]):
            res = np.round(ress1[resInd,:],roundDepth)
            x1 = res[0]
            y1 = res[1]
            z1 = res[2]
            
            x0 = res[3]
            y0 = res[4]
            z0 = res[5]

            plusPlus = np.where((plusEnds == (x1, y1,z1)).all(axis=1))[0]
            minusMinus = np.where((minusEnds == (x0, y0,z0)).all(axis=1))[0]
            
            plusMinus = np.where((minusEnds == (x1, y1,z1)).all(axis=1))[0] #plus end of new res, minus end of old
            minusPlus = np.where((plusEnds == (x0, y0,z0)).all(axis=1))[0]
            
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
        linkMat = linkMat[~np.all(linkMat == 0, axis=1)]  
        
        return linkMat
    
    ######
    #Bloch theory calculation functions
    ######
    def generate_Bloch_matrix(self, kx, ky,kz, modeType = 'FW', t = 1, phase = 0):
        BlochMat = np.zeros((self.numSites, self.numSites))*(0 + 0j)
        
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
                            phaseFactor = np.exp(1j *phase) #e^i phi in one corner
                        elif startInd < targetInd:
                            phaseFactor = np.exp(-1j *phase) #e^-i phi in one corner, so it's Hermitian
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
            
            phaseFactor = np.exp(1j*kx*deltaX)*np.exp(1j*ky*deltaY)*np.exp(1j*kz*deltaZ)
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
        
        kxs = np.linspace(kx_0, kx_1,numsteps)
        kys = np.linspace(ky_0, ky_1,numsteps)
        kzs = np.linspace(kz_0, kz_1,numsteps)
        
        bandCut = np.zeros((self.numSites, numsteps))
        
        stateCut = np.zeros((self.numSites, self.numSites, numsteps)).astype('complex')
        
        for ind in range(0, numsteps):
            kvec = [kxs[ind],kys[ind], kzs[ind]]
            
            H = self.generate_Bloch_matrix(kvec[0], kvec[1],kvec[2], modeType = modeType, phase  = phase)
        
            #Psis = np.zeros((self.numSites, self.numSites)).astype('complex')
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
        
        plt.sca(ax)
        
        for ind in range(0,bandCut.shape[0]):
            colorInd = np.mod(ind, len(colorlist))
            if dots:
                plt.plot(bandCut[ind,:], color = colorlist[colorInd] , marker = '.', markersize = '5', linestyle = '', zorder = zorder)
            else:
                plt.plot(bandCut[ind,:], color = colorlist[colorInd] , linewidth = linewidth, linestyle = '-', zorder = zorder)
#            plt.plot(bandCut[ind,:], '.')
        plt.title('some momentum cut')
        plt.ylabel('Energy')
        plt.xlabel('k_something')
    
    def plot_bloch_wave(self, state_vect, ax,
                        theta = np.pi/10, phi = np.pi/10,
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
        Probs = np.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = np.angle(Amps)/np.pi
        
        #move the branch cut to -0.5
        outOfRange = np.where(mColors< -0.5)[0]
        mColors[outOfRange] = mColors[outOfRange] + 2
        
        
        cm = plt.cm.get_cmap(cmap)
        
        plotMat = self.rotate_coordinates(self.SDcoords, theta, phi)
        sdxs = plotMat[:,0]
        sdys = plotMat[:,1]
        sdzs = plotMat[:,2]
        
        #plot the x-z projection
        plt.sca(ax)
        plt.scatter(sdxs, sdzs,c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            print('making colorbar')
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax, theta = theta, phi = phi, 
                              linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        plt.title(title, fontsize=8)
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
#         Probs = np.abs(Amps)**2
#         mSizes = Probs * len(Probs)*30
#         mColors = np.angle(Amps)/np.pi
        
#         #build full state with value on both ends of the resonators
#         mSizes_end = np.zeros(len(Amps)*2)
#         mSizes_end[0::2] = mSizes
#         mSizes_end[1::2] = mSizes
        
#         mColors_end = np.zeros(len(Amps)*2)
#         mColors_end[0::2] = mColors
#         if modeType == 'FW':
#             mColors_end[1::2] = mColors
#         elif modeType == 'HW':
#             #put opposite phase on other side
#             oppositeCols = mColors + 1
#             #rectify the phases back to between -0.5 and 1.5 pi radians
#             overflow = np.where(oppositeCols > 1.5)[0]
#             newCols = oppositeCols
#             newCols[overflow] = oppositeCols[overflow] - 2
            
#             mColors_end[1::2] = newCols
#         else:
#             raise ValueError('You screwed around with the mode type. It must be FW or HW.')
        
#         cm = plt.cm.get_cmap(cmap)
        
#         #get coordinates for the two ends of the resonator
#         plotPoints = self._get_orientation_plot_points(scaleFactor = scaleFactor)
#         xs = plotPoints[:,0]
#         ys = plotPoints[:,1]
        
#         plt.sca(ax)
#         plt.scatter(xs, ys,c =  mColors_end, s = mSizes_end, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
#         if colorbar:
#             print('making colorbar')
#             cbar = plt.colorbar(fraction=0.046, pad=0.04)
#             cbar.set_label('phase (pi radians)', rotation=270)
              
#         if plot_links:
#             self.draw_SDlinks(ax,linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
#         plt.title(title, fontsize=8)
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
        
        newResonators = np.zeros((newNum,4))
        
        for rind in range(0, oldNum):
            oldRes = resMat[rind,:]
            xstart = oldRes[0]
            ystart = oldRes[1]
            zstart = oldRes[2]
            xend = oldRes[3]
            yend = oldRes[4]
            zend = oldRes[5]
            
            xs = np.linspace(xstart, xend, splitIn+1)
            ys = np.linspace(ystart, yend, splitIn+1)
            zs = np.linspace(zstart, zend, splitIn+1)
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
        newResonators = np.zeros((self.SDHWlinks.shape[0], 6))
        
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
                    
                    res = np.asarray([x0, y0,z0, x1, y1, z1])
                    newResonators[lind, :] = res
                    
        
        #clean out balnk rows that were for redundant resonators
        newResonators = newResonators[~np.all(newResonators == 0, axis=1)]  

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
        allCoords = np.round(self.coords[:,:], roundDepth)
        # svec_all = allCoords[:,0] + 1j*allCoords[:,1]
        svec_all = allCoords
        
        
        
        def check_redundnacy(site, svec_all, shift1, shift2, shift3):
            # vec1 = np.round(self.a1[0] + 1j*self.a1[1], roundDepth)
            # vec2 = np.round(self.a2[0] + 1j*self.a2[1], roundDepth)
            
            shiftVec = shift1*self.a1 + shift2*self.a2 + shift3*self.a3
            
            shiftedCoords = np.zeros(svec_all.shape)
            shiftedCoords[:,0] = svec_all[:,0] + shiftVec[0]
            shiftedCoords[:,1] = svec_all[:,1] + shiftVec[1]
            shiftedCoords[:,2] = svec_all[:,2] + shiftVec[2]

            check  = np.round(site - shiftedCoords, roundDepth)

            redundancies = np.where((check  == (0, 0,0)).all(axis=1))[0] 
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
                                redundancyDict[cind] = np.concatenate((redundancyDict[cind], redundancies))
            
            
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
                        minCellInds = np.concatenate((minCellInds, [cind]))
            else:
                #no redundant sites
                minCellInds = np.concatenate((minCellInds, [cind]))
                
        minCellInds = np.asarray(minCellInds, dtype = 'int')
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
        allCoords = np.round(self.coords[:,:], roundDepth)
        # svec_all = allCoords[:,0] + 1j*allCoords[:,1]
        svec_all = allCoords
        
        #get the coordinates of the minimum unit cell
        coords = np.round(self.rootCoords, roundDepth)
        # svec = np.zeros((coords.shape[0]))*(1 + 1j)
        # svec[:] = coords[:,0] + 1j*coords[:,1]
        svec = coords
        
        #store away the resonators, which tell me about all possible links
        resonators = np.round(self.resonators, roundDepth)
        # zmat = np.zeros((resonators.shape[0],2))*(1 + 1j)
        # zmat[:,0] = resonators[:,0] + 1j*resonators[:,1]
        # zmat[:,1] = resonators[:,2] + 1j*resonators[:,3]
        zmat = resonators

        self.rootLinks = np.zeros((self.resonators.shape[0]*2,5))
        
        def check_cell_relation(site, svec, shift1, shift2, shift3):
            ''' check if a given point in a copy of the unit cell translated by
            shift1*a1 +shift2*a2'''
            shiftVec = shift1*self.a1 + shift2*self.a2 + shift3*self.a3
            
            shiftedCoords = np.zeros(svec.shape)
            shiftedCoords[:,0] = svec[:,0] + shiftVec[0]
            shiftedCoords[:,1] = svec[:,1] + shiftVec[1]
            shiftedCoords[:,2] = svec[:,2] + shiftVec[2]
            
            check  = np.round(site - shiftedCoords, roundDepth)
            
            matches = np.where((check  == (0, 0,0)).all(axis=1))[0]
            # matches = np.where(np.isclose(site,shiftedCoords, atol = 2*10**(-roundDepth)))[0] #rounding is causing issues. Hopefully this is better
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
            
            check  = np.round(source - svec_all, roundDepth)
            sourceInd = np.where((check  == (0, 0,0)).all(axis=1))[0][0]
            # sourceInd = np.where(np.round(source,roundDepth) == np.round(svec_all,roundDepth))[0][0]
            check  = np.round(target - svec_all, roundDepth)
            targetInd = np.where((check  == (0, 0,0)).all(axis=1))[0][0]
            # targetInd = np.where(np.round(target,roundDepth) == np.round(svec_all,roundDepth))[0][0]
    
    
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
            sourceMatInd = np.where(sourceClass == self.rootCellInds)[0][0]
            targetMatInd = np.where(targetClass == self.rootCellInds)[0][0]
            
            
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
#        self.rootlinks = self.rootlinks[~np.all(self.rootlinkss == 0, axis=1)] 
        
        return
    
    def generate_root_Bloch_matrix(self, kx, ky,kz, t = 1):
        ''' 
        generates a Bloch matrix for the root graph of the layout for a given kx and ky
        
        needs the root cell and its links to be found first
        
        
        11-23-21: rewrote this guy to directly use the root links.
        
        '''
        BlochMat = np.zeros((self.numRootSites, self.numRootSites))*(0 + 0j)
        
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
            
            phaseFactor = np.exp(1j*kx*deltaX)*np.exp(1j*ky*deltaY)*np.exp(1j*kz*deltaZ)
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
            
            kxs = np.linspace(kx_0, kx_1,numsteps)
            kys = np.linspace(ky_0, ky_1,numsteps)
            kzs = np.linspace(kz_0, kz_1,numsteps)
            
            #check if the root cell has already been found
            #if it is there, do nothing, otherwise make it.
            try:
                self.rootCellInds[0]
            except:
                self.find_root_cell()
            minCellInds = self.rootCellInds
#            redundancyDict = self.rootVertexRedundnacy
                
            numLayoutSites = len(minCellInds)
            
            bandCut = np.zeros((numLayoutSites, numsteps))
            
            stateCut = np.zeros((numLayoutSites, numLayoutSites, numsteps)).astype('complex')
            
            for ind in range(0, numsteps):
                kvec = [kxs[ind],kys[ind],kzs[ind]]
                
                H = self.generate_root_Bloch_matrix(kvec[0], kvec[1], kvec[2])
            
                #Psis = np.zeros((self.numSites, self.numSites)).astype('complex')
                Es, Psis = scipy.linalg.eigh(H)
                
                bandCut[:,ind] = Es
                stateCut[:,:,ind] = Psis
            if returnStates:
                return kxs, kys,kzs, bandCut, stateCut
            else:
                return kxs, kys,kzs, bandCut
    
    def plot_root_bloch_wave(self, state_vect, ax,
                             theta = np.pi/10, phi = np.pi/10,
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
        Probs = np.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = np.angle(Amps)/np.pi
        
        #move the branch cut to -0.5
        outOfRange = np.where(mColors< -0.5)[0]
        mColors[outOfRange] = mColors[outOfRange] + 2
        
        cm = plt.cm.get_cmap(cmap)
        
        plotMat = self.rotate_coordinates(self.rootCoords, theta, phi)
        xs = plotMat[:,0]
        ys = plotMat[:,1]
        zs = plotMat[:,2]
        
        #plot the x-z projection
        plt.sca(ax)
        plt.scatter(xs, zs,c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            print( 'making colorbar')
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            # self.draw_SDlinks(ax, linewidth = 0.5, color = 'firebrick', zorder = zorder)
            self.draw_resonators(ax, linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        plt.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return        
 