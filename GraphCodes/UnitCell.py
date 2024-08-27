import re
import scipy
import numpy as np
import matplotlib.pyplot as plt

"""
UnitCell Class
    Object to conveniently hold and define a single unit cell. Will store the number
    of site, where they are, what the links are between them and neighboring unit cells,
    and which sites are needed to close an incomplete unit cell
    
    Supported Types:
        Huse (v0)
    
    Methods:
        ########
        #generating the cell
        ########
        _generate_kagome_cell
        _generate_Huse_cell
        _generate_PeterChain_cell
        _generate_PeterChain2_cell
        _generate_square_cell
        _generate_84Huse_cell
        _generate_74Huse_cell
        _generate_123Huse_cell
        _generate_Hk_composite_cell
        _generate_arbitrary_Cell
        
        ########
        #drawing the cell
        ########
        draw_resonators
        draw_resonator_end_points
        draw_sites
        draw_SDlinks
        _get_orientation_plot_points
        draw_site_orientations
        
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
        plot_bloch_wave_end_state
        
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
        testCell = UnitCell(lattice_type = 'Huse', side = 1)
        
        #custom cell
        testCell = UnitCell(lattice_type = 'name', side = 1, resonators = resonatorMat, a1 = vec1, a2 = vec2)
        
"""
class UnitCell(object):
    '''
    UnitCell _summary_

    :param object: _description_
    :type object: _type_
    '''    
    def __init__(self, lattice_type, 
                       side = 1, 
                       resonators = '', 
                       a1 = [1,0], 
                       a2 = [0,1],
                       maxDegree = 3):
        '''
        optional resonator and a1, a2 reciprocal lattice vector input arguments will only be used 
        if making a cell of non-built-in type using _generate_arbitrary_cell
        '''
        
        self.side = side*1.0
        
        #auto parse variants on Huse-type lattices
        match = re.search(r'(\d*)(Huse)(\d*)(\_*)(\d*)', lattice_type)
        if match:
            #Huse type lattice of some sort
            HuseType = match.groups()[0] + match.groups()[1]
            
            if match.groups()[3]  =='':
                #regular unit cell
                self.type = lattice_type
                generateMethod = getattr(self, '_generate_' + HuseType + '_cell')
                generateMethod(side= self.side)
            else:
                #composite unit cell
                defect_type = HuseType
                xtrans = int(match.groups()[2])
                ytrans = int(match.groups()[4])
                
                self.type = lattice_type
                self._generate_Hk_composite_cell(xtrans, ytrans, side = self.side, defect_type = defect_type)
        
        elif lattice_type == 'kagome':
            self.type = lattice_type
            self._generate_kagome_cell(self.side)
            
        elif lattice_type == 'PeterChain':
            self.type = lattice_type
            self._generate_PeterChain_cell(self.side)
            
        elif lattice_type == 'PeterChain_tail':
            self.type = lattice_type
            self._generate_PeterChain2_cell(self.side)
            
        elif lattice_type == 'square':
            self.type = lattice_type
            self._generate_square_cell(self.side)  
            
        else:
            #arbitrary lattice type
            self.type = lattice_type
            self.maxDegree = maxDegree
            self._generate_arbitrary_cell(resonators, a1, a2)
            
            
    ########
    #generator functions for unit cells
    ########        
    def _generate_kagome_cell(self, side = 1):
        '''
        generate kagome-type unit cell
        '''
        self.maxDegree = 3
        
        #set up the sites
        self.numSites = 3
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        
        #set up the lattice vectors
        self.a1 = np.asarray([self.side*np.sqrt(3)/2, self.side/2])
        self.a2 = np.asarray([0, self.side])
        dy = self.a1[1]/2
        dx = self.a1[0]/2
        xcorr = self.side/np.sqrt(3)/2/2
        
        #set up the positions of the sites of the effective lattice. ! look to newer functions for auto way to do these
        xs = np.asarray([-dx, -dx, 0])
        ys = np.asarray([dy, -dy, -2*dy])
        self.SDx = xs
        self.SDy = ys
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = np.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
        self.coords = np.zeros((self.numSites,2)) #set of all resonator start points
        
        a = self.side/np.sqrt(3)
        b = self.a1[0]-a
        #xo,yo,x1,y1
        #define them so their orientation matches the chosen one. First entry is plus end, second is minus
        self.resonators[0,:] = [-a/2, 2*dy, -b-a/2,  0]
        self.resonators[1,:] = [-a/2-b, 0, -a/2,  -2*dy]
        self.resonators[2,:] = [a/2, -2*dy, -a/2,  -2*dy]
        
        self.coords = self.get_coords(self.resonators)
        
        
        #####manual population of the SD links
#        #matrix to hold all the bonds
#        #starting site, ending site, number units cells over in a1, number unit cells over in a2, initial end type, final end type
#        oldlinks = np.zeros((self.numSites*4, 4)) #without orientation
#        links = np.zeros((self.numSites*4, 6))   #with orientation
#        #orientation defines by +x or +y is the +end 
#        
#        #fill in the links
#        links[0,:] = [0,2,0,1,   1,0]
#        links[1,:] = [0,1,0,0,   0,1]
#        links[2,:] = [0,2,-1,1,  0,1]
#        links[3,:] = [0,1,0,1,   1,0]
#        
#        links[4,:] = [1,2,0,0,   0,0]
#        links[5,:] = [1,0,0,0,   1,0]
#        links[6,:] = [1,2,-1,1,  1,1]
#        links[7,:] = [1,0,0,-1,  0,1]
#        
#        links[8,:] = [2,1,0,0,   0,0]
#        links[9,:] = [2,0,0,-1,  0,1]
#        links[10,:] =[2,0,1,-1,  1,0]
#        links[11,:] =[2,1,1,-1,  1,1]
#        
#        oldlinks = links[:,0:4]
#        self.SDlinks = oldlinks
#        self.SDHWlinks = links
        
        #####auto population of the SD links
        self._auto_generate_SDlinks()
        
        
        
        #make note of which resonator you need in order to close the unit cell
        closure = {}
        
        #a1 direction (x)
        closure[(1,0)] =np.asarray([1])
        #-a1 direction (-x)
        closure[(-1,0)] =np.asarray([])
        
        #a2 direction (y)
        closure[(0,1)] =np.asarray([2])
        #-a2 direction (y)
        closure[(0,-1)] =np.asarray([])
        
         #a1,a2 direction (x,y)
        closure[(1,1)] =np.asarray([])
        #-a1,a2 direction (-x,y)
        closure[(-1,1)] =np.asarray([])
        #a1,-a2 direction (x,-y)
        closure[(1,-1)] =np.asarray([0])
        #-a1,-a2 direction (-x,-y)
        closure[(-1,-1)] =np.asarray([])
        self.closure = closure
        
        
        return
    
    def _generate_Huse_cell(self, side = 1):        
        '''
        generate standard Huse-type unit cell (7,5)
        '''
        self.maxDegree = 3
        
        #set up the sites
        self.numSites = 12
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        
        #set up the lattice vectors
        self.a1 = np.asarray([self.side*np.sqrt(3)/2, self.side/2])
        self.a2 = np.asarray([0, 3*self.side])
        dy = self.a1[1]/2
        dx = self.a1[0]/2
        xcorr = self.side/np.sqrt(3)/2/2
        
        
        #set up the positions of the sites of the effective lattice
        xs = np.asarray([-dx, -dx,-dx,-dx,-dx,-dx, 0, -xcorr, +xcorr, 0, -xcorr, +xcorr])
        ys = np.asarray([5*dy, 3*dy, dy, -dy, -3*dy, -5*dy, -6*dy, 2*dy, 2*dy, 0, -2*dy, -2*dy])
        self.SDx = xs
        self.SDy = ys
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = np.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
        self.coords = np.zeros((self.numSites,2)) #set of all resoantor start points
        
        a = self.side/np.sqrt(3)
        b = self.a1[0]-a
        #xo,yo,x1,y1
        #define them so their orientation matches the chosen one. First entry is plus end, second is minus
        self.resonators[0,:] = [-a/2, 6*dy, -b-a/2,  4*dy]
        self.resonators[1,:] = [-a/2-b, 4*dy, -a/2,  2*dy]
        self.resonators[2,:] = [-a/2, 2*dy, -b-a/2,  0]
        self.resonators[3,:] = [-a/2-b, 0, -a/2,  -2*dy]
        self.resonators[4,:] = [-a/2, -2*dy, -b-a/2,  -4*dy]
        self.resonators[5,:] = [-a/2-b, -4*dy, -a/2,  -6*dy]
        self.resonators[6,:] = [a/2, -6*dy, -a/2,  -6*dy]
        self.resonators[7,:] = [0, 2*dy, -a/2,  2*dy]
        self.resonators[8,:] = [a/2, 2*dy, 0,  2*dy]
        self.resonators[9,:] = [0, 2*dy, 0,  -2*dy]
        self.resonators[10,:] = [0, -2*dy, -a/2,  -2*dy]
        self.resonators[11,:] = [a/2, -2*dy, 0,  -2*dy]
        
        self.coords = self.get_coords(self.resonators)
        
        
        
        ######manual population of the SD links
#        #matrix to hold all the bonds
#        #starting site, ending site, number units cells over in a1, number unit cells over in a2, initial end type, final end type
#        oldlinks = np.zeros((self.numSites*4, 4)) #without orientation
#        links = np.zeros((self.numSites*4, 6))   #with orientation
#        #orientation defines by +x or +y is the +end 
#        
#        #fill in the links
#        links[0,:] = [0 , 6, 0,1,  1,0]
#        links[1,:] = [0 , 6, -1,1, 0,1]
#        links[2,:] = [0 , 5, 0,1,  1,0]
#        links[3,:] = [0 , 1, 0,0,  0,1]
#        
#        links[4,:] = [1,2, 0,0,    0,1]
#        links[5,:] = [1,0, 0,0,    1,0]
#        links[6,:] = [1,7, 0,0,    0,0]
#        links[7,:] = [1,6, -1,1,   1,1]
#        
#        links[8,:] = [2,1, 0,0,    1,0]
#        links[9,:] = [2,7, 0,0,    1,0]
#        links[10,:] = [2,3, 0,0,   0,1]
#        links[11,:] = [2,8, -1,0,  0,1]
#        
#        links[12,:] = [3,2, 0,0,   1,0]
#        links[13,:] = [3,4, 0,0,   0,1]
#        links[14,:] = [3,10, 0,0,  0,0]
#        links[15,:] = [3,8, -1,0,  1,1]
#        
#        links[16,:] = [4,3, 0,0,   1,0]
#        links[17,:] = [4,10, 0,0,  1,0]
#        links[18,:] = [4,5, 0,0,   0,1]
#        links[19,:] = [4,11,-1,0,  0,1]
#        
#        links[20,:] = [5,4, 0,0,   1,0]
#        links[21,:] = [5,6, 0,0,   0,0]
#        links[22,:] = [5,11,-1,0,  1,1]
#        links[23,:] = [5,0, 0,-1,  0,1]
#        
#        links[24,:] = [6,5, 0,0,   0,0]
#        links[25,:] = [6,0, 0,-1,  0,1]
#        links[26,:] = [6,1, 1,-1,  1,1]
#        links[27,:] = [6,0, 1,-1,  1,0]
#        
#        links[28,:] = [7,1, 0,0,   0,0]
#        links[29,:] = [7,2, 0,0,   0,1]
#        links[30,:] = [7,9, 0,0,   1,1]
#        links[31,:] = [7,8, 0,0,   1,0]
#        
#        links[32,:] = [8,7, 0,0,   0,1]
#        links[33,:] = [8,9, 0,0,   0,1]
#        links[34,:] = [8,2, 1,0,   1,0]
#        links[35,:] = [8,3, 1,0,   1,1]
#        
#        links[36,:] = [9,7, 0,0,   1,1]
#        links[37,:] = [9,8, 0,0,   1,0]
#        links[38,:] = [9,10, 0,0,  0,1]
#        links[39,:] = [9,11, 0,0,  0,0]
#        
#        links[40,:] = [10,9, 0,0,  1,0]
#        links[41,:] = [10,11,0,0,  1,0]
#        links[42,:] = [10,3, 0,0,  0,0]
#        links[43,:] = [10,4, 0,0,  0,1]
#        
#        links[44,:] = [11,9, 0,0,  0,0]
#        links[45,:] = [11,10,0,0,  0,1]
#        links[46,:] = [11,4, 1,0,  1,0]
#        links[47,:] = [11,5, 1,0,  1,1]
#        
#        oldlinks = links[:,0:4]
#        self.SDlinks = oldlinks
#        self.SDHWlinks = links
        
        #####auto population of the SD links
        self._auto_generate_SDlinks()
        
        
        #make note of which resonator you need in order to close the unit cell
        closure = {}
        
        #a1 direction (x)
        closure[(1,0)] =np.asarray([1,2,3,4,5])
        #-a1 direction (-x)
        closure[(-1,0)] =np.asarray([])
        
        #a2 direction (y)
        closure[(0,1)] =np.asarray([6])
        #-a2 direction (y)
        closure[(0,-1)] =np.asarray([])
        
         #a1,a2 direction (x,y)
        closure[(1,1)] =np.asarray([])
        #-a1,a2 direction (-x,y)
        closure[(-1,1)] =np.asarray([])
        #a1,-a2 direction (x,-y)
        closure[(1,-1)] =np.asarray([0])
        #-a1,-a2 direction (-x,-y)
        closure[(-1,-1)] =np.asarray([])
        self.closure = closure
        
        return
    
    def _generate_PeterChain_cell(self, side = 1):
        '''
        generate Pater-Sarnak chain unit cell
        '''
        self.maxDegree = 3
        
        #set up the sites
        self.numSites = 6
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        
        #set up the lattice vectors
        self.a1 = np.asarray([self.side, 0])
        self.a2 = np.asarray([0, 2*self.side])
        dy = self.a1[1]/2
        dx = self.a1[0]/2
        xcorr = self.side/np.sqrt(3)/2/2
        
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = np.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
        self.coords = np.zeros((self.numSites,2)) #set of all resonator start points
        
        a = self.side/(2*np.sqrt(2) + 1)
        b = np.sqrt(2)*a
        #xo,yo,x1,y1
        #define them so their orientation matches the chosen one. First entry is plus end, second is minus
        self.resonators[0,:] = [-a-b, 0, -b,  0]
        self.resonators[1,:] = [-b, 0, 0,  b]
        self.resonators[2,:] = [0, b, b,  0]
        self.resonators[3,:] = [-b, 0, 0,  -b]
        self.resonators[4,:] = [0, -b, b,  0]
        self.resonators[5,:] = [0, -b, 0,  b]
        
        self.coords = self.get_coords(self.resonators)
        
        #set up the positions of the sites of the effective lattice
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        for rind in range(0, self.resonators.shape[0]):
            res = self.resonators[rind,:]
            xs[rind] = (res[0] + res[2])/2
            ys[rind] = (res[1] + res[3])/2
        self.SDx = xs
        self.SDy = ys
        
        
        #####auto population of the SD links
        self._auto_generate_SDlinks()
        
        
        
        #make note of which resonator you need in order to close the unit cell
        closure = {}
        
        #a1 direction (x)
        closure[(1,0)] =np.asarray([])
        #-a1 direction (-x)
        closure[(-1,0)] =np.asarray([])
        
        #a2 direction (y)
        closure[(0,1)] =np.asarray([])
        #-a2 direction (y)
        closure[(0,-1)] =np.asarray([])
        
         #a1,a2 direction (x,y)
        closure[(1,1)] =np.asarray([])
        #-a1,a2 direction (-x,y)
        closure[(-1,1)] =np.asarray([])
        #a1,-a2 direction (x,-y)
        closure[(1,-1)] =np.asarray([])
        #-a1,-a2 direction (-x,-y)
        closure[(-1,-1)] =np.asarray([])
        self.closure = closure
        
        
        return
    
    def _generate_PeterChain2_cell(self, side = 1):
        '''
        generate Pater-Sarnak chain unit cell
        '''
        self.maxDegree = 3
        
        #set up the sites
        self.numSites = 7
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        
        #set up the lattice vectors
        self.a1 = np.asarray([self.side, 0])
        self.a2 = np.asarray([0, 2*self.side])
        dy = self.a1[1]/2
        dx = self.a1[0]/2
        xcorr = self.side/np.sqrt(3)/2/2
        
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = np.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
        self.coords = np.zeros((self.numSites,2)) #set of all resonator start points
        
        a = self.side/(2*np.sqrt(2) + 2)
        b = np.sqrt(2)*a
        #xo,yo,x1,y1
        #define them so their orientation matches the chosen one. First entry is plus end, second is minus
        self.resonators[0,:] = [-a-b, 0, -b,  0]
        self.resonators[1,:] = [-b, 0, 0,  b]
        self.resonators[2,:] = [0, b, b,  0]
        self.resonators[3,:] = [-b, 0, 0,  -b]
        self.resonators[4,:] = [0, -b, b,  0]
        self.resonators[5,:] = [0, -b, 0,  b]
        self.resonators[6,:] = [b, 0, a+b, 0]
        
        self.coords = self.get_coords(self.resonators)
        
        #set up the positions of the sites of the effective lattice
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        for rind in range(0, self.resonators.shape[0]):
            res = self.resonators[rind,:]
            xs[rind] = (res[0] + res[2])/2
            ys[rind] = (res[1] + res[3])/2
        self.SDx = xs
        self.SDy = ys
        
        
        #####auto population of the SD links
        self._auto_generate_SDlinks()
        
        
        
        #make note of which resonator you need in order to close the unit cell
        closure = {}
        
        #a1 direction (x)
        closure[(1,0)] =np.asarray([])
        #-a1 direction (-x)
        closure[(-1,0)] =np.asarray([])
        
        #a2 direction (y)
        closure[(0,1)] =np.asarray([])
        #-a2 direction (y)
        closure[(0,-1)] =np.asarray([])
        
         #a1,a2 direction (x,y)
        closure[(1,1)] =np.asarray([])
        #-a1,a2 direction (-x,y)
        closure[(-1,1)] =np.asarray([])
        #a1,-a2 direction (x,-y)
        closure[(1,-1)] =np.asarray([])
        #-a1,-a2 direction (-x,-y)
        closure[(-1,-1)] =np.asarray([])
        self.closure = closure
        
        
        return
    
    def _generate_square_cell(self, side = 1):
        '''
        generate sqare lattice unit cell
        '''
        self.maxDegree = 4
        
        #set up the sites
        self.numSites = 2
#        self.numSites = 4
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        
        #set up the lattice vectors
        self.a1 = np.asarray([self.side, 0])
        self.a2 = np.asarray([0, self.side])
        dy = self.a1[1]/2
        dx = self.a1[0]/2
        xcorr = self.side/np.sqrt(3)/2/2
        
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = np.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
        self.coords = np.zeros((self.numSites,2)) #set of all resonator start points
        
        a = self.side
        #xo,yo,x1,y1
        #define them so their orientation matches the chosen one. First entry is plus end, second is minus
        self.resonators[0,:] = [0, 0, 0, a]
        self.resonators[1,:] = [a, 0, 0, 0]
        
#        self.resonators[0,:] = [0, 0, 0, a/2.]
#        self.resonators[1,:] = [0, 0, a/2., 0]
#        self.resonators[2,:] = [-a/2., 0, 0, 0]
#        self.resonators[3,:] = [0, -a/2., 0, 0]
        
        self.coords = self.get_coords(self.resonators)
        
        #set up the positions of the sites of the effective lattice
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        for rind in range(0, self.resonators.shape[0]):
            res = self.resonators[rind,:]
            xs[rind] = (res[0] + res[2])/2
            ys[rind] = (res[1] + res[3])/2
        self.SDx = xs
        self.SDy = ys
        
        
        #####auto population of the SD links
        self._auto_generate_SDlinks()
        
        
        
        #make note of which resonator you need in order to close the unit cell
        closure = {}
        
        #a1 direction (x)
        closure[(1,0)] =np.asarray([])
        #-a1 direction (-x)
        closure[(-1,0)] =np.asarray([])
        
        #a2 direction (y)
        closure[(0,1)] =np.asarray([])
        #-a2 direction (y)
        closure[(0,-1)] =np.asarray([])
        
         #a1,a2 direction (x,y)
        closure[(1,1)] =np.asarray([])
        #-a1,a2 direction (-x,y)
        closure[(-1,1)] =np.asarray([])
        #a1,-a2 direction (x,-y)
        closure[(1,-1)] =np.asarray([])
        #-a1,-a2 direction (-x,-y)
        closure[(-1,-1)] =np.asarray([])
        self.closure = closure
        
        
        return
    
    def _generate_84Huse_cell(self, side = 1):        
        '''
        generate 8,4 Huse variant unit cell
        '''
        self.maxDegree = 3
        
        #set up the sites
        self.numSites = 12
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        
        #set up the lattice vectors
        self.a1 = np.asarray([self.side*np.sqrt(3)/2, self.side/2])
        self.a2 = np.asarray([0, 3*self.side])
        dy = self.a1[1]/2
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = np.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
        self.coords = np.zeros((self.numSites,2)) #set of all resoantor start points
        
        a = self.side/np.sqrt(3)
        b = self.a1[0]-a
        #xo,yo,x1,y1
        #define them so their orientation matches the chosen one. First entry is plus end, second is minus
        self.resonators[0,:] = [-a/2, 6*dy, -b-a/2,  4*dy]
        self.resonators[1,:] = [-a/2-b, 4*dy, -a/2,  2*dy]
        self.resonators[2,:] = [-a/2, 2*dy, -b-a/2,  0]
        self.resonators[3,:] = [-a/2-b, 0, -a/2,  -2*dy]
        self.resonators[4,:] = [-a/2, -2*dy, -b-a/2,  -4*dy]
        self.resonators[5,:] = [-a/2-b, -4*dy, -a/2,  -6*dy]
        self.resonators[6,:] = [a/2, -6*dy, -a/2,  -6*dy]
        
        self.resonators[7,:] = [-a/2, 2*dy, -a/4,  0]
        self.resonators[8,:] = [a/4,0, -a/4,  0]
        self.resonators[9,:] = [ a/2,  2*dy,a/4, 0]
        self.resonators[10,:] = [-a/4, 0, -a/2, -2*dy]
        self.resonators[11,:] = [a/4, 0, a/2, -2*dy]
#        self.resonators[7,:] = [-a/2, 2*dy, -a/2,  0]
#        self.resonators[8,:] = [a/2,0, -a/2,  0]
#        self.resonators[9,:] = [ a/2,  2*dy,a/2, 0]
#        self.resonators[10,:] = [-a/2, 0, -a/2, -2*dy]
#        self.resonators[11,:] = [a/2, 0, a/2, -2*dy]
        
        self.coords = self.get_coords(self.resonators)
        
        #set up the positions of the sites of the effective lattice
#        xs = np.asarray([-dx, -dx,-dx,-dx,-dx,-dx, 0, -xcorr, +xcorr, 0, -xcorr, +xcorr])
#        ys = np.asarray([5*dy, 3*dy, dy, -dy, -3*dy, -5*dy, -6*dy, 2*dy, 2*dy, 0, -2*dy, -2*dy])
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        for rind in range(0, self.resonators.shape[0]):
            res = self.resonators[rind,:]
            xs[rind] = (res[0] + res[2])/2
            ys[rind] = (res[1] + res[3])/2
        self.SDx = xs
        self.SDy = ys
        
        
        #####auto population of the SD links
        self._auto_generate_SDlinks()
        
        #remove bad links from 4-way coupler
        badLinks= []
        for lind in range(self.SDHWlinks.shape[0]):
            link = self.SDHWlinks[lind,:]
            site1 = link[0]
            site2 = link[1]
            if site1 ==7 and site2 ==10:
                badLinks.append(lind)
            if site1 ==10 and site2 ==7:
                badLinks.append(lind)
            if site1 ==9 and site2 ==8:
                badLinks.append(lind)
            if site1 ==8 and site2 ==9:
                badLinks.append(lind)
#        #mark the bad rows
#        self.SDHWlinks[badLinks,:] = -4
#        #excise the bad rows
#        self.SDHWlinks = self.SDHWlinks[~np.all(self.SDHWlinks == -4, axis=1)] 
#        #also store the old link format
#        oldlinks = self.SDHWlinks[:,0:4]
#        self.SDlinks = oldlinks
        
        
        #make note of which resonator you need in order to close the unit cell
        closure = {}
        
        #a1 direction (x)
        closure[(1,0)] =np.asarray([1,2,3,4,5])
        #-a1 direction (-x)
        closure[(-1,0)] =np.asarray([])
        
        #a2 direction (y)
        closure[(0,1)] =np.asarray([6])
        #-a2 direction (y)
        closure[(0,-1)] =np.asarray([])
        
         #a1,a2 direction (x,y)
        closure[(1,1)] =np.asarray([])
        #-a1,a2 direction (-x,y)
        closure[(-1,1)] =np.asarray([])
        #a1,-a2 direction (x,-y)
        closure[(1,-1)] =np.asarray([0])
        #-a1,-a2 direction (-x,-y)
        closure[(-1,-1)] =np.asarray([])
        self.closure = closure
        
        return
    
    def _generate_74Huse_cell(self, side = 1):        
        '''
        generate 7,4 Huse variant unit cell
        
        Note: this graph is not three regular, but it is triangle protected
        '''
        self.maxDegree = 4
        
        #set up the sites
        self.numSites = 11
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        
        #set up the lattice vectors
        self.a1 = np.asarray([self.side*np.sqrt(3)/2, self.side/2])
        self.a2 = np.asarray([0, 3*self.side])
        dy = self.a1[1]/2
        dx = self.a1[0]/2
        xcorr = self.side/np.sqrt(3)/2/2
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = np.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
        self.coords = np.zeros((self.numSites,2)) #set of all resoantor start points
        
        a = self.side/np.sqrt(3)
        b = self.a1[0]-a
        #xo,yo,x1,y1
        #define them so their orientation matches the chosen one. First entry is plus end, second is minus
        self.resonators[0,:] = [-a/2, 6*dy, -b-a/2,  4*dy]
        self.resonators[1,:] = [-a/2-b, 4*dy, -a/2,  2*dy]
        self.resonators[2,:] = [-a/2, 2*dy, -b-a/2,  0]
        self.resonators[3,:] = [-a/2-b, 0, -a/2,  -2*dy]
        self.resonators[4,:] = [-a/2, -2*dy, -b-a/2,  -4*dy]
        self.resonators[5,:] = [-a/2-b, -4*dy, -a/2,  -6*dy]
        self.resonators[6,:] = [a/2, -6*dy, -a/2,  -6*dy]
        
        self.resonators[7,:] = [-a/2, 2*dy, 0,  0]
        self.resonators[8,:] = [a/2, 2*dy, 0,  0]
        self.resonators[9,:] = [ 0,  0,-a/2, -2*dy]
        self.resonators[10,:] = [0, 0, a/2, -2*dy]
        
        self.coords = self.get_coords(self.resonators)
        
        #set up the positions of the sites of the effective lattice
#        xs = np.asarray([-dx, -dx,-dx,-dx,-dx,-dx, 0, -xcorr, +xcorr, 0, -xcorr, +xcorr])
#        ys = np.asarray([5*dy, 3*dy, dy, -dy, -3*dy, -5*dy, -6*dy, 2*dy, 2*dy, 0, -2*dy, -2*dy])
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        for rind in range(0, self.resonators.shape[0]):
            res = self.resonators[rind,:]
            xs[rind] = (res[0] + res[2])/2
            ys[rind] = (res[1] + res[3])/2
        self.SDx = xs
        self.SDy = ys
        
        
        #####auto population of the SD links
        self._auto_generate_SDlinks()
        
        #remove bad links from 4-way coupler
        badLinks= []
        for lind in range(self.SDHWlinks.shape[0]):
            link = self.SDHWlinks[lind,:]
            site1 = link[0]
            site2 = link[1]
            if site1 ==7 and site2 ==10:
                badLinks.append(lind)
            if site1 ==10 and site2 ==7:
                badLinks.append(lind)
            if site1 ==9 and site2 ==8:
                badLinks.append(lind)
            if site1 ==8 and site2 ==9:
                badLinks.append(lind)
#        #mark the bad rows
#        self.SDHWlinks[badLinks,:] = -4
#        #excise the bad rows
#        self.SDHWlinks = self.SDHWlinks[~np.all(self.SDHWlinks == -4, axis=1)] 
#        #also store the old link format
#        oldlinks = self.SDHWlinks[:,0:4]
#        self.SDlinks = oldlinks
        
        
        #make note of which resonator you need in order to close the unit cell
        closure = {}
        
        #a1 direction (x)
        closure[(1,0)] =np.asarray([1,2,3,4,5])
        #-a1 direction (-x)
        closure[(-1,0)] =np.asarray([])
        
        #a2 direction (y)
        closure[(0,1)] =np.asarray([6])
        #-a2 direction (y)
        closure[(0,-1)] =np.asarray([])
        
         #a1,a2 direction (x,y)
        closure[(1,1)] =np.asarray([])
        #-a1,a2 direction (-x,y)
        closure[(-1,1)] =np.asarray([])
        #a1,-a2 direction (x,-y)
        closure[(1,-1)] =np.asarray([0])
        #-a1,-a2 direction (-x,-y)
        closure[(-1,-1)] =np.asarray([])
        self.closure = closure
        
        return
    
    def _generate_123Huse_cell(self, side = 1): 
        '''
        generate 12,3 Huse variant unit cell
        '''
        self.maxDegree = 3
        
        #set up the sites
        self.numSites = 9
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        
        #set up the lattice vectors
        self.a1 = np.asarray([self.side*np.sqrt(3)/2, self.side/2])
        self.a2 = np.asarray([0, 3*self.side])
        dy = self.a1[1]/2
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = np.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
        self.coords = np.zeros((self.numSites,2)) #set of all resoantor start points
        
        a = self.side/np.sqrt(3)
        b = self.a1[0]-a
        #xo,yo,x1,y1
        #define them so their orientation matches the chosen one. First entry is plus end, second is minus
        self.resonators[0,:] = [-a/2, 6*dy, -b-a/2,  4*dy]
        self.resonators[1,:] = [-a/2-b, 4*dy, -a/2,  2*dy]
        self.resonators[2,:] = [-a/2, 2*dy, -b-a/2,  0]
        self.resonators[3,:] = [-a/2-b, 0, -a/2,  -2*dy]
        self.resonators[4,:] = [-a/2, -2*dy, -b-a/2,  -4*dy]
        self.resonators[5,:] = [-a/2-b, -4*dy, -a/2,  -6*dy]
        self.resonators[6,:] = [a/2, -6*dy, -a/2,  -6*dy]
        
        self.resonators[7,:] = [-a/2, 2*dy, -a/2,  -2*dy]
        self.resonators[8,:] = [a/2, 2*dy, a/2,  -2*dy]
        
        self.coords = self.get_coords(self.resonators)
        
        #set up the positions of the sites of the effective lattice
#        xs = np.asarray([-dx, -dx,-dx,-dx,-dx,-dx, 0, -xcorr, +xcorr, 0, -xcorr, +xcorr])
#        ys = np.asarray([5*dy, 3*dy, dy, -dy, -3*dy, -5*dy, -6*dy, 2*dy, 2*dy, 0, -2*dy, -2*dy])
        xs = np.zeros(self.numSites)
        ys = np.zeros(self.numSites)
        for rind in range(0, self.resonators.shape[0]):
            res = self.resonators[rind,:]
            xs[rind] = (res[0] + res[2])/2
            ys[rind] = (res[1] + res[3])/2
        self.SDx = xs
        self.SDy = ys
        
#        #alternate drawing
#        self.SDx[7] = -a/15
#        self.SDx[8] = a/15
        
        
        #####auto population of the SD links
        self._auto_generate_SDlinks()
         
        #make note of which resonator you need in order to close the unit cell
        closure = {}
        
        #a1 direction (x)
        closure[(1,0)] =np.asarray([1,2,3,4,5])
        #-a1 direction (-x)
        closure[(-1,0)] =np.asarray([])
        
        #a2 direction (y)
        closure[(0,1)] =np.asarray([6])
        #-a2 direction (y)
        closure[(0,-1)] =np.asarray([])
        
         #a1,a2 direction (x,y)
        closure[(1,1)] =np.asarray([])
        #-a1,a2 direction (-x,y)
        closure[(-1,1)] =np.asarray([])
        #a1,-a2 direction (x,-y)
        closure[(1,-1)] =np.asarray([0])
        #-a1,-a2 direction (-x,-y)
        closure[(-1,-1)] =np.asarray([])
        self.closure = closure
        
        return
    
    def _generate_Hk_composite_cell(self, xtrans, ytrans, side = 1, defect_type = 'Huse'):
        '''
        make a composite unit cell.
        
        if xtrans =1, huse cells will touch in x direction
        if x trans = 2, Huse cells will have a collumn of hexagona inbetween them
        
        same for ytrans
        
        haven't yet taken care of the closure properly
        
        defect_type tells it what kind of variation to use. Currently accepts all Huse-type cells
        Hopefully will auto accept other tings as they are added, but some funny business may crop up
        if certtain special symetries or things aren't respected
        '''
        self.maxDegree = 4 #in case of 7,4
        
        cellH = UnitCell(defect_type, side = side)
        cellk = UnitCell('kagome', side = side)
        self.cellH = cellH
        self.cellk = cellk
        
        #set up the sites
        self.numSites = cellH.numSites + (xtrans-1)*cellk.numSites*3 +  (xtrans)*(ytrans-1)*cellk.numSites*3
        
        #set up the lattice vectors
        self.a1 = cellH.a1*xtrans
        self.a2 = cellH.a2*ytrans
        
        #allocate for the resonators
        self.resonators = np.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
        self.coords = np.zeros((self.numSites,2)) #set of all resoantor start points
        
        #maks for shifting resonators
        xmask = np.zeros((cellk.numSites,4))
        ymask = np.zeros((cellk.numSites,4))
        xmask[:,0] = 1
        xmask[:,2] = 1
        ymask[:,1] = 1
        ymask[:,3] = 1
        
        #compile all the resonators
        rind = 0
        #all the others
        for indx in range(0, xtrans):
            for indy in range(0, ytrans):
                if (indx==0) and (indy ==0):
                    #the Huse cell
                    self.resonators[rind:rind+cellH.resonators.shape[0],:] = cellH.resonators
                    rind =  rind+ cellH.resonators.shape[0]
                else:  
                    for subind in range(0,3):
                        xOffset = indx*cellH.a1[0] + indy*cellH.a2[0] + (subind-1)*cellk.a2[0]
                        yOffset = indx*cellH.a1[1] + indy*cellH.a2[1] + (subind-1)*cellk.a2[1]
                        
                        ress = cellk.resonators + xOffset*xmask + yOffset*ymask
                        self.resonators[rind:rind+ress.shape[0],:] = ress
                        rind =  rind+ ress.shape[0]
                        
        #####auto population of the SD links
        self._auto_generate_SDlinks()
        
        #set up the positions of the sites of the effective lattice
        x0 = self.resonators[:,0]
        y0 = self.resonators[:,1]
        x1 = self.resonators[:,2]
        y1 = self.resonators[:,3]
        self.SDx = (x0+x1)/2.
        self.SDy = (y0+y1)/2.
        
        self.coords = self.get_coords(self.resonators)
        
        #make note of which resonator you need in order to close the unit cell
        ####!!!!!!!!!! incomplete 
        closure = {}
        
        #a1 direction (x)
        closure[(1,0)] =np.asarray([])
        #-a1 direction (-x)
        closure[(-1,0)] =np.asarray([])
        
        #a2 direction (y)
        closure[(0,1)] =np.asarray([])
        #-a2 direction (y)
        closure[(0,-1)] =np.asarray([])
        
         #a1,a2 direction (x,y)
        closure[(1,1)] =np.asarray([])
        #-a1,a2 direction (-x,y)
        closure[(-1,1)] =np.asarray([])
        #a1,-a2 direction (x,-y)
        closure[(1,-1)] =np.asarray([])
        #-a1,-a2 direction (-x,-y)
        closure[(-1,-1)] =np.asarray([])
        self.closure = closure
            
        
        return
    
    def _generate_arbitrary_cell(self, resonators, a1 = [1,0], a2 = [0,1]):
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
            if resonators.shape[1] != 4:
                raise ValueError('provided resonators are not the right shape')
        
        
        #set up the lattice vectors
        self.a1 = np.asarray(a1)
        self.a2 = np.asarray(a2)    
        
        if self.a1.shape != (2,):
            raise ValueError('first lattice vector has invalid shape')
            
        if self.a2.shape != (2,):
            raise ValueError('first lattice vector has invalid shape')
        
        #set up the sites
        self.numSites = resonators.shape[0]
        
        
        
        
        #set up the poisitions of all the resonators  and their end points
        self.resonators = np.zeros((self.numSites,4)) #pairs of resonator end points for each resonator
        self.resonators=resonators*self.side
        
        self.coords = self.get_coords(self.resonators)
        
        #set up the positions of the sites of the effective lattice
        x0 = self.resonators[:,0]
        y0 = self.resonators[:,1]
        x1 = self.resonators[:,2]
        y1 = self.resonators[:,3]
        self.SDx = (x0+x1)/2.
        self.SDy = (y0+y1)/2.
        
        #####auto population of the SD links
        self._auto_generate_SDlinks()
        
        
        
        #make note of which resonator you need in order to close the unit cell
        #this is not handled well with an arbitrary cell
        closure = {}
        
        #a1 direction (x)
        closure[(1,0)] =np.asarray([])
        #-a1 direction (-x)
        closure[(-1,0)] =np.asarray([])
        
        #a2 direction (y)
        closure[(0,1)] =np.asarray([])
        #-a2 direction (y)
        closure[(0,-1)] =np.asarray([])
        
         #a1,a2 direction (x,y)
        closure[(1,1)] =np.asarray([])
        #-a1,a2 direction (-x,y)
        closure[(-1,1)] =np.asarray([])
        #a1,-a2 direction (x,-y)
        closure[(1,-1)] =np.asarray([])
        #-a1,-a2 direction (-x,-y)
        closure[(-1,-1)] =np.asarray([])
        self.closure = closure
        
        
        return
    
    def get_coords(self, resonators, roundDepth = 3):
        '''
        take in a set of resonators and calculate the set of end points.
        
        Will round all coordinates the the specified number of decimals.
        
        Should remove all redundancies.
        '''
        
        coords_overcomplete = np.zeros((resonators.shape[0]*2, 1)).astype('complex')
        coords_overcomplete =  np.concatenate((resonators[:,0], resonators[:,2])) + 1j * np.concatenate((resonators[:,1], resonators[:,3]))
        
        coords_complex = np.unique(np.round(coords_overcomplete, roundDepth))
    
        coords = np.zeros((coords_complex.shape[0],2))
        coords[:,0] = np.real(coords_complex)
        coords[:,1] = np.imag(coords_complex)
        
        return coords
    

    #######
    #draw functions
    #######  
    def draw_resonators(self, ax, color = 'g', alpha = 1 , linewidth = 0.5, zorder = 1):
        '''
        draw each resonator as a line
        '''
        for res in range(0,self.resonators.shape[0] ):
            [x0, y0, x1, y1]  = self.resonators[res,:]
            ax.plot([x0, x1],[y0, y1] , color = color, alpha = alpha, linewidth = linewidth, zorder = zorder)
        return
    
    def draw_resonator_end_points(self, ax, color = 'g', edgecolor = 'k',  marker = 'o' , size = 10, zorder = 1):
        '''will double draw some points'''
        x0s = self.resonators[:,0]
        y0s = self.resonators[:,1]
        
        x1s = self.resonators[:,2]
        y1s = self.resonators[:,3]
        
        plt.sca(ax)
        plt.scatter(x0s, y0s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        plt.scatter(x1s, y1s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        return
      
    def draw_sites(self, ax, color = 'g', edgecolor = 'k',  marker = 'o' , size = 10, zorder=1):
        '''
        draw sites of the semidual (effective lattice)
        '''
        xs = self.SDx
        ys = self.SDy
        plt.sca(ax)
        plt.scatter(xs, ys ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        ax.set_aspect('equal')
        return
    
    def draw_SDlinks(self, ax, color = 'firebrick', linewidth = 0.5, HW = False, minus_color = 'goldenrod', zorder = 1, alpha = 1):
        '''
        draw all the links of the semidual lattice
        
        if extra is True it will draw only the edge sites required to fix the edge of the tiling
        
        set HW to true if you want the links color coded by sign
        minus_color sets the sign of the negative links
        '''

        links = self.SDHWlinks[:]
        
        for link in range(0, links.shape[0]):
            [startSite, endSite, deltaA1, deltaA2]  = links[link,0:4]
            startSite = int(startSite)
            endSite = int(endSite)
            
            [x0,y0] = [self.SDx[startSite], self.SDy[startSite]]
            [x1,y1] = np.asarray([self.SDx[endSite], self.SDy[endSite]]) + deltaA1*self.a1 + deltaA2*self.a2
            
            if HW:
                ends = links[link,4:6]
                if ends[0]==ends[1]:
                    #++ or --, use normal t
                    ax.plot([x0, x1],[y0, y1] , color = color, linewidth = linewidth, zorder = zorder, alpha = alpha)
                else:
                    #+- or -+, use inverted t
                    ax.plot([x0, x1],[y0, y1] , color = minus_color, linewidth = linewidth, zorder = zorder, alpha = alpha)
            else :
                ax.plot([x0, x1],[y0, y1] , color = color, linewidth = linewidth, zorder = zorder, alpha = alpha)
                
        return
    
    def _get_orientation_plot_points(self,scaleFactor = 0.5):
        '''
        find end coordinate locations part way along each resonator so that
        they can be used to plot the field at both ends of the resonator.
        
        Scale factor says how far appart the two points will be: +- sclaeFactor.2 of the total length
        
        returns the polt points as collumn matrix
        '''
        if scaleFactor> 1:
            raise ValueError('scale factor too big')
            
            
        size = len(self.SDx)
        plot_points = np.zeros((size*2, 2))
        
        resonators = self.resonators
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
    
    def draw_site_orientations(self,ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'jet_r', scaleFactor = 0.5, mSizes = 60, zorder = 1):
        Amps = np.ones(len(self.SDx))
        Probs = np.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = Amps
       
        mSizes = 60
        
        #build full state with value on both ends of the resonators 
        mColors_end = np.zeros(len(Amps)*2)
        mColors_end[0::2] = mColors

        #put opposite sign on other side
        mColors_end[1::2] = -mColors
#        mColors_end[1::2] = 5
        
        cm = plt.cm.get_cmap(cmap)
        
        #get coordinates for the two ends of the resonator
        plotPoints = self._get_orientation_plot_points(scaleFactor = scaleFactor)
        xs = plotPoints[:,0]
        ys = plotPoints[:,1]
        
        plt.sca(ax)
#        plt.scatter(xs, ys,c =  mColors_end, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -1, vmax = 1, zorder = zorder)
        plt.scatter(xs, ys,c =  mColors_end, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -1.5, vmax = 2.0, zorder = zorder)
        if colorbar:
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax,linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        plt.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        
        return mColors_end
    
    #####
    #auto construction functions for SD links
    ######
    def _auto_generate_SDlinks(self):
        '''
        start from all the resonators of a unit cell auto generate the full link matrix,
        including neighboring cells
        '''
        xmask = np.zeros((self.numSites,4))
        ymask = np.zeros((self.numSites,4))
        
        xmask[:,0] = 1
        xmask[:,2] = 1
        
        ymask[:,1] = 1
        ymask[:,3] = 1
        
        if self.type[0:2] == '74':
            self.SDHWlinks = np.zeros((self.numSites*4+4,6))
        elif self.type == 'square':
            self.SDHWlinks = np.zeros((self.numSites*6,6))
        elif self.type[0:4] == 'kite':
            self.SDHWlinks = np.zeros((self.numSites*10,6))
        else:
            # self.SDHWlinks = np.zeros((self.numSites*4,6))
            # self.SDHWlinks = np.zeros((self.numSites*8,6)) #temporary hack to allow some line graph games
            
            # #now fixing to use the max degree given
            maxLineCoordination = (self.maxDegree-1)*2
            self.SDHWlinks = np.zeros((self.numSites*maxLineCoordination, 6))
        
        lind = 0
        for da1 in range(-1,2):
            for da2 in range(-1,2):
                links = self._auto_generate_cell_SDlinks(da1, da2)
                newLinks = links.shape[0]
                self.SDHWlinks[lind:lind+newLinks,:] = links
                lind = lind + newLinks
        
        #remove blank links (needed for some types of arbitrary cells)
        self.SDHWlinks = self.SDHWlinks[~np.all(self.SDHWlinks == 0, axis=1)] 
        
        #also store the old link format
        oldlinks = self.SDHWlinks[:,0:4]
        self.SDlinks = oldlinks 
        
        return
            
    def _auto_generate_cell_SDlinks(self, deltaA1, deltaA2):
        '''
        function to autogenerate the links between two sets of resonators
        deltaA1 and deltaA2 specify how many lattice vectors the two cells are seperated by
        in the first (~x) and second  (~y) lattice directions
        
        could be twice the same set, or it could be two different unit cells.
        
        will return a matrix of all the links [start, target, deltaA1, deltaA2, start_polarity, end_polarity]
        
        '''
        ress1 = self.resonators
        len1 = ress1.shape[0]
        
        #find the new unit cell
        xmask = np.zeros((self.numSites,4))
        ymask = np.zeros((self.numSites,4))
        xmask[:,0] = 1
        xmask[:,2] = 1
        ymask[:,1] = 1
        ymask[:,3] = 1
        xOffset = deltaA1*self.a1[0] + deltaA2*self.a2[0]
        yOffset = deltaA1*self.a1[1] + deltaA2*self.a2[1]
        ress2 = ress1 + xOffset*xmask + yOffset*ymask

        #place to store the links
        # linkMat = np.zeros((len1*4+len1*4,6))  #I'm pretty sure this only works for low degree
        maxLineCoordination = (self.maxDegree-1)*2
        linkMat = np.zeros((len1*maxLineCoordination+len1*maxLineCoordination,6))
        
        #find the links
        
        #round the coordinates to prevent stupid mistakes in finding the connections
        plusEnds = np.round(ress2[:,0:2],3)
        minusEnds = np.round(ress2[:,2:4],3)
        
        extraLinkInd = 0
        for resInd in range(0,ress1.shape[0]):
            res = np.round(ress1[resInd,:],3)
            x1 = res[0]
            y1 = res[1]
            x0 = res[2]
            y0 = res[3]

            plusPlus = np.where((plusEnds == (x1, y1)).all(axis=1))[0]
            minusMinus = np.where((minusEnds == (x0, y0)).all(axis=1))[0]
            
            plusMinus = np.where((minusEnds == (x1, y1)).all(axis=1))[0] #plus end of new res, minus end of old
            minusPlus = np.where((plusEnds == (x0, y0)).all(axis=1))[0]
            
            for ind in plusPlus:
                if ind == resInd:
                    #self link
                    pass
                else:
                    linkMat[extraLinkInd,:] = [resInd, ind, deltaA1, deltaA2, 1,1]
                    extraLinkInd = extraLinkInd+1
                    
            for ind in minusMinus:
                if ind == resInd:
                    #self link
                    pass
                else:
                    linkMat[extraLinkInd,:] = [resInd, ind, deltaA1, deltaA2,  0,0]
                    extraLinkInd = extraLinkInd+1
                    
            for ind in plusMinus:
                linkMat[extraLinkInd,:] = [resInd, ind, deltaA1, deltaA2,  1,0]
                extraLinkInd = extraLinkInd+1
                
            for ind in minusPlus:
                linkMat[extraLinkInd,:] = [ resInd, ind, deltaA1, deltaA2,  0,1]
                extraLinkInd = extraLinkInd+1
        
        #clean the skipped links away 
        linkMat = linkMat[~np.all(linkMat == 0, axis=1)]  
        
        return linkMat
    
    ######
    #Bloch theory calculation functions
    ######
    def generate_Bloch_matrix(self, kx, ky, modeType = 'FW', t = 1, phase = 0):
        BlochMat = np.zeros((self.numSites, self.numSites))*(0 + 0j)
        
        for lind in range(0, self.SDHWlinks.shape[0]):
            link = self.SDHWlinks[lind,:]
            startInd = int(link[0]) #within the unit cell
            targetInd = int(link[1])
            deltaA1 = int(link[2])
            deltaA2   = int(link[3])
            startPol = int(link[4])
            targetPol = int(link[5])
            
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
            
            #coordinates of target site
            x1 = self.SDx[targetInd] + deltaA1*self.a1[0] + deltaA2*self.a2[0]
            y1 = self.SDy[targetInd] + deltaA1*self.a1[1] + deltaA2*self.a2[1]
            
            deltaX = x1-x0
            deltaY = y1-y0
            
            phaseFactor = np.exp(1j*kx*deltaX)*np.exp(1j*ky*deltaY)
            BlochMat[startInd, targetInd] = BlochMat[startInd, targetInd]+ t*phaseFactor*signum
        return BlochMat
    
    def compute_band_structure(self, kx_0, ky_0, kx_1, ky_1, numsteps = 100, modeType = 'FW', returnStates = False, phase  = 0):
        '''
        from scipy.linalg.eigh:
        The normalized selected eigenvector corresponding to the eigenvalue w[i] is the column v[:,i].
        
        This returns same format with two additional kx, ky indices
        '''
        
        kxs = np.linspace(kx_0, kx_1,numsteps)
        kys = np.linspace(ky_0, ky_1,numsteps)
        
        bandCut = np.zeros((self.numSites, numsteps))
        
        stateCut = np.zeros((self.numSites, self.numSites, numsteps)).astype('complex')
        
        for ind in range(0, numsteps):
            kvec = [kxs[ind],kys[ind]]
            
            H = self.generate_Bloch_matrix(kvec[0], kvec[1], modeType = modeType, phase  = phase)
        
            #Psis = np.zeros((self.numSites, self.numSites)).astype('complex')
            Es, Psis = scipy.linalg.eigh(H)
            
            bandCut[:,ind] = Es
            stateCut[:,:,ind] = Psis
        if returnStates:
            return kxs, kys, bandCut, stateCut
        else:
            return kxs, kys, bandCut
    
    def plot_band_cut(self, ax, bandCut, colorlist = '', zorder = 1, dots = False, linewidth = 2.5):
        if colorlist == '':
            colorlist = ['firebrick', 'dodgerblue', 'blueviolet', 'mediumblue', 'goldenrod', 'cornflowerblue']
        
        plt.sca(ax)
        
        for ind in range(0,self.numSites):
            colorInd = np.mod(ind, len(colorlist))
            if dots:
                plt.plot(bandCut[ind,:], color = colorlist[colorInd] , marker = '.', markersize = '5', linestyle = '', zorder = zorder)
            else:
                plt.plot(bandCut[ind,:], color = colorlist[colorInd] , linewidth = linewidth, linestyle = '-', zorder = zorder)
#            plt.plot(bandCut[ind,:], '.')
        plt.title('some momentum cut')
        plt.ylabel('Energy')
        plt.xlabel('k_something')
    
    def plot_bloch_wave(self, state_vect, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', zorder = 1):
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
        
        plt.sca(ax)
        plt.scatter(self.SDx, self.SDy,c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            print('making colorbar')
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax, linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        plt.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return
    
    def plot_bloch_wave_end_state(self, state_vect, ax, modeType, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', scaleFactor = 0.5, zorder = 1):
        '''
        plot a state (wavefunction) on the graph of semidual points, but with a 
        value plotted for each end of the resonator
        
        If you just want a single value for the resonator use plot_layout_state
        
        Takes states defined on only one end of each resonator. Will autogenerate 
        the value on other end based on mode type.
        
        
        SOMETHING may be hinky with the range and flipping the sign
        
        '''
        Amps = state_vect
        Probs = np.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = np.angle(Amps)/np.pi
        
        #build full state with value on both ends of the resonators
        mSizes_end = np.zeros(len(Amps)*2)
        mSizes_end[0::2] = mSizes
        mSizes_end[1::2] = mSizes
        
        mColors_end = np.zeros(len(Amps)*2)
        mColors_end[0::2] = mColors
        if modeType == 'FW':
            mColors_end[1::2] = mColors
        elif modeType == 'HW':
            #put opposite phase on other side
            oppositeCols = mColors + 1
            #rectify the phases back to between -0.5 and 1.5 pi radians
            overflow = np.where(oppositeCols > 1.5)[0]
            newCols = oppositeCols
            newCols[overflow] = oppositeCols[overflow] - 2
            
            mColors_end[1::2] = newCols
        else:
            raise ValueError('You screwed around with the mode type. It must be FW or HW.')
        
        cm = plt.cm.get_cmap(cmap)
        
        #get coordinates for the two ends of the resonator
        plotPoints = self._get_orientation_plot_points(scaleFactor = scaleFactor)
        xs = plotPoints[:,0]
        ys = plotPoints[:,1]
        
        plt.sca(ax)
        plt.scatter(xs, ys,c =  mColors_end, s = mSizes_end, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            print('making colorbar')
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax,linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        plt.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        
#        return plotPoints
        return mColors
    
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
            xend = oldRes[2]
            yend = oldRes[3]
            
            xs = np.linspace(xstart, xend, splitIn+1)
            ys = np.linspace(ystart, yend, splitIn+1)
            for sind in range(0, splitIn):
                newResonators[splitIn*rind + sind,:] = [xs[sind], ys[sind], xs[sind+1], ys[sind+1]]
                
        newCell = UnitCell(name, 
                           resonators = newResonators, 
                           a1 = self.a1, 
                           a2 = self.a2, 
                           maxDegree = self.maxDegree)
        return newCell
    
    def line_graph_cell(self, name = 'TBD', resonatorsOnly = False):
        newResonators = np.zeros((self.SDHWlinks.shape[0], 4))
        
        for lind in range(0, self.SDHWlinks.shape[0]):
            link = self.SDHWlinks[lind,:]
            startInd = int(link[0]) #within the unit cell
            targetInd = int(link[1])
            
            deltaA1 = int(link[2])
            deltaA2   = int(link[3])
            
            startPol = int(link[4])
            targetPol = int(link[5])
            
            if (deltaA1,deltaA2) == (-1,1):
#                print 'skipping -1,1'
                pass
            elif (deltaA1,deltaA2) == (-1,0):
#                print 'skipping -1,0'
                pass
            elif (deltaA1,deltaA2) == (-1,-1):
#                print 'skipping -1,-1'
                pass
            elif (deltaA1,deltaA2) == (0,-1):
#                print 'skipping 0,-1'
                pass
            else:
                if (deltaA1,deltaA2) == (0,0) and  startInd > targetInd:
                    pass
                    #don't want to double count going the other way within the cell
                    #links to neighboring cells won't get double counted in this same way
                else:
                    #corrdiates of origin site
                    x0 = self.SDx[startInd]
                    y0 = self.SDy[startInd]
                    
                    #coordinates of target site
                    x1 = self.SDx[targetInd] + deltaA1*self.a1[0] + deltaA2*self.a2[0]
                    y1 = self.SDy[targetInd] + deltaA1*self.a1[1] + deltaA2*self.a2[1]
                    
                    res = np.asarray([x0, y0, x1, y1])
                    newResonators[lind, :] = res
                    
        
        #clean out balnk rows that were for redundant resonators
        newResonators = newResonators[~np.all(newResonators == 0, axis=1)]  

        if resonatorsOnly:
            return newResonators
        else:
            newCell = UnitCell(name, 
                               resonators = newResonators, 
                               a1 = self.a1, 
                               a2 = self.a2,
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
        svec_all = allCoords[:,0] + 1j*allCoords[:,1]
        
        
        
        def check_redundnacy(site, svec_all, shift1, shift2):
            vec1 = np.round(self.a1[0] + 1j*self.a1[1], roundDepth)
            vec2 = np.round(self.a2[0] + 1j*self.a2[1], roundDepth)
            shiftedCoords = svec_all + shift1*(vec1) + shift2*(vec2)
#            redundancies = np.where(np.round(site,roundDepth) == np.round(shiftedCoords,roundDepth))[0]
            redundancies = np.where(np.isclose(site,shiftedCoords, atol = 2*10**(-roundDepth)))[0] #rounding is causing issues. Hopefully this is better
            return redundancies
            
        
        #determine coordinate equivalences
        redundancyDict = {}
        for cind in range(0, allCoords.shape[0]):
            site = svec_all[cind] #the site to compare
            
            redundancyDict[cind] = []
            for shift1 in (-1,0,1):
                for shift2 in (-1,0,1):
                    redundancies = check_redundnacy(site, svec_all, shift1, shift2)
                    
                    if len(redundancies) > 0:
                        if not (shift1 ==0 and shift2 == 0):
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
        svec_all = allCoords[:,0] + 1j*allCoords[:,1]
        
        #get the coordinates of the minimum unit cell
        coords = np.round(self.rootCoords, roundDepth)
        svec = np.zeros((coords.shape[0]))*(1 + 1j)
        svec[:] = coords[:,0] + 1j*coords[:,1]
        
        #store away the resonators, which tell me about all possible links
        resonators = np.round(self.resonators, roundDepth)
        zmat = np.zeros((resonators.shape[0],2))*(1 + 1j)
        zmat[:,0] = resonators[:,0] + 1j*resonators[:,1]
        zmat[:,1] = resonators[:,2] + 1j*resonators[:,3]
    #    print zmat
        
        self.rootLinks = np.zeros((self.resonators.shape[0]*2,4))
        
        
        def check_cell_relation(site, svec, shift1, shift2):
            ''' check if a given point in a copy of the unit cell translated by
            shift1*a1 +shift2*a2'''
            vec1 = np.round(self.a1[0] + 1j*self.a1[1], roundDepth)
            vec2 = np.round(self.a2[0] + 1j*self.a2[1], roundDepth)
            shiftedCoords = svec + shift1*(vec1) + shift2*(vec2)
#            matches = np.where(np.round(site,roundDepth) == np.round(shiftedCoords,roundDepth))[0]
            matches = np.where(np.isclose(site,shiftedCoords, atol = 2*10**(-roundDepth)))[0] #rounding is causing issues. Hopefully this is better
            if len(matches)>0:
                return True
            else:
                return False
            
        def find_cell_relation(site, svec):
            ''' find out which translate of the unit cell a cite is in'''
            for da1 in range(-1,2):
                for da2 in range(-1,2):
                    match = check_cell_relation(site, svec, da1 , da2)
                    if match:
                        return da1, da2
            
            #raise a an error if no match found
            raise ValueError('not match found')


        lind = 0
        #convert the resonator matrix to links, basically fold it back to the unit cell
        for rind in range(0, resonators.shape[0]):
            #default to an internal link
            xpol = 0
            ypol = 0
            
            source = zmat[rind,0]
            target = zmat[rind,1]
            
            sourceInd = np.where(np.round(source,roundDepth) == np.round(svec_all,roundDepth))[0][0]
            targetInd = np.where(np.round(target,roundDepth) == np.round(svec_all,roundDepth))[0][0]
    
    
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
            pos0X, pos0Y = find_cell_relation(source, svec)
            pos1X, pos1Y = find_cell_relation(target, svec)
            
            xPol = pos1X - pos0X
            yPol = pos1Y - pos0Y
            
            self.rootLinks[lind,:] = [sourceMatInd, targetMatInd, xPol, yPol]
            self.rootLinks[lind+1,:] = [targetMatInd, sourceMatInd, -xPol, -yPol]
            lind = lind+2

        
#        #remove blank links (needed for some types of arbitrary cells)
#        self.rootlinks = self.rootlinks[~np.all(self.rootlinkss == 0, axis=1)] 
        
        return
    
    def generate_root_Bloch_matrix(self, kx, ky, t = 1, modeType = 'FW', phase = 0):
        ''' 
        generates a Bloch matrix for the root graphof the layout for a given kx and ky
        
        needs the root cell and its links to be found first
        
        
        11-23-21: rewrote this guy to directly use the root links.
        
        mode type argument is a hold-over from the line graph band structure and is
        ignored. Same for phase.
        
        '''
        BlochMat = np.zeros((self.numRootSites, self.numRootSites))*(0 + 0j)
        
        for lind in range(0, self.rootLinks.shape[0]):
            link = self.rootLinks[lind,:]
            startInd = int(link[0]) #within the unit cell
            targetInd = int(link[1])
            deltaA1 = int(link[2])
            deltaA2   = int(link[3])
            
            #corrdiates of origin site
            x0 = self.rootCoords[startInd,0]
            y0 = self.rootCoords[startInd,1]
            
            #coordinates of target site
            x1 = self.rootCoords[targetInd,0] + deltaA1*self.a1[0] + deltaA2*self.a2[0] 
            y1 = self.rootCoords[targetInd,1] + deltaA1*self.a1[1] + deltaA2*self.a2[1] 
            
            deltaX = x1-x0
            deltaY = y1-y0
            
            phaseFactor = np.exp(1j*kx*deltaX)*np.exp(1j*ky*deltaY)
            BlochMat[startInd, targetInd] = BlochMat[startInd, targetInd]+ t*phaseFactor
        return BlochMat
    
    def compute_root_band_structure(self, kx_0, ky_0, kx_1, ky_1, numsteps = 100, modeType = 'FW', returnStates = False, phase  = 0):
            '''
            computes a cut through the band structure of the root graph of the layout
            
            from scipy.linalg.eigh:
            The normalized selected eigenvector corresponding to the eigenvalue w[i] is the column v[:,i].
            
            This returns same format with two additional kx, ky indices
            '''
            
            kxs = np.linspace(kx_0, kx_1,numsteps)
            kys = np.linspace(ky_0, ky_1,numsteps)
            
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
                kvec = [kxs[ind],kys[ind]]
                
                H = self.generate_root_Bloch_matrix(kvec[0], kvec[1], modeType = modeType, phase  = phase)
            
                #Psis = np.zeros((self.numSites, self.numSites)).astype('complex')
                Es, Psis = scipy.linalg.eigh(H)
                
                bandCut[:,ind] = Es
                stateCut[:,:,ind] = Psis
            if returnStates:
                return kxs, kys, bandCut, stateCut
            else:
                return kxs, kys, bandCut
    
    def plot_root_bloch_wave(self, state_vect, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', zorder = 1):
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
        
#        print mColors
        
        cm = plt.cm.get_cmap(cmap)
        
        plt.sca(ax)
        plt.scatter(self.rootCoords[:,0], self.rootCoords[:,1],c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            print( 'making colorbar')
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax, linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        plt.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return        
 