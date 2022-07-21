import pickle
import scipy
import numpy as np
import matplotlib.pyplot as plt
from .resonator_utility import shift_resonators, rotate_resonators

"""
TreeResonators Class
    generates resoantors that form a regular tree of a certain degree
    
    v0 - self.coords is wierd, and may not contain all the capacitor points
     
     Methods:
        ###########
        #automated construction, saving, loading
        ##########
        save
        load
        
        ########
        #functions to generate the resonator lattice
        #######
        generate_lattice
         
        #######
        #resonator lattice get /view functions
        #######
        get_xs
        get_ys
        draw_resonator_lattice
        draw_resonator_end_points
        get_all_resonators
        get_coords

    Sample syntax:
        #####
        #loading precalculated resonator config
        #####
        from GeneralLayoutGenerator import TreeResonators
        testTree = TreeResonators(file_path = 'name.pkl')

        #####
        #making new layout
        #####
        from GeneralLayoutGenerator import TreeResonators
        Tree = TreeResonators(degree = 3, iterations = 3, side = 1, file_path = '', modeType = 'FW')
"""

    
class TreeResonators(object):
    def __init__(self, isRegular = True, degree = 3, iterations = 3, side = 1, file_path = '', modeType = 'FW', cell = '', roundDepth = 3):
        if file_path != '':
            self.load(file_path)
        else:
            #create plank planar layout object with the bare bones that you can build on
            self.isRegular = isRegular
            self.degree = degree
            self.side = side*1.0
            self.iterations = iterations
            
            self.modeType = modeType
            
            self.cell = cell #type of thing to be treed
            
            self.roundDepth = roundDepth

            if self.cell == '':
                self.generate_lattice_basic()
            else:
                self.generate_lattice()
            
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
            name = str(self.degree) + 'regularTree_ ' + str(self.iterations) + '_' + waveStr + '.pkl'
        
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

    def generate_lattice_basic(self):
        maxItt = self.iterations

        self.resDict = {}
        
        totalSize = 0
        for itt in range(1, maxItt+1):
            radius = itt*self.side*1.0
            
            if itt==1:
                oldEnds = 1
                newEnds = self.degree
            else:
                #gather the uncapped ends
                oldRes = self.resDict[itt-1]
                oldEnds = oldRes.shape[0]
                newEnds = (self.degree-1)*oldEnds
            
#            thetas = scipy.arange(0,2*np.pi,2*np.pi/newEnds)
            thetas = scipy.arange(0,newEnds,1)*2*np.pi/newEnds
            
            
            if itt == 1:
                #first layer of the tree
                
                xs = radius * np.cos(thetas)
                ys = radius * np.sin(thetas)
                
                #no old resonators to start with, so make the first set
                newRes = np.zeros((newEnds, 4))
                for nrind in range(0, self.degree):
                    newRes[nrind,:] = [0, 0, xs[nrind],  ys[nrind]]
                    
                #store the newly created resonators
                self.resDict[itt] = newRes
                totalSize = totalSize + newEnds
            else:   
                #higher layer of the tree
                
                deltaTheta = thetas[1] - thetas[0]
                
                endInd = 0 #index of the old uncapped ends
                newRes = np.zeros((newEnds, 4))
                for orind in range(0, oldEnds):
                    #starting point for the new resonators
                    xstart = oldRes[orind,2]
                    ystart = oldRes[orind,3]
                    oldTheta = np.arctan2(ystart, xstart)
                    
                    #loop over teh resonators that need to be attached to each old end
                    for nrind in range(0, self.degree-1):
                        newTheta = oldTheta + deltaTheta*(nrind - (self.degree-2)/2.)
                        
                        xend = radius*np.cos(newTheta)
                        yend = radius*np.sin(newTheta)
                        newRes[endInd,:] = [xstart, ystart, xend,  yend]
                        endInd = endInd +1
                self.resDict[itt] = newRes
                totalSize = totalSize + newEnds
                 
        #shuffle resoantor dictionary into an array                       
        self.resonators = np.zeros((totalSize, 4))   
        currRes = 0
        for itt in range(1, maxItt+1):
            news = self.resDict[itt]
            numNews = news.shape[0]
            self.resonators[currRes:currRes+numNews,:] = news
            currRes = currRes + numNews
            
        
        self.coords = self.get_coords(self.resonators)
        
    def generate_lattice(self):
        maxItt = self.iterations

        self.resDict = {}
        
        if self.cell == 'Peter':
            self.cellSites = 7
            self.cellResonators = np.zeros((7,4))
            #set up the poisitions of all the resonators  and their end points
        
            a = self.side/(2*np.sqrt(2) + 2)
            b = np.sqrt(2)*a
            #xo,yo,x1,y1
            #define them so their orientation matches the chosen one. First entry is plus end, second is minus
            tempRes = np.zeros((7,4))
            tempRes[0,:] = [-a-b, 0, -b,  0]
            tempRes[1,:] = [-b, 0, 0,  b]
            tempRes[2,:] = [0, b, b,  0]
            tempRes[3,:] = [-b, 0, 0,  -b]
            tempRes[4,:] = [0, -b, b,  0]
            tempRes[5,:] = [0, -b, 0,  b]
            tempRes[6,:] = [b, 0, a+b, 0]
            
            self.cellResonators = shift_resonators(tempRes, self.side/2,0) #now one end of the cell is at zeo and the other at [self.side,0]
#            self.cellResonators = tempRes
        
        totalSize = 0
        for itt in range(1, maxItt+1):
            radius = itt*self.side*1.0
            
            if itt==1:
                oldEnds = 1
                newEnds = self.degree
            else:
                #gather the uncapped ends
                oldRes = self.resDict[itt-1]
                oldEnds = oldRes.shape[0]/self.cellSites
                #oldEndThetas = self.side*(itt-1)*scipy.arange(0,2*np.pi,2*np.pi/oldEnds)
                newEnds = (self.degree-1)*oldEnds
            
            thetas = scipy.arange(0,2*np.pi,2*np.pi/newEnds)
            
            
            if itt == 1:
                #first layer of the tree
                
                #no old resonators to start with, so make the first set
                newRes = np.zeros((newEnds*self.cellSites, 4))
                for cind in range(0, self.degree):
                    newRes[cind*self.cellSites:(cind+1)*self.cellSites,:] = rotate_resonators(self.cellResonators,thetas[cind])
                    
                #store the newly created resonators
                self.resDict[itt] = newRes
                totalSize = totalSize + newEnds
                
                #store the polar coordinates of the end points
                oldEndThetas = thetas
            else:   
                #higher layer of the tree
                
                deltaTheta = thetas[1] - thetas[0]
                
                endInd = 0 #index of the old uncapped ends
                newRes = np.zeros((newEnds*self.cellSites, 4))
                newEndThetas = np.zeros(newEnds) #place to store polar coordinates of the new end points
                for orind in range(0, oldEnds):
                    #starting point for the new resonators
                    xstart = self.side*(itt-1)*np.cos(oldEndThetas[orind])
                    ystart = self.side*(itt-1)*np.sin(oldEndThetas[orind])
                    oldTheta = np.arctan2(ystart, xstart)
                    
                    #loop over the cells that need to be attached to each old end
                    for cind in range(0, self.degree-1):
                        newTheta = oldTheta + deltaTheta*(cind - (self.degree-2)/2.) #polar coordinate of the end point
                        
                        xend = radius*np.cos(newTheta)
                        yend = radius*np.sin(newTheta)
                        
                        armLength = np.sqrt((xend-xstart)**2 + (yend-ystart)**2) #length that the cell has to fit in
                        armTheta = np.arctan2(yend-ystart, xend-xstart) #angle that the cell has to be at
                        
                        tempRes = np.copy(self.cellResonators)
                        tempRes = tempRes*armLength/self.side #rescale to the right length
                        tempRes = rotate_resonators(tempRes, armTheta) #rotate into poition
                        tempRes = shift_resonators(tempRes, xstart,ystart) #shift into position
                        
                        #store them away
                        newRes[endInd:endInd+1*self.cellSites,:] = tempRes
                        newEndThetas[endInd/self.cellSites] = newTheta #store the absolute polar coorinate of this arm
                        endInd = endInd +self.cellSites
                self.resDict[itt] = newRes
                totalSize = totalSize + newEnds
                oldEndThetas = newEndThetas 
                 
        #shuffle resoantor dictionary into an array                       
        self.resonators = np.zeros((totalSize*self.cellSites, 4))   
        currRes = 0
        for itt in range(1, maxItt+1):
            news = self.resDict[itt]
            numNews = news.shape[0]
            self.resonators[currRes:currRes+numNews,:] = news
            currRes = currRes + numNews
            
        
        self.coords = self.get_coords(self.resonators)

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

    def get_all_resonators(self, maxItter = -1):
        '''
        function to get all resonators as a pair of end points
        '''
        return self.resonators
    
    def get_coords(self, resonators):
        '''
        take in a set of resonators and calculate the set of end points.
        
        Will round all coordinates the the specified number of decimals.
        
        Should remove all redundancies.
        '''
        
        coords_overcomplete = np.zeros((resonators.shape[0]*2, 1)).astype('complex')
        coords_overcomplete =  np.concatenate((resonators[:,0], resonators[:,2])) + 1j * np.concatenate((resonators[:,1], resonators[:,3]))
        
        coords_complex = np.unique(np.round(coords_overcomplete, self.roundDepth))
    
        coords = np.zeros((coords_complex.shape[0],2))
        coords[:,0] = np.real(coords_complex)
        coords[:,1] = np.imag(coords_complex)
        
        return coords
    
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
        
        plt.sca(ax)
        plt.scatter(x0s, y0s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder, alpha = alpha)
        plt.scatter(x1s, y1s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder, alpha = alpha)
        return  

    
 