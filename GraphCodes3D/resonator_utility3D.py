import numpy as np


    
#######
#resonator array processing functions
#######
    

# def split_resonators(resMat, splitIn = 2):
#     '''take in a matrix of resonators, and split them all in half.
#     Return the new resonators
#     (for use in making things like the McLaughlin graph)
    
#     set SplitIn > 2 to split the resonators in more than just half
#     '''
#     oldNum = resMat.shape[0]
    
#     if type(splitIn) != int:
#         raise ValueError('need an integer split')
#     newNum = oldNum*splitIn
    
#     newResonators = np.zeros((newNum,4))
    
#     for rind in range(0, oldNum):
#         oldRes = resMat[rind,:]
#         xstart = oldRes[0]
#         ystart = oldRes[1]
#         xend = oldRes[2]
#         yend = oldRes[3]
        
#         xs = np.linspace(xstart, xend, splitIn+1)
#         ys = np.linspace(ystart, yend, splitIn+1)
#         for sind in range(0, splitIn):
#             newResonators[splitIn*rind + sind,:] = [xs[sind], ys[sind], xs[sind+1], ys[sind+1]]
# #            newResonators[2*rind+1,:] = [xmid, ymid, xend, yend]
         
#     return newResonators
    
    
# def generate_line_graph(resMat, roundDepth = 3):
#     '''
#         function to autogenerate the links between a set of resonators and itself
#         will calculate a matrix of all the links [start, target, start_polarity, end_polarity]
        
#         then use that to make new resonators that consitute the line graph
#         '''


#     ress1 = resMat
#     len1 = ress1.shape[0]
    
#     ress2 = resMat

#     #place to store the links
#     linkMat = np.zeros((len1*4+len1*4,4))
    
#     #find the links
    
#     #round the coordinates to prevent stupid mistakes in finding the connections
#     plusEnds = np.round(ress2[:,0:2],roundDepth)
#     minusEnds = np.round(ress2[:,2:4],roundDepth)
    
#     extraLinkInd = 0
#     for resInd in range(0,ress1.shape[0]):
#         res = np.round(ress1[resInd,:],roundDepth)
#         x1 = res[0]
#         y1 = res[1]
#         x0 = res[2]
#         y0 = res[3]

#         plusPlus = np.where((plusEnds == (x1, y1)).all(axis=1))[0]
#         minusMinus = np.where((minusEnds == (x0, y0)).all(axis=1))[0]
        
#         plusMinus = np.where((minusEnds == (x1, y1)).all(axis=1))[0] #plus end of new res, minus end of old
#         minusPlus = np.where((plusEnds == (x0, y0)).all(axis=1))[0]
        
#         for ind in plusPlus:
#             if ind == resInd:
#                 #self link
#                 pass
#             else:
#                 linkMat[extraLinkInd,:] = [resInd, ind, 1,1]
#                 extraLinkInd = extraLinkInd+1
                
#         for ind in minusMinus:
#             if ind == resInd:
#                 #self link
#                 pass
#             else:
#                 linkMat[extraLinkInd,:] = [resInd, ind, 0,0]
#                 extraLinkInd = extraLinkInd+1
                
#         for ind in plusMinus:
#             if ind == resInd: #this is a self loop edge
#                 linkMat[extraLinkInd,:] = [resInd, ind,  1,0]
#                 extraLinkInd = extraLinkInd+1
#             elif ind in plusPlus: #don't double count if you hit a self loop edge 
#                 pass 
#             elif ind in minusMinus:
#                 pass 
#             else:
#                 linkMat[extraLinkInd,:] = [resInd, ind,  1,0]
#                 extraLinkInd = extraLinkInd+1
            
#         for ind in minusPlus:
#             if ind == resInd:#this is a self loop edge
#                 linkMat[extraLinkInd,:] = [ resInd, ind,  0,1]
#                 extraLinkInd = extraLinkInd+1
#             elif ind in plusPlus: #don't double count if you hit a self loop edge 
#                 pass 
#             elif ind in minusMinus:
#                 pass 
#             else:
#                 linkMat[extraLinkInd,:] = [ resInd, ind,  0,1]
#                 extraLinkInd = extraLinkInd+1
    
#     #clean the skipped links away 
#     linkMat = linkMat[~np.all(linkMat == 0, axis=1)]  
    
#     newNum = linkMat.shape[0]/2 #number of resonators in the line graph
#     newResonators = np.zeros((int(newNum), 4))
    
    

#     xs = np.zeros(resMat.shape[0])
#     ys = np.zeros(resMat.shape[0])
#     for rind in range(0, resMat.shape[0]):
#         res = resMat[rind,:]
#         xs[rind] = (res[0] + res[2])/2
#         ys[rind] = (res[1] + res[3])/2
#     SDx = xs
#     SDy = ys
    
#     #process into a Hamiltonian because it's a little friendlier to read from and doesn't double count
#     totalSize = len(SDx)
#     H = np.zeros((totalSize, totalSize))
#     #loop over the links and fill the Hamiltonian
#     for link in range(0, linkMat.shape[0]):
#         [sourceInd, targetInd] = linkMat[link, 0:2]
#         source = int(sourceInd)
#         target = int(targetInd)
#         H[source, target] = 1
        
#     #loop over one half of the Hamiltonian
#     rind = 0
#     for sind in range(0, totalSize):
#         for tind in range(sind+1,totalSize):
#             if H[sind,tind] == 0:
#                 #no connections
#                 pass
#             else:
#                 #sites are connected. Need to make a resoantors
#                 newResonators[rind,:] = np.asarray([SDx[sind], SDy[sind], SDx[tind], SDy[tind]])
#                 rind = rind+1
    
#     return newResonators  

def shift_resonators(resonators, dx, dy, dz):
    '''
    take array of resonators and shfit them by dx inthe x direction and dy in the y direction
    
    returns modified resonators
    '''
    newResonators = np.zeros(resonators.shape)
    
    newResonators[:,0] = resonators[:,0] + dx
    newResonators[:,1] = resonators[:,1] + dy
    newResonators[:,2] = resonators[:,2] + dz
    newResonators[:,3] = resonators[:,3] + dx
    newResonators[:,4] = resonators[:,4] + dy
    newResonators[:,5] = resonators[:,5] + dz
    
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

# def decorate_layout(layoutResonators, cellResonators):
#     '''
#     Take a layout and decorate each resonator in it with a cell of resonators.
    
#     NOTE: cell must run between (-1/2,0) and (1/2,0) otherwise this will give garbage
#     '''
#     oldRes = layoutResonators.shape[0]
#     cellSites = cellResonators.shape[0]
#     newResonators = np.zeros((oldRes*cellSites,4))
    
#     for rind in range(0, oldRes):
#         [xstart,ystart, xend, yend] = layoutResonators[rind,:]
        
#         armLength = np.sqrt((xend-xstart)**2 + (yend-ystart)**2) #length that the cell has to fit in
#         armTheta = np.arctan2(yend-ystart, xend-xstart) #angle that the cell has to be at
        
#         tempRes = np.copy(cellResonators)
#         tempRes = shift_resonators(cellResonators, 0.5,0)
#         tempRes = tempRes*armLength #rescale to the right length
#         tempRes = rotate_resonators(tempRes, armTheta) #rotate into poition
#         tempRes = shift_resonators(tempRes, xstart,ystart) #shift into position
        
#         #store them away
#         newResonators[rind*cellSites:(rind+1)*cellSites,:] = tempRes
    
#     return newResonators

def get_coords(resonators, roundDepth = 3):
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

def draw_resonators(resMat, ax, theta = np.pi/10, phi = np.pi/10, 
                        color = 'g', 
                        alpha = 1 , 
                        linewidth = 0.5, 
                        zorder = 1):
        '''
        draw each resonator as a line, x-z projection only
        '''
        plotRes = rotate_resonators(resMat, theta, phi)
        for res in range(0,plotRes.shape[0] ):
            [x0, y0,z0, x1, y1,z1]  = plotRes[res,:]
            #plot the x, z projection
            ax.plot([x0, x1],[z0, z1] , color = color, alpha = alpha, linewidth = linewidth, zorder = zorder)
        return


  
    
  