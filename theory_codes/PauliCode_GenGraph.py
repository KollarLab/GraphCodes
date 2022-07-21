#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:12:20 2021


4-21-22 Forked from PauliCode.py to be modified so that it works on general layout
        objects and not just Euclidean lattices. Will ry to inherit as much as possible
        from Pauli Code

@author: kollar2
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
from flint import nmod_mat
#https://fredrikj.net/python-flint/nmod_mat.html
#conda install -c conda-forge python-flint
#det
#modulus
#ncols
#nrows
#nullspace
#rank
#rref (reduced row echeleon)
#solve
#transpose
#table


KollarLabClassPath = r'/Users/kollar2/Documents/KollarLab/MainClasses/GraphCodes'
if not KollarLabClassPath in sys.path:
    sys.path.append(KollarLabClassPath)

    
from CDSconfig import CDSconfig


   
from GeneralLayoutGenerator import GeneralLayout
from GeneralLayoutGenerator import TreeResonators

from EuclideanLayoutGenerator2 import UnitCell
from EuclideanLayoutGenerator2 import EuclideanLayout

from LayoutGenerator5 import PlanarLayout


from GeneralLayoutGenerator import split_resonators
from GeneralLayoutGenerator import rotate_resonators
from GeneralLayoutGenerator import generate_line_graph
from GeneralLayoutGenerator import shift_resonators

from PauliCode import PauliCode



#############
#color defaults
##########

#colorList = ['firebrick', 'darkgoldenrod', 'dodgerblue', 'forestgreen', 'k', 'grey', 'turquoise', 'orange']
colorList = ['firebrick', 'darkgoldenrod', 'dodgerblue', 'forestgreen', 'grey','turquoise', 'k', 'orange']
colorList2 = ['firebrick', 'darkgoldenrod', 'dodgerblue', 'orange', 'forestgreen', 'grey','turquoise', 'k', 'orange']
heights = [-0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

    
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
  
heavytext_params = {'ha': 'center', 'va': 'center', 'family': 'calibri',
               'fontweight': 'bold'}
lighttext_params = {'ha': 'center', 'va': 'center', 'family': 'calibri',
               'fontweight': 'regular'}
smallTextSize = 11
textSize = 12
labelSize = 14






# class PauliCode_GenGraph(object):
class PauliCode_GenGraph(PauliCode):
    def __init__(self, graph, name = '', 
                 bosonization_type = 'edge',
                 fiducial = True,
                 generate_dimers = False,
                 verbose = False, 
                 file_path = ''):
        '''
        
        '''
        
        if file_path != '':
            self.load(file_path)
        else:
        
            self.name = name
            self.verbose = verbose
            
            if fiducial == True:
                self.fiducial = True
                
                bosonization_type = 'edge'
            else:
                self.fiducial = False
            
            
            if bosonization_type == 'edge':
                self.bosonization_type = 'edge'
            elif bosonization_type == 'vertex':
                self.bosonization_type = 'vertex'
            else:
                raise(ValueError, 'unknown bosonization type')
            
            # self.unitcell = unitcell
            # self.unitcell.find_root_cell()
            self.lattice = graph
            
            try:
                self.lattice.SDx[0]
            except:
                self.lattice.generate_semiduals()
                
            try:
                self.lattice.rootLinks[0,0]
            except:
                self.lattice.generate_root_graph()
        
            self.maxDegree = self.lattice.maxDegree #manually input parameter that is the
            #maximum degree of the root graph. 
            #default is set to 4 so it will automatically allocate enough space
            #for the link matrices. For higher-degree graphs, you will have to 
            #tell it to allocate more space

            
            self.numVertices = self.lattice.rootCoords.shape[0]
            self.numEdges = self.lattice.SDx.shape[0]
            
            if self.bosonization_type == 'edge':
                self.numQubits = self.numEdges
            elif self.bosonization_type == 'vertex':
                self.numQubits = self.numVertices
            
            
            ####
            #lets get the positions
            #####
            if self.bosonization_type == 'edge':
                self.Xs = self.lattice.SDx
                self.Ys = self.lattice.SDy
            if self.bosonization_type == 'vertex':
                self.Xs = self.lattice.rootCoords[:,0]
                self.Ys = self.lattice.rootCoords[:,1]
                
            # if self.bosonization_type == 'edge':
            #     self.cellXs = self.lattice.SDx
            #     self.cellYs = self.lattice.SDy
            # if self.bosonization_type == 'vertex':
            #     self.cellXs = self.lattice.rootCoords[:,0]
            #     self.cellYs = self.lattice.rootCoords[:,1]
            
            
            # ####
            # #now to fill in the on the torus stuff
            # ####
            # if self.bosonization_type == 'edge':
            #     self.torusXs = self.lattice.SDx
            #     self.torusYs = self.lattice.SDy
            # if self.bosonization_type == 'vertex':
            #     self.make_torus_vertices()
            
            
            #####
            #compute the incidence matrix of the unit cell
            #####
            self.make_incidence_matrix()
            self.make_edge_incidence_matrix()
            
            
            #####
            # compute the fiducial Hamiltonian and the dimers
            #####
            
            
            if self.fiducial == True:
                
                self.fiducialH = self.make_fiducial_H(check_plot = self.verbose)
                
                if generate_dimers:
                    self.fiducialDimers = self.make_fiducial_dimers(check_plot = self.verbose)
                
                self.H0 = self.fiducialH
                
                # self.H_torus = self.unpack_to_torus(self.H0)
            
           
    #######
    #saving and loading functions
    #######
    
    # def save(self, name = ''):
    #     '''
    #     save structure to a pickle file
        
    #     if name is blank, will use dafualt name
    #     '''
            
    #     if self.fiducial:
    #         fidStr = '_fiducial'
    #     else:
    #         fidStr = ''
    #     if name == '':
    #         name = self.name + fidStr + '.pkl'
        
    #     savedict = self.__dict__
    #     pickle.dump(savedict, open(name, 'wb'))
    #     return
    
    # def load(self, file_path):
    #     '''
    #     laod structure from pickle file
    #     '''
    #     pickledict = pickle.load(open(file_path, "rb" ) )
        
    #     for key in pickledict.keys():
    #         setattr(self, key, pickledict[key])
           
    #     return

    # ######
    # #matrix type casters to go between the various type of matrices
    # ######
    # def convert_np_to_list(self, mat):
    #     '''cast an integer-valued numpy array to a single 1D list
        
    #     This is a prestep to passing to nmod_mat
    #     '''
    #     if len(mat.shape) == 1:
    #         #this is a vector
    #         vec = mat
    #     else:
    #         vec = mat.reshape(mat.shape[0]*mat.shape[1])
    #     newList = []
    #     for vind in range(0,len(vec)):
    #         entry = int(vec[vind])
    #         newList.append(entry)
    #     return newList
            
    # def convert_list_to_np(self, oldList, shape0, shape1):
    #     '''covert flattened integer list of entires back into a
    #     2D numpy array'''
    #     vec = numpy.asarray(oldList)
    #     mat = vec.reshape(shape0, shape1)
    #     return mat
    
    # def convert_nmod_to_np(self, modmat):
    #     '''Goes through the stupid hassle of converting an nmod_mat
    #     back into a standard numpy array'''
    #     shape0 = int(modmat.nrows())
    #     shape1 = int(modmat.ncols())
    #     npmat = numpy.zeros((shape0, shape1))
    #     for nind in range(0,shape0):
    #         for mind in range(0, shape1):
    #             npmat[nind, mind] = int(modmat[nind, mind])
    #     return npmat
    
    # def convert_np_to_nmod(self, npmat, modulus = 2):
    #     '''wrapper to produce the corresponding nmod_mat
    #     from the 2D numpy array directly
        
    #     defualts to mod 2
    #     '''
    #     newList = self.convert_np_to_list(npmat)
    #     if len(npmat.shape) == 1:
    #         #this is a vector
    #         shape0 = npmat.shape[0]
    #         shape1 = 1
    #     else:
    #         shape0 = npmat.shape[0]
    #         shape1 = npmat.shape[1]
    #     modmat = nmod_mat(shape0, shape1, newList, modulus)
    #     return modmat
    
    # def convert_string_to_Haah(self,inputThing):
    #     if type(inputThing) == numpy.ndarray:
    #         oldMat = inputThing
    #         numQubits = oldMat.shape[0]
    #         numTerms = oldMat.shape[1]
            
    #         newMat = numpy.zeros((numQubits*2, numTerms))
            
    #         for qind in range(0, numQubits):
    #             for tind in range(0, numTerms):
    #                 if oldMat[qind,tind] == 'I':
    #                     newMat[2*qind, tind] = 0
    #                     newMat[2*qind+1, tind] = 0
    #                 if oldMat[qind,tind] == 'X':
    #                     newMat[2*qind, tind] = 1
    #                     newMat[2*qind+1, tind] = 0
    #                 if oldMat[qind,tind] == 'Y':
    #                     newMat[2*qind, tind] = 1
    #                     newMat[2*qind+1, tind] = 1
    #                 if oldMat[qind,tind] == 'Z':
    #                     newMat[2*qind, tind] = 0
    #                     newMat[2*qind+1, tind] = 1
    #         return newMat
        
    #     elif type(inputThing) == dict:
    #         pauliDict = inputThing
    #         newDict = {}
            
    #         numQubits = pauliDict[(0,0)].shape[0]
    #         numTerms = pauliDict[(0,0)].shape[1]
    #         for key in pauliDict.keys():
    #             oldMat = pauliDict[key]
                
    #             newMat = numpy.zeros((numQubits*2, numTerms))
                
    #             for qind in range(0, numQubits):
    #                 for tind in range(0, numTerms):
    #                     if oldMat[qind,tind] == 'I':
    #                         newMat[2*qind, tind] = 0
    #                         newMat[2*qind+1, tind] = 0
    #                     if oldMat[qind,tind] == 'X':
    #                         newMat[2*qind, tind] = 1
    #                         newMat[2*qind+1, tind] = 0
    #                     if oldMat[qind,tind] == 'Y':
    #                         newMat[2*qind, tind] = 1
    #                         newMat[2*qind+1, tind] = 1
    #                     if oldMat[qind,tind] == 'Z':
    #                         newMat[2*qind, tind] = 0
    #                         newMat[2*qind+1, tind] = 1
    #             newDict[key] = newMat
    #         return newDict
    #     else:
    #         raise ValueError('invalid input type. should be numpy array or dictionary')
    
    # def convert_Haah_to_string(self, inputThing):
    #     if type(inputThing) == numpy.ndarray:
    #         oldMat = inputThing
            
    #         numQubits = int(oldMat.shape[0]/2)
    #         numTerms = oldMat.shape[1]
            
    #         newMat = numpy.zeros((numQubits, numTerms), dtype = 'U21')
    #         for qind in range(0, numQubits):
    #             for tind in range(0, numTerms):
    #                 e0 = oldMat[2*qind, tind]
    #                 e1 = oldMat[2*qind+1, tind]
                    
    #                 if (e0 == 0) and (e1 == 0):
    #                     newMat[qind, tind] = 'I'
    #                 if (e0 == 1) and (e1 == 0):
    #                     newMat[qind, tind] = 'X'
    #                 if (e0 == 1) and (e1 == 1):
    #                     newMat[qind, tind] = 'Y'
    #                 if (e0 == 0) and (e1 == 1):
    #                     newMat[qind, tind] = 'Z'
    #         return newMat
            
    #     elif type(inputThing) == dict:
    #         HaahDict = inputThing
    #         newDict = {}
            
    #         numQubits = int(HaahDict[(0,0)].shape[0]/2)
    #         numTerms = HaahDict[(0,0)].shape[1]
                
    #         for key in HaahDict.keys():
    #             oldMat = HaahDict[key]
                
    #             newMat = numpy.zeros((numQubits, numTerms), dtype = 'U21')
    #             for qind in range(0, numQubits):
    #                 for tind in range(0, numTerms):
    #                     e0 = oldMat[2*qind, tind]
    #                     e1 = oldMat[2*qind+1, tind]
                        
    #                     if (e0 == 0) and (e1 == 0):
    #                         newMat[qind, tind] = 'I'
    #                     if (e0 == 1) and (e1 == 0):
    #                         newMat[qind, tind] = 'X'
    #                     if (e0 == 1) and (e1 == 1):
    #                         newMat[qind, tind] = 'Y'
    #                     if (e0 == 0) and (e1 == 1):
    #                         newMat[qind, tind] = 'Z'
    #             newDict[key] = newMat
    #         return newDict
    #     else:
    #         raise ValueError('invalid input. Should be np array or dict')
    
    # def combine_pauli_terms(self, dict1, dict2):
    #     ''' takes two pauli dictionaries in string form
    #     and combines them to make a larger one.
        
    #     intended for combining Hamiltonians with added stabilizers
    #     '''
        
    #     shape1 = dict1[(0,0)].shape
    #     shape2 = dict2[(0,0)].shape
        
    #     if not shape1[0] == shape2[0]:
    #         raise ValueError('cannot combine sets of operators on different numbers of qubits')
        
    #     newDict = {}
        
    #     for key in dict1.keys():
    #         if key in dict2.keys():
    #             #both of these guys have stuff in this unit cell
    #             newDict[key] = numpy.concatenate((dict1[key], dict2[key]), axis = 1)
                
    #         else:
    #             #dict2 doesn't have stuff in this unit cell
                
    #             temp2 = self.blank_pauli_mat(shape2[0], shape2[1])
    #             newDict[key] = numpy.concatenate((dict1[key], temp2), axis = 1)
                
    #     for key in dict2.keys():
    #         if not key in dict1.keys():
    #             #this guy is in dictionary 2, but not in dictionary 1, 
    #             #so I need to add it too
    #             temp1 = self.blank_pauli_mat(shape1[0], shape1[1])
    #             newDict[key] = numpy.concatenate((temp1, dict2[key]), axis = 1)
                
    #     return newDict  

    # def _multiply_pauli_matrices(self, vec1, vec2, strFlag):
    #     '''multiply two vectors that describe paulis
        
    #     must be of the same type, and function must be
    #     told what the type is (str v Haah)
        
    #     '''
    #     if not len(vec1) == len(vec2):
    #         raise ValueError('lengths do not match')
    #     else:
    #         if strFlag:
    #             newVec = numpy.zeros(len(vec1), dtype = 'U21')
    #             for ind in range(0,len(vec1)):
    #                 v1 = vec1[ind]
    #                 v2 = vec2[ind]
                    
    #                 if v1 == 'I':
    #                     if v2 == 'I':
    #                         newV = 'I'
    #                     elif v2 == 'X':
    #                         newV = 'X'
    #                     elif v2 == 'Y':
    #                         newV = 'Y'
    #                     else:
    #                         newV = 'Z'
    #                 elif v1 == 'X':
    #                     if v2 == 'I':
    #                         newV = 'X'
    #                     elif v2 == 'X':
    #                         newV = 'I'
    #                     elif v2 == 'Y':
    #                         newV = 'Z'
    #                     else:
    #                         newV = 'Y'
    #                 elif v1 == 'Y':
    #                     if v2 == 'I':
    #                         newV = 'Y'
    #                     elif v2 == 'X':
    #                         newV = 'Z'
    #                     elif v2 == 'Y':
    #                         newV = 'I'
    #                     else:
    #                         newV = 'X'
    #                 elif v1 == 'Z':
    #                     if v2 == 'I':
    #                         newV = 'Z'
    #                     elif v2 == 'X':
    #                         newV = 'Y'
    #                     elif v2 == 'Y':
    #                         newV = 'X'
    #                     else:
    #                         newV = 'I'
                    
    #                 newVec[ind] = newV
    #         else:
    #             newVec = numpy.zeros(len(vec1))
    #             newVec = numpy.mod(vec1+vec2, 2)
                
    #     return newVec
                        

    # def multiply_paulis(self, thing1, thing2):
    #     '''take in two representations of paulis and multiply them 
        
    #     first argument should be a single multi-qubit Pauli
    #     second can be multiple and it would multiply them all
    #     with the first argument
        
    #     first argument must be a collumn vector
        
    #     '''
        
    #     #figure out what the input things are
        
    #     dictFlag1 = False
    #     strFlag1 = False
    #     HaahFlag1 = False
    #     if type(thing1) == dict:
    #         dictFlag1 = True
            
    #         shape1 = thing1[(0,0)].shape
            
    #         if (type(thing1[(0,0)][0,0]) == numpy.str_):
    #             strFlag1 = True
    #         else:
    #             HaahFlag1 = True
    #     else:
    #         shape1 = thing1.shape
            
    #         if len(shape1)>1:
    #             if type(thing1[0,0]) == numpy.str_:
    #                 strFlag1 = True
    #             else:
    #                 HaahFlag1 = True
    #         else:
    #             if type(thing1[0]) == numpy.str_:
    #                 strFlag1 = True
    #             else:
    #                 HaahFlag1 = True
                    
    #     dictFlag2 = False
    #     strFlag2 = False
    #     HaahFlag2 = False
    #     if type(thing2) == dict:
    #         dictFlag2 = True
            
    #         shape2 = thing2[(0,0)].shape
            
    #         if (type(thing2[(0,0)][0,0]) == numpy.str_):
    #             strFlag2 = True
    #         else:
    #             HaahFlag2 = True
    #     else:
    #         shape2 = thing2.shape
            
    #         if len(shape2)>1:
    #             if type(thing2[0,0]) == numpy.str_:
    #                 strFlag2 = True
    #             else:
    #                 HaahFlag2 = True
    #         else:
    #             if type(thing2[0]) == numpy.str_:
    #                 strFlag2 = True
    #             else:
    #                 HaahFlag2 = True
                    
    #     if not dictFlag1 == dictFlag2:
    #         raise ValueError('incompatible types: not both dictionaries')
            
    #     if not strFlag1 == strFlag2:
    #         raise ValueError('incompatible types: not both str type storage')
            
    #     if not shape1[0] == shape2[0]:
    #         raise ValueError('incompatible types: unequal numbers of qubits')
                
    #     if shape1[1]>1:
    #         raise ValueError('this function cannot multiple two matrices, only a vector and a matrix')
        
    #     if dictFlag1:
    #         #do the multiplication for dictionaries
    #         newDict= {}
    #         for key1 in thing1.keys():
    #             if key1 in thing2.keys():
    #                 #both of these dictionaries have this monomial, so I need to multiply
    #                 vec1 = thing1[key1][:,0]
    #                 mat2 = thing2[key1]
    #                 if strFlag1:
    #                     newMat = numpy.zeros(shape2, dtype = 'U21')
    #                 else:
    #                     newMat = newMat = numpy.zeros(shape2)
                        
    #                 for ind in range(0, shape2[1]):
    #                     vec2 = mat2[:,ind]
    #                     newVec = self._multiply_pauli_matrices(vec1,vec2, strFlag1)
    #                     newMat[:,ind] = newVec
                    
    #                 newDict[key1] = newMat
    #             else:
    #                 #only dict 1 has this guy
    #                 newDict[key1] = thing1[key1]
    #         for key2 in thing2.keys():
    #             if key2 in thing1.keys():
    #                 #already got this guy
    #                 pass
    #             else:
    #                 #only dict 2 has this guy
    #                 newDict[key2] = thing2[key2]
    #         output = newDict
    #     else:
    #         #these are matrices
    #         vec1 = thing1[:,0]    
    #         mat2 = thing2
            
    #         if strFlag1:
    #             newMat = numpy.zeros(shape2, dtype = 'U21')
    #         else:
    #             newMat = newMat = numpy.zeros(shape2)
                
    #         for ind in range(0, shape2[1]):
    #             vec2 = mat2[:,ind]
    #             newVec = self._multiply_pauli_matrices(vec1,vec2, strFlag1)
    #             newMat[:,ind] = newVec
                
    #         output = newMat
        

    #     return output                  
    
    # ########
    # # symplectic dot product and commutation checks
    # ########
    
    # def make_J(self, shape1):
    #     if not numpy.mod(shape1,2) == 0:
    #         raise ValueError('need even numbers of rows for this')
            
    #     Jmat = numpy.zeros((shape1, shape1))
    #     for qind in range(0, int(shape1/2)):
    #         Jmat[2*qind, 2*qind+1] = 1
    #         Jmat[2*qind+1, 2*qind ] = 1
    #     return Jmat
    
    # def _matrix_symplectic_dot(self, mat1, mat2, debugMode = False):
    #     '''This function will define the symplectic dot product
    #     on two numpy arrays
        
    #     There will be a higher lever wrapper to define this on the Lorent 
    #     Pauli dictionaries
        
    #     Define the symplectic inner product mod 2 with X1Z1X2Z2...XnZn bit convention
        
    #     mat1^t * A*mat2 (I think)
        
        
    #     I believe that the output comes out in the following format
        
    #     anticommutation between m1[:,n] and m2[:,m] appears in output[n,m]
        
        
    #     '''
        
    #     m1 = self.convert_np_to_nmod(mat1, modulus = 2)
    #     m2 = self.convert_np_to_nmod(mat2, modulus = 2)
        
    #     shape1 = mat1.shape
    #     shape2 = mat2.shape
        
    #     if not shape1[0] == shape2[0]:
    #         raise ValueError('These two matrices are defined on different numbers of qubits')
    #     elif numpy.mod(shape1[0],2)==1:
    #         raise ValueError('This functions requires Haah style matrices. \n And these matrices have an odd number of rows')
    #     else:
    #         #let's try this
    #         Jmat = self.make_J(shape1[0])
                
    #         J = self.convert_np_to_nmod(Jmat, modulus = 2)
            
    #         finalMat = m1.transpose()*J*m2 #do the mod 2 matrix multiplication
            
    #         outputMat = self.convert_nmod_to_np(finalMat) #convert back to np array
            
    #     if debugMode:
    #         m1_out = self.convert_nmod_to_np(m1)
    #         m2_out = self.convert_nmod_to_np(m2)
    #         J_out = self.convert_nmod_to_np(J)
    #         return outputMat, m1_out, m2_out, J_out, m1, m2, J
    #     else:
    #         return outputMat
    
    # def check_Lorent_commutation(self, dict1, dict2):
    #     '''this function uses the symplectic dot product trick
    #     to check wether the entries in two Lorent-valued dictionaries
    #     of my string form commut or not. Returns a matrix of the anticommutations
        
    #     for now it does not check whether there is a problem with translates'''
        
    #     temp1 = dict1[(0,0)]
    #     temp2 = dict2[(0,0)]
        
    #     nQubits1 = temp1.shape[0]
    #     nQubits2 = temp2.shape[0]
        
    #     if not nQubits1 == nQubits2:
    #         raise ValueError('Numbers of qubits not euql between these. Cannot compare')
            
    #     if len(temp1.shape) ==1:
    #         #this guy is just a vector
    #         nTerms1 = 1
    #     else:
    #         nTerms1 = temp1.shape[1]
            
    #     if len(temp2.shape) ==1:
    #         #this guy is just a vector
    #         nTerms2 = 1
    #     else:
    #         nTerms2 = temp2.shape[1]
            
    #     HaahDict1 =  self.convert_string_to_Haah(dict1)  
    #     HaahDict2 =  self.convert_string_to_Haah(dict2)   
            
    #     anticommutations = numpy.zeros((nTerms1, nTerms2))
        
    #     for key in HaahDict1.keys():
    #         if key in HaahDict2.keys():
    #             #both of these dictionaries include qubits in thi unit cell
    #             #,so we need to actually check.
    #             HaahMat1 = HaahDict1[key]
    #             HaahMat2 = HaahDict2[key]
                
    #             delta = self._matrix_symplectic_dot(HaahMat1, HaahMat2)
                
    #             anticommutations = anticommutations + delta
        
    #     #take the antocommutations mod2
    #     anticommutations = numpy.mod(anticommutations, 2)
        
    #     return anticommutations
    
    # def _extract_nullspace(self, modMat):
    #     '''takes an nmod mat and extracts the nullspace
        
    #     because the syntax is annoying'''
    #     if not type(modMat) == nmod_mat:
    #         raise ValueError('this is for nmod matrices only')
    #     else:
    #         # print(modMat.rank())
    #         # X, nullity= modMat.nullspace()
    #         # print(nullity)
            
    #         #find everything that commutes with H and the stabilizers
    #         temp2 = modMat.nullspace()
    #         num_commuting_things = temp2[1]
    #         oversizeMatrix = temp2[0] #first few collumns of this are the basis of the null space
            
    #         if num_commuting_things == 0:
    #             #there is nothing that commutes
                
    #             return [], []
    #         else:
                
    #             #arg! Taking cust of this matrix is stupid!!
    #             #######
    #             #thing.table gives a 2D list that must be accessed as [n][m]
    #             #it can be trimmed as a list, but to work with it, it 
    #             #then needs to be case to np array, and it needs to be np anyway
    #             #to get an nmod_mat of the reduced size
                
    #             #the other way is to cast the whole thing back to np, then cut
    #             #then cast back
                
    #             out = self.convert_nmod_to_np(oversizeMatrix)
    #             nullspace_basis = out[:,0:num_commuting_things]
    #             nullspace_basis_mod2 = self.convert_np_to_nmod(nullspace_basis)
                
    #             #I think this latter should work too, but something might be going wrong
    #             # nullspace_basis_list = oversizeMatrix.table()[:][0:num_commuting_things]
    #             # nullspace_basis= numpy.asarray(nullspace_basis_list)
    #             # nullspace_basis_mod2 = self.convert_np_to_nmod(nullspace_basis)
    #             # # # nullspace_basis = self.convert_nmod_to_np(nullspace_basis_mod2)
                
    #             return nullspace_basis, nullspace_basis_mod2
            
        
    
    # def find_commutant(self, inputThing, debugMode = False):
    #     ''' 
    #     takes either dictionary definition for a set of paulis in string form
    #     or a straight up mod 2 matrix in Haah form
        
    #     and computes the commutant
        
    #     this will not work on Lorent-valued things
        
        
    #     Finally fixed it so that it  gives actual states, I think.
        
    #     '''
    #     if type(inputThing) == nmod_mat:
    #         G_mod2 = inputThing
    #         Gmat = self.convert_nmod_to_np(inputThing)
    #     elif type(inputThing) == dict:
    #         if len(inputThing.keys())> 1:
    #             raise ValueError('This function can only handle the monomial x^0Y^0')
    #         else:
    #             fullDict_Haah = self.convert_string_to_Haah(inputThing)
    #             Gmat = fullDict_Haah[(0,0)]
    #             G_mod2 = self.convert_np_to_nmod(Gmat)
    #     elif type(inputThing) == numpy.ndarray:
    #         G_mod2 = self.convert_np_to_nmod(inputThing)
    #         Gmat = inputThing
            
    #     # print(G_mod2.ncols())   #number of terms in the thing we are getting the null space of
    #     # print(G_mod2.nrows())   #number of qubits *2    
        
    #     #symplectic interpose matrix
    #     shape0 = G_mod2.nrows()
    #     Jmat = self.make_J(shape0)
    #     J_mod2 = self.convert_np_to_nmod(Jmat)
        
    #     #find everything that commutes with H and the stabilizers
    #     # temp = (G_mod2.transpose()*J_mod2).transpose() 
    #     #whoops I think the collumn vector version has no null space, unless collumns are redundant
    #     temp = (G_mod2.transpose()*J_mod2)
        
    #     ####
    #     #!!!!! I think that this may not detect redundant collumns in the original matrix?
    #     ###
        
    #     # temp2 = temp.nullspace()
    #     # num_commuting_things = temp2[1]
    #     # oversizeMatrix = temp2[0] #first few collumns of this are the basis of the null space
    #     # out = self.convert_nmod_to_np(oversizeMatrix)
    #     # nullspace_basis = out[:,0:num_commuting_things]
    #     nullspace_basis, nullspace_basis_mod2 = self._extract_nullspace(temp)
        
        

    #     # print(nullspace_basis_mod2.ncols()) #this ends up being the number of terms in nulls space
    #     # print(nullspace_basis_mod2.nrows()) #number of terms in the thing we are getting the null space of
        
        
    #     # SL_mod2 = G_mod2 * nullspace_basis_mod2
    #     # SL_mod2 = (G_mod2.transpose() *nullspace_basis_mod2).transpose()
    #     # SL_mod2 = (G_mod2.transpose()*J_mod2).transpose()*nullspace_basis_mod2
    #     SL_mod2 =nullspace_basis_mod2
    #     SL = self.convert_nmod_to_np(SL_mod2)
        
    #     if debugMode:
    #         return nullspace_basis, nullspace_basis_mod2, SL, SL_mod2, Gmat, G_mod2, J_mod2
        
    #     #these now contain a basis for the things that commute
    #     #with the union of H and the stabilizers
    #     # SL = nullspace_basis
        
    #     if type(inputThing) == nmod_mat:
    #         output = self.convert_np_to_nmod(SL)
    #     elif type(inputThing) == dict:
    #         output = {}
    #         output[(0,0)] = self.convert_Haah_to_string(SL)

    #     elif type(inputThing) == numpy.ndarray:
    #         output = SL
        
    #     return output
    
    
    # def canonical_basis(self, Hdict, Sdict, verbose = False, nonCommutant = False):
    #     '''my attempt at copying Steve's canonical basis function
        
    #     give it a Hamiltonian and stabilziers
        
    #     and it should find eveyrthing that commutes with them and
    #     divide them into stabilizers, logical Xs and logical Zs
        
        
    #     If there is no Hamiltonain, you can pass a blank string in the 
    #     place of the Hamiltonian
        
    #     If there are no stabilizers, you can pass a blank string in their place
        
        
    #     Safety checks added
    #     -S should commute with H
    #     -S should commute with S
        
    #     If you don't know what S should be, then just leave it blank and the code will find them.'
        
        
    #     If nonCommutant = True, then it will run symplectic basis on the Hamiltonian 
    #     instead of the commutant.
        
    #     '''
    #     if self.verbose:
    #         verbose = True
        
    #     if type(Hdict) == str:
    #         #use stabilizers only
    #         fullDict = Sdict
            
    #         #check whether the stabilizers commute witheachother
    #         flagMat = self.check_Lorent_commutation(Sdict, Sdict)
    #         if len(numpy.where(flagMat !=0)[0])>0:
    #             raise ValueError('This set of stabilizers do not commute with themselves')
                
    #     elif type(Sdict) == str:
    #         #use Hamiltonian only
    #         fullDict = Hdict
    #     else:
    #         #put eveyrthing together
    #         fullDict = self.combine_pauli_terms(Hdict, Sdict)
            
    #         #check whether the stabilizers commute with H
    #         flagMat = self.check_Lorent_commutation(Hdict, Sdict)
    #         if len(numpy.where(flagMat !=0)[0])>0:
    #             raise ValueError('This Hamiltonian does not commute with this set of stabilizers')
                
    #         #check whether the stabilizers commute witheachother
    #         flagMat = self.check_Lorent_commutation(Sdict, Sdict)
    #         if len(numpy.where(flagMat !=0)[0])>0:
    #             raise ValueError('This set of stabilizers do not commute with themselves')
        
    #     #find everything that commutes with what we already have
    #     if nonCommutant:
    #         SLdict = Hdict
    #     else:
    #         SLdict = self.find_commutant(fullDict)
    #     SL = self.convert_string_to_Haah(SLdict[(0,0)])
    #     SL_mod2 = self.convert_np_to_nmod(SL)
        
    #     if verbose:
    #         #do some size checking
    #         Heff_Haah = self.convert_string_to_Haah(fullDict[(0,0)])
    #         Heff_mod2 = self.convert_np_to_nmod(Heff_Haah)
            
    #         everything = self.combine_pauli_terms(fullDict , SLdict )
    #         everythingHaah = self.convert_string_to_Haah(everything[(0,0)])
    #         everything_mod2 = self.convert_np_to_nmod(everythingHaah )
            
    #         print('total hamiltonian and stabilizer terms = ' + str(Heff_mod2.ncols()))
    #         print('independent hamiltonian and stabilizer terms = ' + str(Heff_mod2.rank()))
            
    #         print('size of the commutant of H and S = ' + str(SL_mod2.ncols()))
    #         print('independent things in the commutant of H and S = ' + str(SL_mod2.rank()))
            
    #         print('size of the commutant(H and S) and H,S = ' + str(everything_mod2.ncols()))
    #         print('independent things in the H,S, commutant together = ' + str(everything_mod2.rank()))
        
    #     #now I need to monogamize 
    #     M = numpy.copy(SL)
    #     M_mod2 = self.convert_np_to_nmod(SL)
        
    #     R_mod2, dim = M_mod2.rref() #I'm not sure what I'm supposed to do with this
    #     #reduced row eschelon form
    #     #e.g. M_mod2 = M_mod2.transpose()*R_mod2
        
    #     # #symplectic interposer matrix
    #     # shape0 = M_mod2.nrows()
    #     # Jmat = self.make_J(shape0)
    #     # J_mod2 = self.convert_np_to_nmod(Jmat)
        
    #     #now we need to process everything else into monogamous pairs and stabilizers
    #     newM = numpy.copy(M)
    #     newXs = numpy.zeros(M.shape) #will trim excess size later
    #     newZs = numpy.zeros(M.shape) #will trim excess size later
    #     newSs = numpy.zeros(M.shape) #will trim excess size later
            
    #     nind = 0
    #     sind = 0
    #     while newM.shape[1] > 0:
    #         v = newM[:,0]
    #         newM = newM[:,1:]
            
            
    #         y = self._matrix_symplectic_dot(v, newM)
            
    #         hits = numpy.where(y != 0) [1] #collumn vector stuff, so I take the second argument
    #         # print(newM.shape[1])
    #         # print(hits)
            
    #         if len(hits) > 0:
    #             #there are some things that this vector does not commute with
        
    #             #take the first anticommuting thing and keep it
    #             currInd = hits[0]
    #             w = newM[:,currInd]
                
    #             #cut this vector out
    #             newM = numpy.concatenate((newM[:,0:currInd], newM[:,currInd+1:]), axis = 1)
                
    #             #now I need to add v to the xs, and w to the Zs
    #             newXs[:,nind] = v
    #             newZs[:,nind] = w
    #             nind = nind+1
        
    #             #fix the remaining things that anticommute
        
        
    #             #look for stuff in the reamainder which anticommut with v and w
    #             test1 = self._matrix_symplectic_dot(v, newM)
    #             hits1 = numpy.where(test1 != 0) [1]
    #             if len(hits1) ==0:
    #                 #nothing else anticommutes with v
    #                 pass
    #             else:
    #                 #now we have to fix stuff
    #                 for ind1 in hits1:
    #                     #these are the locations of the things that need to be fixed
    #                     oldVec = newM[:,ind1]
    #                     newVec = numpy.mod(oldVec+w,2)
    #                     newM[:,ind1] = newVec 
                
    #             test2 = self._matrix_symplectic_dot(w, newM)
    #             hits2 = numpy.where(test2 != 0) [1]
    #             if len(hits2) == 0:
    #                 #nothing else anticommutes with w
    #                 pass
    #             else:
    #                 #now we have to fix stuff
    #                 for ind2 in hits2:
    #                     #these are the locations of the things that need to be fixed
    #                     oldVec = newM[:,ind2]
    #                     newVec = numpy.mod(oldVec+v,2)
    #                     newM[:,ind2] = newVec   
                
    #         else:
    #             #this commutes with everything, so file it as a stabilizer
    #             newSs[:,sind] = v
    #             sind = sind+1
        
    #     #trim excess zeros
    #     newXs = newXs[:,0:nind]
    #     newZs = newZs[:,0:nind]
    #     newSs = newSs[:,0:sind]
        
        
    #     #return to convenient dictionary forms
    #     xDict_Haah = {}
    #     xDict_Haah[(0,0)] = newXs
    #     xDict = self.convert_Haah_to_string(xDict_Haah)
        
    #     zDict_Haah = {}
    #     zDict_Haah[(0,0)] = newZs
    #     zDict = self.convert_Haah_to_string(zDict_Haah)
        
    #     sDict_Haah = {}
    #     sDict_Haah[(0,0)] = newSs
    #     sDict = self.convert_Haah_to_string(sDict_Haah)
        
    #     return sDict, xDict, zDict
             
    
    
    ########
    #very general utilities
    ########
    
    ''' Thise functions no longer have any meaning '''
    def make_torus_vertices(self):
        print('This function is meaningless for this data type')
        pass
        return
    def find_torus_index(self, siteIndex, xPos, yPos ):
        print('This function is meaningless for this data type')
        pass
        return 
    
    
    
    def make_incidence_matrix(self, flagEnds = False):
        '''
        loops through the positions of the vertices in the graph
        and compares to the resonators and figures out who is connected to whom
        
        This will let me know which qubits are incident on which vertices (i.e. Hamiltonian terms)
        
        makes a matrix are the rows  of this format
        
        [vertex number, edge/qubit number]
        this says the the given edge, whien moved to the unit cell given by dX and dY is
        incident at this vertex
        
        optional argument flagEnds will trigger it to keep a record of which end of the resonator
        touches that vertex.
        This will help with fiducial bosonizations.
        0 for the first end, 1 for the second
        
        '''
        #complex encoding of the locations of the vertices
        vertVec = self.lattice.rootCoords[:,0] + 1j*self.lattice.rootCoords[:,1]
        
        #complex encodings of the start points of all resonators
        sVec = self.lattice.resonators[:,0] + 1j * self.lattice.resonators[:,1]
        #complex encoding of the end points of all the resonators
        tVec = self.lattice.resonators[:,2] + 1j * self.lattice.resonators[:,3]
        
        blankList = []
        for vind in range(0, self.numVertices):
            vertPos = vertVec[vind]
            
            for rind in range(0, len(sVec)):
                ss = sVec[rind]
                tt = tVec[rind]
                      
                hitFlag = 0
                endFlag = -1
                if numpy.round(vertPos,3) == numpy.round(ss, 3):
                    hitFlag = 1
                    endFlag = 0
                if numpy.round(vertPos,3) == numpy.round(tt, 3):
                    hitFlag = 1
                    endFlag = 1
                
                if hitFlag:
                    #this vertex is connected to this resonator at this offset
                    if flagEnds:
                        blankList.append([vind, rind, endFlag])
                    else:
                        blankList.append([vind, rind])
        self.Incidence = numpy.asarray(blankList)
        return
    
    def make_edge_incidence_matrix(self):
        '''
        loops through the positions of the vertices in the graph
        and compares to the resonators and figures out who is connected to whom
        
        this will keep track of what vertices lie at each end of the resonator
        
        makes a matrix are the rows  of this format
        
        [vertex1, x shift1, y shift1, vertex2, xshift2, yshift2]
        this says the given edge starts at vertex 1 in cell x1, y1 and ends ate vertex in cell x2, y2
        
        
        '''
        #complex encoding of the locations of the vertices
        vertVec = self.lattice.rootCoords[:,0] + 1j*self.lattice.rootCoords[:,1]
        
        #complex encodings of the start points of all resonators
        sVec = self.lattice.resonators[:,0] + 1j * self.lattice.resonators[:,1]
        #complex encoding of the end points of all the resonators
        tVec = self.lattice.resonators[:,2] + 1j * self.lattice.resonators[:,3]
        
        blankList = numpy.zeros((self.lattice.resonators.shape[0], 2))
        
            
        for rind in range(0, len(sVec)):
            ss = sVec[rind]
            tt = tVec[rind]
            
            v0 = numpy.NaN
            
            v1 = numpy.NaN
            
            for vind in range(0, self.numVertices):
                vertPos = vertVec[vind]
                
                        
                if numpy.round(vertPos,3) == numpy.round(ss, 3):
                    v0 = vind
                if numpy.round(vertPos,3) == numpy.round(tt, 3):
                    v1 = vind
                        

            blankList[rind,:] = [v0, v1]
        self.edgeIncidence = numpy.asarray(blankList)
        return
    
    
    def make_fiducial_H(self, check_plot = False):
        #user has not specificied a coloring for the vertices, so I will do something simple. 
        pauliDict = {}
        pauliDict[(0,0)] = self.blank_pauli_mat(self.numQubits, self.numVertices)
    
        for qind in range(0, self.numQubits):
            couplings = numpy.where(self.Incidence[:,1] == qind)[0]
            
            if not len(couplings)==2:
                print('warning. making fiducial H went wrong')
            else:
                for cind in range(0,2):
                    vert, qNum= self.Incidence[couplings[cind],:]
                    #vertex is connected to this qubit/edge
                    
                    #the vertex we are grabbing is the far end of the resonator
                    #so , I think we should keep dX and dY
                    
                    key = (0, 0) #this graph only has the zero, zero position
                    if cind == 0:
                        pauli = 'X'
                    else:
                        pauli = 'Z'
                    if key in pauliDict.keys():
                        pauliDict[key][qind,vert] = pauli
                    else:
                        pauliDict[key] = self.blank_pauli_mat(self.numQubits, self.numVertices)
                        pauliDict[key][qind,vert] = pauli
        if check_plot:
            fig = pylab.figure(44)
            pylab.clf()
            
            for vind in range(0, self.numVertices):
                ax = pylab.subplot(1, self.numVertices, vind+1)
                self.plot_single_term(pauliDict, vind, fignum = -1, 
                     numberSites = True,
                     spotSize = 400,
                     axis = ax)
                self.lattice.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
                pylab.title(str(vind))
                
            pylab.tight_layout()
            pylab.tight_layout()
            pylab.show()
            
            pylab.suptitle('fiducial H check')
        return pauliDict
            
             
        
    def make_fiducial_dimers(self, check_plot = False):
        ''' 
        makes all the dimer operators
        
        For the moment it only works for 3-regular graphs, because I need to predict with number of dimers
        in order to allocate space.
        
        
        5-25-21 I think it now works for all lattices
        
        '''

        pauliDict = {}
        fudgeFactor = 16
        pauliDict[(0,0)] = self.blank_pauli_mat(self.numQubits, self.numVertices*fudgeFactor)
        
        
        tind = 0
        for vind in range(0, self.numVertices):
            
            linkedVertInds = numpy.where(self.lattice.rootLinks[:,0] == vind)[0]
            linkedVerts = numpy.unique(self.lattice.rootLinks[linkedVertInds,1])
            
            linkedQubitInds = numpy.where(self.Incidence[:,0] == vind)[0]
            linkedQubits = numpy.unique(self.Incidence[linkedQubitInds,1])
            
            # print(linkedQubits)
            
            #find all the qubits that are linked to this guy, inside and outside of cell
            linkList = []
            for qind1 in linkedQubits:
                if self.edgeIncidence[qind1][0] == vind:
                    #this guy vertex is at the starting end of the resonator
                    
                    # #if resonator in the origin unit cell hits this vertex in the unit cell, great
                    # # but if reconator in the origina unit cell hits the (1,0) copy of this vertes
                    # #then I need to shift the resonator back to hit this guy in this cell
                    # resX = int(-self.cellEdgeIncidence[qind1][1])
                    # resY = int( -self.cellEdgeIncidence[qind1][2])
                    # linkList.append([qind1, resX, resY, 0])
                    linkList.append([qind1, 0])
                    
                # if self.cellEdgeIncidence[qind1][3] == vind:
                if self.edgeIncidence[qind1][1] == vind:
                    #this guy vertex is at the terminating end of the resonator
                    
                    # #if resonator in the origin unit cell hits this vertex in the unit cell, great
                    # # but if reconator in the origina unit cell hits the (1,0) copy of this vertes
                    # #then I need to shift the resonator back to hit this guy in this cell
                    # resX = int(-self.cellEdgeIncidence[qind1][4])
                    # resY = int(-self.cellEdgeIncidence[qind1][5])
                    # linkList.append([qind1, resX, resY, 1])
                    
                    linkList.append([qind1, 1])
            
            # return linkList
            #now to build the dimers
            for lind1 in range(0,len(linkList)):
                qNum1, end = linkList[lind1]
                key1 = (0, 0) #the location  is always the origin for the non-translation invariant case
                
                # #now I need to find the Pauli on the far end of this edge
                # if end ==0:
                #     otherEnd = self.cellEdgeIncidence[qNum1,3:]
                # else:
                #     otherEnd = self.cellEdgeIncidence[qNum1,0:3]
                # v1 = int(otherEnd[0])
                # dX1 = int(otherEnd[1])
                # dY1 = int(otherEnd[2])
                #now I need to find the Pauli on the far end of this edge
                if end ==0:
                    v1 = self.cellEdgeIncidence[qNum1,1]
                else:
                    v1 = self.cellEdgeIncidence[qNum1,0]
   
                
                #I need to grab from the Hamiltonian term at v1
                #and for the qubit qNum1
                #but, the monomial I need to grab is more complicated
                #I want the pauli that wind up on this edge when I translate by
                #dX1 and dY1, so I think I need to pull from the Hamiltonian at the inverse position
                
                # pauli1 = self.fiducialH[(-dX1, -dY1)][qNum1, v1]
                pauli1 = self.fiducialH[(0, 0)][qNum1, v1]
                
                for lind2 in range(lind1+1,len(linkList)):
                    if lind1 == lind2:
                        pass
                    else:
                        
                        qNum2, end = linkList[lind2]
                        key2 = (0,0) #the location of the qubit that we need to put a pauli on for this dimer
                        
                        # #now I need to find the Pauli on the far end of this edge
                        # if end ==0:
                        #     otherEnd = self.cellEdgeIncidence[qNum2,3:]
                        # else:
                        #     otherEnd = self.cellEdgeIncidence[qNum2,0:3]
                        # v2 = int(otherEnd[0])
                        # dX2 = int(otherEnd[1])
                        # dY2 = int(otherEnd[2])
                        #now I need to find the Pauli on the far end of this edge
                        if end ==0:
                            v2 = self.cellEdgeIncidence[qNum2,1]
                        else:
                            v1 = self.cellEdgeIncidence[qNum2,0]
                        
                        #I need to grab from the Hamiltonian term at v1
                        #and for the qubit qNum1
                        #but, the monomial I need to grab is more complicated
                        # pauli2 = self.fiducialH[(-dX2, -dY2)][qNum2, v2]
                        pauli2 = self.fiducialH[(0,0)][qNum2, v2]
                        
                        
                        if key1 in pauliDict.keys(): 
                            pauliDict[key1][qNum1, tind] = pauli1
                        else:
                            pauliDict[key1] = self.blank_pauli_mat(self.numQubits, self.numVertices*fudgeFactor)
                            pauliDict[key1][qNum1, tind] = pauli1
                            
                        if key2 in pauliDict.keys(): 
                            pauliDict[key2][qNum2, tind] = pauli2
                        else:
                            pauliDict[key2] = self.blank_pauli_mat(self.numQubits, self.numVertices*fudgeFactor)
                            pauliDict[key2][qNum2, tind] = pauli2
        
                        tind = tind+1
                        
        #trim off trailing balnk stuff
        for key in pauliDict.keys():
            pauliDict[key] = pauliDict[key][:,0:tind]
            
        if check_plot:
            fig = pylab.figure(45)
            pylab.clf()
            
            numTerms = pauliDict[(0,0)].shape[1]
            for tind in range(0, numTerms):
                ax = pylab.subplot(2, int(numpy.ceil(numTerms/2)), tind+1)
                self.plot_single_term(pauliDict, tind, fignum = -1, 
                     numberSites = True,
                     spotSize = 400,
                     axis = ax)
                self.lattice.draw_resonators(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
                pylab.title(str(vind))
                
            pylab.tight_layout()
            pylab.tight_layout()
            pylab.show()
            
            pylab.suptitle('fiducial dimer check')
        return pauliDict
                
    def check_unit_cell_labels(self, title = '', fignum = 1, figx = 6, figy = 6):
        print('This function is meaningless for this data type')
        pass
        return
    
    def check_labels(self, title = '', fignum = 1, figx = 6, figy = 6):
            
        fig1 = pylab.figure(fignum)
        pylab.clf()
        
        ax = pylab.subplot(1,1,1)
        pylab.scatter(self.Xs, self.Ys,c =  layoutCapColor, s = 400, marker = 'o', edgecolors = 'k',zorder = 2)
        
        
        for ind in range(0, self.Xs.shape[0]):
            coords = [self.Xs[ind], self.Ys[ind]]
            pylab.text(coords[0], coords[1], str(ind), color='k', size=labelSize, fontdict = lighttext_params)
        
        if self.bosonization_type == 'edge':
            #plot the vertex numbering too
            Xs = self.lattice.rootCoords[:,0]
            Ys = self.lattice.rootCoords[:,1]
            pylab.scatter(Xs, Ys,c =  FWsiteColor, s = 200, marker = 'o', edgecolors = 'royalblue',zorder = 2)
        
            for ind in range(0, Xs.shape[0]):
                coords = [Xs[ind], Ys[ind]]
                pylab.text(coords[0], coords[1], str(ind), color='k', size=10, fontdict = lighttext_params)
        
        self.lattice.draw_resonators(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
        
        
        ax.axis('off')
        ax.set_aspect('equal')
        pylab.title('unit cell')
        pylab.suptitle(title)
        
        
        fig1.set_size_inches([figx,figy])
        pylab.show()
        
    def check_vertex_assignment(self, title = '', fignum = 1, figx = 6, figy = 6):
       
        print(' this function is meaningless for this data type')   
    
        return
    
    # def blank_pauli_mat(self, n,m):
    #     temp = numpy.zeros((n,m), dtype = 'U21')
    #     for nind in range(0, n):
    #         for mind in range(0, m):
    #             temp[nind, mind] = 'I'
    #     return temp
    
    
    
    
    ########
    #term plotters for Haah valued stuff in the unit cell
    ########

    
    def plot_single_term(self, pauliDict, tind, fignum = 7, 
                     title = 'blank', 
                     numberSites = False,
                     spotSize = 400,
                     figx = 6,
                     figy = 6,
                     axis = ''):
        ''' Takes a single (possibly laurent-valued) pauli matrix in dictionary form, 
        pulls one term, and plots that
        
        set figure number negative and pass an axis object in order to plot in a subplot
        
        '''

        if fignum < 0:
            ax = axis
            pylab.sca(ax)
        else:
            currFig = pylab.figure(fignum)
            pylab.clf()
            ax = pylab.subplot(1,1,1)
            
        pylab.scatter(self.Xs, self.Ys,c =  layoutCapColor, s = spotSize, marker = 'o', edgecolors = 'k',zorder = 2)
        if numberSites:
            for ind in range(0 , len(self.Xs)):
                coords = [self.Xs[ind], self.Ys[ind]]
                coords = coords + 0.06*numpy.asarray([1,1])
                pylab.text(coords[0], coords[1], str(ind), color='k', size=labelSize, fontdict = lighttext_params)
        ax.axis('off')
        ax.set_aspect('equal')
          
        pylab.suptitle(title)
        
        self.lattice.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
        
        if self.bosonization_type == 'edge':
            #plot the vertex numbering too
            Xs = self.lattice.rootCoords[:,0]
            Ys = self.lattice.rootCoords[:,1]
            pylab.scatter(Xs, Ys,c =  FWsiteColor, s = 45, marker = 'o', edgecolors = 'royalblue',zorder = 2)
        
        
        
        for key in pauliDict.keys():
            if not key == (0,0):
                raise ValueError('This pauli dict has non-zero keys. It shouldnt')
            else:
            
                n = int(key[0])
                m = int(key[1])
                
                pauliVec = pauliDict[key][:,tind]
                numQubits = pauliVec.shape[0]
            
                
                for qind in range(0, numQubits):
    
                    [x0,y0] = [self.Xs[qind], self.Ys[qind]]
                    [x,y] = numpy.asarray([x0,y0])
                    
                    pauliLabel = pauliVec[qind]
                    
                    if pauliLabel == 'I':
                        pass
                    else:
                        
                        #high vis plots
                        if pauliLabel =='X':
                            color =  'dodgerblue' #'deepskyblue' 
                        elif pauliLabel == 'Y':
                            color = 'mediumblue'#'forestgreen'
                        else:
                            color=  'firebrick'#'indianred' #'dodgerblue'
                        textcolor = 'whitesmoke'
                        
                        pylab.scatter([x], [y],c =  color, s = (spotSize + 50), marker = 'o', edgecolors = 'midnightblue',zorder = 2)
                        pylab.text(x, y, pauliLabel, color=textcolor, size=labelSize, fontdict = lighttext_params)
            
        if fignum > 0:
            currFig.set_size_inches([figx,figy])
            return currFig   
        else:
            return
    
    
    def plot_terms(self, pauliDict, saveExtension = '_blah', 
                   videoMode = False,
                   inlineMode = False,
                   fignum = 4, 
                   jog = 2.25,
                   jogy = 2.25,
                   figx = 6,
                   figy = 6,
                   videoTime = 1):
        
        baseStr = self.name + saveExtension
        saveDir = os.path.join('codeAutoPlots', baseStr)
        
        if not os.path.isdir('codeAutoPlots'):
            os.mkdir('codeAutoPlots')
        
        if not os.path.isdir(saveDir):
            os.mkdir(saveDir)
        
        for cind in range(0, pauliDict[(0,0)].shape[1]):
            print(cind)
            
            if inlineMode:
                fignum = fignum+1
                
            figName = baseStr + '_' +  str(cind)# + '.png'
            currFig = self.plot_single_term(pauliDict, cind, 
                                       fignum = fignum, 
                                       title = figName, 
                                       figx = figx,
                                       figy = figy)
            if videoMode:
                currFig.canvas.draw()
                currFig.canvas.flush_events()
                time.sleep(videoTime)
            else:
                if not inlineMode:
                    savePath = os.path.join(saveDir, figName + '.png')
                    currFig.savefig(savePath, dpi = 200)  
                
        return
    
    ########
    #term plotters for stuff on the torus
    ########
    
    
    def plot_single_torus_term(self, torusPauliDict, tind, fignum = 7, 
                     title = 'blank', 
                     numberSites = False,
                     drawLinks = False,
                     spotSize = 250,
                     figx = 6,
                     figy = 6,
                     axis = ''):
        print(' this function is invalid for this data type')
        return
    
    def plot_torus_terms(self, torusPauliDict, saveExtension = '_torusBlah',
                         videoMode = False,
                         inlineMode = False,
                         saveDir = '', 
                         fignum = 4,
                         drawLinks = False,
                         spotSize = 250,
                         figx = 6,
                         figy = 6,
                         videoTime = 0.5):
        print(' this function is invalid for this data type')
        return
    

    ########
    #unpacker to generate matrices on the torus from Lorent-valued stuff on the unit cell
    ########

    def unpack_to_torus(self, cellPauliDict):
        print(' this function is invalid for this data type')
        return                                
                        
                    
                
                
        
        


if __name__=="__main__":      
    
    print('main loop')
    
    test0 = PlanarLayout(5,4)
    test0.populate(2)
    test0.generate_semiduals()
    res = test0.get_all_resonators()
    
    test= GeneralLayout(name = '5_4', resonators = res, Hamiltonian = False)
    test.generate_root_graph()
    
    
    pylab.figure(1)
    pylab.clf()
    
    ax = pylab.subplot(1,1,1)
    
    test.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
    test.draw_resonator_end_points(ax, color = layoutCapColor, edgecolor = 'k',  marker = 'o' , size = smallCdefault, zorder = 5)
    # test.draw_SDlinks(ax, color = FWlinkColor, linewidth = 1, minus_links = True, minus_color = 'gold', alpha=1)
    # test.draw_SD_points(ax,color =  FWsiteColor, size = smallCdefault, edgecolor = FWsiteEdgeColor, zorder = 5)
    
    for lind in range(test.rootLinks.shape[0]):
        c1, c2 = test.rootLinks[lind,:]
        c1 = int(c1)
        c2 = int(c2)
        x0, y0 = test.rootCoords[c1, :]
        x1, y1 = test.rootCoords[c2, :]
        
        pylab.plot([ x0, x1], [y0,y1], 'deepskyblue', linewidth = 0.5)
    
    ax.axis('off')
    
    pylab.show()
    
    
    
    ####
    #now to make a Pauli code object from this lattice
    #####
    
    
    
    code = PauliCode_GenGraph(test, name = 'temp', 
                        bosonization_type = 'edge',
                        fiducial = False,
                        verbose = False)
    
    
    fiducialH = code.make_fiducial_H(check_plot = False)
    
    
    pylab.figure(2)
    pylab.clf()
    
    numPlots = 4
    offset = 50
    for ind in range(0, numPlots):
        ax = pylab.subplot(1,numPlots, ind+1)
        
        tind = offset + ind
        code.plot_single_term(fiducialH, tind, fignum = -1, 
                         title = '', 
                         numberSites = False,
                         spotSize = 100,
                         figx = 6,
                         figy = 6,
                         axis = ax)
        pylab.title(str(tind))
    
    pylab.tight_layout()


