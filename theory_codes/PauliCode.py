#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:12:20 2021


5-18-21 Fixed a silly bug in the Lz returned by canonical basis. Fixed vectors were being
slotted back in the wrong place.   Also found an indexing big in find_torus_index. Stupidly,
I ws modding the abolsute index, instead of the x and y position of the unit cell and then
computing the absolute index. This did icky things to the torus unpacker for terms
that straddled the vertical edge.

5-19-21 Changed how canonical basis works.
-should reuqire that the stabilizers commute with eachother and H
-should not require that H commutes with itself
-should be able to work when only one of S and H are given
(basically If you don't know what good stabilizers are, then it should just find them')

5-20-21 Added a function to allow paulis to be multiplied even in string
or dictionary form.

5-25-21 I think I fixed the bug thatr made fiducial dimers fail for lattices like
the square lattice in which edges connect equivalent sites in different unit cells.
-Along the way added an indcidence matrix for the edges, as well as the vertices

6-6-21 Trying to add a new control parameter so that the underlying layers allocate enough
space for link matrices even for graphs with large coordination numbers (eg. L(Square))
New parameter is called maxDegree. -- actually. I'm moving it to the unit cells, so 
that it will get taken care of automatically. It's too much of a mess otherwise.'


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
labelSize = 18





class PauliCode(object):
    def __init__(self, unitcell = UnitCell('kagome'), name = '', 
                 bosonization_type = 'edge',
                 fiducial = True,
                 xSize = 2, ySize = 2, 
                 vertex_assignment = [],
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
            self.vertex_assignment = vertex_assignment
            
            
            if bosonization_type == 'edge':
                self.bosonization_type = 'edge'
            elif bosonization_type == 'vertex':
                self.bosonization_type = 'vertex'
            else:
                raise(ValueError, 'unknown bosonization type')
            
            self.unitcell = unitcell
            self.unitcell.find_root_cell()
        
            self.maxDegree = self.unitcell.maxDegree #manually input parameter that is the
            #maximum degree of the root graph. 
            #default is set to 4 so it will automatically allocate enough space
            #for the link matrices. For higher-degree graphs, you will have to 
            #tell it to allocate more space
        
            self.xSize = xSize
            self.ySize = ySize
            self.lattice = EuclideanLayout(xcells= xSize,
                                           ycells =  ySize, 
                                           initialCell = self.unitcell)
            
            
            self.verticesPerCell = self.unitcell.rootCoords.shape[0]
            self.edgesPerCell = self.unitcell.SDx.shape[0]
            
            if self.bosonization_type == 'edge':
                self.qubitsPerCell = self.edgesPerCell
            elif self.bosonization_type == 'vertex':
                self.qubitsPerCell = self.verticesPerCell
            self.numQubits = self.qubitsPerCell*self.xSize*self.ySize
            
            
            ####
            #lets get the positions
            #####
            if self.bosonization_type == 'edge':
                self.cellXs = self.unitcell.SDx
                self.cellYs = self.unitcell.SDy
            if self.bosonization_type == 'vertex':
                self.cellXs = self.unitcell.rootCoords[:,0]
                self.cellYs = self.unitcell.rootCoords[:,1]
            
            
            ####
            #now to fill in the on the torus stuff
            ####
            if self.bosonization_type == 'edge':
                self.torusXs = self.lattice.SDx
                self.torusYs = self.lattice.SDy
            if self.bosonization_type == 'vertex':
                self.make_torus_vertices()
            
            
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
                self.fiducialDimers = self.make_fiducial_dimers(check_plot = self.verbose)
                
                self.H0 = self.fiducialH
                
                self.H_torus = self.unpack_to_torus(self.H0)
            
            ###
            #fill in Lorent-valued matrices for the unit cell
            ####
            
            # #hack to do stuff before I have made the autogenerate functions for the Hamiltonian and stabilizers
            # if self.name == 'kaphene':
            #     self.Lx0_raw = numpy.asarray([
            #               [0, 0, 0, 0, 'X', 'X+XY'],
            #               [0, 0, 0, 0, 'X', 'X+XY'],
            #               [0, 1, 0, 1, 0, 0],
            #               [0, 0, 0, 0, 0, '1+X'],
            #               [0, 0, 0, 1, 0, 1],
            #               [0, 0, 0, 0, 0, '1+X'],
            #               [0, 0, 0, 0, 0, 'X+XY'],
            #               [0, 0, 0, 0, 'X', '1+XY'],
            #               [0, 0, 0, 0, 0, '1+X'],
            #               [0, 1, 0, 0, 0, 'X'],
            #               [0, 0, 0, 0, 0, '1+XY'],
            #               [1, 0, 1, 0, 0, 0],
            #               [1, 0, 0, 0, 0, 1],
            #               [0, 0, 0, 0, 0, '1+XY'],
            #               [0, 0, 0, 0, 1, 'Y+XY'],
            #               [0, 0, 0, 0, 0, '1+Y'],
            #               [0, 0, 0, 0, 0, '1+XY'],
            #               [0, 0, 1, 0, 0, 'XY']
            #             ])
            #     self.Stab0_raw = numpy.zeros((self.qubitsPerCell, 6))
            #     self.H0_raw = numpy.asarray([
            #                              [1, 0, 0, 0, 0, 0],
            #                              [0, 0, 0, 0, 0, 'X'],
            #                              [0, 1, 0, 0, 1, 0],
            #                              [0, 1, 0, 0, 0, 0],
            #                              [1, 1, 0, 0, 0, 0],
            #                              [0, 1, 0, 0, 0, 0],
            #                              [1, 0, 0, 0, 0, 0],
            #                              [0, 0, 1, 0, 0, 0],
            #                              [0, 1, 0, 0, 0, 0],
            #                              [0, 1, 1, 0, 0, 0],
            #                              [0, 0, 0, 1, 0, 0],
            #                              [0, 0, 1, 1, 0, 0],
            #                              [0, 0, 0, 1, 'Y', 0],
            #                              [0, 0, 0, 1, 0, 0],
            #                              [0, 0, 0, 0, 'Y', 0],
            #                              [0, 0, 0, 0, 0, 1],
            #                              [0, 0, 0, 1, 0, 0],
            #                              [0, 0, 0, 1, 0, 1]
            #                             ])
            #     self.H0 = self.parse_text_operator_matrix(self.H0_raw)
            #     self.Lx0 = self.parse_text_operator_matrix(self.Lx0_raw) 
            
            # #hamitonian from the lattice, in Haah form
            # self.H0 = self.generate_H0()
            
            # #stabilizers from checking the commutations, in Haah form
            # self.Stab0, self.Lx0, self.Lz0 = self.Lorent_commutant_basis()
            
            
            
            
            # self.H0 = self.parse_operator_matrix(self.H0_raw)
            
            # self.Lx0 = self.parse_operator_matrix(self.Lx0_raw)  
            # try:
            #     self.Stab0 = self.parse_operator_matrix(self.Stab0_raw)
            #     self.Lz0 = self.parse_operator_matrix(self.Lz0_raw)
            # except:
            #     pass
            
            # return
            
            
            ###
            #now to fill in the on the torus stuff
            ####
            # self.H_torus = self.unpack_to_torus(self.H0)
            #self.Lx_torus = self.unpack_to_torus(self.Lx0)

    
    #######
    #saving and loading functions
    #######
    
    def save(self, name = ''):
        '''
        save structure to a pickle file
        
        if name is blank, will use dafualt name
        '''
            
        if self.fiducial:
            fidStr = '_fiducial'
        else:
            fidStr = ''
        if name == '':
            name = self.name + fidStr + '.pkl'
        
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
           
        return

    ######
    #matrix type casters to go between the various type of matrices
    ######
    def convert_np_to_list(self, mat):
        '''cast an integer-valued numpy array to a single 1D list
        
        This is a prestep to passing to nmod_mat
        '''
        if len(mat.shape) == 1:
            #this is a vector
            vec = mat
        else:
            vec = mat.reshape(mat.shape[0]*mat.shape[1])
        newList = []
        for vind in range(0,len(vec)):
            entry = int(vec[vind])
            newList.append(entry)
        return newList
            
    def convert_list_to_np(self, oldList, shape0, shape1):
        '''covert flattened integer list of entires back into a
        2D numpy array'''
        vec = numpy.asarray(oldList)
        mat = vec.reshape(shape0, shape1)
        return mat
    
    def convert_nmod_to_np(self, modmat):
        '''Goes through the stupid hassle of converting an nmod_mat
        back into a standard numpy array'''
        shape0 = int(modmat.nrows())
        shape1 = int(modmat.ncols())
        npmat = numpy.zeros((shape0, shape1))
        for nind in range(0,shape0):
            for mind in range(0, shape1):
                npmat[nind, mind] = int(modmat[nind, mind])
        return npmat
    
    def convert_np_to_nmod(self, npmat, modulus = 2):
        '''wrapper to produce the corresponding nmod_mat
        from the 2D numpy array directly
        
        defualts to mod 2
        '''
        newList = self.convert_np_to_list(npmat)
        if len(npmat.shape) == 1:
            #this is a vector
            shape0 = npmat.shape[0]
            shape1 = 1
        else:
            shape0 = npmat.shape[0]
            shape1 = npmat.shape[1]
        modmat = nmod_mat(shape0, shape1, newList, modulus)
        return modmat
    
    def convert_string_to_Haah(self,inputThing):
        if type(inputThing) == numpy.ndarray:
            oldMat = inputThing
            numQubits = oldMat.shape[0]
            numTerms = oldMat.shape[1]
            
            newMat = numpy.zeros((numQubits*2, numTerms))
            
            for qind in range(0, numQubits):
                for tind in range(0, numTerms):
                    if oldMat[qind,tind] == 'I':
                        newMat[2*qind, tind] = 0
                        newMat[2*qind+1, tind] = 0
                    if oldMat[qind,tind] == 'X':
                        newMat[2*qind, tind] = 1
                        newMat[2*qind+1, tind] = 0
                    if oldMat[qind,tind] == 'Y':
                        newMat[2*qind, tind] = 1
                        newMat[2*qind+1, tind] = 1
                    if oldMat[qind,tind] == 'Z':
                        newMat[2*qind, tind] = 0
                        newMat[2*qind+1, tind] = 1
            return newMat
        
        elif type(inputThing) == dict:
            pauliDict = inputThing
            newDict = {}
            
            numQubits = pauliDict[(0,0)].shape[0]
            numTerms = pauliDict[(0,0)].shape[1]
            for key in pauliDict.keys():
                oldMat = pauliDict[key]
                
                newMat = numpy.zeros((numQubits*2, numTerms))
                
                for qind in range(0, numQubits):
                    for tind in range(0, numTerms):
                        if oldMat[qind,tind] == 'I':
                            newMat[2*qind, tind] = 0
                            newMat[2*qind+1, tind] = 0
                        if oldMat[qind,tind] == 'X':
                            newMat[2*qind, tind] = 1
                            newMat[2*qind+1, tind] = 0
                        if oldMat[qind,tind] == 'Y':
                            newMat[2*qind, tind] = 1
                            newMat[2*qind+1, tind] = 1
                        if oldMat[qind,tind] == 'Z':
                            newMat[2*qind, tind] = 0
                            newMat[2*qind+1, tind] = 1
                newDict[key] = newMat
            return newDict
        else:
            raise ValueError('invalid input type. should be numpy array or dictionary')
    
    def convert_Haah_to_string(self, inputThing):
        if type(inputThing) == numpy.ndarray:
            oldMat = inputThing
            
            numQubits = int(oldMat.shape[0]/2)
            numTerms = oldMat.shape[1]
            
            newMat = numpy.zeros((numQubits, numTerms), dtype = 'U21')
            for qind in range(0, numQubits):
                for tind in range(0, numTerms):
                    e0 = oldMat[2*qind, tind]
                    e1 = oldMat[2*qind+1, tind]
                    
                    if (e0 == 0) and (e1 == 0):
                        newMat[qind, tind] = 'I'
                    if (e0 == 1) and (e1 == 0):
                        newMat[qind, tind] = 'X'
                    if (e0 == 1) and (e1 == 1):
                        newMat[qind, tind] = 'Y'
                    if (e0 == 0) and (e1 == 1):
                        newMat[qind, tind] = 'Z'
            return newMat
            
        elif type(inputThing) == dict:
            HaahDict = inputThing
            newDict = {}
            
            numQubits = int(HaahDict[(0,0)].shape[0]/2)
            numTerms = HaahDict[(0,0)].shape[1]
                
            for key in HaahDict.keys():
                oldMat = HaahDict[key]
                
                newMat = numpy.zeros((numQubits, numTerms), dtype = 'U21')
                for qind in range(0, numQubits):
                    for tind in range(0, numTerms):
                        e0 = oldMat[2*qind, tind]
                        e1 = oldMat[2*qind+1, tind]
                        
                        if (e0 == 0) and (e1 == 0):
                            newMat[qind, tind] = 'I'
                        if (e0 == 1) and (e1 == 0):
                            newMat[qind, tind] = 'X'
                        if (e0 == 1) and (e1 == 1):
                            newMat[qind, tind] = 'Y'
                        if (e0 == 0) and (e1 == 1):
                            newMat[qind, tind] = 'Z'
                newDict[key] = newMat
            return newDict
        else:
            raise ValueError('invalid input. Should be np array or dict')
    
    def combine_pauli_terms(self, dict1, dict2):
        ''' takes two pauli dictionaries in string form
        and combines them to make a larger one.
        
        intended for combining Hamiltonians with added stabilizers
        '''
        
        shape1 = dict1[(0,0)].shape
        shape2 = dict2[(0,0)].shape
        
        if not shape1[0] == shape2[0]:
            raise ValueError('cannot combine sets of operators on different numbers of qubits')
        
        newDict = {}
        
        for key in dict1.keys():
            if key in dict2.keys():
                #both of these guys have stuff in this unit cell
                newDict[key] = numpy.concatenate((dict1[key], dict2[key]), axis = 1)
                
            else:
                #dict2 doesn't have stuff in this unit cell
                
                temp2 = self.blank_pauli_mat(shape2[0], shape2[1])
                newDict[key] = numpy.concatenate((dict1[key], temp2), axis = 1)
                
        for key in dict2.keys():
            if not key in dict1.keys():
                #this guy is in dictionary 2, but not in dictionary 1, 
                #so I need to add it too
                temp1 = self.blank_pauli_mat(shape1[0], shape1[1])
                newDict[key] = numpy.concatenate((temp1, dict2[key]), axis = 1)
                
        return newDict  

    def _multiply_pauli_matrices(self, vec1, vec2, strFlag):
        '''multiply two vectors that describe paulis
        
        must be of the same type, and function must be
        told what the type is (str v Haah)
        
        '''
        if not len(vec1) == len(vec2):
            raise ValueError('lengths do not match')
        else:
            if strFlag:
                newVec = numpy.zeros(len(vec1), dtype = 'U21')
                for ind in range(0,len(vec1)):
                    v1 = vec1[ind]
                    v2 = vec2[ind]
                    
                    if v1 == 'I':
                        if v2 == 'I':
                            newV = 'I'
                        elif v2 == 'X':
                            newV = 'X'
                        elif v2 == 'Y':
                            newV = 'Y'
                        else:
                            newV = 'Z'
                    elif v1 == 'X':
                        if v2 == 'I':
                            newV = 'X'
                        elif v2 == 'X':
                            newV = 'I'
                        elif v2 == 'Y':
                            newV = 'Z'
                        else:
                            newV = 'Y'
                    elif v1 == 'Y':
                        if v2 == 'I':
                            newV = 'Y'
                        elif v2 == 'X':
                            newV = 'Z'
                        elif v2 == 'Y':
                            newV = 'I'
                        else:
                            newV = 'X'
                    elif v1 == 'Z':
                        if v2 == 'I':
                            newV = 'Z'
                        elif v2 == 'X':
                            newV = 'Y'
                        elif v2 == 'Y':
                            newV = 'X'
                        else:
                            newV = 'I'
                    
                    newVec[ind] = newV
            else:
                newVec = numpy.zeros(len(vec1))
                newVec = numpy.mod(vec1+vec2, 2)
                
        return newVec
                        

    def multiply_paulis(self, thing1, thing2):
        '''take in two representations of paulis and multiply them 
        
        first argument should be a single multi-qubit Pauli
        second can be multiple and it would multiply them all
        with the first argument
        
        first argument must be a collumn vector
        
        '''
        
        #figure out what the input things are
        
        dictFlag1 = False
        strFlag1 = False
        HaahFlag1 = False
        if type(thing1) == dict:
            dictFlag1 = True
            
            shape1 = thing1[(0,0)].shape
            
            if (type(thing1[(0,0)][0,0]) == numpy.str_):
                strFlag1 = True
            else:
                HaahFlag1 = True
        else:
            shape1 = thing1.shape
            
            if len(shape1)>1:
                if type(thing1[0,0]) == numpy.str_:
                    strFlag1 = True
                else:
                    HaahFlag1 = True
            else:
                if type(thing1[0]) == numpy.str_:
                    strFlag1 = True
                else:
                    HaahFlag1 = True
                    
        dictFlag2 = False
        strFlag2 = False
        HaahFlag2 = False
        if type(thing2) == dict:
            dictFlag2 = True
            
            shape2 = thing2[(0,0)].shape
            
            if (type(thing2[(0,0)][0,0]) == numpy.str_):
                strFlag2 = True
            else:
                HaahFlag2 = True
        else:
            shape2 = thing2.shape
            
            if len(shape2)>1:
                if type(thing2[0,0]) == numpy.str_:
                    strFlag2 = True
                else:
                    HaahFlag2 = True
            else:
                if type(thing2[0]) == numpy.str_:
                    strFlag2 = True
                else:
                    HaahFlag2 = True
                    
        if not dictFlag1 == dictFlag2:
            raise ValueError('incompatible types: not both dictionaries')
            
        if not strFlag1 == strFlag2:
            raise ValueError('incompatible types: not both str type storage')
            
        if not shape1[0] == shape2[0]:
            raise ValueError('incompatible types: unequal numbers of qubits')
                
        if shape1[1]>1:
            raise ValueError('this function cannot multiple two matrices, only a vector and a matrix')
        
        if dictFlag1:
            #do the multiplication for dictionaries
            newDict= {}
            for key1 in thing1.keys():
                if key1 in thing2.keys():
                    #both of these dictionaries have this monomial, so I need to multiply
                    vec1 = thing1[key1][:,0]
                    mat2 = thing2[key1]
                    if strFlag1:
                        newMat = numpy.zeros(shape2, dtype = 'U21')
                    else:
                        newMat = newMat = numpy.zeros(shape2)
                        
                    for ind in range(0, shape2[1]):
                        vec2 = mat2[:,ind]
                        newVec = self._multiply_pauli_matrices(vec1,vec2, strFlag1)
                        newMat[:,ind] = newVec
                    
                    newDict[key1] = newMat
                else:
                    #only dict 1 has this guy
                    newDict[key1] = thing1[key1]
            for key2 in thing2.keys():
                if key2 in thing1.keys():
                    #already got this guy
                    pass
                else:
                    #only dict 2 has this guy
                    newDict[key2] = thing2[key2]
            output = newDict
        else:
            #these are matrices
            vec1 = thing1[:,0]    
            mat2 = thing2
            
            if strFlag1:
                newMat = numpy.zeros(shape2, dtype = 'U21')
            else:
                newMat = newMat = numpy.zeros(shape2)
                
            for ind in range(0, shape2[1]):
                vec2 = mat2[:,ind]
                newVec = self._multiply_pauli_matrices(vec1,vec2, strFlag1)
                newMat[:,ind] = newVec
                
            output = newMat
        

        return output                  
    
    ########
    # symplectic dot product and commutation checks
    ########
    
    def make_J(self, shape1):
        if not numpy.mod(shape1,2) == 0:
            raise ValueError('need even numbers of rows for this')
            
        Jmat = numpy.zeros((shape1, shape1))
        for qind in range(0, int(shape1/2)):
            Jmat[2*qind, 2*qind+1] = 1
            Jmat[2*qind+1, 2*qind ] = 1
        return Jmat
    
    def _matrix_symplectic_dot(self, mat1, mat2, debugMode = False):
        '''This function will define the symplectic dot product
        on two numpy arrays
        
        There will be a higher lever wrapper to define this on the Lorent 
        Pauli dictionaries
        
        Define the symplectic inner product mod 2 with X1Z1X2Z2...XnZn bit convention
        
        mat1^t * A*mat2 (I think)
        
        
        I believe that the output comes out in the following format
        
        anticommutation between m1[:,n] and m2[:,m] appears in output[n,m]
        
        
        '''
        
        m1 = self.convert_np_to_nmod(mat1, modulus = 2)
        m2 = self.convert_np_to_nmod(mat2, modulus = 2)
        
        shape1 = mat1.shape
        shape2 = mat2.shape
        
        if not shape1[0] == shape2[0]:
            raise ValueError('These two matrices are defined on different numbers of qubits')
        elif numpy.mod(shape1[0],2)==1:
            raise ValueError('This functions requires Haah style matrices. \n And these matrices have an odd number of rows')
        else:
            #let's try this
            Jmat = self.make_J(shape1[0])
                
            J = self.convert_np_to_nmod(Jmat, modulus = 2)
            
            finalMat = m1.transpose()*J*m2 #do the mod 2 matrix multiplication
            
            outputMat = self.convert_nmod_to_np(finalMat) #convert back to np array
            
        if debugMode:
            m1_out = self.convert_nmod_to_np(m1)
            m2_out = self.convert_nmod_to_np(m2)
            J_out = self.convert_nmod_to_np(J)
            return outputMat, m1_out, m2_out, J_out, m1, m2, J
        else:
            return outputMat
    
    def check_Lorent_commutation(self, dict1, dict2):
        '''this function uses the symplectic dot product trick
        to check wether the entries in two Lorent-valued dictionaries
        of my string form commut or not. Returns a matrix of the anticommutations
        
        for now it does not check whether there is a problem with translates'''
        
        temp1 = dict1[(0,0)]
        temp2 = dict2[(0,0)]
        
        nQubits1 = temp1.shape[0]
        nQubits2 = temp2.shape[0]
        
        if not nQubits1 == nQubits2:
            raise ValueError('Numbers of qubits not euql between these. Cannot compare')
            
        if len(temp1.shape) ==1:
            #this guy is just a vector
            nTerms1 = 1
        else:
            nTerms1 = temp1.shape[1]
            
        if len(temp2.shape) ==1:
            #this guy is just a vector
            nTerms2 = 1
        else:
            nTerms2 = temp2.shape[1]
            
        HaahDict1 =  self.convert_string_to_Haah(dict1)  
        HaahDict2 =  self.convert_string_to_Haah(dict2)   
            
        anticommutations = numpy.zeros((nTerms1, nTerms2))
        
        for key in HaahDict1.keys():
            if key in HaahDict2.keys():
                #both of these dictionaries include qubits in thi unit cell
                #,so we need to actually check.
                HaahMat1 = HaahDict1[key]
                HaahMat2 = HaahDict2[key]
                
                delta = self._matrix_symplectic_dot(HaahMat1, HaahMat2)
                
                anticommutations = anticommutations + delta
        
        #take the antocommutations mod2
        anticommutations = numpy.mod(anticommutations, 2)
        
        return anticommutations
    
    def _extract_nullspace(self, modMat):
        '''takes an nmod mat and extracts the nullspace
        
        because the syntax is annoying'''
        if not type(modMat) == nmod_mat:
            raise ValueError('this is for nmod matrices only')
        else:
            # print(modMat.rank())
            # X, nullity= modMat.nullspace()
            # print(nullity)
            
            #find everything that commutes with H and the stabilizers
            temp2 = modMat.nullspace()
            num_commuting_things = temp2[1]
            oversizeMatrix = temp2[0] #first few collumns of this are the basis of the null space
            
            if num_commuting_things == 0:
                #there is nothing that commutes
                
                return [], []
            else:
                
                #arg! Taking cust of this matrix is stupid!!
                #######
                #thing.table gives a 2D list that must be accessed as [n][m]
                #it can be trimmed as a list, but to work with it, it 
                #then needs to be case to np array, and it needs to be np anyway
                #to get an nmod_mat of the reduced size
                
                #the other way is to cast the whole thing back to np, then cut
                #then cast back
                
                out = self.convert_nmod_to_np(oversizeMatrix)
                nullspace_basis = out[:,0:num_commuting_things]
                nullspace_basis_mod2 = self.convert_np_to_nmod(nullspace_basis)
                
                #I think this latter should work too, but something might be going wrong
                # nullspace_basis_list = oversizeMatrix.table()[:][0:num_commuting_things]
                # nullspace_basis= numpy.asarray(nullspace_basis_list)
                # nullspace_basis_mod2 = self.convert_np_to_nmod(nullspace_basis)
                # # # nullspace_basis = self.convert_nmod_to_np(nullspace_basis_mod2)
                
                return nullspace_basis, nullspace_basis_mod2
            
        
    
    def find_commutant(self, inputThing, debugMode = False):
        ''' 
        takes either dictionary definition for a set of paulis in string form
        or a straight up mod 2 matrix in Haah form
        
        and computes the commutant
        
        this will not work on Lorent-valued things
        
        
        Finally fixed it so that it  gives actual states, I think.
        
        '''
        if type(inputThing) == nmod_mat:
            G_mod2 = inputThing
            Gmat = self.convert_nmod_to_np(inputThing)
        elif type(inputThing) == dict:
            if len(inputThing.keys())> 1:
                raise ValueError('This function can only handle the monomial x^0Y^0')
            else:
                fullDict_Haah = self.convert_string_to_Haah(inputThing)
                Gmat = fullDict_Haah[(0,0)]
                G_mod2 = self.convert_np_to_nmod(Gmat)
        elif type(inputThing) == numpy.ndarray:
            G_mod2 = self.convert_np_to_nmod(inputThing)
            Gmat = inputThing
            
        # print(G_mod2.ncols())   #number of terms in the thing we are getting the null space of
        # print(G_mod2.nrows())   #number of qubits *2    
        
        #symplectic interpose matrix
        shape0 = G_mod2.nrows()
        Jmat = self.make_J(shape0)
        J_mod2 = self.convert_np_to_nmod(Jmat)
        
        #find everything that commutes with H and the stabilizers
        # temp = (G_mod2.transpose()*J_mod2).transpose() 
        #whoops I think the collumn vector version has no null space, unless collumns are redundant
        temp = (G_mod2.transpose()*J_mod2)
        
        ####
        #!!!!! I think that this may not detect redundant collumns in the original matrix?
        ###
        
        # temp2 = temp.nullspace()
        # num_commuting_things = temp2[1]
        # oversizeMatrix = temp2[0] #first few collumns of this are the basis of the null space
        # out = self.convert_nmod_to_np(oversizeMatrix)
        # nullspace_basis = out[:,0:num_commuting_things]
        nullspace_basis, nullspace_basis_mod2 = self._extract_nullspace(temp)
        
        

        # print(nullspace_basis_mod2.ncols()) #this ends up being the number of terms in nulls space
        # print(nullspace_basis_mod2.nrows()) #number of terms in the thing we are getting the null space of
        
        
        # SL_mod2 = G_mod2 * nullspace_basis_mod2
        # SL_mod2 = (G_mod2.transpose() *nullspace_basis_mod2).transpose()
        # SL_mod2 = (G_mod2.transpose()*J_mod2).transpose()*nullspace_basis_mod2
        SL_mod2 =nullspace_basis_mod2
        SL = self.convert_nmod_to_np(SL_mod2)
        
        if debugMode:
            return nullspace_basis, nullspace_basis_mod2, SL, SL_mod2, Gmat, G_mod2, J_mod2
        
        #these now contain a basis for the things that commute
        #with the union of H and the stabilizers
        # SL = nullspace_basis
        
        if type(inputThing) == nmod_mat:
            output = self.convert_np_to_nmod(SL)
        elif type(inputThing) == dict:
            output = {}
            output[(0,0)] = self.convert_Haah_to_string(SL)

        elif type(inputThing) == numpy.ndarray:
            output = SL
        
        return output
    
    
    def canonical_basis(self, Hdict, Sdict, verbose = False, nonCommutant = False):
        '''my attempt at copying Steve's canonical basis function
        
        give it a Hamiltonian and stabilziers
        
        and it should find eveyrthing that commutes with them and
        divide them into stabilizers, logical Xs and logical Zs
        
        
        If there is no Hamiltonain, you can pass a blank string in the 
        place of the Hamiltonian
        
        If there are no stabilizers, you can pass a blank string in their place
        
        
        Safety checks added
        -S should commute with H
        -S should commute with S
        
        If you don't know what S should be, then just leave it blank and the code will find them.'
        
        
        If nonCommutant = True, then it will run symplectic basis on the Hamiltonian 
        instead of the commutant.
        
        '''
        if self.verbose:
            verbose = True
        
        if type(Hdict) == str:
            #use stabilizers only
            fullDict = Sdict
            
            #check whether the stabilizers commute witheachother
            flagMat = self.check_Lorent_commutation(Sdict, Sdict)
            if len(numpy.where(flagMat !=0)[0])>0:
                raise ValueError('This set of stabilizers do not commute with themselves')
                
        elif type(Sdict) == str:
            #use Hamiltonian only
            fullDict = Hdict
        else:
            #put eveyrthing together
            fullDict = self.combine_pauli_terms(Hdict, Sdict)
            
            #check whether the stabilizers commute with H
            flagMat = self.check_Lorent_commutation(Hdict, Sdict)
            if len(numpy.where(flagMat !=0)[0])>0:
                raise ValueError('This Hamiltonian does not commute with this set of stabilizers')
                
            #check whether the stabilizers commute witheachother
            flagMat = self.check_Lorent_commutation(Sdict, Sdict)
            if len(numpy.where(flagMat !=0)[0])>0:
                raise ValueError('This set of stabilizers do not commute with themselves')
        
        #find everything that commutes with what we already have
        if nonCommutant:
            SLdict = Hdict
        else:
            SLdict = self.find_commutant(fullDict)
        SL = self.convert_string_to_Haah(SLdict[(0,0)])
        SL_mod2 = self.convert_np_to_nmod(SL)
        
        if verbose:
            #do some size checking
            Heff_Haah = self.convert_string_to_Haah(fullDict[(0,0)])
            Heff_mod2 = self.convert_np_to_nmod(Heff_Haah)
            
            everything = self.combine_pauli_terms(fullDict , SLdict )
            everythingHaah = self.convert_string_to_Haah(everything[(0,0)])
            everything_mod2 = self.convert_np_to_nmod(everythingHaah )
            
            print('total hamiltonian and stabilizer terms = ' + str(Heff_mod2.ncols()))
            print('independent hamiltonian and stabilizer terms = ' + str(Heff_mod2.rank()))
            
            print('size of the commutant of H and S = ' + str(SL_mod2.ncols()))
            print('independent things in the commutant of H and S = ' + str(SL_mod2.rank()))
            
            print('size of the commutant(H and S) and H,S = ' + str(everything_mod2.ncols()))
            print('independent things in the H,S, commutant together = ' + str(everything_mod2.rank()))
        
        #now I need to monogamize 
        M = numpy.copy(SL)
        M_mod2 = self.convert_np_to_nmod(SL)
        
        R_mod2, dim = M_mod2.rref() #I'm not sure what I'm supposed to do with this
        #reduced row eschelon form
        #e.g. M_mod2 = M_mod2.transpose()*R_mod2
        
        # #symplectic interposer matrix
        # shape0 = M_mod2.nrows()
        # Jmat = self.make_J(shape0)
        # J_mod2 = self.convert_np_to_nmod(Jmat)
        
        #now we need to process everything else into monogamous pairs and stabilizers
        newM = numpy.copy(M)
        newXs = numpy.zeros(M.shape) #will trim excess size later
        newZs = numpy.zeros(M.shape) #will trim excess size later
        newSs = numpy.zeros(M.shape) #will trim excess size later
            
        nind = 0
        sind = 0
        while newM.shape[1] > 0:
            v = newM[:,0]
            newM = newM[:,1:]
            
            
            y = self._matrix_symplectic_dot(v, newM)
            
            hits = numpy.where(y != 0) [1] #collumn vector stuff, so I take the second argument
            # print(newM.shape[1])
            # print(hits)
            
            if len(hits) > 0:
                #there are some things that this vector does not commute with
        
                #take the first anticommuting thing and keep it
                currInd = hits[0]
                w = newM[:,currInd]
                
                #cut this vector out
                newM = numpy.concatenate((newM[:,0:currInd], newM[:,currInd+1:]), axis = 1)
                
                #now I need to add v to the xs, and w to the Zs
                newXs[:,nind] = v
                newZs[:,nind] = w
                nind = nind+1
        
                #fix the remaining things that anticommute
        
        
                #look for stuff in the reamainder which anticommut with v and w
                test1 = self._matrix_symplectic_dot(v, newM)
                hits1 = numpy.where(test1 != 0) [1]
                if len(hits1) ==0:
                    #nothing else anticommutes with v
                    pass
                else:
                    #now we have to fix stuff
                    for ind1 in hits1:
                        #these are the locations of the things that need to be fixed
                        oldVec = newM[:,ind1]
                        newVec = numpy.mod(oldVec+w,2)
                        newM[:,ind1] = newVec 
                
                test2 = self._matrix_symplectic_dot(w, newM)
                hits2 = numpy.where(test2 != 0) [1]
                if len(hits2) == 0:
                    #nothing else anticommutes with w
                    pass
                else:
                    #now we have to fix stuff
                    for ind2 in hits2:
                        #these are the locations of the things that need to be fixed
                        oldVec = newM[:,ind2]
                        newVec = numpy.mod(oldVec+v,2)
                        newM[:,ind2] = newVec   
                
            else:
                #this commutes with everything, so file it as a stabilizer
                newSs[:,sind] = v
                sind = sind+1
        
        #trim excess zeros
        newXs = newXs[:,0:nind]
        newZs = newZs[:,0:nind]
        newSs = newSs[:,0:sind]
        
        
        #return to convenient dictionary forms
        xDict_Haah = {}
        xDict_Haah[(0,0)] = newXs
        xDict = self.convert_Haah_to_string(xDict_Haah)
        
        zDict_Haah = {}
        zDict_Haah[(0,0)] = newZs
        zDict = self.convert_Haah_to_string(zDict_Haah)
        
        sDict_Haah = {}
        sDict_Haah[(0,0)] = newSs
        sDict = self.convert_Haah_to_string(sDict_Haah)
        
        return sDict, xDict, zDict
             
    
    
    ########
    #very general utilities
    ########


    def make_torus_vertices(self):

        self.torusXs = numpy.zeros(self.numQubits)
        self.torusYs = numpy.zeros(self.numQubits)
                
        mind = 0
        for xind in range(0, self.xSize):
            offset1 = self.unitcell.a1*xind
            for yind in range(0, self.ySize):
                offset2 = self.unitcell.a2*yind
                
                for qind in range(0, self.qubitsPerCell):
                    x0 = self.cellXs[qind]
                    y0 = self.cellYs[qind]
                    
                    [x,y] = numpy.asarray([x0,y0]) + offset1 + offset2
                    
                    self.torusXs[mind] = x
                    self.torusYs[mind] = y
                    
                    mind = mind+1
        return

    def find_torus_index(self, siteIndex, xPos, yPos ):
        '''takes in the number of a site inside the unit cell,
        and the x and y locations of the unit cell, and returns the final
        absolute index of that qubit on the torus
        '''
        
        ###tying to rmod the coordinates individually instead of the absolute index
        
        x_eff = numpy.mod(xPos, self.xSize)
        y_eff = numpy.mod(yPos, self.ySize)
        newInd= x_eff*self.ySize*self.qubitsPerCell + y_eff*self.qubitsPerCell + siteIndex
        
        # #old version
        # rawInd = xPos*self.ySize*self.qubitsPerCell + yPos*self.qubitsPerCell + siteIndex
        # # p1 = 2*colPos*codeSize + 2*rowPos
        # # p2 = 2*colPos*codeSize + 2*(rowPos+1)
        # # p3 = 2*(colPos)*codeSize + 2*(rowPos+1) + 1
        # # p4 = 2*(colPos-1)*codeSize + 2*(rowPos+1) + 1
        # newInd = numpy.mod(rawInd, self.xSize*self.ySize*self.qubitsPerCell)
        
        return newInd
    
    
    
    def make_incidence_matrix(self, flagEnds = False):
        '''
        loops through the positions of the vertices in the unit cell
        and compares to the resonators and figures out who is connected to whom
        
        This will let me know which qubits are incident on which vertices (i.e. Hamiltonian terms)
        
        makes a matrix are the rows  of this format
        
        [vertex number, edge/qubit number, x shift, y shift]
        this says the the given edge, whien moved to the unit cell given by dX and dY is
        incident at this vertex
        
        optional argument flagEnds will trigger it to keep a record of which end of the resonator
        touches that vertex.
        This will help with fiducial bosonizations.
        0 for the first end, 1 for the second
        
        '''
        #complex encoding of the locations of the vertices
        vertVec = self.unitcell.rootCoords[:,0] + 1j*self.unitcell.rootCoords[:,1]
        
        #complex encodings of the start points of all resonators
        sVec = self.unitcell.resonators[:,0] + 1j * self.unitcell.resonators[:,1]
        #complex encoding of the end points of all the resonators
        tVec = self.unitcell.resonators[:,2] + 1j * self.unitcell.resonators[:,3]
        
        blankList = []
        for vind in range(0, self.verticesPerCell):
            vertPos = vertVec[vind]
            
            for rind in range(0, len(sVec)):
                ss = sVec[rind]
                tt = tVec[rind]
                
                for dX in [-1,0,1]:
                    for dY in [-1,0,1]:
                        temp = dX*self.unitcell.a1 + dY*self.unitcell.a2
                        shift = temp[0] + 1j*temp[1]
                        
                        hitFlag = 0
                        endFlag = -1
                        if numpy.round(vertPos,3) == numpy.round(ss+shift, 3):
                            hitFlag = 1
                            endFlag = 0
                        if numpy.round(vertPos,3) == numpy.round(tt+shift, 3):
                            hitFlag = 1
                            endFlag = 1
                        
                        if hitFlag:
                            #this vertex is connected to this resonator at this offset
                            if flagEnds:
                                blankList.append([vind, rind, dX, dY, endFlag])
                            else:
                                blankList.append([vind, rind, dX, dY])
        self.cellIncidence = numpy.asarray(blankList)
        return
    
    def make_edge_incidence_matrix(self):
        '''
        loops through the positions of the vertices in the unit cell
        and compares to the resonators and figures out who is connected to whom
        
        this will keep track of what vertices lie at each end of the resonator
        
        makes a matrix are the rows  of this format
        
        [vertex1, x shift1, y shift1, vertex2, xshift2, yshift2]
        this says the given edge starts at vertex 1 in cell x1, y1 and ends ate vertex in cell x2, y2
        
        
        '''
        #complex encoding of the locations of the vertices
        vertVec = self.unitcell.rootCoords[:,0] + 1j*self.unitcell.rootCoords[:,1]
        
        #complex encodings of the start points of all resonators
        sVec = self.unitcell.resonators[:,0] + 1j * self.unitcell.resonators[:,1]
        #complex encoding of the end points of all the resonators
        tVec = self.unitcell.resonators[:,2] + 1j * self.unitcell.resonators[:,3]
        
        blankList = numpy.zeros((self.unitcell.resonators.shape[0], 6))
        
            
        for rind in range(0, len(sVec)):
            ss = sVec[rind]
            tt = tVec[rind]
            
            v0 = numpy.NaN
            x0 = numpy.NaN
            y0 = numpy.NaN
            
            v1 = numpy.NaN
            x1 = numpy.NaN
            y1 = numpy.NaN
            
            for vind in range(0, self.verticesPerCell):
                vertPos = vertVec[vind]
                
                for dX in [-1,0,1]:
                    for dY in [-1,0,1]:
                        temp = dX*self.unitcell.a1 + dY*self.unitcell.a2
                        shift = temp[0] + 1j*temp[1]
                        
                        if numpy.round(vertPos,3) == numpy.round(ss+shift, 3):
                            v0 = vind
                            x0 = -dX #minu sign because I was shifting the cell
                            y0 = -dY
                        if numpy.round(vertPos,3) == numpy.round(tt+shift, 3):
                            v1 = vind
                            x1 = -dX
                            y1 = -dY
                        

            blankList[rind,:] = [v0, x0, y0, v1, x1, y1]
        self.cellEdgeIncidence = numpy.asarray(blankList)
        return
    
    
    def make_fiducial_H(self, check_plot = False):
        if len(self.vertex_assignment) > 0:
            #user has specified how the vertices should be colored
            
            if not len(self.vertex_assignment) == self.verticesPerCell:
                #this assignment is not the correct size
                raise ValueError('Vertex assignment is the wrong length. There are ' + str(self.verticesPerCell) + ' vertices')
            else:
                pauliDict = {}
                pauliDict[(0,0)] = self.blank_pauli_mat(self.qubitsPerCell, self.verticesPerCell)
                
                
                for vind in range(0, self.verticesPerCell):
                    currPauli = self.vertex_assignment[vind]
                    if not currPauli in ['I', 'X', 'Y', 'Z']:
                        raise ValueError('Invalid Pauli. Should be I, X, Y, or Z')
                    
                    couplings = numpy.where(self.cellIncidence[:,0] == vind)[0]
                    
                    for cind in couplings:
                        qNum, dX, dY = self.cellIncidence[cind,1:]
                        
                        dX = int(dX)
                        dY = int(dY)
                        key = (dX, dY)
                        if key in pauliDict.keys():
                            #we already have some terms of this type
                            pauliDict[key][qNum, vind] = currPauli
                        else:
                            pauliDict[key] = self.blank_pauli_mat(self.qubitsPerCell, self.verticesPerCell)
                            pauliDict[key][qNum, vind] = currPauli
                

        else:
            #user has not specificied a coloring for the vertices, so I will do something simple. 
            pauliDict = {}
            pauliDict[(0,0)] = self.blank_pauli_mat(self.qubitsPerCell, self.verticesPerCell)
        
            for qind in range(0, self.qubitsPerCell):
                couplings = numpy.where(self.cellIncidence[:,1] == qind)[0]
                
                if not len(couplings)==2:
                    print('warning. making fiducial H went wrong')
                else:
                    for cind in range(0,2):
                        vert, qNum, dX, dY = self.cellIncidence[couplings[cind],:]
                        #vertex is connected to this qubit/edge when that it moved
                        #to the cell given by dX, dY
                        
                        #the vertex we are grabbing is the far end of the resonator
                        #so , I think we should keep dX and dY
                        
                        key = (dX, dY)
                        if cind == 0:
                            pauli = 'X'
                        else:
                            pauli = 'Z'
                        if key in pauliDict.keys():
                            pauliDict[key][qind,vert] = pauli
                        else:
                            pauliDict[key] = self.blank_pauli_mat(self.qubitsPerCell, self.verticesPerCell)
                            pauliDict[key][qind,vert] = pauli
        if check_plot:
            fig = pylab.figure(44)
            pylab.clf()
            
            for vind in range(0, self.verticesPerCell):
                ax = pylab.subplot(1, self.verticesPerCell, vind+1)
                self.plot_single_term(pauliDict, vind, fignum = -1, 
                     numberSites = True,
                     spotSize = 400,
                     axis = ax)
                self.unitcell.draw_resonators(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
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
        pauliDict[(0,0)] = self.blank_pauli_mat(self.qubitsPerCell, self.verticesPerCell*fudgeFactor)
        
        
        tind = 0
        for vind in range(0, self.verticesPerCell):
            
            linkedVertInds = numpy.where(self.unitcell.rootLinks[:,0] == vind)[0]
            linkedVerts = numpy.unique(self.unitcell.rootLinks[linkedVertInds,1])
            
            linkedQubitInds = numpy.where(self.cellIncidence[:,0] == vind)[0]
            linkedQubits = numpy.unique(self.cellIncidence[linkedQubitInds,1])
            
            # print(linkedQubits)
            
            #find all the qubits that are linked to this guy, inside and outside of cell
            linkList = []
            for qind1 in linkedQubits:
                if self.cellEdgeIncidence[qind1][0] == vind:
                    #this guy vertex is at the starting end of the resonator
                    
                    #if resonator in the origin unit cell hits this vertex in the unit cell, great
                    # but if reconator in the origina unit cell hits the (1,0) copy of this vertes
                    #then I need to shift the resonator back to hit this guy in this cell
                    resX = int(-self.cellEdgeIncidence[qind1][1])
                    resY = int( -self.cellEdgeIncidence[qind1][2])
                    linkList.append([qind1, resX, resY, 0])
                    
                if self.cellEdgeIncidence[qind1][3] == vind:
                    #this guy vertex is at the terminating end of the resonator
                    
                    #if resonator in the origin unit cell hits this vertex in the unit cell, great
                    # but if reconator in the origina unit cell hits the (1,0) copy of this vertes
                    #then I need to shift the resonator back to hit this guy in this cell
                    resX = int(-self.cellEdgeIncidence[qind1][4])
                    resY = int(-self.cellEdgeIncidence[qind1][5])
                    linkList.append([qind1, resX, resY, 1])
            
            # return linkList
            #now to build the dimers
            for lind1 in range(0,len(linkList)):
                qNum1, dX1, dY1, end = linkList[lind1]
                key1 = (dX1, dY1) #the location of the qubit that we need to put a pauli on for this dimer
                
                #now I need to find the Pauli on the far end of this edge
                if end ==0:
                    otherEnd = self.cellEdgeIncidence[qNum1,3:]
                else:
                    otherEnd = self.cellEdgeIncidence[qNum1,0:3]
                v1 = int(otherEnd[0])
                dX1 = int(otherEnd[1])
                dY1 = int(otherEnd[2])
                
                #I need to grab from the Hamiltonian term at v1
                #and for the qubit qNum1
                #but, the monomial I need to grab is more complicated
                #I want the pauli that wind up on this edge when I translate by
                #dX1 and dY1, so I think I need to pull from the Hamiltonian at the inverse position
                
                pauli1 = self.fiducialH[(-dX1, -dY1)][qNum1, v1]
                
                for lind2 in range(lind1+1,len(linkList)):
                    if lind1 == lind2:
                        pass
                    else:
                        
                        qNum2, dX2, dY2, end = linkList[lind2]
                        key2 = (dX2, dY2) #the location of the qubit that we need to put a pauli on for this dimer
                        
                        #now I need to find the Pauli on the far end of this edge
                        if end ==0:
                            otherEnd = self.cellEdgeIncidence[qNum2,3:]
                        else:
                            otherEnd = self.cellEdgeIncidence[qNum2,0:3]
                        v2 = int(otherEnd[0])
                        dX2 = int(otherEnd[1])
                        dY2 = int(otherEnd[2])
                        
                        #I need to grab from the Hamiltonian term at v1
                        #and for the qubit qNum1
                        #but, the monomial I need to grab is more complicated
                        pauli2 = self.fiducialH[(-dX2, -dY2)][qNum2, v2]
                        
                        
                        if key1 in pauliDict.keys(): 
                            pauliDict[key1][qNum1, tind] = pauli1
                        else:
                            pauliDict[key1] = self.blank_pauli_mat(self.qubitsPerCell, self.verticesPerCell*fudgeFactor)
                            pauliDict[key1][qNum1, tind] = pauli1
                            
                        if key2 in pauliDict.keys(): 
                            pauliDict[key2][qNum2, tind] = pauli2
                        else:
                            pauliDict[key2] = self.blank_pauli_mat(self.qubitsPerCell, self.verticesPerCell*fudgeFactor)
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
                self.unitcell.draw_resonators(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
                pylab.title(str(vind))
                
            pylab.tight_layout()
            pylab.tight_layout()
            pylab.show()
            
            pylab.suptitle('fiducial dimer check')
        return pauliDict
                
    def make_fiducial_dimers_old_version(self, check_plot = False):
        ''' 
        makes all the dimer operators
        
        For the moment it only works for 3-regular graphs, because I need to predict with number of dimers
        in order to allocate space.
        
        
        This currently fails if there is only one vertex in the unit cell.
        This means that the same qubits participates in a term twice,
        and it gets confused which end to use.
        
        '''

        pauliDict = {}
        fudgeFactor = 8
        pauliDict[(0,0)] = self.blank_pauli_mat(self.qubitsPerCell, self.verticesPerCell*fudgeFactor)
        
        
        tind = 0
        for vind in range(0, self.verticesPerCell):
            
            couplings = numpy.where(self.cellIncidence[:,0] == vind)[0]
            
            for cind in couplings:
                qNum1, dX1, dY1 = self.cellIncidence[cind,1:]
                dX1 = int(dX1)
                dY1 = int(dY1)
                for cind2 in couplings:
                    if cind < cind2:
                        #this is a mixed pair so I want to include it
                        #and I get rid of dupilcates by keeping the first index smaller than the second
                        qNum2, dX2, dY2 = self.cellIncidence[cind2,1:]
                        dX2 = int(dX2)
                        dY2 = int(dY2)
                
                        #imagine the first one is in the unit cell, so I need to comine the shifts
                        dX = dX2 - dX1
                        dY = dY2 - dY1
                
                        key = (dX, dY)
                        
                        #now to figure out the right paulis. 
                        # need the far end of each resonator
                        incidences1 = numpy.where(self.cellIncidence[:,1] == qNum1)[0]
                        incidences2 = numpy.where(self.cellIncidence[:,1] == qNum2)[0]
                        
                        #these are the vertices that this edge touches
                        Vs1 = self.cellIncidence[incidences1,0] 
                        Vs2 = self.cellIncidence[incidences2,0]
                        
                        # #find out which ones are not the middle vertex
                        # temp1 = numpy.where(Vs1 != vind)[0][0]
                        # temp2 = numpy.where(Vs1 != vind)[0][0]
                        # #and grab them
                        # v1 = Vs1[temp1]
                        # v2 = Vs2[temp2]
                        
                        # #find out which ones are not the middle vertex
                        hits1 = numpy.where(Vs1 != vind)[0]
                        hits2 = numpy.where(Vs1 != vind)[0]
                        if len(hits1) > 0:
                            temp1 = hits1[0]
                            #and grab them
                            v1 = Vs1[temp1]
                        else:
                            #this is the apthological case where both ends are the same vertex
                            #, just in a different unit cell
                            v1 = vind
                            
                            
                        if len(hits2) > 0:
                            temp2 = hits2[0]
                            #and grab them
                            v2 = Vs2[temp2]
                        else:
                            #this is the apthological case where both ends are the same vertex
                            #, just in a different unit cell
                            v2 = vind

                        
                        #now I need to find out what value each
                        #qubit has at the corresponding vertex 
                        #in the fiducial bosoniztion
                        
                        
                        
                        #!!!! this version works if the edge only hits a qubit once
                        #!!!!for more complicated cases, we have to figure out which end to use.
                        
                        for tempkey in self.fiducialH.keys():
                            temp = self.fiducialH[tempkey][qNum1, v1]
                            if not temp == 'I':
                                pauli1 = temp

                        for tempkey in self.fiducialH.keys():
                            temp = self.fiducialH[tempkey][qNum2, v2]
                            # print( [qNum2, v2])
                            if not temp == 'I':
                                pauli2 = temp
                    
                        
                        # pauli1 = self.vertex_assignment[v1]
                        # pauli2 = self.vertex_assignment[v2]
                        if key in pauliDict.keys():
                            #we already have some terms of this type
                            pauliDict[key][qNum2, tind] = pauli2
                            pauliDict[(0,0)][qNum1, tind] = pauli1
                        else:
                            pauliDict[key] = self.blank_pauli_mat(self.qubitsPerCell, self.verticesPerCell*fudgeFactor)
                            # print( [qNum2, v2])
                            pauliDict[key][qNum2, tind] = pauli2
                            pauliDict[(0,0)][qNum1, tind] = pauli1
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
                self.unitcell.draw_resonators(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
                pylab.title(str(vind))
                
            pylab.tight_layout()
            pylab.tight_layout()
            pylab.show()
            
            pylab.suptitle('fiducial dimer check')
        return pauliDict

    
    def check_unit_cell_labels(self, title = '', fignum = 1, figx = 6, figy = 6):
            
        fig1 = pylab.figure(fignum)
        pylab.clf()
        
        ax = pylab.subplot(1,1,1)
        pylab.scatter(self.cellXs, self.cellYs,c =  layoutCapColor, s = 400, marker = 'o', edgecolors = 'k',zorder = 2)
        
        
        for ind in range(0, self.cellXs.shape[0]):
            coords = [self.cellXs[ind], self.cellYs[ind]]
            pylab.text(coords[0], coords[1], str(ind), color='k', size=labelSize, fontdict = lighttext_params)
        
        if self.bosonization_type == 'edge':
            #plot the vertex numbering too
            Xs = self.unitcell.rootCoords[:,0]
            Ys = self.unitcell.rootCoords[:,1]
            pylab.scatter(Xs, Ys,c =  FWsiteColor, s = 200, marker = 'o', edgecolors = 'royalblue',zorder = 2)
        
            for ind in range(0, Xs.shape[0]):
                coords = [Xs[ind], Ys[ind]]
                pylab.text(coords[0], coords[1], str(ind), color='k', size=10, fontdict = lighttext_params)
        
        self.unitcell.draw_resonators(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
        
        
        ax.axis('off')
        ax.set_aspect('equal')
        pylab.title('unit cell')
        pylab.suptitle(title)
        
        
        fig1.set_size_inches([figx,figy])
        pylab.show()
        
    def check_vertex_assignment(self, title = '', fignum = 1, figx = 6, figy = 6):
            
        if len(self.vertex_assignment) == 0:
            print('there is no vertex assignment and I cannot check it')
            return
        else:
            fig1 = pylab.figure(fignum)
            pylab.clf()
            
            ax = pylab.subplot(1,1,1)
            pylab.scatter(self.cellXs, self.cellYs,c =  layoutCapColor, s = 400, marker = 'o', edgecolors = 'k',zorder = 2)
            
            if self.bosonization_type == 'edge':
                for ind in range(0, self.cellXs.shape[0]):
                    coords = [self.cellXs[ind], self.cellYs[ind]]
                    pylab.text(coords[0], coords[1], str(ind), color='k', size=labelSize, fontdict = lighttext_params)
                
    
            Xs = self.unitcell.rootCoords[:,0]
            Ys = self.unitcell.rootCoords[:,1]
            pylab.scatter(Xs, Ys,c =  FWsiteColor, s = 400, marker = 'o', edgecolors = 'royalblue',zorder = 2)
        
            for ind in range(0, Xs.shape[0]):
                coords = [Xs[ind], Ys[ind]]
                labelStr = str(ind) + ',' + self.vertex_assignment[ind]
                pylab.text(coords[0], coords[1], labelStr, color='k', size=10, fontdict = lighttext_params)
            
            self.unitcell.draw_resonators(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
            
            
            ax.axis('off')
            ax.set_aspect('equal')
            pylab.title('unit cell')
            pylab.suptitle(title)
            
            
            fig1.set_size_inches([figx,figy])
            pylab.show()
            
    
            return
    
    def blank_pauli_mat(self, n,m):
        temp = numpy.zeros((n,m), dtype = 'U21')
        for nind in range(0, n):
            for mind in range(0, m):
                temp[nind, mind] = 'I'
        return temp
    
    ########
    #import helpers for checking with the mathematica
    ########
    
    def clean_binary_txt(self, rawMat):
        '''takes a .txt from matlab that has only numerical entries and coverts
        it to a python string matrix, stripping out the curly brackets and the commas.
        
        The direct import numpy.loadtxt comes back with a lot of crap. this cleans it out.'''
        newMat = numpy.zeros(rawMat.shape, dtype = 'U21')
        for nind in range(0, rawMat.shape[0]):
            for mind in range(0, rawMat.shape[1]):
                currStr = rawMat[nind, mind]
        
                match = re.search(r'\{*(\w*)', currStr)
                if match:
                    # print(match.groups()[0])
                    currStr = match.groups()[0]
                else:
                    currStr = 'ERROR!'
                    
                newMat[nind, mind] = currStr
        return newMat
    
    def load_unpacked_mat_from_mathematica(self, filename, fromMathematica = False):
        '''takes in a matrix on the torus from mathamtica.
        
        It should have entries all 0 or 1 and is encoded in steve's way with 
        two bits for each qubit.
        
        this function loads it form a text file and cleans it.
        
        Then it is ready to be parsed.
        
        '''
        mat_raw = numpy.loadtxt(filename, dtype = 'U21')
        newMat = self.clean_binary_txt(mat_raw)
        return newMat
    
    
    
    def parse_text_operator_matrix(self, rawMat, debugMode = False, textMode = False):
        ''' 
        takes a possibly haah valued matrix and parses it into Pauli identifiers
        
        
        It does double duty and is meant to be run either on a Lorent valued matrix 
        describing the unit cell,
        or on a binary matrix describing operators on the torus.
        
        '''
        AdrianMat = numpy.asarray(rawMat)
        
        numQubits = int(AdrianMat.shape[0]/2)
        numTerms = AdrianMat.shape[1]
        
        pauliMat = numpy.zeros((numQubits, numTerms), dtype = 'U21')
        pauliDict = {}
        pauliDict[(0,0)] = self.blank_pauli_mat(numQubits, numTerms)
        for cind in range(0, AdrianMat.shape[1]):
            currCol = AdrianMat[:,cind]
            
            hits = numpy.where(currCol != '0')[0]
            
            pauliString = numpy.zeros(numQubits, dtype = 'U21')
            for ind in range(0,numQubits): #looping over qubits
                XFlag = 0
                ZFlag = 0
                
                
                matInd = 0
                locationList_X = []
                locationList_Z = []
                if 2*ind in hits:
                    #I need an X from this qubit
                    matInd = 2*ind
                    
                    currString = currCol[matInd] #eg. 1+X+XY
                    match = re.search(r'(\w*)\+*(\w*)\+*(\w*)\+*(\w*)\+*', currString)
    
                    
                    if match:
                        locationsFound = match.groups()
                        #print(locationsFound)
                        
                        for tind in range(0, len(locationsFound)):
                            location = locationsFound[tind]
                            if location == '':
                                pass
                            else:
                                locationList_X.append(location)
                 
                    
                if 2*ind + 1 in hits:
                    #I need a Z from this qubit  
                    matInd = 2*ind + 1
                    
                    currString = currCol[matInd]
                    match = re.search(r'(\w*)\+*(\w*)\+*(\w*)\+*(\w*)\+*', currString)
    
                    
                    if match:
                        locationsFound = match.groups()
                        #print(locationsFound)
                        
                        for tind in range(0, len(locationsFound)):
                            location = locationsFound[tind]
                            if location == '':
                                pass
                            else:
                                locationList_Z.append(location)
                    
                #put together all the locations that there should be Paulis for this qubit
                temp = numpy.concatenate((locationList_X, locationList_Z)) 
                     
                allLocations = numpy.unique(temp) #all the unit cells that have a pauli at this site in this operator
                #print(allLocations)
                
                if len(allLocations) == 0:
                    #there are no paulis here
                    operators = 'I0'
                else:
                    #parse
                    operators = ''
                    for location in allLocations:
                        XFlag = 0
                        ZFlag = 0
                        
                        #figure out if I get an X and a Z here
                        if location in locationList_X:
                            XFlag = 1
                        if location in locationList_Z:
                            ZFlag = 1
                            
                        #compile the net Pauli for this position and unit cell
                        if XFlag ==0 and ZFlag == 0:
                            #this qubit does not participate
                            pauliType = 'I'
                        if XFlag ==1 and ZFlag == 0:
                            #this qubit is X
                            pauliType = 'X'
                        if XFlag ==0 and ZFlag == 1:
                            #this qubit is Z
                            pauliType = 'Z'
                        if XFlag ==1 and ZFlag == 1:
                            #this qubit is Y
                            pauliType = 'Y'    
                            
                        
                        #parse which lorent polynomial I need
                        
                        if location == '1':
                            posType = '0'
                            n = 0
                            m = 0
                        elif location == 'X':
                            posType = 'a'
                            n = 1
                            m = 0
                        elif location == 'Y':
                            posType = 'b'
                            n=0
                            m=1
                        elif location == 'XY':
                            posType = 'ab'
                            n =1
                            m = 1
                        else:
                            posType = 'error'
                          
                        #compile the string needed
                        newPauli = pauliType + posType + '*'
                        operators = operators + newPauli
                        
                        
                        #store in the dictionary form
                        currKey = (n,m)
                        if currKey in pauliDict.keys():
                            pauliDict[currKey][ind, cind] = pauliType
                        else:
                            #haven't had a Lorent monomial like this before
                            pauliDict[currKey] = self.blank_pauli_mat(numQubits, numTerms)
                            
                            pauliDict[currKey][ind, cind] = pauliType
                        
                    #have compiled all the locations that contribute terms    
                    operators = operators[:-1] #trim the last underscore
                        
                #now we have added together all of the paulis that this qubit contributes in different unit cells 
                pauliString[ind] = operators    
                
    
                
                
                # pauliString[ind] = pauliType + posType
            pauliMat[:,cind] = pauliString
        if debugMode:
            return pauliMat, pauliString, numQubits, numTerms, pauliDict
        else:
            if textMode:
                return pauliMat
            else:
                return pauliDict
    
     
    def plot_single_pauli_string(self, pauliString, fignum = 7, 
                     title = 'blank', 
                     numberSites = True,
                     spotSize = 400,
                     figx = 6,
                     figy = 6):
        ''' Takes a single (possibly laurent-valued) pauli operator in string vector format
        and plots.
        
        e.g. [I0, X0, Y0, Xa_Yb, Zab] '''
    
        # if self.bosonization_type == 'edge':
        #     vertXs = self.unitcell.SDx
        #     vertYs = self.unitcell.SDy
        # if self.bosonization_type == 'vertex':
        #     vertXs = self.unitcell.rootCoords[:,0]
        #     vertYs = self.unitcell.rootCoords[:,1]
        
        currFig = pylab.figure(fignum)
        pylab.clf()
        ax = pylab.subplot(1,1,1)
        pylab.scatter(self.cellXs, self.cellYs,c =  layoutCapColor, s = spotSize, marker = 'o', edgecolors = 'k',zorder = 2)
        if numberSites:
            for ind in range(0 , len(self.cellXs)):
                coords = [self.cellXs[ind], self.cellYs[ind]]
                coords = coords + 0.06*self.unitcell.a2
                pylab.text(coords[0], coords[1], str(ind), color='k', size=labelSize, fontdict = lighttext_params)
        ax.axis('off')
        ax.set_aspect('equal')
          
        pylab.suptitle(title)
        
        for qind in range(0, len(pauliString)):
            allCurrPaulis = pauliString[qind]
        

            [x0,y0] = [self.cellXs[qind], self.cellYs[qind]]
            
            # match = re.search(r'(\w*)\_*(\w*)\_*(\w*)\_*(\w*)', allCurrPaulis)
            match = re.search(r'(\w*)\**(\w*)\**(\w*)\**(\w*)', allCurrPaulis)
            if match:
                paulisFound = match.groups()
                
                for currPauli in paulisFound:
                    if currPauli == '':
                        pass
                    else:
                        #print(currPauli)
                
                        pauliLabel = currPauli[0]
                        cellLabel = currPauli[1:]
                        
                        
                        delta1 = 0
                        delta2 = 0
                        if cellLabel == 'a':
                            delta1 = 1
                        elif cellLabel == 'b':
                            delta2 = 1
                        elif cellLabel == 'ab':
                            delta1 =1
                            delta2 = 1
                        [x,y] = numpy.asarray([x0,y0]) + delta1*self.unitcell.a1 + delta2*self.unitcell.a2
                        
                        # if cellLabel == 'a':
                        #     x = x0+jog
                        #     y = y0
                        # elif cellLabel == 'b':
                        #     x = x0
                        #     y = y0 + jogy
                        # elif cellLabel == 'ab':
                        #     x = x0+jog
                        #     y = y0 + jogy
                        # else:
                        #     x = x0
                        #     y = y0
                        
                        if pauliLabel == 'I':
                            pass
                        else: 
                            pylab.scatter([x], [y],c =  'deepskyblue', s = (spotSize + 50), marker = 'o', edgecolors = 'midnightblue',zorder = 2)
                            pylab.text(x, y, pauliLabel, color='k', size=labelSize, fontdict = lighttext_params)
            else:
                print( 'Error!!! :Failed to understand this term' )
        currFig.set_size_inches([figx,figy])
        return currFig    
    
    
    ########
    #term plotters for Haah valued stuff in the unit cell
    ########

    
    def plot_single_term(self, pauliDict, tind, fignum = 7, 
                     title = 'blank', 
                     numberSites = True,
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
            
        pylab.scatter(self.cellXs, self.cellYs,c =  layoutCapColor, s = spotSize, marker = 'o', edgecolors = 'k',zorder = 2)
        if numberSites:
            for ind in range(0 , len(self.cellXs)):
                coords = [self.cellXs[ind], self.cellYs[ind]]
                coords = coords + 0.06*self.unitcell.a2
                pylab.text(coords[0], coords[1], str(ind), color='k', size=labelSize, fontdict = lighttext_params)
        ax.axis('off')
        ax.set_aspect('equal')
          
        pylab.suptitle(title)
        
        self.unitcell.draw_resonators(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
        
        if self.bosonization_type == 'edge':
            #plot the vertex numbering too
            Xs = self.unitcell.rootCoords[:,0]
            Ys = self.unitcell.rootCoords[:,1]
            pylab.scatter(Xs, Ys,c =  FWsiteColor, s = 100, marker = 'o', edgecolors = 'royalblue',zorder = 2)
        
        
        
        for key in pauliDict.keys():
            n = int(key[0])
            m = int(key[1])
            
            pauliVec = pauliDict[key][:,tind]
            numQubits = pauliVec.shape[0]
            
            positionShift = n*self.unitcell.a1 + m * self.unitcell.a2
            
            for qind in range(0, numQubits):

                [x0,y0] = [self.cellXs[qind], self.cellYs[qind]]
                [x,y] = numpy.asarray([x0,y0]) + positionShift
                
                pauliLabel = pauliVec[qind]
                
                if pauliLabel == 'I':
                    pass
                else:
                    # #blue plots
                    # if pauliLabel =='X':
                    #     color =  'deepskyblue' #'deepskyblue' 'dodgerblue'
                    # elif pauliLabel == 'Y':
                    #     color = 'powderblue'#'lightskyblue'#'forestgreen'
                    # else:
                    #     color=  'dodgerblue' #'firebrick' #'indianred' #'dodgerblue'
                    # textcolor= 'k'
                    
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
        ''' Takes a single pauli matrix in dictionary form, 
        that is setup on the torus 
        
        should have only x^0y^0 monomials'''
    
        if len(torusPauliDict.keys()) > 1:
            #this matrix has too many Lorent monomials. 
            #It's not suitable for the real-space torus picture
            raise ValueError('This disctionary contains non-zero monomials. It shouldnt')
        
        if fignum < 0:
            ax = axis
            pylab.sca(ax)
        else:
            currFig = pylab.figure(fignum)
            pylab.clf()
            ax = pylab.subplot(1,1,1)
        
        pylab.scatter(self.torusXs, self.torusYs,c =  layoutCapColor, s = spotSize, marker = 'o', edgecolors = 'k',zorder = 2)
        if numberSites:
            for ind in range(0 , len(self.cellXs)):
                coords = [self.torusXs[ind], self.toruslYs[ind]]
                coords = coords + 0.06*self.unitcell.a2
                pylab.text(coords[0], coords[1], str(ind), color='k', size=labelSize/2, fontdict = lighttext_params)
        ax.axis('off')
        ax.set_aspect('equal')
          
        pylab.suptitle(title)
        
            
        pauliVec = torusPauliDict[(0,0)][:,tind]
        
        if not len(pauliVec) == len(self.torusXs):
            print('WARNING!!!!! This state is the WRONG LENGTH!!')
        
        numQubits = pauliVec.shape[0]
        
        positionShift = 0*self.unitcell.a1 + 0 * self.unitcell.a2
        
        for qind in range(0, numQubits):

            [x0,y0] = [self.torusXs[qind], self.torusYs[qind]]
            [x,y] = numpy.asarray([x0,y0]) + positionShift
            
            pauliLabel = pauliVec[qind]
            
            if pauliLabel == 'I':
                pass
            else:
                # #blue plots
                # if pauliLabel =='X':
                #     color =  'deepskyblue' #'deepskyblue' 'dodgerblue'
                # elif pauliLabel == 'Y':
                #     color = 'powderblue'#'lightskyblue'#'forestgreen'
                # else:
                #     color=  'dodgerblue' #'firebrick' #'indianred' #'dodgerblue'
                # textcolor= 'k'
                
                #high vis plots
                if pauliLabel =='X':
                    color =  'dodgerblue' #'deepskyblue' 
                elif pauliLabel == 'Y':
                    color = 'mediumblue'#'forestgreen'
                else:
                    color=  'firebrick'#'indianred' #'dodgerblue'
                textcolor = 'whitesmoke'
                    
                pylab.scatter([x], [y],c =  color, s = (spotSize + 50), marker = 'o', edgecolors = 'midnightblue',zorder = 2)
                pylab.text(x, y, pauliLabel, color=textcolor, size=labelSize/1.5, fontdict = lighttext_params)

        if drawLinks:
                self.lattice.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)

        # currFig.set_size_inches([figx,figy])
        # return currFig
        if fignum > 0:
            currFig.set_size_inches([figx,figy])
            return currFig   
        else:
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
        codeName= self.name + '_' + saveExtension + '_' + str(self.xSize) + '_' + str(self.ySize)
        
        saveDir = os.path.join('codeAutoPlots', codeName)
    
        if not os.path.isdir('codeAutoPlots'):
            os.mkdir('codeAutoPlots')
        
        if not os.path.isdir(saveDir):
            os.mkdir(saveDir)
        
        if len(torusPauliDict.keys()) > 1:
            #this matrix has too many Lorent monomials. 
            #It's not suitable for the real-space torus picture
            raise ValueError('This disctionary contains non-zero monomials. It shouldnt')
        
        baseStr = codeName + '_'
        for cind in range(0, torusPauliDict[(0,0)].shape[1]):
            print(cind)
            
            if inlineMode:
                fignum = fignum+1
            
            figName = baseStr + str(cind)# + '.png'
            currFig = self.plot_single_torus_term(torusPauliDict, cind, 
                                   fignum = fignum, 
                                   title = figName, 
                                   drawLinks = drawLinks,
                                   spotSize = spotSize,
                                   figx = figx,
                                   figy = figy)
            if drawLinks:
                # ax = pylab.subplot(1,1,1)
                ax = currFig.axes[0]
                self.lattice.draw_resonator_lattice(ax, color = layoutLineColor, alpha = 1 , linewidth = 2.5)
            if videoMode:
                currFig.canvas.draw()
                currFig.canvas.flush_events()
                time.sleep(videoTime)
            else:
                if not inlineMode:
                    savePath = os.path.join(saveDir, figName + '.png')
                    currFig.savefig(savePath, dpi = 200)  
                # savePath = os.path.join(saveDir, figName + '.png')
                # currFig.savefig(savePath, dpi = 200)
        
        return
    

    ########
    #unpacker to generate matrices on the torus from Lorent-valued stuff on the unit cell
    ########

    def unpack_to_torus(self, cellPauliDict):
        torusPauliDict = {}
        
        termsPerCell = cellPauliDict[(0,0)].shape[1]
        numTerms = termsPerCell * self.xSize *self.ySize
        torusPauliDict[(0,0)] = self.blank_pauli_mat(self.numQubits, numTerms)
        
        mind = 0
        for tind in range(0, termsPerCell): 
            for xind in range(0, self.xSize):
                for yind in range(0, self.ySize):
                    for monomialKey in cellPauliDict.keys():
                        n = int(monomialKey[0])
                        m = int(monomialKey[1])
                        
                        currVec = cellPauliDict[monomialKey][:, tind]
                        
                        for qind in range(0, self.qubitsPerCell):
                            if currVec[qind] == 'I':
                                #identity. Nothing to do
                                pass
                            else:
                                xPos = xind + n
                                yPos = yind + m
                                newInd = self.find_torus_index(qind, xPos, yPos)
                                torusPauliDict[(0,0)][newInd, mind] = currVec[qind]
                    mind = mind+1
        return torusPauliDict                                
                        
                    
                
                
        
        


if __name__=="__main__":      
    
    print('main loop')
    
    
    # grapheneCell = UnitCell('kagome')
    # splitGrapheneCell = grapheneCell.split_cell()
    # kapheneCell = splitGrapheneCell.line_graph_cell()

    # # code = PauliCode(kapheneCell, name = 'kaphene', xSize = 4, ySize = 4, 
    # #                  bosonization_type = 'edge',
    # #                  fiducial = True,
    # #                  vertex_assignment = ['X', 'Y', 'Z', 'X', 'Y', 'Z'],
    # #                  verbose = True)
    
    # code = PauliCode(kapheneCell, name = 'kaphene', xSize = 4, ySize = 4, 
    #                  bosonization_type = 'edge',
    #                  fiducial = True,
    #                  vertex_assignment = [],
    #                  verbose = True)
    

    # code.check_unit_cell_labels()
    
    # ###check that the text matrices are parsing and plitting correctly
    # # testMat = code.parse_text_operator_matrix(code.H0_raw, textMode = True)
    # # tind = 2
    # # testStr = testMat[:,tind]
    # # code.plot_single_pauli_string(testStr, fignum = 7, 
    # #                   title = 'string version', 
    # #                   numberSites = True,
    # #                   spotSize = 400,
    # #                   figx = 6,
    # #                   figy = 6)
    
    # # code.plot_single_term(code.H0, tind, fignum = 8, 
    # #                   title = 'dictionary version', 
    # #                   numberSites = True,
    # #                   spotSize = 400,
    # #                   figx = 6,
    # #                   figy = 6)

    # # # # ####test the term plotting on the Hamiltonian
    # # code.plot_terms(code.H0, saveExtension = '_LorentH')
    # # code.plot_terms(code.Lx0, saveExtension = '_LorentLx')
    
    
    
    # #####make some fake torus terms and plot them
    # lattice_size = code.numQubits
    
    # latticeTerms = {}
    # latticeTerms['00'] = code.blank_pauli_mat(lattice_size, 2)
    
    # latticeTerms['00'][1,0] = 'X'
    # latticeTerms['00'][3,0] = 'Y'
    # latticeTerms['00'][5,0] = 'Z'
    

    # latticeTerms['00'][1+code.qubitsPerCell,1] = 'X'
    # latticeTerms['00'][3+code.qubitsPerCell,1] = 'Y'
    # latticeTerms['00'][5+code.qubitsPerCell,1] = 'Z'

    # # tind = 0
    # # code.plot_single_torus_term(latticeTerms, tind, fignum = 14, 
    # #                   title = 'test torus term 1', 
    # #                   numberSites = False,
    # #                   drawLinks = True,
    # #                   spotSize = 250,
    # #                   figx = 6,
    # #                   figy = 6)
    
    # # tind = 1
    # # code.plot_single_torus_term(latticeTerms, tind, fignum = 15, 
    # #                   title = 'test torus term 1', 
    # #                   numberSites = False,
    # #                   drawLinks = True,
    # #                   spotSize = 250,
    # #                   figx = 6,
    # #                   figy = 6)

    
    # # code.plot_torus_terms(latticeTerms, saveExtension = 'dummyTerms',
    # #                      saveDir = '', 
    # #                      fignum = 4,
    # #                      drawLinks = False,
    # #                      spotSize = 250,
    # #                      figx = 6,
    # #                      figy = 6)




    # # ###test the torus plotting on the Hamiltonian and stabilizers
    # # code.plot_single_term(code.H0, 0, fignum = 3, 
    # #                   title = 'first H term', 
    # #                   numberSites = False,
    # #                   spotSize = 150,
    # #                   figx = 6,
    # #                   figy = 6)
    

    # # code.plot_torus_terms(code.H_torus, saveExtension = 'torusH',
    # #                       saveDir = '', 
    # #                       fignum = 4,
    # #                       drawLinks = True,
    # #                       spotSize = 150,
    # #                       figx = 6,
    # #                       figy = 6)
    
    # # code.plot_torus_terms(code.Lx_torus, saveExtension = 'torusLx',
    # #                       saveDir = '', 
    # #                       fignum = 4,
    # #                       drawLinks = True,
    # #                       spotSize = 150,
    # #                       figx = 6,
    # #                       figy = 6)



    code = PauliCode(UnitCell('square'), name = 'temp', 
                         xSize = 2, ySize = 2, 
                         bosonization_type = 'edge',
                         fiducial = True,
                         vertex_assignment = [],
                         verbose = True)
    
    H_torus = code.unpack_to_torus(code.H0)
    dimer_torus = code.unpack_to_torus(code.fiducialDimers)
    
    cellCheck = code.check_Lorent_commutation(code.H0, code.fiducialDimers)
    cellErrors = numpy.where(cellCheck !=0)
    print(cellErrors)
    
    torusCheck = code.check_Lorent_commutation(H_torus, dimer_torus)
    torusErrors = numpy.where(torusCheck !=0)
    print(torusErrors)
    
    
    # out = code.make_fiducial_dimers_new_version()
    # code.plot_terms(out, saveExtension = '_blah', 
    #                videoMode = True,
    #                fignum = 4, 
    #                jog = 2.25,
    #                jogy = 2.25,
    #                figx = 6,
    #                figy = 6,
    #                videoTime = 1)



