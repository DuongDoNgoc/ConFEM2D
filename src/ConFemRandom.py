# ConFemRandom -- 2014-01-13
# Copyright (C) [2014] [Joerg Weselek]
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License (GNU GPLv3) as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program; if not, see <http://www.gnu.org/licenses
#
'''
Created on 15.11.2013

@author: josh
'''
from ConFEM2D_Basics import ZeroD

class RandomField_Routines(object):
    def add_elset(self,elset):
        try:
            if elset not in RandomField_Routines.elsets:
                RandomField_Routines.elsets.append(elset)
        except AttributeError:
            RandomField_Routines.elsets=[]
            RandomField_Routines.elsets.append(elset)
            
    def add_propertyEntry(self,prop,elset,label):
        try:
#             RandomField_Routines.property[elset].append((label,prop))
            RandomField_Routines.property[elset][label].append(prop)
        except AttributeError:
            RandomField_Routines.property={elset:{}}
            RandomField_Routines.property[elset][label]=[]
            RandomField_Routines.property[elset][label].append(prop)
        except KeyError:
            RandomField_Routines.property[elset][label]=[]
            RandomField_Routines.property[elset][label].append(prop)

    def get_property(self,elset,label,original):
        '''Problem: current property of the material will be overwritten when called
        and will not be cached for further use'''
        try:
            return RandomField_Routines.property[elset][label]
        except (AttributeError,KeyError): 
            return [original] 

    def more_properties(self,fct,Gf,cCrit,epsct,wcr,cVal,elset,label):
        try:
            if RandomField_Routines.property[elset][label][0]>ZeroD: wcr = 2*Gf/RandomField_Routines.property[elset][label][0]           # critical crack width
            else:         wcr=0
#            if cCrit == 0: cVal = RandomField_Routines.property[elset][label][0]
            if cCrit == 0: cVal = 0.  # stress criterion
            else:          cVal = epsct # strain criterion
            return wcr,cVal
        except (AttributeError,KeyError): 
            return wcr,cVal
    def overwrite_fct(self):
        return None