##============================================================================
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##============================================================================

set(headers
  Predicates.h
  TopVolumeBranchData.h
  AboveThresholdWorklet.h
  AssignValueWorklet.h
  BinarySearchWorklet.h
  BranchSaddleIsKnownWorklet.h
  ClarifyBranchEndSupernodeTypeWorklet.h
  CollectOuterSaddleWorklet.h
  UpdateOuterSaddleWorklet.h
  UpdateInfoByBranchDirectionWorklet.h
  GetParentBranchWorklet.h
  GetBranchHierarchyWorklet.h
  GetBranchVolumeWorklet.h
  BranchParentComparator.h
  BranchVolumeComparator.h
  BranchDecompositionTreeMaker.h
)
#-----------------------------------------------------------------------------

vtkm_declare_headers(${headers})
