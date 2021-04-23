//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_cont_tbb_DeviceAdapterTBB_h
#define vtk_m_cont_tbb_DeviceAdapterTBB_h

#include <vtkm/cont/tbb/internal/DeviceAdapterRuntimeDetectorTBB.h>
#include <vtkm/cont/tbb/internal/DeviceAdapterTagTBB.h>

#ifdef VTKM_ENABLE_TBB
#include <vtkm/cont/tbb/internal/DeviceAdapterAlgorithmTBB.h>
#include <vtkm/cont/tbb/internal/DeviceAdapterMemoryManagerTBB.h>
#ifndef VTKM_NO_DEPRECATED_VIRTUAL
#include <vtkm/cont/tbb/internal/VirtualObjectTransferTBB.h>
#endif //VTKM_NO_DEPRECATED_VIRTUAL
#endif

#endif //vtk_m_cont_tbb_DeviceAdapterTBB_h
