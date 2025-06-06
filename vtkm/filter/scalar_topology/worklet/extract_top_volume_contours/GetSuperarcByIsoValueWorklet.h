//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
// Copyright (c) 2018, The Regents of the University of California, through
// Lawrence Berkeley National Laboratory (subject to receipt of any required approvals
// from the U.S. Dept. of Energy).  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice, this
//     list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// (3) Neither the name of the University of California, Lawrence Berkeley National
//     Laboratory, U.S. Dept. of Energy nor the names of its contributors may be
//     used to endorse or promote products derived from this software without
//     specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.
//
//=============================================================================
//
//  This code is an extension of the algorithm presented in the paper:
//  Parallel Peak Pruning for Scalable SMP Contour Tree Computation.
//  Hamish Carr, Gunther Weber, Christopher Sewell, and James Ahrens.
//  Proceedings of the IEEE Symposium on Large Data Analysis and Visualization
//  (LDAV), October 2016, Baltimore, Maryland.
//
//  The PPP2 algorithm and software were jointly developed by
//  Hamish Carr (University of Leeds), Gunther H. Weber (LBNL), and
//  Oliver Ruebel (LBNL)
//==============================================================================

#ifndef vtk_m_filter_scalar_topology_worklet_extract_top_volume_contours_get_superarc_by_isovalue_worklet_h
#define vtk_m_filter_scalar_topology_worklet_extract_top_volume_contours_get_superarc_by_isovalue_worklet_h

#include <vtkm/worklet/WorkletMapField.h>

namespace vtkm
{
namespace worklet
{
namespace scalar_topology
{
namespace extract_top_volume_contours
{
/// Worklet for getting the superarc of a branch given an isovalue
class GetSuperarcByIsoValueWorklet : public vtkm::worklet::WorkletMapField
{
public:
  using ControlSignature =
    void(FieldIn upperEndLocalId,     // (input) upper end of the branch
         FieldIn lowerEndLocalId,     // (input) lower end of the branch
         FieldIn isoValue,            // (input) isoValue
         FieldIn saddleEndGRId,       // (input) saddle end global regular id
         FieldIn branchSaddleEpsilon, // (input) whether the branch is on top or at the bottom
         FieldOut superarc,           // (output) local superarc that intersects the isosurface
         ExecObject findSuperarcByNode);
  using ExecutionSignature = _6(_1, _2, _3, _4, _5, _7);
  using InputDomain = _1;

  /// <summary>
  /// Constructor
  /// </summary>
  VTKM_EXEC_CONT
  GetSuperarcByIsoValueWorklet(vtkm::Id totNumPoints, bool isContourByValue)
    : HighValue(totNumPoints)
    , IsContourByValue(isContourByValue)
  {
  }

  /// <summary>
  /// Implementation of GetSuperarcByIsoValueWorklet.
  /// Check vtkm::worklet::contourtree_distributed::FindSuperArcForUnknownNode
  /// for the execution object description.
  /// </summary>
  /// <typeparam name="ValueType">data value type</typeparam>
  /// <typeparam name="findSuperarcType">execution object type of findSuperarc</typeparam>
  /// <param name="upperEndLocalId">local id of the upper end vertex of the branch</param>
  /// <param name="lowerEndLocalId">local id of the lower end vertex of the branch</param>
  /// <param name="isoValue">isovalue</param>
  /// <param name="branchSaddleEpsilon">the direction for tiebreaking when comparing values</param>
  /// <param name="findSuperarc">execution object</param>
  /// <returns></returns>
  template <typename ValueType, typename findSuperarcType>
  VTKM_EXEC vtkm::Id operator()(const vtkm::Id upperEndLocalId,
                                const vtkm::Id lowerEndLocalId,
                                const ValueType isoValue,
                                const vtkm::Id saddleEndGRId,
                                const vtkm::Id branchSaddleEpsilon,
                                const findSuperarcType& findSuperarc) const
  {
    VTKM_ASSERT(branchSaddleEpsilon != 0);

    // Update 01/06/2025:
    // We need the global regular ID of the saddle end,
    // which is used for simulation of simplicity when looking for the closest superarc to the saddle end.
    // If we extract contours solely by value (i.e., ignore simulation of simplicity),
    // the contour global regular ID should either be inf small or inf large;
    // otherwise, it is offset by 1 from the saddle end of the branch.
    vtkm::Id contourGRId;
    if (IsContourByValue)
      contourGRId = branchSaddleEpsilon < 0 ? -1 : HighValue;
    else
      contourGRId = branchSaddleEpsilon < 0 ? saddleEndGRId - 1 : saddleEndGRId + 1;

    return findSuperarc.FindSuperArcForUnknownNode(
      contourGRId, isoValue, upperEndLocalId, lowerEndLocalId);
  }

private:
  const vtkm::Id HighValue;
  const bool IsContourByValue;
}; // GetSuperarcByIsoValueWorklet

} // namespace extract_top_volume_contours
} // namespace scalar_topology
} // namespace worklet
} // namespace vtkm

#endif
