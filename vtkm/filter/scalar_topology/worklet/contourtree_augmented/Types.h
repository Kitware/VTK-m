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


#ifndef vtk_m_worklet_contourtree_augmented_types_h
#define vtk_m_worklet_contourtree_augmented_types_h

#include <vtkm/Assert.h>
#include <vtkm/Types.h>
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleConstant.h>
#include <vtkm/cont/CellSetStructured.h>

namespace vtkm
{
namespace worklet
{
namespace contourtree_augmented
{

// constexpr for bit flags
// clang-format off
constexpr vtkm::Id NO_SUCH_ELEMENT = std::numeric_limits<vtkm::Id>::min();
constexpr vtkm::Id TERMINAL_ELEMENT = std::numeric_limits<vtkm::Id>::max() / 2 + 1; //0x40000000 || 0x4000000000000000
constexpr vtkm::Id IS_SUPERNODE = std::numeric_limits<vtkm::Id>::max() / 4 + 1; //0x20000000 || 0x2000000000000000
constexpr vtkm::Id IS_HYPERNODE = std::numeric_limits<vtkm::Id>::max() / 8 + 1; //0x10000000 || 0x1000000000000000
constexpr vtkm::Id IS_ASCENDING = std::numeric_limits<vtkm::Id>::max() / 16 + 1; //0x08000000 || 0x0800000000000000
constexpr vtkm::Id INDEX_MASK = std::numeric_limits<vtkm::Id>::max() / 16; //0x07FFFFFF || 0x07FFFFFFFFFFFFFF
constexpr vtkm::Id CV_OTHER_FLAG = std::numeric_limits<vtkm::Id>::max() / 8 + 1; //0x10000000 || 0x1000000000000000
constexpr vtkm::Id ELEMENT_EXISTS = std::numeric_limits<vtkm::Id>::max() / 4 + 1; //0x20000000 || 0x2000000000000000 , same as IS_SUPERNODE

// flags for testing regular vertices
constexpr vtkm::Id IS_LOWER_LEAF = static_cast<vtkm::Id>(0);
constexpr vtkm::Id IS_UPPER_LEAF = static_cast<vtkm::Id>(1);
constexpr vtkm::Id IS_REGULAR = static_cast<vtkm::Id>(2);
constexpr vtkm::Id IS_SADDLE = static_cast<vtkm::Id>(3);
constexpr vtkm::Id IS_ATTACHMENT = static_cast<vtkm::Id>(4);

// NOTE: 29/08/2024
// After discussion between Mingzhe and Hamish, warning left in.
// The logic for the choice is that we are already using 5 bits out of 64, leaving us with 0.5 exa-indices available.
// Using an extra bit flag would reduce it to 0.25 exa-indices, which may pose a problem in the future.
// We therefore have chosen to reuse an existing bit flag, which is not used in the section of code in question.
// (HierarchicalHypersweeper.h: ComputeSuperarcTransferWeights())
// WARNING 11/07/2023
// TERMINAL_ELEMENT is primarily used for optimisation of memory access during pointer doubling operations
// We now need to distinguish between a supernode and superarc when sorting by superarc(node) IDs
// This only (at present) comes up when processing attachment points, which have null superarcs, so it
// is reasonable to reuse TERMINAL_ELEMENT for this purpose.  However, we give it a separate macro name with
// the same value to aid comprehension
constexpr vtkm::Id TRANSFER_TO_SUPERARC = TERMINAL_ELEMENT;

// clang-format on
using IdArrayType = vtkm::cont::ArrayHandle<vtkm::Id>;

using EdgePair = vtkm::Pair<vtkm::Id, vtkm::Id>; // here EdgePair.first=low and EdgePair.second=high
using EdgePairArray = vtkm::cont::ArrayHandle<EdgePair>; // Array of edge pairs


// inline functions for retrieving flags or index
VTKM_EXEC_CONT
inline bool NoSuchElement(vtkm::Id flaggedIndex)
{ // NoSuchElement()
  return ((flaggedIndex & (vtkm::Id)NO_SUCH_ELEMENT) != 0);
} // NoSuchElement()

VTKM_EXEC_CONT
inline bool IsTerminalElement(vtkm::Id flaggedIndex)
{ // IsTerminalElement()
  return ((flaggedIndex & TERMINAL_ELEMENT) != 0);
} // IsTerminalElement()

VTKM_EXEC_CONT
inline bool IsSupernode(vtkm::Id flaggedIndex)
{ // IsSupernode()
  return ((flaggedIndex & IS_SUPERNODE) != 0);
} // IsSupernode()

VTKM_EXEC_CONT
inline bool IsHypernode(vtkm::Id flaggedIndex)
{ // IsHypernode()
  return ((flaggedIndex & IS_HYPERNODE) != 0);
} // IsHypernode()

VTKM_EXEC_CONT
inline bool IsAscending(vtkm::Id flaggedIndex)
{ // IsAscending()
  return ((flaggedIndex & IS_ASCENDING) != 0);
} // IsAscending()

VTKM_EXEC_CONT
inline vtkm::Id MaskedIndex(vtkm::Id flaggedIndex)
{ // MaskedIndex()
  return (flaggedIndex & INDEX_MASK);
} // MaskedIndex()

/// Used in the context of CombinedVector class used in ContourTreeMesh to merge the mesh of contour trees
VTKM_EXEC_CONT
inline bool IsThis(vtkm::Id flaggedIndex)
{ // IsThis
  return ((flaggedIndex & CV_OTHER_FLAG) == 0);
} // IsThis

// Helper function: Ensure no flags are set
VTKM_EXEC_CONT
inline bool NoFlagsSet(vtkm::Id flaggedIndex)
{ // NoFlagsSet
  return (flaggedIndex & ~INDEX_MASK) == 0;
} // NoFlagsSet

// Helper function: to check that the TRANSFER_TO_SUPERARC flag is set
VTKM_EXEC_CONT
inline bool TransferToSuperarc(vtkm::Id flaggedIndex)
{ // transferToSuperarc()
  return ((flaggedIndex & TRANSFER_TO_SUPERARC) != 0);
} // transferToSuperarc()

// Debug helper function: Assert that an index array has no element with any flags set
template <typename S>
VTKM_CONT inline void AssertArrayHandleNoFlagsSet(const vtkm::cont::ArrayHandle<vtkm::Id, S>& ah)
{
#ifndef VTKM_NO_ASSERT
  auto rp = ah.ReadPortal();
  for (vtkm::Id i = 0; i < ah.GetNumberOfValues(); ++i)
  {
    VTKM_ASSERT(NoFlagsSet(rp.Get(i)));
  }
#else
  (void)ah;
#endif
}


/// Helper function to set a single array valye with CopySubRange to avoid pulling the array to the control environment
VTKM_CONT
inline void IdArraySetValue(vtkm::Id index, vtkm::Id value, IdArrayType& arr)
{ // IdArraySetValue
  vtkm::cont::Algorithm::CopySubRange(
    vtkm::cont::ArrayHandleConstant<vtkm::Id>(value, 1), 0, 1, arr, index);
} // IdArraySetValues


/// Helper function used to resize a 1D ArrayHandle and initalize new values with a
/// given fillValue. For resizing ArrayHandles without initalizing new values VTKm
/// supports the vtkm::CopyFlag::On setting as part of the ArrayHandle.Allocate
/// method.
/// @param[in] thearray The 1D array to be resized
/// @param[in] newSize The new size the array should be changed to
/// @param[in] fillValue The value to be used to fill the array
template <typename ValueType>
void ResizeVector(vtkm::cont::ArrayHandle<ValueType>& thearray,
                  vtkm::Id newSize,
                  ValueType fillValue)
{
  vtkm::Id oldSize = thearray.GetNumberOfValues();
  // Simply return if the size of the array does not change
  if (oldSize == newSize)
  {
    return;
  }

  // Resize the array but keep the original values
  thearray.Allocate(newSize, vtkm::CopyFlag::On);

  // Add the fill values to the array if we increased the size of the array
  if (oldSize < newSize)
  {
    vtkm::cont::Algorithm::CopySubRange(
      vtkm::cont::ArrayHandleConstant<ValueType>(fillValue, newSize - oldSize), // copy
      0,                 // start copying from first index
      newSize - oldSize, // num values to copy
      thearray,          // target array to copy to
      oldSize            // start copy to after oldSize
    );
  }
}

template <typename T>
struct MaskedIndexFunctor
{
  VTKM_EXEC_CONT
  MaskedIndexFunctor() {}

  VTKM_EXEC_CONT
  vtkm::Id operator()(T x) const { return MaskedIndex(x); }
};

inline std::string FlagString(vtkm::Id flaggedIndex)
{ // FlagString()
  std::string fString("");
  fString += (NoSuchElement(flaggedIndex) ? "n" : ".");
  fString += (IsTerminalElement(flaggedIndex) ? "t" : ".");
  fString += (IsSupernode(flaggedIndex) ? "s" : ".");
  fString += (IsHypernode(flaggedIndex) ? "h" : ".");
  fString += (IsAscending(flaggedIndex) ? "a" : ".");
  return fString;
} // FlagString()


// == comparison operator for edges
inline bool edgeEqual(const EdgePair& LHS, const EdgePair& RHS)
{ // operator ==

  if (LHS.first != RHS.first)
  {
    return false;
  }
  if (LHS.second != RHS.second)
  {
    return false;
  }
  return true;
} // operator ==


class EdgeDataHeight
{
public:
  // RegularNodeID (or sortIndex)
  Id I;
  // RegularNodeID (or sortIndex)
  Id J;
  // RegularNodeID (or sortIndex)
  Id SubtreeMin;
  // RegularNodeID (or sortIndex)
  Id SubtreeMax;
  bool UpEdge;
  Float64 SubtreeHeight;

  VTKM_EXEC
  bool operator<(const EdgeDataHeight& b) const
  {
    if (this->I == b.I)
    {
      if (this->UpEdge == b.UpEdge)
      {
        if (this->SubtreeHeight == b.SubtreeHeight)
        {
          if (this->SubtreeMin == b.SubtreeMin)
          {
            return this->SubtreeMax > b.SubtreeMax;
          }
          else
          {
            return this->SubtreeMin < b.SubtreeMin;
          }
        }
        else
        {
          return this->SubtreeHeight > b.SubtreeHeight;
        }
      }
      else
      {
        return this->UpEdge < b.UpEdge;
      }
    }
    else
    {
      return this->I < b.I;
    }
  }
};

class EdgeDataVolume
{
public:
  // RegularNodeID (or sortIndex)
  Id I;
  // RegularNodeID (or sortIndex)
  Id J;
  bool UpEdge;
  Id SubtreeVolume;

  VTKM_EXEC
  bool operator<(const EdgeDataVolume& b) const
  {
    if (this->I == b.I)
    {
      if (this->UpEdge == b.UpEdge)
      {
        if (this->SubtreeVolume == b.SubtreeVolume)
        {
          if (this->UpEdge == true)
          {
            return this->J > b.J;
          }
          else
          {
            return this->J < b.J;
          }
        }
        else
        {
          return this->SubtreeVolume > b.SubtreeVolume;
        }
      }
      else
      {
        return this->UpEdge < b.UpEdge;
      }
    }
    else
    {
      return this->I < b.I;
    }
  }
};


///
/// Helper struct to collect sizing information from a dataset.
/// The struct is used in the contour tree filter implementation
/// to determine the rows, cols, slices parameters from the
/// datasets so we can call the contour tree worklet properly.
///
struct GetPointDimensions
{
  ///@{
  /// Get the number of rows, cols, and slices of a vtkm::cont::CellSetStructured.
  /// @param[in] cells  The input vtkm::cont::CellSetStructured.
  /// @param[out] pointDimensions mesh size (often referred to as columns, rows, and slices)
  ///   with last dimension having a value of 1 for 2D data.
  void operator()(const vtkm::cont::CellSetStructured<2>& cells, vtkm::Id3& pointDimensions) const
  {
    vtkm::Id2 pointDimensions2D = cells.GetPointDimensions();
    pointDimensions[0] = pointDimensions2D[0];
    pointDimensions[1] = pointDimensions2D[1];
    pointDimensions[2] = 1;
  }
  void operator()(const vtkm::cont::CellSetStructured<3>& cells, vtkm::Id3& pointDimensions) const
  {
    pointDimensions = cells.GetPointDimensions();
  }
  ///@}

  ///  Raise ErrorBadValue if the input cell set is not a vtkm::cont::CellSetStructured<2> or <3>
  template <typename T>
  void operator()(const T&, vtkm::Id3&) const
  {
    throw vtkm::cont::ErrorBadValue("Expected 2D or 3D structured cell cet! ");
  }
};


struct GetLocalAndGlobalPointDimensions
{
  void operator()(const vtkm::cont::CellSetStructured<2>& cells,
                  vtkm::Id3& pointDimensions,
                  vtkm::Id3& globalPointDimensions,
                  vtkm::Id3& globalPointIndexStart) const
  {
    vtkm::Id2 pointDimensions2D = cells.GetPointDimensions();
    pointDimensions[0] = pointDimensions2D[0];
    pointDimensions[1] = pointDimensions2D[1];
    pointDimensions[2] = 1;
    vtkm::Id2 globalPointDimensions2D = cells.GetGlobalPointDimensions();
    globalPointDimensions[0] = globalPointDimensions2D[0];
    globalPointDimensions[1] = globalPointDimensions2D[1];
    globalPointDimensions[2] = 1;
    vtkm::Id2 pointIndexStart2D = cells.GetGlobalPointIndexStart();
    globalPointIndexStart[0] = pointIndexStart2D[0];
    globalPointIndexStart[1] = pointIndexStart2D[1];
    globalPointIndexStart[2] = 0;
  }
  void operator()(const vtkm::cont::CellSetStructured<3>& cells,
                  vtkm::Id3& pointDimensions,
                  vtkm::Id3& globalPointDimensions,
                  vtkm::Id3& globalPointIndexStart) const
  {
    pointDimensions = cells.GetPointDimensions();
    globalPointDimensions = cells.GetGlobalPointDimensions();
    globalPointIndexStart = cells.GetGlobalPointIndexStart();
  }


  ///  Raise ErrorBadValue if the input cell set is not a vtkm::cont::CellSetStructured<2> or <3>
  template <typename T>
  void operator()(const T&, vtkm::Id3&, vtkm::Id3&, vtkm::Id3&) const
  {
    throw vtkm::cont::ErrorBadValue("Expected 2D or 3D structured cell cet! ");
  }
};

} // namespace contourtree_augmented
} // worklet
} // vtkm

#endif
