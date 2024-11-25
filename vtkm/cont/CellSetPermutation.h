//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_cont_CellSetPermutation_h
#define vtk_m_cont_CellSetPermutation_h

#include <vtkm/CellShape.h>
#include <vtkm/CellTraits.h>
#include <vtkm/Deprecated.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandleCast.h>
#include <vtkm/cont/ArrayHandleGroupVecVariable.h>
#include <vtkm/cont/ArrayHandlePermutation.h>
#include <vtkm/cont/CellSet.h>
#include <vtkm/cont/CellSetExplicit.h>
#include <vtkm/cont/Invoker.h>
#include <vtkm/cont/internal/ConnectivityExplicitInternals.h>
#include <vtkm/cont/internal/ConvertNumComponentsToOffsetsTemplate.h>
#include <vtkm/cont/internal/ReverseConnectivityBuilder.h>
#include <vtkm/internal/ConnectivityStructuredInternals.h>
#include <vtkm/worklet/WorkletMapTopology.h>

#include <vtkm/exec/ConnectivityPermuted.h>

#ifndef VTKM_DEFAULT_CELLSET_PERMUTATION_STORAGE_TAG
#define VTKM_DEFAULT_CELLSET_PERMUTATION_STORAGE_TAG VTKM_DEFAULT_STORAGE_TAG
#endif

namespace vtkm
{
namespace cont
{

namespace internal
{

// To generate the reverse connectivity table with the
// ReverseConnectivityBuilder, we need a compact connectivity array that
// contains only the cell definitions from the permuted dataset, and an offsets
// array. These helpers are used to generate these arrays so that they can be
// converted in the reverse conn table.
class RConnTableHelpers
{
public:
  struct WriteNumIndices : public vtkm::worklet::WorkletVisitCellsWithPoints
  {
    using ControlSignature = void(CellSetIn cellset, FieldOutCell numIndices);
    using ExecutionSignature = void(PointCount, _2);
    using InputDomain = _1;

    VTKM_EXEC void operator()(vtkm::IdComponent pointCount, vtkm::IdComponent& numIndices) const
    {
      numIndices = pointCount;
    }
  };

  struct WriteConnectivity : public vtkm::worklet::WorkletVisitCellsWithPoints
  {
    using ControlSignature = void(CellSetIn cellset, FieldOutCell connectivity);
    using ExecutionSignature = void(PointCount, PointIndices, _2);
    using InputDomain = _1;

    template <typename PointIndicesType, typename OutConnectivityType>
    VTKM_EXEC void operator()(vtkm::IdComponent pointCount,
                              const PointIndicesType& pointIndices,
                              OutConnectivityType& connectivity) const
    {
      for (vtkm::IdComponent i = 0; i < pointCount; ++i)
      {
        connectivity[i] = pointIndices[i];
      }
    }
  };

  template <typename CellSetPermutationType>
  static VTKM_CONT vtkm::cont::ArrayHandle<vtkm::IdComponent> GetNumIndicesArray(
    const CellSetPermutationType& cs,
    vtkm::cont::DeviceAdapterId device)
  {
    vtkm::cont::ArrayHandle<vtkm::IdComponent> numIndices;
    vtkm::cont::Invoker{ device }(WriteNumIndices{}, cs, numIndices);
    return numIndices;
  }

  template <typename NumIndicesStorageType>
  static VTKM_CONT vtkm::cont::ArrayHandle<vtkm::Id> GetOffsetsArray(
    const vtkm::cont::ArrayHandle<vtkm::IdComponent, NumIndicesStorageType>& numIndices,
    vtkm::Id& connectivityLength /* outparam */,
    vtkm::cont::DeviceAdapterId)
  {
    return vtkm::cont::internal::ConvertNumComponentsToOffsetsTemplate(numIndices,
                                                                       connectivityLength);
  }

  template <typename CellSetPermutationType, typename OffsetsStorageType>
  static vtkm::cont::ArrayHandle<vtkm::Id> GetConnectivityArray(
    const CellSetPermutationType& cs,
    const vtkm::cont::ArrayHandle<vtkm::Id, OffsetsStorageType>& offsets,
    vtkm::Id connectivityLength,
    vtkm::cont::DeviceAdapterId device)
  {
    vtkm::cont::ArrayHandle<vtkm::Id> connectivity;
    connectivity.Allocate(connectivityLength);
    auto connWrap = vtkm::cont::make_ArrayHandleGroupVecVariable(connectivity, offsets);
    vtkm::cont::Invoker{ device }(WriteConnectivity{}, cs, connWrap);
    return connectivity;
  }
};

// This holds the temporary input arrays for the ReverseConnectivityBuilder
// algorithm.
template <typename ConnectivityStorageTag = VTKM_DEFAULT_STORAGE_TAG,
          typename OffsetsStorageTag = VTKM_DEFAULT_STORAGE_TAG,
          typename NumIndicesStorageTag = VTKM_DEFAULT_STORAGE_TAG>
struct RConnBuilderInputData
{
  using ConnectivityArrayType = vtkm::cont::ArrayHandle<vtkm::Id, ConnectivityStorageTag>;
  using OffsetsArrayType = vtkm::cont::ArrayHandle<vtkm::Id, OffsetsStorageTag>;
  using NumIndicesArrayType = vtkm::cont::ArrayHandle<vtkm::IdComponent, NumIndicesStorageTag>;

  ConnectivityArrayType Connectivity;
  OffsetsArrayType Offsets; // Includes the past-the-end offset.
  NumIndicesArrayType NumIndices;
};

// default for CellSetPermutations of any cell type
template <typename CellSetPermutationType>
class RConnBuilderInput
{
public:
  using ConnectivityArrays = vtkm::cont::internal::RConnBuilderInputData<>;

  static ConnectivityArrays Get(const CellSetPermutationType& cellset,
                                vtkm::cont::DeviceAdapterId device)
  {
    using Helper = RConnTableHelpers;
    ConnectivityArrays conn;
    vtkm::Id connectivityLength = 0;

    conn.NumIndices = Helper::GetNumIndicesArray(cellset, device);
    conn.Offsets = Helper::GetOffsetsArray(conn.NumIndices, connectivityLength, device);
    conn.Connectivity =
      Helper::GetConnectivityArray(cellset, conn.Offsets, connectivityLength, device);

    return conn;
  }
};

// Specialization for CellSetExplicit/CellSetSingleType
template <typename InShapesST,
          typename InConnST,
          typename InOffsetsST,
          typename PermutationArrayHandleType>
class RConnBuilderInput<CellSetPermutation<CellSetExplicit<InShapesST, InConnST, InOffsetsST>,
                                           PermutationArrayHandleType>>
{
private:
  using BaseCellSetType = CellSetExplicit<InShapesST, InConnST, InOffsetsST>;
  using CellSetPermutationType = CellSetPermutation<BaseCellSetType, PermutationArrayHandleType>;

  using InShapesArrayType = typename BaseCellSetType::ShapesArrayType;
  using InNumIndicesArrayType = typename BaseCellSetType::NumIndicesArrayType;

  using ConnectivityStorageTag = vtkm::cont::ArrayHandle<vtkm::Id>::StorageTag;
  using OffsetsStorageTag = vtkm::cont::ArrayHandle<vtkm::Id>::StorageTag;
  using NumIndicesStorageTag =
    typename vtkm::cont::ArrayHandlePermutation<PermutationArrayHandleType,
                                                InNumIndicesArrayType>::StorageTag;


public:
  using ConnectivityArrays = vtkm::cont::internal::
    RConnBuilderInputData<ConnectivityStorageTag, OffsetsStorageTag, NumIndicesStorageTag>;

  static ConnectivityArrays Get(const CellSetPermutationType& cellset,
                                vtkm::cont::DeviceAdapterId device)
  {
    using Helper = RConnTableHelpers;

    static constexpr vtkm::TopologyElementTagCell cell{};
    static constexpr vtkm::TopologyElementTagPoint point{};

    auto fullCellSet = cellset.GetFullCellSet();

    vtkm::Id connectivityLength = 0;
    ConnectivityArrays conn;

    fullCellSet.GetOffsetsArray(cell, point);

    // We can use the implicitly generated NumIndices array to save a bit of
    // memory:
    conn.NumIndices = vtkm::cont::make_ArrayHandlePermutation(
      cellset.GetValidCellIds(), fullCellSet.GetNumIndicesArray(cell, point));

    // Need to generate the offsets from scratch so that they're ordered for the
    // lower-bounds binary searches in ReverseConnectivityBuilder.
    conn.Offsets = Helper::GetOffsetsArray(conn.NumIndices, connectivityLength, device);

    // Need to create a copy of this containing *only* the permuted cell defs,
    // in order, since the ReverseConnectivityBuilder will process every entry
    // in the connectivity array and we don't want the removed cells to be
    // included.
    conn.Connectivity =
      Helper::GetConnectivityArray(cellset, conn.Offsets, connectivityLength, device);

    return conn;
  }
};

// Specialization for CellSetStructured
template <vtkm::IdComponent DIMENSION, typename PermutationArrayHandleType>
class RConnBuilderInput<
  CellSetPermutation<CellSetStructured<DIMENSION>, PermutationArrayHandleType>>
{
private:
  using CellSetPermutationType =
    CellSetPermutation<CellSetStructured<DIMENSION>, PermutationArrayHandleType>;

public:
  using ConnectivityArrays = vtkm::cont::internal::RConnBuilderInputData<
    VTKM_DEFAULT_STORAGE_TAG,
    typename vtkm::cont::ArrayHandleCounting<vtkm::Id>::StorageTag,
    typename vtkm::cont::ArrayHandleConstant<vtkm::IdComponent>::StorageTag>;

  static ConnectivityArrays Get(const CellSetPermutationType& cellset,
                                vtkm::cont::DeviceAdapterId device)
  {
    vtkm::Id numberOfCells = cellset.GetNumberOfCells();
    vtkm::IdComponent numPointsInCell =
      vtkm::internal::ConnectivityStructuredInternals<DIMENSION>::NUM_POINTS_IN_CELL;
    vtkm::Id connectivityLength = numberOfCells * numPointsInCell;

    ConnectivityArrays conn;
    conn.NumIndices = make_ArrayHandleConstant(numPointsInCell, numberOfCells);
    conn.Offsets = ArrayHandleCounting<vtkm::Id>(0, numPointsInCell, numberOfCells + 1);
    conn.Connectivity =
      RConnTableHelpers::GetConnectivityArray(cellset, conn.Offsets, connectivityLength, device);

    return conn;
  }
};

template <typename CellSetPermutationType>
struct CellSetPermutationTraits;

template <typename OriginalCellSet_, typename PermutationArrayHandleType_>
struct CellSetPermutationTraits<CellSetPermutation<OriginalCellSet_, PermutationArrayHandleType_>>
{
  using OriginalCellSet = OriginalCellSet_;
  using PermutationArrayHandleType = PermutationArrayHandleType_;
};

template <typename OriginalCellSet_,
          typename OriginalPermutationArrayHandleType,
          typename PermutationArrayHandleType_>
struct CellSetPermutationTraits<
  CellSetPermutation<CellSetPermutation<OriginalCellSet_, OriginalPermutationArrayHandleType>,
                     PermutationArrayHandleType_>>
{
  using PreviousCellSet = CellSetPermutation<OriginalCellSet_, OriginalPermutationArrayHandleType>;
  using PermutationArrayHandleType = vtkm::cont::ArrayHandlePermutation<
    PermutationArrayHandleType_,
    typename CellSetPermutationTraits<PreviousCellSet>::PermutationArrayHandleType>;
  using OriginalCellSet = typename CellSetPermutationTraits<PreviousCellSet>::OriginalCellSet;
  using Superclass = CellSetPermutation<OriginalCellSet, PermutationArrayHandleType>;
};

template <typename VisitTopology,
          typename IncidentTopology,
          typename OriginalCellSetType,
          typename PermutationArrayHandleType>
struct CellSetPermutationConnectivityChooser;

template <typename OriginalCellSetType, typename PermutationArrayHandleType>
struct CellSetPermutationConnectivityChooser<vtkm::TopologyElementTagCell,
                                             vtkm::TopologyElementTagPoint,
                                             OriginalCellSetType,
                                             PermutationArrayHandleType>
{
  using ExecPortalType = typename PermutationArrayHandleType::ReadPortalType;
  using OrigExecObjectType =
    typename OriginalCellSetType::template ExecConnectivityType<vtkm::TopologyElementTagCell,
                                                                vtkm::TopologyElementTagPoint>;

  using ExecConnectivityType =
    vtkm::exec::ConnectivityPermutedVisitCellsWithPoints<ExecPortalType, OrigExecObjectType>;
};

template <typename OriginalCellSetType, typename PermutationArrayHandleType>
struct CellSetPermutationConnectivityChooser<vtkm::TopologyElementTagPoint,
                                             vtkm::TopologyElementTagCell,
                                             OriginalCellSetType,
                                             PermutationArrayHandleType>
{
  using ConnectivityPortalType = typename vtkm::cont::ArrayHandle<vtkm::Id>::ReadPortalType;
  using NumIndicesPortalType = typename vtkm::cont::ArrayHandle<vtkm::IdComponent>::ReadPortalType;
  using OffsetPortalType = typename vtkm::cont::ArrayHandle<vtkm::Id>::ReadPortalType;

  using ExecConnectivityType =
    vtkm::exec::ConnectivityPermutedVisitPointsWithCells<ConnectivityPortalType, OffsetPortalType>;
};

} // internal

/// @brief Rearranges the cells of one cell set to create another cell set.
///
/// This restructuring of cells is not done by copying data to a new structure.
/// Rather, `CellSetPermutation` establishes a look-up from one cell structure to
/// another. Cells are permuted on the fly while algorithms are run.
///
/// A `CellSetPermutation` is established by providing a mapping array that for every
/// cell index provides the equivalent cell index in the cell set being permuted.
/// `CellSetPermutation` is most often used to mask out cells in a data set so that
/// algorithms will skip over those cells when running.
template <typename OriginalCellSetType_,
          typename PermutationArrayHandleType_ =
            vtkm::cont::ArrayHandle<vtkm::Id, VTKM_DEFAULT_CELLSET_PERMUTATION_STORAGE_TAG>>
class CellSetPermutation : public CellSet
{
  VTKM_IS_CELL_SET(OriginalCellSetType_);
  VTKM_IS_ARRAY_HANDLE(PermutationArrayHandleType_);
  VTKM_STATIC_ASSERT_MSG(
    (std::is_same<vtkm::Id, typename PermutationArrayHandleType_::ValueType>::value),
    "Must use ArrayHandle with value type of Id for permutation array.");

public:
  using OriginalCellSetType = OriginalCellSetType_;
  using PermutationArrayHandleType = PermutationArrayHandleType_;

  /// @brief Create a `CellSetPermutation`.
  ///
  /// @param[in] validCellIds An array that defines the permutation. If index @a i
  ///   is value @a j, then the @a ith cell of this cell set will be the same as
  ///   the @a jth cell in the original @a cellset.
  /// @param[in] cellset The original cell set that this one is permuting.
  VTKM_CONT CellSetPermutation(const PermutationArrayHandleType& validCellIds,
                               const OriginalCellSetType& cellset)
    : CellSet()
    , ValidCellIds(validCellIds)
    , FullCellSet(cellset)
  {
  }

  VTKM_CONT CellSetPermutation()
    : CellSet()
    , ValidCellIds()
    , FullCellSet()
  {
  }

  ~CellSetPermutation() override {}

  CellSetPermutation(const CellSetPermutation& src)
    : CellSet()
    , ValidCellIds(src.ValidCellIds)
    , FullCellSet(src.FullCellSet)
  {
  }


  CellSetPermutation& operator=(const CellSetPermutation& src)
  {
    this->ValidCellIds = src.ValidCellIds;
    this->FullCellSet = src.FullCellSet;
    return *this;
  }

  /// @brief Returns the original `CellSet` that this one is permuting.
  VTKM_CONT
  const OriginalCellSetType& GetFullCellSet() const { return this->FullCellSet; }

  /// @brief Returns the array used to permute the cell indices.
  VTKM_CONT
  const PermutationArrayHandleType& GetValidCellIds() const { return this->ValidCellIds; }

  VTKM_CONT
  vtkm::Id GetNumberOfCells() const override { return this->ValidCellIds.GetNumberOfValues(); }

  VTKM_CONT
  vtkm::Id GetNumberOfPoints() const override { return this->FullCellSet.GetNumberOfPoints(); }

  VTKM_CONT
  vtkm::Id GetNumberOfFaces() const override { return -1; }

  VTKM_CONT
  vtkm::Id GetNumberOfEdges() const override { return -1; }

  VTKM_CONT
  void ReleaseResourcesExecution() override
  {
    this->ValidCellIds.ReleaseResourcesExecution();
    this->FullCellSet.ReleaseResourcesExecution();
    this->VisitPointsWithCells.ReleaseResourcesExecution();
  }

  VTKM_CONT
  vtkm::IdComponent GetNumberOfPointsInCell(vtkm::Id cellIndex) const override
  {
    // Looping over GetNumberOfPointsInCell is a performance bug.
    return this->FullCellSet.GetNumberOfPointsInCell(
      this->ValidCellIds.ReadPortal().Get(cellIndex));
  }

  VTKM_DEPRECATED(1.6,
                  "Calling GetCellShape(cellid) is a performance bug. Call ShapesReadPortal() "
                  "and loop over the Get.")
  vtkm::UInt8 GetCellShape(vtkm::Id id) const override
  {
    // Looping over GetCellShape is a performance bug.
    VTKM_DEPRECATED_SUPPRESS_BEGIN
    return this->FullCellSet.GetCellShape(this->ValidCellIds.ReadPortal().Get(id));
    VTKM_DEPRECATED_SUPPRESS_END
  }

  void GetCellPointIds(vtkm::Id id, vtkm::Id* ptids) const override
  {
    // Looping over GetCellPointsIdx is a performance bug.
    return this->FullCellSet.GetCellPointIds(this->ValidCellIds.ReadPortal().Get(id), ptids);
  }

  std::shared_ptr<CellSet> NewInstance() const override
  {
    return std::make_shared<CellSetPermutation>();
  }

  void DeepCopy(const CellSet* src) override
  {
    const auto* other = dynamic_cast<const CellSetPermutation*>(src);
    if (!other)
    {
      throw vtkm::cont::ErrorBadType("CellSetPermutation::DeepCopy types don't match");
    }

    this->FullCellSet.DeepCopy(&(other->GetFullCellSet()));
    vtkm::cont::ArrayCopy(other->GetValidCellIds(), this->ValidCellIds);
  }

  /// @brief Set the topology.
  ///
  /// @param[in] validCellIds An array that defines the permutation. If index @a i
  ///   is value @a j, then the @a ith cell of this cell set will be the same as
  ///   the @a jth cell in the original @a cellset.
  /// @param[in] cellset The original cell set that this one is permuting.
  VTKM_CONT
  void Fill(const PermutationArrayHandleType& validCellIds, const OriginalCellSetType& cellset)
  {
    this->ValidCellIds = validCellIds;
    this->FullCellSet = cellset;
  }

  VTKM_CONT vtkm::Id GetSchedulingRange(vtkm::TopologyElementTagCell) const
  {
    return this->ValidCellIds.GetNumberOfValues();
  }

  VTKM_CONT vtkm::Id GetSchedulingRange(vtkm::TopologyElementTagPoint) const
  {
    return this->FullCellSet.GetNumberOfPoints();
  }

  template <typename VisitTopology, typename IncidentTopology>
  using ExecConnectivityType = typename internal::CellSetPermutationConnectivityChooser<
    VisitTopology,
    IncidentTopology,
    OriginalCellSetType,
    PermutationArrayHandleType>::ExecConnectivityType;

  /// @brief Prepares the data for a particular device and returns the execution object for it.
  ///
  /// @param device Specifies the device on which the cell set will ve available.
  /// @param visitTopology Specifies the "visit" topology element. This is the element
  /// that will be indexed in the resulting connectivity object. This is typically
  /// `vtkm::TopologyElementTagPoint` or `vtkm::TopologyElementTagCell`.
  /// @param incidentTopology Specifies the "incident" topology element. This is the element
  /// that will incident to the elements that are visited. This is typically
  /// `vtkm::TopologyElementTagPoint` or `vtkm::TopologyElementTagCell`.
  /// @param token Provides a `vtkm::cont::Token` object that will define the span which
  /// the return execution object must be valid.
  ///
  /// @returns A connectivity object that can be used in the execution environment on the
  /// specified device.
  VTKM_CONT ExecConnectivityType<vtkm::TopologyElementTagCell, vtkm::TopologyElementTagPoint>
  PrepareForInput(vtkm::cont::DeviceAdapterId device,
                  vtkm::TopologyElementTagCell visitTopology,
                  vtkm::TopologyElementTagPoint incidentTopology,
                  vtkm::cont::Token& token) const
  {
    using ConnectivityType =
      ExecConnectivityType<vtkm::TopologyElementTagCell, vtkm::TopologyElementTagPoint>;
    return ConnectivityType(
      this->ValidCellIds.PrepareForInput(device, token),
      this->FullCellSet.PrepareForInput(device, visitTopology, incidentTopology, token));
  }

  /// @copydoc PrepareForInput
  VTKM_CONT ExecConnectivityType<vtkm::TopologyElementTagPoint, vtkm::TopologyElementTagCell>
  PrepareForInput(vtkm::cont::DeviceAdapterId device,
                  vtkm::TopologyElementTagPoint visitTopology,
                  vtkm::TopologyElementTagCell incidentTopology,
                  vtkm::cont::Token& token) const
  {
    (void)visitTopology;
    (void)incidentTopology;
    if (!this->VisitPointsWithCells.ElementsValid)
    {
      auto connTable = internal::RConnBuilderInput<CellSetPermutation>::Get(*this, device);
      internal::ComputeRConnTable(
        this->VisitPointsWithCells, connTable, this->GetNumberOfPoints(), device);
    }

    using ConnectivityType =
      ExecConnectivityType<vtkm::TopologyElementTagPoint, vtkm::TopologyElementTagCell>;
    return ConnectivityType(this->VisitPointsWithCells.Connectivity.PrepareForInput(device, token),
                            this->VisitPointsWithCells.Offsets.PrepareForInput(device, token));
  }

  VTKM_CONT
  void PrintSummary(std::ostream& out) const override
  {
    out << "CellSetPermutation of: " << std::endl;
    this->FullCellSet.PrintSummary(out);
    out << "Permutation Array: " << std::endl;
    vtkm::cont::printSummary_ArrayHandle(this->ValidCellIds, out);
  }

private:
  using VisitPointsWithCellsConnectivity = vtkm::cont::internal::ConnectivityExplicitInternals<
    typename ArrayHandleConstant<vtkm::UInt8>::StorageTag>;

  PermutationArrayHandleType ValidCellIds;
  OriginalCellSetType FullCellSet;
  mutable VisitPointsWithCellsConnectivity VisitPointsWithCells;
};

template <typename CellSetType,
          typename PermutationArrayHandleType1,
          typename PermutationArrayHandleType2>
class CellSetPermutation<CellSetPermutation<CellSetType, PermutationArrayHandleType1>,
                         PermutationArrayHandleType2>
  : public internal::CellSetPermutationTraits<
      CellSetPermutation<CellSetPermutation<CellSetType, PermutationArrayHandleType1>,
                         PermutationArrayHandleType2>>::Superclass
{
private:
  using Superclass = typename internal::CellSetPermutationTraits<CellSetPermutation>::Superclass;

public:
  VTKM_CONT
  CellSetPermutation(const PermutationArrayHandleType2& validCellIds,
                     const CellSetPermutation<CellSetType, PermutationArrayHandleType1>& cellset)
    : Superclass(vtkm::cont::make_ArrayHandlePermutation(validCellIds, cellset.GetValidCellIds()),
                 cellset.GetFullCellSet())
  {
  }

  VTKM_CONT
  CellSetPermutation()
    : Superclass()
  {
  }

  ~CellSetPermutation() override {}

  VTKM_CONT
  void Fill(const PermutationArrayHandleType2& validCellIds,
            const CellSetPermutation<CellSetType, PermutationArrayHandleType1>& cellset)
  {
    this->ValidCellIds = make_ArrayHandlePermutation(validCellIds, cellset.GetValidCellIds());
    this->FullCellSet = cellset.GetFullCellSet();
  }

  std::shared_ptr<CellSet> NewInstance() const override
  {
    return std::make_shared<CellSetPermutation>();
  }
};

template <typename OriginalCellSet, typename PermutationArrayHandleType>
vtkm::cont::CellSetPermutation<OriginalCellSet, PermutationArrayHandleType> make_CellSetPermutation(
  const PermutationArrayHandleType& cellIndexMap,
  const OriginalCellSet& cellSet)
{
  VTKM_IS_CELL_SET(OriginalCellSet);
  VTKM_IS_ARRAY_HANDLE(PermutationArrayHandleType);

  return vtkm::cont::CellSetPermutation<OriginalCellSet, PermutationArrayHandleType>(cellIndexMap,
                                                                                     cellSet);
}
}
} // namespace vtkm::cont

//=============================================================================
// Specializations of serialization related classes
/// @cond SERIALIZATION
namespace vtkm
{
namespace cont
{

template <typename CSType, typename AHValidCellIds>
struct SerializableTypeString<vtkm::cont::CellSetPermutation<CSType, AHValidCellIds>>
{
  static VTKM_CONT const std::string& Get()
  {
    static std::string name = "CS_Permutation<" + SerializableTypeString<CSType>::Get() + "," +
      SerializableTypeString<AHValidCellIds>::Get() + ">";
    return name;
  }
};
}
} // vtkm::cont

namespace mangled_diy_namespace
{

template <typename CSType, typename AHValidCellIds>
struct Serialization<vtkm::cont::CellSetPermutation<CSType, AHValidCellIds>>
{
private:
  using Type = vtkm::cont::CellSetPermutation<CSType, AHValidCellIds>;

public:
  static VTKM_CONT void save(BinaryBuffer& bb, const Type& cs)
  {
    vtkmdiy::save(bb, cs.GetFullCellSet());
    vtkmdiy::save(bb, cs.GetValidCellIds());
  }

  static VTKM_CONT void load(BinaryBuffer& bb, Type& cs)
  {
    CSType fullCS;
    vtkmdiy::load(bb, fullCS);
    AHValidCellIds validCellIds;
    vtkmdiy::load(bb, validCellIds);

    cs = make_CellSetPermutation(validCellIds, fullCS);
  }
};

} // diy
/// @endcond SERIALIZATION

#endif //vtk_m_cont_CellSetPermutation_h
