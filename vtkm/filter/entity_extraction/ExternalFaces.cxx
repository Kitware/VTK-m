//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#include <vtkm/cont/UncertainCellSet.h>
#include <vtkm/filter/MapFieldPermutation.h>
#include <vtkm/filter/clean_grid/CleanGrid.h>
#include <vtkm/filter/entity_extraction/ExternalFaces.h>
#include <vtkm/filter/entity_extraction/worklet/ExternalFaces.h>

namespace vtkm
{
namespace filter
{
namespace entity_extraction
{
//-----------------------------------------------------------------------------
ExternalFaces::ExternalFaces()
  : Worklet(std::make_unique<vtkm::worklet::ExternalFaces>())
{
  this->SetPassPolyData(true);
}

ExternalFaces::~ExternalFaces() = default;

//-----------------------------------------------------------------------------
void ExternalFaces::SetPassPolyData(bool value)
{
  this->PassPolyData = value;
  this->Worklet->SetPassPolyData(value);
}

//-----------------------------------------------------------------------------
vtkm::cont::DataSet ExternalFaces::GenerateOutput(const vtkm::cont::DataSet& input,
                                                  vtkm::cont::UnknownCellSet& outCellSet)
{
  //3. Check the fields of the dataset to see what kinds of fields are present, so
  //   we can free the cell mapping array if it won't be needed.
  const vtkm::Id numFields = input.GetNumberOfFields();
  bool hasCellFields = false;
  for (vtkm::Id fieldIdx = 0; fieldIdx < numFields && !hasCellFields; ++fieldIdx)
  {
    const auto& f = input.GetField(fieldIdx);
    hasCellFields = f.IsCellField();
  }

  if (!hasCellFields)
  {
    this->Worklet->ReleaseCellMapArrays();
  }

  //4. create the output dataset
  auto mapper = [&](auto& result, const auto& f) {
    // New Design: We are still using the old MapFieldOntoOutput to demonstrate the transition
    this->MapFieldOntoOutput(result, f);
  };
  return this->CreateResult(input, outCellSet, mapper);
}

//-----------------------------------------------------------------------------
vtkm::cont::DataSet ExternalFaces::DoExecute(const vtkm::cont::DataSet& input)
{
  //1. extract the cell set
  const vtkm::cont::UnknownCellSet& cells = input.GetCellSet();

  //2. using the policy convert the dynamic cell set, and run the
  // external faces worklet
  vtkm::cont::UnknownCellSet outCellSet;

  if (cells.CanConvert<vtkm::cont::CellSetStructured<3>>())
  {
    outCellSet = this->Worklet->Run(cells.AsCellSet<vtkm::cont::CellSetStructured<3>>());
  }
  else
  {
    outCellSet =
      this->Worklet->Run(cells.ResetCellSetList<VTKM_DEFAULT_CELL_SET_LIST_UNSTRUCTURED>());
  }

  // New Filter Design: we generate new output and map the fields first.
  auto output = this->GenerateOutput(input, outCellSet);

  // New Filter Design: then we remove entities if requested.
  if (this->CompactPoints)
  {
    vtkm::filter::clean_grid::CleanGrid compactor;
    compactor.SetCompactPointFields(true);
    compactor.SetMergePoints(false);
    return compactor.Execute(output);
  }
  else
  {
    return output;
  }
}

bool ExternalFaces::MapFieldOntoOutput(vtkm::cont::DataSet& result, const vtkm::cont::Field& field)
{
  if (field.IsPointField())
  {
    result.AddField(field);
    return true;
  }
  else if (field.IsCellField())
  {
    return vtkm::filter::MapFieldPermutation(field, this->Worklet->GetCellIdMap(), result);
  }
  else if (field.IsWholeDataSetField())
  {
    result.AddField(field);
    return true;
  }
  else
  {
    return false;
  }
}
} // namespace entity_extraction
} // namespace filter
} // namespace vtkm
