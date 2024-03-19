//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_filter_resampling_HistSampling_h
#define vtk_m_filter_resampling_HistSampling_h

#include <vtkm/filter/Filter.h>
#include <vtkm/filter/resampling/vtkm_filter_resampling_export.h>

#include <vtkm/Deprecated.h>

namespace vtkm
{
namespace filter
{
namespace resampling
{

/// @brief Adaptively sample points to preserve tail features.
///
/// This filter randomly samples the points of a `vtkm::cont::DataSet` and generates
/// a new `vtkm::cont::DataSet` with a subsampling of the points. The sampling is
/// adaptively selected to preserve tail and outlying features of the active field.
/// That is, the more rare a field value is, the more likely the point will be
/// selected in the sampling. This is done by creating a histogram of the field
/// and using that to derive the importance level of each field value. Details of
/// the algorithm can be found in the paper "In Situ Data-Driven Adaptive Sampling
/// for Large-scale Simulation Data Summarization" by Biswas, Dutta, Pulido, and Ahrens
/// as published in _In Situ Infrastructures for Enabling Extreme-scale Analysis and
/// Visualization_ (ISAV 2018).
///
/// The cell set of the input data is removed and replaced with a set with a vertex
/// cell for each point. This effectively converts the data to a point cloud.
class VTKM_FILTER_RESAMPLING_EXPORT HistSampling : public vtkm::filter::Filter
{
public:
  /// @brief Specify the number of bins used when computing the histogram.
  ///
  /// The histogram is used to select the importance of each field value.
  /// More rare field values are more likely to be selected.
  VTKM_CONT void SetNumberOfBins(vtkm::Id numberOfBins) { this->NumberOfBins = numberOfBins; }
  /// @copydoc SetNumberOfBins
  VTKM_CONT vtkm::Id GetNumberOfBins() { return this->NumberOfBins; }

  /// @brief Specify the fraction of points to create in the sampled data.
  ///
  /// A fraction of 1 means that all the points will be sampled and be in the output.
  /// A fraction of 0 means that none of the points will be sampled. A fraction of 0.5 means
  /// that half the points will be selected to be in the output.
  VTKM_CONT void SetSampleFraction(vtkm::FloatDefault fraction) { this->SampleFraction = fraction; }
  /// @copydoc SetSampleFraction
  VTKM_CONT vtkm::FloatDefault GetSampleFraction() const { return this->SampleFraction; }

  VTKM_DEPRECATED(2.2, "Use SetSampleFraction().")
  VTKM_CONT void SetSamplePercent(vtkm::FloatDefault samplePercent)
  {
    this->SetSampleFraction(samplePercent);
  }
  VTKM_DEPRECATED(2.2, "Use GetSampleFraction().")
  VTKM_CONT vtkm::FloatDefault GetSamplePercent() const { return this->GetSampleFraction(); }

  /// @brief Specify the seed used for random number generation.
  ///
  /// The random numbers are used to select which points to pull from the input. If
  /// the same seed is used for multiple invocations, the results will be the same.
  VTKM_CONT void SetSeed(vtkm::UInt32 seed) { this->Seed = seed; }
  /// @copydoc SetSeed
  VTKM_CONT vtkm::UInt32 GetSeed() { return this->Seed; }

private:
  VTKM_CONT vtkm::cont::DataSet DoExecute(const vtkm::cont::DataSet& input) override;
  vtkm::Id NumberOfBins = 10;
  vtkm::FloatDefault SampleFraction = 0.1f;
  vtkm::UInt32 Seed = 0;
};

} // namespace resampling
} // namespace filter
} // namespace vtkm

#endif // vtk_m_filter_resampling_HistSampling_h
