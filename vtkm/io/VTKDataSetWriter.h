//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_io_DataSetWriter_h
#define vtk_m_io_DataSetWriter_h

#include <vtkm/cont/DataSet.h>

#include <vtkm/io/vtkm_io_export.h>

namespace vtkm
{
namespace io
{

// Might want to place this somewhere else.
enum struct FileType
{
  ASCII,
  BINARY
};

/// @brief Reads a legacy VTK file.
///
/// By convention, legacy VTK files have a `.vtk` extension.
/// This class should be constructed with a filename, and the data
/// read with `ReadDataSet`.
class VTKM_IO_EXPORT VTKDataSetWriter
{
public:
  VTKM_CONT VTKDataSetWriter(const char* fileName);
  /// @brief Construct a writer to save data to the given file.
  VTKM_CONT VTKDataSetWriter(const std::string& fileName);

  /// @brief Write data from the given `DataSet` object to the file specified in the constructor.
  VTKM_CONT void WriteDataSet(const vtkm::cont::DataSet& dataSet) const;

  /// @brief Get whether the file will be written in ASCII or binary format.
  ///
  VTKM_CONT vtkm::io::FileType GetFileType() const;

  /// @brief Set whether the file will be written in ASCII or binary format.
  VTKM_CONT void SetFileType(vtkm::io::FileType type);
  /// @brief Set whether the file will be written in ASCII or binary format.
  VTKM_CONT void SetFileTypeToAscii() { this->SetFileType(vtkm::io::FileType::ASCII); }
  /// @brief Set whether the file will be written in ASCII or binary format.
  VTKM_CONT void SetFileTypeToBinary() { this->SetFileType(vtkm::io::FileType::BINARY); }

private:
  std::string FileName;
  vtkm::io::FileType FileType = vtkm::io::FileType::ASCII;

}; //struct VTKDataSetWriter
}
} //namespace vtkm::io

#endif //vtk_m_io_DataSetWriter_h
