//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#ifndef vtk_m_Range_h
#define vtk_m_Range_h

#include <vtkm/Assert.h>
#include <vtkm/Math.h>
#include <vtkm/Types.h>
#include <vtkm/VecTraits.h>

namespace vtkm
{

/// \brief Represent a continuous scalar range of values.
///
/// \c vtkm::Range is a helper class for representing a range of floating point
/// values from a minimum value to a maximum value. This is specified simply
/// enough with a \c Min and \c Max value.
///
/// \c Range also contains several helper functions for computing and
/// maintaining the range.
///
struct Range
{
  /// The minumum value of the range (inclusive).
  vtkm::Float64 Min;
  /// Tha maximum value of the range (inclusive).
  vtkm::Float64 Max;

  /// Construct a range with a given minimum and maximum. If no minimum or maximum is
  /// given, the range will be empty.
  VTKM_EXEC_CONT
  Range()
    : Min(vtkm::Infinity64())
    , Max(vtkm::NegativeInfinity64())
  {
  }

  Range(const Range&) = default;
  Range(Range&&) = default;

  template <typename T1, typename T2>
  VTKM_EXEC_CONT Range(const T1& min, const T2& max)
    : Min(static_cast<vtkm::Float64>(min))
    , Max(static_cast<vtkm::Float64>(max))
  {
  }

  vtkm::Range& operator=(const vtkm::Range& src) = default;
  vtkm::Range& operator=(vtkm::Range&& src) = default;

  /// \b Determine if the range is valid (i.e. has at least one valid point).
  ///
  /// \c IsNonEmpty return true if the range contains some valid values between
  /// \c Min and \c Max. If \c Max is less than \c Min, then no values satisfy
  /// the range and \c IsNonEmpty returns false. Otherwise, return true.
  ///
  /// \c IsNonEmpty assumes \c Min and \c Max are inclusive. That is, if they
  /// are equal then true is returned.
  ///
  VTKM_EXEC_CONT
  bool IsNonEmpty() const { return (this->Min <= this->Max); }

  /// \b Determines if a value is within the range.
  ///
  /// \c Contains returns true if the give value is within the range, false
  /// otherwise. \c Contains treats the min and max as inclusive. That is, if
  /// the value is exactly the min or max, true is returned.
  ///
  template <typename T>
  VTKM_EXEC_CONT bool Contains(const T& value) const
  {
    return ((this->Min <= static_cast<vtkm::Float64>(value)) &&
            (this->Max >= static_cast<vtkm::Float64>(value)));
  }

  /// \b Returns the length of the range.
  ///
  /// \c Length computes the distance between the min and max. If the range
  /// is empty, 0 is returned.
  ///
  VTKM_EXEC_CONT
  vtkm::Float64 Length() const
  {
    if (this->IsNonEmpty())
    {
      return (this->Max - this->Min);
    }
    else
    {
      return 0.0;
    }
  }

  /// \b Returns the center of the range.
  ///
  /// \c Center computes the middle value of the range. If the range is empty,
  /// NaN is returned.
  ///
  VTKM_EXEC_CONT
  vtkm::Float64 Center() const
  {
    if (this->IsNonEmpty())
    {
      return 0.5 * (this->Max + this->Min);
    }
    else
    {
      return vtkm::Nan64();
    }
  }

  /// \b Expand range to include a value.
  ///
  /// This version of \c Include expands the range just enough to include the
  /// given value. If the range already includes this value, then nothing is
  /// done.
  ///
  template <typename T>
  VTKM_EXEC_CONT void Include(const T& value)
  {
    this->Min = vtkm::Min(this->Min, static_cast<vtkm::Float64>(value));
    this->Max = vtkm::Max(this->Max, static_cast<vtkm::Float64>(value));
  }

  /// \b Expand range to include other range.
  ///
  /// This version of \c Include expands this range just enough to include that
  /// of another range. Essentially it is the union of the two ranges.
  ///
  VTKM_EXEC_CONT
  void Include(const vtkm::Range& range)
  {
    if (range.IsNonEmpty())
    {
      this->Min = vtkm::Min(this->Min, range.Min);
      this->Max = vtkm::Max(this->Max, range.Max);
    }
  }

  /// \b Return the union of this and another range.
  ///
  /// This is a nondestructive form of \c Include.
  ///
  VTKM_EXEC_CONT
  vtkm::Range Union(const vtkm::Range& otherRange) const
  {
    vtkm::Range unionRange(*this);
    unionRange.Include(otherRange);
    return unionRange;
  }

  /// \b Return the intersection of this and another range.
  ///
  VTKM_EXEC_CONT
  vtkm::Range Intersection(const vtkm::Range& otherRange) const
  {
    return vtkm::Range(vtkm::Max(this->Min, otherRange.Min), vtkm::Min(this->Max, otherRange.Max));
  }

  /// \b Operator for union
  ///
  VTKM_EXEC_CONT
  vtkm::Range operator+(const vtkm::Range& otherRange) const { return this->Union(otherRange); }

  VTKM_EXEC_CONT
  bool operator==(const vtkm::Range& otherRange) const
  {
    return ((this->Min == otherRange.Min) && (this->Max == otherRange.Max));
  }

  VTKM_EXEC_CONT
  bool operator!=(const vtkm::Range& otherRange) const
  {
    return ((this->Min != otherRange.Min) || (this->Max != otherRange.Max));
  }
};

/// Helper function for printing ranges during testing
///
inline VTKM_CONT std::ostream& operator<<(std::ostream& stream, const vtkm::Range& range)
{
  return stream << "[" << range.Min << ".." << range.Max << "]";
} // Declared inside of vtkm namespace so that the operator work with ADL lookup

template <>
struct VTKM_NEVER_EXPORT VecTraits<vtkm::Range>
{
  using ComponentType = vtkm::Float64;
  using BaseComponentType = vtkm::Float64;

  static constexpr vtkm::IdComponent NUM_COMPONENTS = 2;
  static constexpr vtkm::IdComponent GetNumberOfComponents(const vtkm::Range&)
  {
    return NUM_COMPONENTS;
  }
  using HasMultipleComponents = vtkm::VecTraitsTagMultipleComponents;
  using IsSizeStatic = vtkm::VecTraitsTagSizeStatic;

  VTKM_EXEC_CONT
  static const ComponentType& GetComponent(const vtkm::Range& range, vtkm::IdComponent component)
  {
    VTKM_ASSERT((component == 0) || (component == 1));
    return (component == 0) ? range.Min : range.Max;
  }
  VTKM_EXEC_CONT
  static ComponentType& GetComponent(vtkm::Range& range, vtkm::IdComponent component)
  {
    VTKM_ASSERT((component == 0) || (component == 1));
    return (component == 0) ? range.Min : range.Max;
  }

  VTKM_EXEC_CONT
  static void SetComponent(vtkm::Range& range, vtkm::IdComponent component, ComponentType value)
  {
    VTKM_ASSERT((component == 0) || (component == 1));
    if (component == 0)
    {
      range.Min = value;
    }
    else
    {
      range.Max = value;
    }
  }

  template <typename NewComponentType>
  using ReplaceComponentType = vtkm::Vec<NewComponentType, NUM_COMPONENTS>;
  template <typename NewComponentType>
  using ReplaceBaseComponentType = vtkm::Vec<NewComponentType, NUM_COMPONENTS>;

  template <vtkm::IdComponent destSize>
  VTKM_EXEC_CONT static void CopyInto(const vtkm::Range& src,
                                      vtkm::Vec<ComponentType, destSize>& dest)
  {
    const vtkm::IdComponent maxComponent = (destSize < NUM_COMPONENTS) ? destSize : NUM_COMPONENTS;
    for (vtkm::IdComponent component = 0; component < maxComponent; ++component)
    {
      dest[component] = GetComponent(src, component);
    }
  }
};

} // namespace vtkm


#endif //vtk_m_Range_h
