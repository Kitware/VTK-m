//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#ifndef vtk_m_Bounds_h
#define vtk_m_Bounds_h

#include <vtkm/Range.h>

namespace vtkm
{

/// \brief Represent an axis-aligned 3D bounds in space.
///
/// \c vtkm::Bounds is a helper class for representing the axis-aligned box
/// representing some region in space. The typical use of this class is to
/// express the containing box of some geometry. The box is specified as ranges
/// in the x, y, and z directions.
///
/// \c Bounds also contains several helper functions for computing and
/// maintaining the bounds.
///
struct Bounds
{
  /// The range of values in the X direction. The `vtkm::Range` struct provides
  /// the minimum and maximum along that axis.
  vtkm::Range X;
  /// The range of values in the Y direction. The `vtkm::Range` struct provides
  /// the minimum and maximum along that axis.
  vtkm::Range Y;
  /// The range of values in the Z direction. The `vtkm::Range` struct provides
  /// the minimum and maximum along that axis.
  vtkm::Range Z;

  /// Construct an empty bounds. The bounds will represent no space until
  /// otherwise modified.
  VTKM_EXEC_CONT
  Bounds() {}

  Bounds(const Bounds&) = default;

  /// Construct a bounds with a given range in the x, y, and z dimensions.
  VTKM_EXEC_CONT
  Bounds(const vtkm::Range& xRange, const vtkm::Range& yRange, const vtkm::Range& zRange)
    : X(xRange)
    , Y(yRange)
    , Z(zRange)
  {
  }

  /// Construct a bounds with the minimum and maximum coordinates in the x, y, and z
  /// directions.
  template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
  VTKM_EXEC_CONT Bounds(const T1& minX,
                        const T2& maxX,
                        const T3& minY,
                        const T4& maxY,
                        const T5& minZ,
                        const T6& maxZ)
    : X(vtkm::Range(minX, maxX))
    , Y(vtkm::Range(minY, maxY))
    , Z(vtkm::Range(minZ, maxZ))
  {
  }

  /// Initialize bounds with an array of 6 values in the order xmin, xmax,
  /// ymin, ymax, zmin, zmax.
  ///
  template <typename T>
  VTKM_EXEC_CONT explicit Bounds(const T bounds[6])
    : X(vtkm::Range(bounds[0], bounds[1]))
    , Y(vtkm::Range(bounds[2], bounds[3]))
    , Z(vtkm::Range(bounds[4], bounds[5]))
  {
  }

  /// Initialize bounds with the minimum corner point and the maximum corner
  /// point.
  ///
  template <typename T>
  VTKM_EXEC_CONT Bounds(const vtkm::Vec<T, 3>& minPoint, const vtkm::Vec<T, 3>& maxPoint)
    : X(vtkm::Range(minPoint[0], maxPoint[0]))
    , Y(vtkm::Range(minPoint[1], maxPoint[1]))
    , Z(vtkm::Range(minPoint[2], maxPoint[2]))
  {
  }

  vtkm::Bounds& operator=(const vtkm::Bounds& src) = default;

  /// \b Determine if the bounds are valid (i.e. has at least one valid point).
  ///
  /// \c IsNonEmpty returns true if the bounds contain some valid points. If
  /// the bounds are any real region, even if a single point or it expands to
  /// infinity, true is returned.
  ///
  VTKM_EXEC_CONT
  bool IsNonEmpty() const
  {
    return (this->X.IsNonEmpty() && this->Y.IsNonEmpty() && this->Z.IsNonEmpty());
  }

  /// \b Determines if a point coordinate is within the bounds.
  ///
  template <typename T>
  VTKM_EXEC_CONT bool Contains(const vtkm::Vec<T, 3>& point) const
  {
    return (this->X.Contains(point[0]) && this->Y.Contains(point[1]) && this->Z.Contains(point[2]));
  }

  /// \b Returns the volume of the bounds.
  ///
  /// \c Volume computes the product of the lengths of the ranges in each dimension. If the bounds
  /// are empty, 0 is returned.
  ///
  VTKM_EXEC_CONT
  vtkm::Float64 Volume() const
  {
    if (this->IsNonEmpty())
    {
      return (this->X.Length() * this->Y.Length() * this->Z.Length());
    }
    else
    {
      return 0.0;
    }
  }

  /// \b Returns the area of the bounds in the X-Y-plane.
  ///
  /// \c Area computes the product of the lengths of the ranges in dimensions X and Y. If the bounds
  /// are empty, 0 is returned.
  ///
  VTKM_EXEC_CONT
  vtkm::Float64 Area() const
  {
    if (this->IsNonEmpty())
    {
      return (this->X.Length() * this->Y.Length());
    }
    else
    {
      return 0.0;
    }
  }

  /// \b Returns the center of the range.
  ///
  /// \c Center computes the point at the middle of the bounds. If the bounds
  /// are empty, the results are undefined.
  ///
  VTKM_EXEC_CONT
  vtkm::Vec3f_64 Center() const
  {
    return vtkm::Vec3f_64(this->X.Center(), this->Y.Center(), this->Z.Center());
  }

  /// \b Returns the min point of the bounds
  ///
  /// \c MinCorder returns the minium point of the bounds.If the bounds
  /// are empty, the results are undefined.
  ///
  VTKM_EXEC_CONT
  vtkm::Vec3f_64 MinCorner() const { return vtkm::Vec3f_64(this->X.Min, this->Y.Min, this->Z.Min); }

  /// \b Returns the max point of the bounds
  ///
  /// \c MaxCorder returns the minium point of the bounds.If the bounds
  /// are empty, the results are undefined.
  ///
  VTKM_EXEC_CONT
  vtkm::Vec3f_64 MaxCorner() const { return vtkm::Vec3f_64(this->X.Max, this->Y.Max, this->Z.Max); }

  /// \b Expand bounds to include a point.
  ///
  /// This version of \c Include expands the bounds just enough to include the
  /// given point coordinates. If the bounds already include this point, then
  /// nothing is done.
  ///
  template <typename T>
  VTKM_EXEC_CONT void Include(const vtkm::Vec<T, 3>& point)
  {
    this->X.Include(point[0]);
    this->Y.Include(point[1]);
    this->Z.Include(point[2]);
  }

  /// \b Expand bounds to include other bounds.
  ///
  /// This version of \c Include expands these bounds just enough to include
  /// that of another bounds. Essentially it is the union of the two bounds.
  ///
  VTKM_EXEC_CONT
  void Include(const vtkm::Bounds& bounds)
  {
    this->X.Include(bounds.X);
    this->Y.Include(bounds.Y);
    this->Z.Include(bounds.Z);
  }

  /// \b Return the union of this and another bounds.
  ///
  /// This is a nondestructive form of \c Include.
  ///
  VTKM_EXEC_CONT
  vtkm::Bounds Union(const vtkm::Bounds& otherBounds) const
  {
    vtkm::Bounds unionBounds(*this);
    unionBounds.Include(otherBounds);
    return unionBounds;
  }

  /// \b Return the intersection of this and another range.
  ///
  VTKM_EXEC_CONT
  vtkm::Bounds Intersection(const vtkm::Bounds& otherBounds) const
  {
    return vtkm::Bounds(this->X.Intersection(otherBounds.X),
                        this->Y.Intersection(otherBounds.Y),
                        this->Z.Intersection(otherBounds.Z));
  }

  /// \b Operator for union
  ///
  VTKM_EXEC_CONT
  vtkm::Bounds operator+(const vtkm::Bounds& otherBounds) const { return this->Union(otherBounds); }

  VTKM_EXEC_CONT
  bool operator==(const vtkm::Bounds& bounds) const
  {
    return ((this->X == bounds.X) && (this->Y == bounds.Y) && (this->Z == bounds.Z));
  }

  VTKM_EXEC_CONT
  bool operator!=(const vtkm::Bounds& bounds) const
  {
    return ((this->X != bounds.X) || (this->Y != bounds.Y) || (this->Z != bounds.Z));
  }
};

/// Helper function for printing bounds during testing
///
inline VTKM_CONT std::ostream& operator<<(std::ostream& stream, const vtkm::Bounds& bounds)
{
  return stream << "{ X:" << bounds.X << ", Y:" << bounds.Y << ", Z:" << bounds.Z << " }";
}

template <>
struct VTKM_NEVER_EXPORT VecTraits<vtkm::Bounds>
{
  using ComponentType = vtkm::Range;
  using BaseComponentType = vtkm::VecTraits<vtkm::Range>::BaseComponentType;

  static constexpr vtkm::IdComponent NUM_COMPONENTS = 3;
  static constexpr vtkm::IdComponent GetNumberOfComponents(const vtkm::Bounds&)
  {
    return NUM_COMPONENTS;
  }
  using HasMultipleComponents = vtkm::VecTraitsTagMultipleComponents;
  using IsSizeStatic = vtkm::VecTraitsTagSizeStatic;

  VTKM_EXEC_CONT
  static const ComponentType& GetComponent(const vtkm::Bounds& bounds, vtkm::IdComponent component)
  {
    VTKM_ASSERT((component >= 0) || (component < 3));
    switch (component)
    {
      case 0:
        return bounds.X;
      case 1:
        return bounds.Y;
      case 2:
        return bounds.Z;
      default:
        // Should never reach here
        return bounds.X;
    }
  }
  VTKM_EXEC_CONT
  static ComponentType& GetComponent(vtkm::Bounds& bounds, vtkm::IdComponent component)
  {
    VTKM_ASSERT((component >= 0) || (component < 3));
    switch (component)
    {
      case 0:
        return bounds.X;
      case 1:
        return bounds.Y;
      case 2:
        return bounds.Z;
      default:
        // Should never reach here
        return bounds.X;
    }
  }

  VTKM_EXEC_CONT
  static void SetComponent(vtkm::Bounds& bounds,
                           vtkm::IdComponent component,
                           const ComponentType& value)
  {
    VTKM_ASSERT((component >= 0) || (component < 3));
    switch (component)
    {
      case 0:
        bounds.X = value;
        break;
      case 1:
        bounds.Y = value;
        break;
      case 2:
        bounds.Z = value;
        break;
    }
  }

  template <typename NewComponentType>
  using ReplaceComponentType = vtkm::Vec<NewComponentType, NUM_COMPONENTS>;
  template <typename NewComponentType>
  using ReplaceBaseComponentType =
    vtkm::Vec<NewComponentType, NUM_COMPONENTS * vtkm::VecTraits<vtkm::Range>::NUM_COMPONENTS>;

  template <vtkm::IdComponent destSize>
  VTKM_EXEC_CONT static void CopyInto(const vtkm::Bounds& src,
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

#endif //vtk_m_Bounds_h
