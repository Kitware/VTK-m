##============================================================================
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##============================================================================
add_library(vtkm_loguru INTERFACE)

vtkm_get_kit_name(kit_name kit_dir)

# loguru needs C++11
target_compile_features(vtkm_loguru INTERFACE cxx_std_11)

target_include_directories(vtkm_loguru INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(vtkm_loguru INTERFACE ${CMAKE_DL_LIBS} Threads::Threads)

vtkm_install_targets(TARGETS vtkm_loguru)
