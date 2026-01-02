project(otmap)

cmake_minimum_required(VERSION 3.2.0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use system libraries over conda/anaconda libraries
# This ensures we link against the correct glibc++ version
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH "/usr/lib/x86_64-linux-gnu")
set(CMAKE_INSTALL_RPATH "/usr/lib/x86_64-linux-gnu")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(USE_CIMAGE true)

# Force MSVC to use the static runtime (/MT or /MTd)
if(MSVC)
  foreach(flag_var CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
                   CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE)
    string(REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
  endforeach()
endif()

if(USE_CIMAGE)
    add_definitions(-D_cimg_)
    add_definitions(-Dcimg_display=0)

    find_package(PNG REQUIRED)
    if(PNG_FOUND)
        add_definitions(-Dcimg_use_png)
        set(ALLLIBS ${ALLLIBS} PNG::PNG)
    endif()

    find_package(JPEG)
    if(JPEG_FOUND)
        add_definitions(-Dcimg_use_jpeg)
        set(ALLLIBS ${ALLLIBS} JPEG::JPEG)
    endif()

   if(WIN32)
      add_definitions(-Dcimg_OS=2)
   else()
      add_definitions(-Dcimg_OS=1)
   endif()
endif()

include_directories(extern/)
include_directories(extern/eigen/)
include_directories(otlib/)

find_package(Cholmod)
if(CHOLMOD_FOUND)
  message(STATUS "Enable Cholmod support")
  add_definitions("-DHAS_CHOLMOD")
  include_directories(${CHOLMOD_INCLUDES})
  set(ALLLIBS ${ALLLIBS} ${CHOLMOD_LIBRARIES})
else()
  message(STATUS "Disable Cholmod support (not found)")
  message(WARNING "Chomod not found, solving will be significantly slower than expected.")
endif()

find_package(Ceres REQUIRED)

## FILES ##########################################################################
set(SURFACE_MESH_SRC_FILES
    otlib/surface_mesh/Surface_mesh.cpp
)

set(SURFACE_MESH_HEADER_FILES
    otlib/surface_mesh/properties.h
    otlib/surface_mesh/Surface_mesh.h
    otlib/surface_mesh/types.h
    otlib/surface_mesh/Vector.h
)

set(OTSOLVER_SRC_FILES
    otlib/transport_map.cpp
    otlib/otsolver_2dgrid.cpp
    otlib/otsolver_trianglemesh.cpp
    otlib/details/line_search.cpp
    otlib/details/nested_dissection.cpp
    otlib/utils/bvh2d.cpp
    otlib/utils/rasterizer.cpp
    otlib/utils/stochastic_rasterizer.cpp
    otlib/utils/mesh_utils.cpp
)

set(OTSOLVER_HEADER_FILES
    otlib/transport_map.h
    otlib/otsolver_2dgrid.h
    otlib/otsolver_trianglemesh.h
    otlib/details/line_search.h
    otlib/details/nested_dissection.h
    otlib/utils/bvh2d.h
    otlib/utils/rasterizer.h
    otlib/utils/stochastic_rasterizer.h
    otlib/utils/mesh_utils.h
)

###################################################################################

add_definitions(-DDATA_DIR="${PROJECT_SOURCE_DIR}/data")

if(NOT WIN32)
   add_definitions(-DENABLE_SSE_MODE)
else()
    add_definitions(-D_USE_MATH_DEFINES)
endif()

add_library(otlib ${SURFACE_MESH_SRC_FILES} ${OTSOLVER_SRC_FILES})



set(OTAPPUTILS_SRC_FILES
    apps/common/image_utils.cpp
    apps/common/inputparser.cpp
    apps/common/analytical_functions.cpp
    apps/common/generic_tasks.cpp
    apps/normal_integration/normal_integration.cpp
    apps/normal_integration/mesh.cpp
)

set(OTAPPUTILS_HEADER_FILES
    apps/common/image_utils.h
    apps/common/inputparser.h
    apps/common/analytical_functions.h
    apps/common/generic_tasks.h
    apps/common/otsolver_options.h
    apps/normal_integration/normal_integration.h
    apps/normal_integration/mesh.h
    apps/normal_integration/costFunctor.h
)

add_library(otapputils ${OTAPPUTILS_SRC_FILES})



add_executable(caustic_design apps/caustic_design.cpp)

target_link_libraries(caustic_design PRIVATE PNG::PNG)

target_link_libraries(otapputils
    PRIVATE
        Ceres::ceres
        PNG::PNG
        # JPEG::JPEG
)

target_link_libraries(caustic_design
    PRIVATE
        otapputils
        otlib
        ${ALLLIBS}
        Ceres::ceres
)
