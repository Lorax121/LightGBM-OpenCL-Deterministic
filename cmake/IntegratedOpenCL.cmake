set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BOOST_VERSION_DOT "1.91")
string(REPLACE "." "_" BOOST_VERSION_UNDERSCORE ${BOOST_VERSION_DOT})

set(OPENCL_HEADER_REPOSITORY "https://github.com/KhronosGroup/OpenCL-Headers.git")
set(OPENCL_HEADER_TAG "v2026.05.29")

set(OPENCL_LOADER_REPOSITORY "https://github.com/KhronosGroup/OpenCL-ICD-Loader.git")
set(OPENCL_LOADER_TAG "v2026.05.29")

set(BOOST_REPOSITORY "https://github.com/boostorg/boost.git")
set(BOOST_TAG "boost-${BOOST_VERSION_DOT}.0")

# Build Independent OpenCL library
include(FetchContent)
# lint_cmake: -readability/wonkycase
FetchContent_Declare(OpenCL-Headers GIT_REPOSITORY ${OPENCL_HEADER_REPOSITORY} GIT_TAG ${OPENCL_HEADER_TAG})
FetchContent_GetProperties(OpenCL-Headers)
# lint_cmake: +readability/wonkycase
if(NOT OpenCL-Headers_POPULATED)
# lint_cmake: -readability/wonkycase
  FetchContent_MakeAvailable(OpenCL-Headers)
# lint_cmake: +readability/wonkycase
  message(STATUS "Populated OpenCL Headers")
endif()
set(OPENCL_ICD_LOADER_HEADERS_DIR ${opencl-headers_SOURCE_DIR} CACHE PATH "") # for OpenCL ICD Loader
set(OpenCL_INCLUDE_DIR ${opencl-headers_SOURCE_DIR} CACHE PATH "") # for Boost::Compute

# lint_cmake: -readability/wonkycase
FetchContent_Declare(
# lint_cmake: +readability/wonkycase
  OpenCL-ICD-Loader
  GIT_REPOSITORY
  ${OPENCL_LOADER_REPOSITORY}
  GIT_TAG
  ${OPENCL_LOADER_TAG}
  EXCLUDE_FROM_ALL
)
# lint_cmake: -readability/wonkycase
FetchContent_GetProperties(OpenCL-ICD-Loader)
# lint_cmake: +readability/wonkycase
if(NOT OpenCL-ICD-Loader_POPULATED)
# lint_cmake: -readability/wonkycase
  FetchContent_MakeAvailable(OpenCL-ICD-Loader)
# lint_cmake: +readability/wonkycase
  if(WIN32)
    set(USE_DYNAMIC_VCXX_RUNTIME ON)
  endif()
  message(STATUS "Populated OpenCL ICD Loader")
endif()
list(APPEND INTEGRATED_OPENCL_INCLUDES ${OPENCL_ICD_LOADER_HEADERS_DIR})
list(APPEND INTEGRATED_OPENCL_DEFINITIONS CL_TARGET_OPENCL_VERSION=120)
if(WIN32)
  list(
    APPEND
    INTEGRATED_OPENCL_LIBRARIES
      ${opencl-icd-loader_BINARY_DIR}/Release/OpenCL.lib
      cfgmgr32.lib
      runtimeobject.lib
  )
else()
  list(
    APPEND
    INTEGRATED_OPENCL_LIBRARIES
      ${opencl-icd-loader_BINARY_DIR}/libOpenCL.a
  )
  set_property(TARGET OpenCL PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

# Build Independent Boost libraries
include(ExternalProject)
include(ProcessorCount)
# lint_cmake: -readability/wonkycase
ProcessorCount(J)
# lint_cmake: +readability/wonkycase
set(BOOST_BASE "${PROJECT_BINARY_DIR}/Boost")
set(BOOST_INCLUDE "${BOOST_BASE}/source" CACHE PATH "")
set(BOOST_LIBRARY "${BOOST_BASE}/source/stage/lib" CACHE PATH "")
if(WIN32)
  if(MSVC)
    # references:
    #
    #  * range of MSVC versions: https://learn.microsoft.com/en-us/cpp/overview/compiler-versions
    #  * MSVC toolchain IDs: not sure...
    #    comments like https://learn.microsoft.com/en-us/answers/questions/769911/visual-studio-2019-build-tools-v143
    #
    if(${MSVC_VERSION} GREATER 1949)
      set(MSVC_TOOLCHAIN_ID "145")
      set(BOOST_MSVC_VERSION "14.5")
    elseif(${MSVC_VERSION} GREATER 1929)
      set(MSVC_TOOLCHAIN_ID "143")
      set(BOOST_MSVC_VERSION "14.3")
    elseif(${MSVC_VERSION} GREATER 1919)
      set(MSVC_TOOLCHAIN_ID "142")
      set(BOOST_MSVC_VERSION "14.2")
    elseif(${MSVC_VERSION} GREATER 1909)
      set(MSVC_TOOLCHAIN_ID "141")
      set(BOOST_MSVC_VERSION "14.1")
    else()
      message(FATAL_ERROR "Unsupported MSVC version number: ${MSVC_VERSION}")
    endif()
    list(
      APPEND
        BOOST_BUILD_BYPRODUCTS
          ${BOOST_LIBRARY}/libboost_filesystem-vc${MSVC_TOOLCHAIN_ID}-mt-x64-${BOOST_VERSION_UNDERSCORE}.lib
          ${BOOST_LIBRARY}/libboost_chrono-vc${MSVC_TOOLCHAIN_ID}-mt-x64-${BOOST_VERSION_UNDERSCORE}.lib
    )
  else()
    message(FATAL_ERROR "Integrated OpenCL build is not yet available for MinGW")
  endif()
  set(BOOST_BOOTSTRAP "${BOOST_BASE}/source/bootstrap.bat")
  set(BOOST_BUILD "${BOOST_BASE}/source/b2.exe")
  # Register the exact cl.exe, force the 64-bit address model (the CMake
  # byproducts below use the -x64 tag), and give b2 an explicit vcvarsall.bat
  # wrapper (VS2017+ keeps vcvarsall.bat outside the MSVC toolset dir).
  get_filename_component(_msvc_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)   # .../bin/Hostx64/x64
  get_filename_component(_msvc_dir "${_msvc_dir}" DIRECTORY)            # .../bin/Hostx64
  get_filename_component(_msvc_dir "${_msvc_dir}" DIRECTORY)            # .../bin
  get_filename_component(_msvc_toolset_dir "${_msvc_dir}" DIRECTORY)    # .../MSVC/<ver>
  get_filename_component(_msvc_toolset_version "${_msvc_toolset_dir}" NAME)
  string(REGEX MATCH "^[0-9]+\\.[0-9]+" _msvc_vcvars_version "${_msvc_toolset_version}")
  get_filename_component(_msvc_dir "${_msvc_toolset_dir}" DIRECTORY)    # .../MSVC
  get_filename_component(_msvc_dir "${_msvc_dir}" DIRECTORY)            # .../Tools
  get_filename_component(_msvc_dir "${_msvc_dir}" DIRECTORY)            # .../VC
  set(BOOST_CONFIGURE_COMMAND
      "${PROJECT_SOURCE_DIR}/cmake/prepare_boost.bat"
      "${BOOST_BASE}/source"
      "${CMAKE_CXX_COMPILER}"
      "${_msvc_dir}/Auxiliary/Build/vcvarsall.bat"
      "${BOOST_MSVC_VERSION}"
      "${_msvc_vcvars_version}")
  set(BOOST_ADDRESS_MODEL "address-model=64")
  unset(BOOST_CXXFLAGS_ARGUMENT)
else()
  set(BOOST_BOOTSTRAP "${BOOST_BASE}/source/bootstrap.sh")
  set(BOOST_BUILD "${BOOST_BASE}/source/b2")
  set(BOOST_CXXFLAGS_ARGUMENT "cxxflags=-fPIC")
  list(
    APPEND
    BOOST_BUILD_BYPRODUCTS
      ${BOOST_LIBRARY}/libboost_filesystem.a
      ${BOOST_LIBRARY}/libboost_chrono.a
  )
endif()
list(
  APPEND
  BOOST_SUBMODULES
    "libs/algorithm"
    "libs/align"
    "libs/any"
    "libs/array"
    "libs/assert"
    "libs/atomic"
    "libs/bind"
    "libs/chrono"
    "libs/compute"
    "libs/compat"
    "libs/concept_check"
    "libs/config"
    "libs/container"
    "libs/container_hash"
    "libs/core"
    "libs/date_time"
    "libs/describe"
    "libs/detail"
    "libs/filesystem"
    "libs/foreach"
    "libs/format"
    "libs/function"
    "libs/function_types"
    "libs/functional"
    "libs/fusion"
    "libs/headers"
    "libs/integer"
    "libs/io"
    "libs/iterator"
    "libs/lexical_cast"
    "libs/move"
    "libs/mp11"
    "libs/mpl"
    "libs/multi_index"
    "libs/numeric/conversion"
    "libs/optional"
    "libs/predef"
    "libs/preprocessor"
    "libs/property_tree"
    "libs/proto"
    "libs/range"
    "libs/ratio"
    "libs/serialization"
    "libs/scope"
    "libs/smart_ptr"
    "libs/static_assert"
    "libs/system"
    "libs/thread"
    "libs/throw_exception"
    "libs/tuple"
    "libs/typeof"
    "libs/type_index"
    "libs/type_traits"
    "libs/utility"
    "libs/uuid"
    "libs/variant2"
    "libs/winapi"
    "tools/boost_install"
    "tools/build"
)
# lint_cmake: -readability/wonkycase
ExternalProject_Add(
# lint_cmake: +readability/wonkycase
  Boost
  TMP_DIR "${BOOST_BASE}/tmp"
  STAMP_DIR "${BOOST_BASE}/stamp"
  DOWNLOAD_DIR "${BOOST_BASE}/download"
  SOURCE_DIR "${BOOST_BASE}/source"
  BINARY_DIR "${BOOST_BASE}/source"
  INSTALL_DIR "${BOOST_BASE}/install"
  GIT_REPOSITORY ${BOOST_REPOSITORY}
  GIT_TAG ${BOOST_TAG}
  GIT_SUBMODULES ${BOOST_SUBMODULES}
  GIT_SHALLOW ON
  GIT_CONFIG core.longpaths=true
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  CONFIGURE_COMMAND ${BOOST_CONFIGURE_COMMAND}
  BUILD_COMMAND
    ${BOOST_BUILD}
    -sBOOST_ROOT=${BOOST_BASE}/source
    --user-config=${BOOST_BASE}/source/user-config.jam
    -a
    -q
    -j ${J}
    --with-headers
    --with-chrono
    --with-filesystem
    link=static
    runtime-link=shared
    variant=release
    threading=multi
    ${BOOST_ADDRESS_MODEL}
    ${BOOST_CXXFLAGS_ARGUMENT}
  INSTALL_COMMAND ""
  # BUILD_BYPRODUCTS is necessary to support 'Ninja' builds.
  # ref:
  #  - https://cmake.org/cmake/help/latest/module/ExternalProject.html
  #  - https://stackoverflow.com/a/65803911/3986677
  BUILD_BYPRODUCTS ${BOOST_BUILD_BYPRODUCTS}
)
list(APPEND INTEGRATED_OPENCL_INCLUDES ${BOOST_INCLUDE})
list(APPEND INTEGRATED_OPENCL_LIBRARIES ${BOOST_BUILD_BYPRODUCTS})

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
