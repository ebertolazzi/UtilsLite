############################################################################
#                                                                          #
#  file: CMakeLists.txt                                                    #
#                                                                          #
############################################################################

# Require a CMake version recent enough for modern target usage,
# FetchContent, generator expressions and install/export rules used below.
cmake_minimum_required(VERSION 3.16)

# --------------------------------------------------------------------------
# CMake policy selection
# --------------------------------------------------------------------------
# Policies make behavior explicit across CMake releases.  The guarded checks
# keep the file compatible with older CMake versions that do not know newer
# policy identifiers.
#
# CMP0177 NEW: use the newer install path normalization behavior when running
# with CMake >= 3.31.
# CMP0135 NEW: FetchContent/external archives get predictable timestamps,
# avoiding stale rebuild decisions after downloads.
# CMP0169 OLD: keep compatibility with the existing FetchContent_Populate()
# usage in this project.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
  cmake_policy(SET CMP0177 NEW)
endif()
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

# --------------------------------------------------------------------------
# Apple FAT/universal build setup
# --------------------------------------------------------------------------
# On Apple platforms a single library can contain slices for several CPU
# architectures, for example arm64 and x86_64.  This is controlled by
# CMAKE_OSX_ARCHITECTURES.
#
# The variable must be set before project(), otherwise CMake may initialize
# compilers and targets before the requested architectures are known.
#
# Default behavior:
#   - ON  on Apple hosts or Apple cross-compilation targets.
#   - OFF on Linux/Windows and other platforms, where there is no equivalent
#     single-file universal library format.
#
# User overrides:
#   -DUTILS_BUILD_FAT=OFF
#   -DUTILS_FAT_ARCHITECTURES=arm64
#   -DCMAKE_OSX_ARCHITECTURES=arm64\;x86_64
set(UTILS_BUILD_FAT_DEFAULT OFF)
if(CMAKE_HOST_APPLE OR CMAKE_SYSTEM_NAME MATCHES "Darwin|iOS|tvOS|watchOS|visionOS")
  set(UTILS_BUILD_FAT_DEFAULT ON)
endif()
option(UTILS_BUILD_FAT "Build Apple universal/fat binaries when supported" ${UTILS_BUILD_FAT_DEFAULT})
set(UTILS_FAT_ARCHITECTURES "arm64;x86_64" CACHE STRING "Apple architectures to build when UTILS_BUILD_FAT is ON")

if(UTILS_BUILD_FAT AND NOT CMAKE_OSX_ARCHITECTURES)
  set(
    CMAKE_OSX_ARCHITECTURES
    "${UTILS_FAT_ARCHITECTURES}"
    CACHE STRING "Build architectures for Apple platforms"
    FORCE
  )
endif()

# --------------------------------------------------------------------------
# Project version discovery
# --------------------------------------------------------------------------
# Prefer the most recent Git tag as project version.  This keeps the CMake
# project version aligned with repository releases.  If Git is not available,
# or the source tree is not a Git checkout with tags, the version falls back
# to 0.0.0.
find_package(Git QUIET)
if(Git_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE UTILS_PROJECT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(UTILS_PROJECT_VERSION MATCHES "^v")
    string(SUBSTRING "${UTILS_PROJECT_VERSION}" 1 -1 UTILS_PROJECT_VERSION)
  endif()
endif()
if(NOT UTILS_PROJECT_VERSION)
  set(UTILS_PROJECT_VERSION "0.0.0")
endif()

# Declare the CMake project after all pre-project settings have been applied.
# The VERSION value is used later for target properties, installation exports
# and package metadata.
project(
  UtilsLite
  VERSION ${UTILS_PROJECT_VERSION}
  HOMEPAGE_URL "https://ebertolazzi.github.io/UtilsLite/"
  LANGUAGES CXX
)

# --------------------------------------------------------------------------
# Project-wide naming and source-root variables
# --------------------------------------------------------------------------
# UTILS_NAMESPACE is used for internal target names and exported aliases.
# SOURCE_DIR centralizes the source tree root used by source discovery, include
# directories and installation rules.
set(UTILS_NAMESPACE utils)
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# --------------------------------------------------------------------------
# Standard CMake modules and project helper modules
# --------------------------------------------------------------------------
# GNUInstallDirs defines portable install directory variables such as
# CMAKE_INSTALL_INCLUDEDIR and CMAKE_INSTALL_LIBDIR.
# CTest enables test registration through add_test().
# FetchContent is used to retrieve header-only/vendor dependencies when local
# checkouts are not available.
# Update3rdParties.cmake provides helper routines used to rewrite and vendor
# third-party headers into src/Utils/3rd.
include(GNUInstallDirs)
include(CTest)
include(FetchContent)
include(CheckIPOSupported)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Update3rdParties.cmake")

# --------------------------------------------------------------------------
# Public build options
# --------------------------------------------------------------------------
# These cache options are visible to users configuring the project with cmake.
# They replace options that were previously provided by cmake_utils.
#
# UTILS_BUILD_SHARED: also build a shared library in addition to the static one.
# UTILS_ENABLE_TESTS: build tests by default only for a top-level checkout.
# UTILS_INSTALL_LOCAL: enable install rules by default only for a top-level
# project, avoiding unexpected install rules when embedded as a dependency.
# UTILS_UPDATE_3RDPARTY: regenerate vendored third-party headers on request.
option(UTILS_BUILD_SHARED "Build the UtilsLite shared library" OFF)
option(UTILS_ENABLE_TESTS "Build UtilsLite tests" ${PROJECT_IS_TOP_LEVEL})
option(UTILS_INSTALL_LOCAL "Enable UtilsLite install rules" ${PROJECT_IS_TOP_LEVEL})
option(UTILS_UPDATE_3RDPARTY "Download and regenerate vendored 3rd-party headers in src/Utils/3rd" OFF)
option(UTILS_ALLOW_NETWORK_FETCH "Allow network downloads when refreshing third-party headers" OFF)

# --------------------------------------------------------------------------
# Check presence of vendored third-party headers
# --------------------------------------------------------------------------
# If the required files are missing and the user has not explicitly requested
# a refresh, stop with a clear error message.
set(UTILS_3RD_REQUIRED_FILES
  "${SOURCE_DIR}/Utils/3rd/Eigen/Core"
  "${SOURCE_DIR}/Utils/3rd/fmt/format.h"
  "${SOURCE_DIR}/Utils/3rd/CLI/CLI.hpp"
  "${SOURCE_DIR}/Utils/3rd/BS_thread_pool.hpp"
  "${SOURCE_DIR}/Utils/3rd/autodiff/forward/dual.hpp"
)
set(UTILS_3RD_MISSING FALSE)
foreach(f IN LISTS UTILS_3RD_REQUIRED_FILES)
  if(NOT EXISTS "${f}")
    set(UTILS_3RD_MISSING TRUE)
    message(STATUS "UtilsLite: missing vendored file: ${f}")
  endif()
endforeach()

if(UTILS_3RD_MISSING AND NOT UTILS_UPDATE_3RDPARTY)
  message(FATAL_ERROR
    "Vendored third-party headers in src/Utils/3rd are incomplete or missing.\n"
    "Please set -DUTILS_UPDATE_3RDPARTY=ON -DUTILS_ALLOW_NETWORK_FETCH=ON to regenerate them.\n"
    "If you are building from a source distribution, ensure the src/Utils/3rd directory is complete."
  )
endif()

# --------------------------------------------------------------------------
# Compilation and optimization options
# --------------------------------------------------------------------------
# Native CPU tuning is enabled for ordinary single-architecture builds, but
# disabled by default for Apple universal binaries: -march=native cannot
# describe both arm64 and x86_64 slices in the same artifact.
set(UTILS_APPLE_MULTI_ARCH FALSE)
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  list(LENGTH CMAKE_OSX_ARCHITECTURES UTILS_OSX_ARCHITECTURE_COUNT)
  if(UTILS_OSX_ARCHITECTURE_COUNT GREATER 1)
    set(UTILS_APPLE_MULTI_ARCH TRUE)
  endif()
endif()

set(UTILS_USE_NATIVE_ARCH_DEFAULT ON)
if(UTILS_APPLE_MULTI_ARCH)
  set(UTILS_USE_NATIVE_ARCH_DEFAULT OFF)
endif()

option(
  UTILS_USE_NATIVE_ARCH
  "Optimize Release builds for the current CPU architecture"
  ${UTILS_USE_NATIVE_ARCH_DEFAULT}
)
option(UTILS_ENABLE_LTO "Enable interprocedural optimization / LTO" ON)
option(UTILS_UNROLL_LOOPS "Enable explicit loop unrolling in Release builds" OFF)
option(
  UTILS_ENABLE_FAST_MATH
  "Enable aggressive floating-point optimizations; may change IEEE-754 semantics"
  OFF
)

# Probe IPO/LTO support once and reuse the result for every UtilsLite target.
set(UTILS_IPO_SUPPORTED FALSE)
set(UTILS_IPO_STATUS "disabled")
if(UTILS_ENABLE_LTO)
  check_ipo_supported(
    RESULT UTILS_IPO_SUPPORTED
    OUTPUT UTILS_IPO_ERROR
    LANGUAGES CXX
  )
  if(UTILS_IPO_SUPPORTED)
    set(UTILS_IPO_STATUS "enabled")
  else()
    set(UTILS_IPO_STATUS "not supported: ${UTILS_IPO_ERROR}")
  endif()
endif()

# --------------------------------------------------------------------------
# Third-party dependency versions
# --------------------------------------------------------------------------
# Each dependency can be provided by a local checkout or fetched from the
# configured upstream archive.  These cache variables let users pin or override
# the versions without editing this file.
set(UTILS_EIGEN_VERSION "5.0.1" CACHE STRING "Eigen version to fetch")
set(UTILS_CLI11_VERSION "2.6.2" CACHE STRING "CLI11 version to fetch")
set(UTILS_FMT_VERSION "12.2.0" CACHE STRING "fmt version to fetch")
set(UTILS_BS_THREAD_POOL_VERSION "5.0.0" CACHE STRING "BS::thread_pool version to fetch")
set(UTILS_AUTODIFF_VERSION "main" CACHE STRING "autodiff branch or tag to fetch")
if(UTILS_AUTODIFF_VERSION STREQUAL "1.1.2")
  set(UTILS_AUTODIFF_VERSION "main" CACHE STRING "autodiff branch or tag to fetch" FORCE)
endif()

# --------------------------------------------------------------------------
# Preferred local dependency locations
# --------------------------------------------------------------------------
# The build first looks for sibling checkouts next to this project.  This is
# useful for development because it avoids network access and allows local
# dependency modifications. A checkout is accepted only when its current Git
# branch matches the configured fetch version; otherwise FetchContent is used.
set(
  UTILS_EIGEN_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../eigen"
  CACHE PATH "Path to a local Eigen checkout to use before FetchContent"
)
set(
  UTILS_CLI11_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../CLI11"
  CACHE PATH "Path to a local CLI11 checkout to use before FetchContent"
)
set(
  UTILS_FMT_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../fmt"
  CACHE PATH "Path to a local fmt checkout to use before FetchContent"
)
set(
  UTILS_BS_THREAD_POOL_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../BS_thread_pool"
  CACHE PATH "Path to a local BS::thread_pool checkout to use before FetchContent"
)
# Backward-compatibility correction for an older local directory name.  If the
# cache still points to ../thread-pool but the newer ../BS_thread_pool checkout
# exists, prefer the newer path automatically.
if(
  UTILS_BS_THREAD_POOL_LOCAL_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/../thread-pool" AND
  EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../BS_thread_pool/include/BS_thread_pool.hpp"
)
  set(
    UTILS_BS_THREAD_POOL_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../BS_thread_pool"
    CACHE PATH "Path to a local BS::thread_pool checkout to use before FetchContent"
    FORCE
  )
endif()
set(
  UTILS_AUTODIFF_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../autodiff"
  CACHE PATH "Path to a local autodiff checkout to use before FetchContent"
)

# --------------------------------------------------------------------------
# Top-level install prefix
# --------------------------------------------------------------------------
# When this repository is configured directly, install into the source tree by
# default so headers/libraries go into the project-local lib/ directory.  When
# the project is included by another project, this block is skipped to avoid
# overriding the parent build's install prefix.
if(
  PROJECT_IS_TOP_LEVEL AND
  (
    CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR
    CMAKE_INSTALL_PREFIX STREQUAL "/usr/local"
  )
)
  set(
    CMAKE_INSTALL_PREFIX
    "${CMAKE_CURRENT_SOURCE_DIR}"
    CACHE PATH "Install path prefix, prepended onto install directories."
    FORCE
  )
endif()

# For top-level builds, use the historical local layout:
#   lib/include  installed headers
#   lib/lib      static/shared libraries
#   lib/bin      executables/tests
# The directories are created early so subsequent build and install steps have
# stable output locations.
if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_INSTALL_INCLUDEDIR "lib/include" CACHE PATH "Install include directory" FORCE)
  set(CMAKE_INSTALL_LIBDIR "lib/lib" CACHE PATH "Install library directory" FORCE)
  set(CMAKE_INSTALL_BINDIR "lib/bin" CACHE PATH "Install binary directory" FORCE)
  file(MAKE_DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/lib"
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/bin"
  )
endif()

# --------------------------------------------------------------------------
# Platform suffix used in generated library names
# --------------------------------------------------------------------------
# The real binary name includes an OS/toolchain suffix such as osx, linux,
# win64 or mingw64.  A compatibility alias without the suffix is created later.
# This preserves existing consumer expectations while keeping artifacts
# unambiguous by platform.
if(APPLE)
  set(UTILS_PLATFORM_SUFFIX osx)
elseif(WIN32)
  if(MINGW)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(UTILS_PLATFORM_SUFFIX mingw64)
    else()
      set(UTILS_PLATFORM_SUFFIX mingw32)
    endif()
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(UTILS_PLATFORM_SUFFIX win64)
  else()
    set(UTILS_PLATFORM_SUFFIX win32)
  endif()
elseif(UNIX)
  set(UTILS_PLATFORM_SUFFIX linux)
else()
  string(TOLOWER "${CMAKE_SYSTEM_NAME}" UTILS_PLATFORM_SUFFIX)
endif()

# --------------------------------------------------------------------------
# C++ language standard
# --------------------------------------------------------------------------
# Require ISO C++20 for all targets, and disable compiler-specific extensions
# so the code is built in a portable standard mode.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --------------------------------------------------------------------------
# Default build type for single-configuration generators
# --------------------------------------------------------------------------
# Makefiles and Ninja normally need CMAKE_BUILD_TYPE.  If the user did not set
# one, use Release.  Multi-configuration generators such as Xcode or Visual
# Studio use CMAKE_CONFIGURATION_TYPES instead and are not touched here.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# --------------------------------------------------------------------------
# Global warning adjustments for Clang-family compilers
# --------------------------------------------------------------------------
# These suppress warnings that are noisy for this codebase or for common
# third-party/header-only usage patterns.  They are only added for C++
# compilation units through generator expressions.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-Wno-poison-system-directories>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-default>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wno-exit-time-destructors>"
  )
endif()

# --------------------------------------------------------------------------
# Optional local customization hook
# --------------------------------------------------------------------------
# A sibling CMakeLists-customize.txt file can inject site-specific settings
# without modifying this tracked build script.  This is useful for local paths,
# toolchain choices or developer-specific overrides.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../CMakeLists-customize.txt")
  include("${CMAKE_CURRENT_LIST_DIR}/../CMakeLists-customize.txt")
endif()

# --------------------------------------------------------------------------
# Optional regeneration of vendored third-party headers
# --------------------------------------------------------------------------
# This is intentionally opt-in because it modifies files under src/Utils/3rd.
# It is enabled only for a top-level build, not when UtilsLite is embedded by
# another project.
if(UTILS_UPDATE_3RDPARTY AND NOT UTILS_ALLOW_NETWORK_FETCH)
  message(FATAL_ERROR
    "UTILS_UPDATE_3RDPARTY=ON requires UTILS_ALLOW_NETWORK_FETCH=ON. "
    "Normal builds use the committed src/Utils/3rd headers and never access the network.")
endif()

# --------------------------------------------------------------------------
# Resolve a dependency from a local checkout or FetchContent
# --------------------------------------------------------------------------
# Arguments:
#   dep_name   FetchContent identifier.
#   local_dir  Preferred local checkout path.
#   probe_rel  File/directory expected inside local_dir to prove it is valid.
#   expected_ref Git ref (branch, tag, or commit) that a local checkout must match.
#   url        Archive URL used by FetchContent if local_dir is not valid.
#   out_root   Output variable receiving the resolved dependency root.
#   out_source Output variable receiving a human-readable source description.
#
# The result variables are returned to the caller with PARENT_SCOPE.
function(utils_resolve_dependency dep_name local_dir probe_rel expected_ref url out_root out_source)
  # Normalize the local path relative to this source directory so callers may
  # pass relative paths consistently.
  get_filename_component(_local_abs "${local_dir}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  set(_local_matches FALSE)
  set(_local_desc "not available")
  if(EXISTS "${_local_abs}/${probe_rel}" AND GIT_EXECUTABLE)
    # Get current commit hash (works for both branches and detached HEAD)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${_local_abs}" rev-parse HEAD
      RESULT_VARIABLE _rev_result
      OUTPUT_VARIABLE _local_commit
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    # Get current branch name if on a branch, otherwise "detached"
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${_local_abs}" symbolic-ref --quiet --short HEAD
      RESULT_VARIABLE _branch_result
      OUTPUT_VARIABLE _local_branch
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_branch_result EQUAL 0)
      set(_local_desc "branch '${_local_branch}'")
    else()
      set(_local_desc "detached at ${_local_commit}")
    endif()

    # Check if local checkout matches expected_ref:
    # 1. If expected_ref looks like a tag (starts with v or is semantic version), verify commit matches tag
    # 2. Otherwise treat as branch name and compare branch
    if(_rev_result EQUAL 0)
      if(expected_ref MATCHES "^v?[0-9]+\\.[0-9]+\\.[0-9]+")
        # Expected ref is a version/tag - check if current commit matches that tag
        execute_process(
          COMMAND "${GIT_EXECUTABLE}" -C "${_local_abs}" rev-parse "refs/tags/${expected_ref}"
          RESULT_VARIABLE _tag_result
          OUTPUT_VARIABLE _tag_commit
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET
        )
        if(_tag_result EQUAL 0 AND _tag_commit STREQUAL _local_commit)
          set(_local_matches TRUE)
        endif()
      else()
        # Expected ref is a branch name
        if(_branch_result EQUAL 0 AND _local_branch STREQUAL expected_ref)
          set(_local_matches TRUE)
        endif()
      endif()
    endif()
  endif()

  if(_local_matches)
    message(STATUS "UtilsLite: ${dep_name} -> local ${_local_abs} (${_local_desc}, matches '${expected_ref}')")
    set(${out_root} "${_local_abs}" PARENT_SCOPE)
    set(${out_source} "local: ${_local_abs} [${_local_desc}]" PARENT_SCOPE)
    return()
  endif()

  if(EXISTS "${_local_abs}/${probe_rel}")
    message(STATUS "UtilsLite: ${dep_name} -> ignoring local ${_local_abs} (${_local_desc}, expected '${expected_ref}')")
  else()
    message(STATUS "UtilsLite: ${dep_name} -> local checkout not found at ${_local_abs}")
  endif()
  message(STATUS "UtilsLite: ${dep_name} -> FetchContent (requested '${expected_ref}')")

  # Local checkout not found or doesn't match: declare and populate the dependency from the
  # configured archive URL.  DOWNLOAD_EXTRACT_TIMESTAMP is aligned with CMP0135
  # so extracted files get deterministic timestamps.
  FetchContent_Declare(
    ${dep_name}
    URL "${url}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_GetProperties(${dep_name})
  if(NOT ${dep_name}_POPULATED)
    FetchContent_Populate(${dep_name})
  endif()

  # Return the FetchContent source directory and a description used in the
  # configuration summary printed at the end of this file.
  set(${out_root} "${${dep_name}_SOURCE_DIR}" PARENT_SCOPE)
  set(${out_source} "FetchContent: ${expected_ref} (${url})" PARENT_SCOPE)
endfunction()

# --------------------------------------------------------------------------
# Write a file only when its content actually changes
# --------------------------------------------------------------------------
# This avoids touching timestamps unnecessarily, which prevents avoidable
# rebuilds in incremental builds.
function(utils_write_file_if_different file_path content)
  if(EXISTS "${file_path}")
    file(READ "${file_path}" _existing)
  else()
    set(_existing "")
  endif()

  if(NOT _existing STREQUAL content)
    file(WRITE "${file_path}" "${content}")
  endif()
endfunction()

# --------------------------------------------------------------------------
# Synchronize resolved third-party headers into src/Utils/3rd
# --------------------------------------------------------------------------
function(utils_sync_resolved_3rdparty)
  set(_third_party_dir "${SOURCE_DIR}/Utils/3rd")
  set(_stage_dir "${CMAKE_CURRENT_BINARY_DIR}/_utils_3rd_stage")
  file(MAKE_DIRECTORY "${_third_party_dir}")
  file(MAKE_DIRECTORY "${_stage_dir}")

  # Eigen: copy the main Eigen include tree.
  message(STATUS "UtilsLite: copying Eigen headers from ${UTILS_EIGEN_INCLUDE_DIR}/Eigen to ${_third_party_dir}/Eigen")
  file(REMOVE_RECURSE "${_third_party_dir}/Eigen")
  # Use REAL_PATH to correctly compare paths (handles symlinks, relative vs absolute)
  file(REAL_PATH "${UTILS_EIGEN_INCLUDE_DIR}/Eigen" _eigen_src_real)
  file(REAL_PATH "${_third_party_dir}/Eigen" _eigen_dst_real)
  if(NOT _eigen_src_real STREQUAL _eigen_dst_real)
    file(COPY "${UTILS_EIGEN_INCLUDE_DIR}/Eigen" DESTINATION "${_third_party_dir}")
  endif()
  # Verifica che la copia sia avvenuta correttamente
  if(NOT EXISTS "${_third_party_dir}/Eigen/Core")
    message(FATAL_ERROR
      "Failed to copy Eigen headers: ${_third_party_dir}/Eigen/Core not found.\n"
      "Check that UTILS_EIGEN_INCLUDE_DIR points to a valid Eigen distribution.\n"
      "Current value: ${UTILS_EIGEN_INCLUDE_DIR}"
    )
  endif()

  # Eigen unsupported modules are optional; copy them only if the resolved
  # Eigen distribution provides the unsupported/ directory.
  if(EXISTS "${UTILS_EIGEN_INCLUDE_DIR}/unsupported")
    file(REMOVE_RECURSE "${_third_party_dir}/unsupported")
    file(COPY "${UTILS_EIGEN_INCLUDE_DIR}/unsupported" DESTINATION "${_third_party_dir}")
  endif()

  # CLI11: stage the headers first, rewrite prefixed includes, then replace
  # the vendored copy atomically.  CLI11.hpp is added as a compatibility entry
  # point that includes CLI/CLI.hpp.
  file(REMOVE_RECURSE "${_third_party_dir}/CLI")
  set(_cli_stage "${_stage_dir}/CLI")
  file(REMOVE_RECURSE "${_cli_stage}")
  file(COPY "${UTILS_CLI11_INCLUDE_DIR}/CLI/" DESTINATION "${_cli_stage}")
  _utils_rewrite_prefixed_includes_recursive("${_cli_stage}" "CLI")
  _utils_replace_dir("${_cli_stage}" "${_third_party_dir}/CLI")
  utils_write_file_if_different(
    "${_third_party_dir}/CLI11.hpp"
    "#pragma once\n#include \"CLI/CLI.hpp\"\n"
  )

  # fmt: copy the header tree.  The target later defines FMT_HEADER_ONLY, so
  # consumers do not need a separately compiled fmt library.
  file(REMOVE_RECURSE "${_third_party_dir}/fmt")
  file(COPY "${UTILS_FMT_INCLUDE_DIR}/fmt" DESTINATION "${_third_party_dir}")

  # BS::thread_pool: remove the previous legacy header name if present, then
  # copy the current upstream single-header implementation.
  file(REMOVE "${_third_party_dir}/task_thread_pool.hpp")

  file(COPY "${UTILS_BS_THREAD_POOL_ROOT}/include/BS_thread_pool.hpp" DESTINATION "${_third_party_dir}")

  # autodiff: stage the source tree and rewrite includes so vendored autodiff
  # headers use the vendored Eigen and relative autodiff paths.  The helper
  # functions come from cmake/Update3rdParties.cmake.
  set(_autodiff_stage "${_stage_dir}/autodiff")
  file(REMOVE_RECURSE "${_autodiff_stage}")
  file(COPY "${UTILS_AUTODIFF_ROOT}/autodiff/" DESTINATION "${_autodiff_stage}")
  file(REMOVE "${_autodiff_stage}/BUILD")
  _utils_rewrite_glob("${_autodiff_stage}/*/*"
    "#include <eigen3/Eigen/([^>]*)>" "#include \"../../../Eigen/\\1\""
    "#include <Eigen/([^>]*)>"        "#include \"../../../Eigen/\\1\""
    "#include <autodiff/([^>]*)>"     "#include \"../\\1\"")
  _utils_rewrite_glob("${_autodiff_stage}/*/*/*"
    "#include <eigen3/Eigen/([^>]*)>" "#include \"../../../Eigen/\\1\""
    "#include <Eigen/([^>]*)>"        "#include \"../../../Eigen/\\1\""
    "#include <autodiff/([^>]*)>"     "#include \"../../\\1\"")
  _utils_patch_autodiff("${_autodiff_stage}")
  _utils_replace_dir("${_autodiff_stage}" "${_third_party_dir}/autodiff")
endfunction()

# --------------------------------------------------------------------------
# Resolve concrete dependency roots
# --------------------------------------------------------------------------
# Each block asks utils_resolve_dependency() to either use a local checkout or
# fetch the requested archive.  The *_ROOT variables are then converted to
# include roots as needed by the synchronization step and target include paths.
if(UTILS_UPDATE_3RDPARTY)
message(STATUS "UtilsLite: refreshing vendored dependencies (network fallback explicitly enabled)")
utils_resolve_dependency(
  utils_eigen
  "${UTILS_EIGEN_LOCAL_DIR}"
  "Eigen/Core"
  "${UTILS_EIGEN_VERSION}"
  "https://gitlab.com/libeigen/eigen/-/archive/${UTILS_EIGEN_VERSION}/eigen-${UTILS_EIGEN_VERSION}.tar.gz"
  UTILS_EIGEN_ROOT
  UTILS_EIGEN_SOURCE
)
set(UTILS_EIGEN_INCLUDE_DIR "${UTILS_EIGEN_ROOT}")

utils_resolve_dependency(
  utils_cli11
  "${UTILS_CLI11_LOCAL_DIR}"
  "include/CLI/CLI.hpp"
  "v${UTILS_CLI11_VERSION}"
  "https://github.com/CLIUtils/CLI11/archive/refs/tags/v${UTILS_CLI11_VERSION}.tar.gz"
  UTILS_CLI11_ROOT
  UTILS_CLI11_SOURCE
)
set(UTILS_CLI11_INCLUDE_DIR "${UTILS_CLI11_ROOT}/include")

utils_resolve_dependency(
  utils_fmt
  "${UTILS_FMT_LOCAL_DIR}"
  "include/fmt/format.h"
  "${UTILS_FMT_VERSION}"
  "https://github.com/fmtlib/fmt/archive/refs/tags/${UTILS_FMT_VERSION}.tar.gz"
  UTILS_FMT_ROOT
  UTILS_FMT_SOURCE
)
set(UTILS_FMT_INCLUDE_DIR "${UTILS_FMT_ROOT}/include")

utils_resolve_dependency(
  utils_bs_thread_pool
  "${UTILS_BS_THREAD_POOL_LOCAL_DIR}"
  "include/BS_thread_pool.hpp"
  "v${UTILS_BS_THREAD_POOL_VERSION}"
  "https://github.com/bshoshany/thread-pool/archive/refs/tags/v${UTILS_BS_THREAD_POOL_VERSION}.tar.gz"
  UTILS_BS_THREAD_POOL_ROOT
  UTILS_BS_THREAD_POOL_SOURCE
)

utils_resolve_dependency(
  utils_autodiff
  "${UTILS_AUTODIFF_LOCAL_DIR}"
  "autodiff/forward/dual.hpp"
  "${UTILS_AUTODIFF_VERSION}"
  "https://github.com/ebertolazzi/autodiff/archive/refs/heads/${UTILS_AUTODIFF_VERSION}.tar.gz"
  UTILS_AUTODIFF_ROOT
  UTILS_AUTODIFF_SOURCE
)

# After all dependency roots are known, refresh the vendored header tree used
# by the build and install rules.
utils_sync_resolved_3rdparty()
else()
  set(UTILS_VENDORED_ROOT "${SOURCE_DIR}/Utils/3rd")
  set(UTILS_EIGEN_ROOT "${UTILS_VENDORED_ROOT}")
  set(UTILS_EIGEN_INCLUDE_DIR "${UTILS_VENDORED_ROOT}")
  set(UTILS_CLI11_ROOT "${UTILS_VENDORED_ROOT}")
  set(UTILS_CLI11_INCLUDE_DIR "${UTILS_VENDORED_ROOT}")
  set(UTILS_FMT_ROOT "${UTILS_VENDORED_ROOT}")
  set(UTILS_FMT_INCLUDE_DIR "${UTILS_VENDORED_ROOT}")
  set(UTILS_BS_THREAD_POOL_ROOT "${UTILS_VENDORED_ROOT}")
  set(UTILS_AUTODIFF_ROOT "${UTILS_VENDORED_ROOT}")
  set(UTILS_EIGEN_SOURCE "vendored: src/Utils/3rd/Eigen")
  set(UTILS_CLI11_SOURCE "vendored: src/Utils/3rd/CLI")
  set(UTILS_FMT_SOURCE "vendored: src/Utils/3rd/fmt")
  set(UTILS_BS_THREAD_POOL_SOURCE "vendored: src/Utils/3rd/BS_thread_pool.hpp")
  set(UTILS_AUTODIFF_SOURCE "vendored: src/Utils/3rd/autodiff")
  message(STATUS "UtilsLite: using committed third-party headers; network access disabled")
endif()

# --------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------
# Only .cc files are compiled into the library.  CONFIGURE_DEPENDS tells CMake
# to re-run configuration when matching source files are added or removed.
file(
  GLOB_RECURSE SOURCES
  CONFIGURE_DEPENDS
  "${SOURCE_DIR}/*.cc"
)

# Discover project headers for IDE visibility and installation.  Vendored
# third-party headers under Utils/3rd are excluded here because they are
# installed by a dedicated rule later.
file(
  GLOB_RECURSE PROJECT_HEADERS_FULL
  CONFIGURE_DEPENDS
  "${SOURCE_DIR}/*.h"
  "${SOURCE_DIR}/*.hh"
  "${SOURCE_DIR}/*.hpp"
  "${SOURCE_DIR}/*.hxx"
  "${SOURCE_DIR}/Utils_nonlinear_system/*.cxx"
)
list(FILTER PROJECT_HEADERS_FULL EXCLUDE REGEX "/Utils/3rd/")

# Mark headers as non-compilable source entries.  This lets IDE generators show
# them in the target without trying to compile them as translation units.
foreach(H IN LISTS PROJECT_HEADERS_FULL)
  set_source_files_properties("${H}" PROPERTIES HEADER_FILE_ONLY TRUE)
endforeach()

# Build a private include-directory list used for compilation.  Public include
# directories are set in utils_configure_target(); this expanded list gives
# implementation files direct access to nested Utils subdirectories.
set(INCLUDE_DIRS
  "${SOURCE_DIR}/Utils/3rd"
  "${SOURCE_DIR}"
  "${SOURCE_DIR}/Utils"
)

file(GLOB_RECURSE UTILS_SUBDIRS CONFIGURE_DEPENDS "${SOURCE_DIR}/Utils/*/")
foreach(DIR IN LISTS UTILS_SUBDIRS)
  if(IS_DIRECTORY "${DIR}")
    # Exclude vendored third-party subdirectories to avoid polluting include paths
    # (e.g. Utils/3rd/Eigen, Utils/3rd/autodiff would allow #include <Core> instead of #include <Eigen/Core>)
    string(FIND "${DIR}" "/Utils/3rd/" _is_3rd)
    if(_is_3rd EQUAL -1)
      list(APPEND INCLUDE_DIRS "${DIR}")
    endif()
  endif()
endforeach()
list(REMOVE_DUPLICATES INCLUDE_DIRS)

# --------------------------------------------------------------------------
# Common target configuration
# --------------------------------------------------------------------------
# Apply the same compile features, include directories, definitions, warnings,
# Apple universal settings and output directories to each library target.
function(utils_configure_target target_name)
  # Publish the required C++ standard to consumers linking this target.
  target_compile_features(${target_name} PUBLIC cxx_std_20)
  # Public include paths distinguish between build-tree and install-tree usage.
  # Private include paths expose implementation-only subdirectories while
  # compiling this target.
  target_include_directories(
    ${target_name}
    PUBLIC
      "$<BUILD_INTERFACE:${SOURCE_DIR}/Utils/3rd>"
      "$<BUILD_INTERFACE:${SOURCE_DIR}>"
      "$<BUILD_INTERFACE:${SOURCE_DIR}/Utils>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
      ${INCLUDE_DIRS}
  )
  # Build fmt as header-only.  This avoids a separate fmt link dependency and
  # matches the vendored-header approach used by this project.
  target_compile_definitions(
    ${target_name}
    PUBLIC
      FMT_HEADER_ONLY
  )

  # Diagnostics and conformance warnings.  Keep them target-local so UtilsLite
  # does not inject warning flags into parent projects when used as a dependency.
  target_compile_options(
    ${target_name}
    PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wextra>
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wpedantic>
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wshadow>
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wnon-virtual-dtor>
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Woverloaded-virtual>
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wcast-align>
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wunused>
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wformat=2>
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wimplicit-fallthrough>
      $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-cond>
      $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-branches>
      $<$<CXX_COMPILER_ID:GNU>:-Wlogical-op>
      $<$<CXX_COMPILER_ID:GNU>:-Wnull-dereference>
      $<$<CXX_COMPILER_ID:GNU>:-Wmisleading-indentation>
      $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-poison-system-directories>
      $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-switch-default>
      $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-exit-time-destructors>
      $<$<CXX_COMPILER_ID:MSVC>:/W4>
      $<$<CXX_COMPILER_ID:MSVC>:/permissive->
      $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
  )

  # Explicit optimization policy.  These generator expressions work with both
  # single-configuration generators (Ninja/Makefiles) and multi-config IDEs.
  target_compile_options(
    ${target_name}
    PRIVATE
      $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O3>
      $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2>
      $<$<AND:$<CONFIG:RelWithDebInfo>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O2>
      $<$<AND:$<CONFIG:RelWithDebInfo>,$<CXX_COMPILER_ID:MSVC>>:/O2>
  )

  target_compile_definitions(
    ${target_name}
    PRIVATE
      $<$<CONFIG:Release>:NDEBUG>
      $<$<CONFIG:Release>:EIGEN_NO_DEBUG>
  )

  if(UTILS_USE_NATIVE_ARCH)
    if(UTILS_APPLE_MULTI_ARCH)
      message(STATUS
        "Skipping native CPU flags for ${target_name}: Apple universal build uses "
        "CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
      )
    else()
      target_compile_options(
        ${target_name}
        PRIVATE
          $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-march=native>
          $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-mtune=native>
          $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:IntelLLVM>>:-xHost>
          $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/arch:AVX2>
      )
    endif()
  endif()

  if(UTILS_UNROLL_LOOPS)
    target_compile_options(
      ${target_name}
      PRIVATE
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-funroll-loops>
    )
  endif()

  # Deliberately OFF by default because it permits reassociation and other
  # transformations that weaken strict IEEE-754 behavior.
  if(UTILS_ENABLE_FAST_MATH)
    target_compile_options(
      ${target_name}
      PRIVATE
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-ffast-math>
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/fp:fast>
    )
  endif()

  if(UTILS_IPO_SUPPORTED)
    set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
  endif()

  # For Apple universal builds, explicitly attach the architecture list to
  # the target.  CMAKE_OSX_ARCHITECTURES was set before project(), but setting
  # OSX_ARCHITECTURES here makes the intended target property unambiguous.
  if(APPLE AND UTILS_BUILD_FAT)
    set_target_properties(
      ${target_name}
      PROPERTIES
        OSX_ARCHITECTURES "${CMAKE_OSX_ARCHITECTURES}"
    )
  endif()

  # Place generated artifacts in the historical project-local directories and
  # expose the source root to Xcode's header search path setting.
  set_target_properties(
    ${target_name}
    PROPERTIES
      VERSION ${PROJECT_VERSION}
      XCODE_ATTRIBUTE_HEADER_SEARCH_PATHS "${SOURCE_DIR}"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib/lib"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib/bin"
  )
endfunction()

# --------------------------------------------------------------------------
# Platform-specific library naming and compatibility alias
# --------------------------------------------------------------------------
# The real output file includes a platform suffix.  A second unsuffixed file is
# created as a compatibility alias for older consumers that expected names such
# as libUtilsLite.a.
function(utils_add_platform_library_alias target_name base_name library_kind)
  # The real library keeps the platform/static suffix, e.g.
  #   libUtilsLite_osx_static.a
  # and the unsuffixed name is kept only as a compatibility alias, e.g.
  #   libUtilsLite.a -> libUtilsLite_osx_static.a
  # Compute output and alias names.  Static libraries get an extra _static
  # marker; shared libraries do not.  Windows uses a lib prefix here to keep
  # naming compatible with the previous layout.
  if(library_kind STREQUAL "STATIC")
    set(output_name "${base_name}_${UTILS_PLATFORM_SUFFIX}_static")
    set(alias_filename "${CMAKE_STATIC_LIBRARY_PREFIX}${base_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    if(WIN32)
      set(real_filename "lib${output_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(output_name "lib${output_name}")
    else()
      set(real_filename "${CMAKE_STATIC_LIBRARY_PREFIX}${output_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endif()
  else()
    set(output_name "${base_name}_${UTILS_PLATFORM_SUFFIX}")
    set(alias_filename "${CMAKE_SHARED_LIBRARY_PREFIX}${base_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(WIN32)
      set(real_filename "lib${output_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
      set(output_name "lib${output_name}")
    else()
      set(real_filename "${CMAKE_SHARED_LIBRARY_PREFIX}${output_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()
  endif()

  # Apply the computed real output name to the CMake target.
  set_target_properties(${target_name} PROPERTIES OUTPUT_NAME "${output_name}")

  # Windows cannot rely on POSIX symlinks in the same way, so the alias is
  # materialized as a copied file.
  if(WIN32)
    add_custom_command(
      TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "$<TARGET_FILE:${target_name}>"
              "$<TARGET_FILE_DIR:${target_name}>/${alias_filename}"
      VERBATIM
    )

    if(UTILS_INSTALL_LOCAL)
      string(CONCAT _alias_install_code
        "file(MAKE_DIRECTORY \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")\n"
        "execute_process(\n"
        "  COMMAND \"${CMAKE_COMMAND}\" -E copy_if_different\n"
        "          \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${real_filename}\"\n"
        "          \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${alias_filename}\"\n"
        ")\n"
      )
      install(CODE "${_alias_install_code}")
    endif()
  else()
    # On Unix-like platforms, create a lightweight symlink alias next to the
    # real library after the target has been built.
    add_custom_command(
      TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E rm -f
              "$<TARGET_FILE_DIR:${target_name}>/${alias_filename}"
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              "$<TARGET_FILE_NAME:${target_name}>"
              "$<TARGET_FILE_DIR:${target_name}>/${alias_filename}"
      VERBATIM
    )

    if(UTILS_INSTALL_LOCAL)
      string(CONCAT _alias_install_code
        "file(MAKE_DIRECTORY \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")\n"
        "execute_process(\n"
        "  COMMAND \"${CMAKE_COMMAND}\" -E rm -f\n"
        "          \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${alias_filename}\"\n"
        ")\n"
        "execute_process(\n"
        "  COMMAND \"${CMAKE_COMMAND}\" -E create_symlink\n"
        "          \"${real_filename}\"\n"
        "          \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${alias_filename}\"\n"
        ")\n"
      )
      install(CODE "${_alias_install_code}")
    endif()
  endif()
endfunction()

# --------------------------------------------------------------------------
# Static library target
# --------------------------------------------------------------------------
# This is always built.  The target name and alias preserve compatibility with
# the previous cmake_utils-based naming scheme.
add_library(${UTILS_NAMESPACE}_${PROJECT_NAME}_Static STATIC ${SOURCES} ${PROJECT_HEADERS_FULL})
add_library(${UTILS_NAMESPACE}::${PROJECT_NAME}_Static ALIAS ${UTILS_NAMESPACE}_${PROJECT_NAME}_Static)
utils_configure_target(${UTILS_NAMESPACE}_${PROJECT_NAME}_Static)
utils_add_platform_library_alias(${UTILS_NAMESPACE}_${PROJECT_NAME}_Static ${PROJECT_NAME} STATIC)

# --------------------------------------------------------------------------
# Optional shared library target
# --------------------------------------------------------------------------
# Built only when UTILS_BUILD_SHARED is ON.  It uses the same source list and
# common target configuration as the static library.
if(UTILS_BUILD_SHARED)
  add_library(${UTILS_NAMESPACE}_${PROJECT_NAME} SHARED ${SOURCES} ${PROJECT_HEADERS_FULL})
  add_library(${UTILS_NAMESPACE}::${PROJECT_NAME} ALIAS ${UTILS_NAMESPACE}_${PROJECT_NAME})
  utils_configure_target(${UTILS_NAMESPACE}_${PROJECT_NAME})
  utils_add_platform_library_alias(${UTILS_NAMESPACE}_${PROJECT_NAME} ${PROJECT_NAME} SHARED)
endif()

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
# Test executables are built from src_tests/*.cc when UTILS_ENABLE_TESTS is ON.
# The tests link against the static UtilsLite library plus platform-specific
# system libraries when needed.
if(UTILS_ENABLE_TESTS)
  enable_testing()

  # Base test link libraries.  CMAKE_DL_LIBS adds the platform dynamic-loader
  # library when one is required.
  set(LIBS ${UTILS_NAMESPACE}_${PROJECT_NAME}_Static ${CMAKE_DL_LIBS})

  # Add operating-system-specific libraries used by the tests or by lower-level
  # utility code.
  if(UNIX AND NOT APPLE)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    list(APPEND LIBS Threads::Threads)
  elseif(WIN32)
    list(APPEND LIBS ws2_32 iphlpapi kernel32 pdh psapi)
  endif()

  # Discover test sources.  CONFIGURE_DEPENDS keeps the build system updated
  # when test files are added or removed.
  file(
    GLOB_RECURSE TEST_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src_tests/*.cc"
  )

  # If at least one test exists, create an aggregate target built by default.
  if(TEST_SOURCES)
    add_custom_target("${PROJECT_NAME}_all_tests" ALL)
  endif()

  # Each .cc file becomes one executable.  The executable name is derived from
  # the filename without extension and is also used as the CTest test name.
  foreach(TEST_SOURCE IN LISTS TEST_SOURCES)
    get_filename_component(TEST_NAME "${TEST_SOURCE}" NAME_WE)
    add_executable(${TEST_NAME} "${TEST_SOURCE}")
    set_target_properties(
      ${TEST_NAME}
      PROPERTIES
        SUFFIX ".exe"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
        XCODE_ATTRIBUTE_HEADER_SEARCH_PATHS "${SOURCE_DIR}"
    )
    # Tests get the same Apple universal architecture setting as the library so
    # they can link correctly against universal artifacts.
    if(APPLE AND UTILS_BUILD_FAT)
      set_target_properties(
        ${TEST_NAME}
        PROPERTIES
          OSX_ARCHITECTURES "${CMAKE_OSX_ARCHITECTURES}"
      )
    endif()
    target_link_libraries(${TEST_NAME} PRIVATE ${LIBS})
    target_include_directories(${TEST_NAME} PRIVATE ${INCLUDE_DIRS})
    add_test(NAME ${TEST_NAME} COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/bin/${TEST_NAME}.exe")
    add_dependencies("${PROJECT_NAME}_all_tests" ${TEST_NAME})
  endforeach()
endif()

# --------------------------------------------------------------------------
# Installation rules
# --------------------------------------------------------------------------
# When enabled, install library targets, project headers, vendored third-party
# headers and an export file usable by downstream CMake projects.
if(UTILS_INSTALL_LOCAL)
  # Start with the always-built static library; append the shared library only
  # when it exists.
  set(_utils_install_targets ${UTILS_NAMESPACE}_${PROJECT_NAME}_Static)
  if(UTILS_BUILD_SHARED)
    list(APPEND _utils_install_targets ${UTILS_NAMESPACE}_${PROJECT_NAME})
  endif()

  # Install compiled targets and generate export metadata for downstream
  # find_package/import use.
  install(
    TARGETS ${_utils_install_targets}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  # Install project headers while excluding vendored third-party files, which
  # are handled by the next install rule to preserve their Utils/3rd location.
  install(
    DIRECTORY "${SOURCE_DIR}/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.hh"
      PATTERN "*.hpp"
      PATTERN "*.hxx"
      PATTERN "*.cxx"
      PATTERN "Utils/3rd/*" EXCLUDE
      PATTERN ".DS_Store" EXCLUDE
      PATTERN ".gitkeep" EXCLUDE
      PATTERN "*.swp" EXCLUDE
      PATTERN "*.swo" EXCLUDE
  )

  # Install the vendored third-party headers in their dedicated subdirectory so
  # installed project headers keep valid include paths.
  install(
    DIRECTORY "${SOURCE_DIR}/Utils/3rd/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Utils/3rd"
    PATTERN ".DS_Store" EXCLUDE
    PATTERN ".gitkeep" EXCLUDE
    PATTERN "*.swp" EXCLUDE
    PATTERN "*.swo" EXCLUDE
  )

  # Export CMake target definitions under the utils:: namespace for consumers
  # that include the generated package files from the install tree.
  install(
    EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${UTILS_NAMESPACE}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
endif()

# --------------------------------------------------------------------------
# CPack packaging metadata
# --------------------------------------------------------------------------
# Configure simple source/binary package generation in TGZ and ZIP formats.
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "UtilsLite C++ utility library")
set(CPACK_GENERATOR "TGZ;ZIP")
include(CPack)

# --------------------------------------------------------------------------
# Configuration summary
# --------------------------------------------------------------------------
# Print the most important configuration decisions at the end of configure so
# users can immediately verify compiler, platform, architecture, install paths
# and dependency origins.
message(STATUS "")
message(STATUS "${PROJECT_NAME} configuration summary:")
message(STATUS "  Version          : ${PROJECT_VERSION}")
message(STATUS "  Compiler         : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Platform suffix  : ${UTILS_PLATFORM_SUFFIX}")
if(APPLE)
  message(STATUS "  Build fat/universal: ${UTILS_BUILD_FAT}")
  if(UTILS_BUILD_FAT)
    message(STATUS "  Apple archs      : ${CMAKE_OSX_ARCHITECTURES}")
  endif()
endif()
message(STATUS "  Build shared     : ${UTILS_BUILD_SHARED}")
message(STATUS "  Native arch      : ${UTILS_USE_NATIVE_ARCH}")
message(STATUS "  Apple multi-arch : ${UTILS_APPLE_MULTI_ARCH}")
message(STATUS "  LTO requested    : ${UTILS_ENABLE_LTO}")
message(STATUS "  LTO status       : ${UTILS_IPO_STATUS}")
message(STATUS "  Loop unrolling   : ${UTILS_UNROLL_LOOPS}")
message(STATUS "  Fast math        : ${UTILS_ENABLE_FAST_MATH}")
message(STATUS "  Build tests      : ${UTILS_ENABLE_TESTS}")
message(STATUS "  Install rules    : ${UTILS_INSTALL_LOCAL}")
message(STATUS "  Headers install  : ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "  Libraries install: ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "  3rd-party include: ${CMAKE_INSTALL_INCLUDEDIR}/Utils/3rd")
message(STATUS "  Eigen            : ${UTILS_EIGEN_SOURCE}")
message(STATUS "  CLI11            : ${UTILS_CLI11_SOURCE}")
message(STATUS "  fmt              : ${UTILS_FMT_SOURCE}")
message(STATUS "  BS thread pool   : ${UTILS_BS_THREAD_POOL_SOURCE}")
message(STATUS "  autodiff         : ${UTILS_AUTODIFF_SOURCE}")

set(UTILS_EFFECTIVE_OPTIMIZATION_FLAGS)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  list(APPEND UTILS_EFFECTIVE_OPTIMIZATION_FLAGS "Release:-O3" "RelWithDebInfo:-O2")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  list(APPEND UTILS_EFFECTIVE_OPTIMIZATION_FLAGS "Release:/O2" "RelWithDebInfo:/O2")
endif()
if(UTILS_USE_NATIVE_ARCH)
  if(UTILS_APPLE_MULTI_ARCH)
    list(APPEND UTILS_EFFECTIVE_OPTIMIZATION_FLAGS "native-arch:disabled-for-Apple-universal-build")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    list(APPEND UTILS_EFFECTIVE_OPTIMIZATION_FLAGS "Release:-march=native" "Release:-mtune=native")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    list(APPEND UTILS_EFFECTIVE_OPTIMIZATION_FLAGS "Release:-xHost")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    list(APPEND UTILS_EFFECTIVE_OPTIMIZATION_FLAGS "Release:/arch:AVX2")
  endif()
endif()
if(UTILS_UNROLL_LOOPS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  list(APPEND UTILS_EFFECTIVE_OPTIMIZATION_FLAGS "Release:-funroll-loops")
endif()
if(UTILS_ENABLE_FAST_MATH)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    list(APPEND UTILS_EFFECTIVE_OPTIMIZATION_FLAGS "Release:-ffast-math")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    list(APPEND UTILS_EFFECTIVE_OPTIMIZATION_FLAGS "Release:/fp:fast")
  endif()
endif()
if(UTILS_IPO_SUPPORTED)
  list(APPEND UTILS_EFFECTIVE_OPTIMIZATION_FLAGS "Release/RelWithDebInfo:IPO/LTO")
endif()
list(APPEND UTILS_EFFECTIVE_OPTIMIZATION_FLAGS "Release:NDEBUG" "Release:EIGEN_NO_DEBUG")
list(JOIN UTILS_EFFECTIVE_OPTIMIZATION_FLAGS " " UTILS_EFFECTIVE_OPTIMIZATION_FLAGS_TEXT)
message(STATUS "  Optimization flags: ${UTILS_EFFECTIVE_OPTIMIZATION_FLAGS_TEXT}")
message(STATUS "")
