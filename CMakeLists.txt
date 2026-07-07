############################################################################
#                                                                          #
#  file: CMakeLists.txt                                                    #
#                                                                          #
############################################################################

cmake_minimum_required(VERSION 3.16)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
  cmake_policy(SET CMP0177 NEW)
endif()
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

# Version from the latest Git tag when available.  This replaces the
# UTILS_PROJECT_VERSION value that used to come from cmake_utils.
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

project(
  UtilsLite
  VERSION ${UTILS_PROJECT_VERSION}
  HOMEPAGE_URL "https://ebertolazzi.github.io/UtilsLite/"
  LANGUAGES CXX
)

include(GNUInstallDirs)
include(CTest)
include(FetchContent)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Update3rdParties.cmake")

# Options formerly provided by cmake_utils.
option(UTILS_BUILD_SHARED "Build the UtilsLite shared library" OFF)
option(UTILS_ENABLE_TESTS "Build UtilsLite tests" ${PROJECT_IS_TOP_LEVEL})
option(UTILS_INSTALL_LOCAL "Enable UtilsLite install rules" ${PROJECT_IS_TOP_LEVEL})
option(UTILS_UPDATE_3RDPARTY "Download and regenerate vendored 3rd-party headers in src/Utils/3rd" OFF)

set(UTILS_EIGEN_VERSION "5.0.0" CACHE STRING "Eigen version to fetch")
set(UTILS_CLI11_VERSION "2.6.2" CACHE STRING "CLI11 version to fetch")
set(UTILS_FMT_VERSION "12.2.0" CACHE STRING "fmt version to fetch")
set(UTILS_SPDLOG_VERSION "1.16.0" CACHE STRING "spdlog version to fetch")
set(UTILS_BS_THREAD_POOL_VERSION "5.0.0" CACHE STRING "BS::thread_pool version to fetch")
set(UTILS_AUTODIFF_VERSION "main" CACHE STRING "autodiff branch or tag to fetch")
if(UTILS_AUTODIFF_VERSION STREQUAL "1.1.2")
  set(UTILS_AUTODIFF_VERSION "main" CACHE STRING "autodiff branch or tag to fetch" FORCE)
endif()

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
  UTILS_SPDLOG_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../spdlog"
  CACHE PATH "Path to a local spdlog checkout to use before FetchContent"
)
set(
  UTILS_BS_THREAD_POOL_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../BS_thread_pool"
  CACHE PATH "Path to a local BS::thread_pool checkout to use before FetchContent"
)
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

set(UTILS_NAMESPACE utils)
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-Wno-poison-system-directories>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-default>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wno-exit-time-destructors>"
  )
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../CMakeLists-customize.txt")
  include("${CMAKE_CURRENT_LIST_DIR}/../CMakeLists-customize.txt")
endif()

if(UTILS_UPDATE_3RDPARTY AND PROJECT_IS_TOP_LEVEL)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Update3rdParties.cmake")
  utils_update_3rdparties()
endif()

function(utils_resolve_dependency dep_name local_dir probe_rel url out_root out_source)
  get_filename_component(_local_abs "${local_dir}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(EXISTS "${_local_abs}/${probe_rel}")
    set(${out_root} "${_local_abs}" PARENT_SCOPE)
    set(${out_source} "local: ${_local_abs}" PARENT_SCOPE)
    return()
  endif()

  FetchContent_Declare(
    ${dep_name}
    URL "${url}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_GetProperties(${dep_name})
  if(NOT ${dep_name}_POPULATED)
    FetchContent_Populate(${dep_name})
  endif()

  set(${out_root} "${${dep_name}_SOURCE_DIR}" PARENT_SCOPE)
  set(${out_source} "FetchContent: ${url}" PARENT_SCOPE)
endfunction()

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

function(utils_sync_resolved_3rdparty)
  set(_third_party_dir "${SOURCE_DIR}/Utils/3rd")
  set(_stage_dir "${CMAKE_CURRENT_BINARY_DIR}/_utils_3rd_stage")
  file(MAKE_DIRECTORY "${_third_party_dir}")
  file(MAKE_DIRECTORY "${_stage_dir}")

  file(REMOVE_RECURSE "${_third_party_dir}/Eigen")
  file(COPY "${UTILS_EIGEN_INCLUDE_DIR}/Eigen" DESTINATION "${_third_party_dir}")

  if(EXISTS "${UTILS_EIGEN_INCLUDE_DIR}/unsupported")
    file(REMOVE_RECURSE "${_third_party_dir}/unsupported")
    file(COPY "${UTILS_EIGEN_INCLUDE_DIR}/unsupported" DESTINATION "${_third_party_dir}")
  endif()

  file(REMOVE_RECURSE "${_third_party_dir}/CLI")
  file(COPY "${UTILS_CLI11_INCLUDE_DIR}/CLI" DESTINATION "${_third_party_dir}")
  utils_write_file_if_different(
    "${_third_party_dir}/CLI11.hpp"
    "#pragma once\n#include <CLI/CLI.hpp>\n"
  )

  file(REMOVE_RECURSE "${_third_party_dir}/fmt")
  file(COPY "${UTILS_FMT_INCLUDE_DIR}/fmt" DESTINATION "${_third_party_dir}")

  file(REMOVE_RECURSE "${_third_party_dir}/spdlog")
  file(COPY "${UTILS_SPDLOG_INCLUDE_DIR}/spdlog" DESTINATION "${_third_party_dir}")

  set(_tweakme "${_third_party_dir}/spdlog/tweakme.h")
  if(EXISTS "${_tweakme}")
    file(READ "${_tweakme}" _tweakme_contents)
    string(REPLACE "// #define SPDLOG_FMT_EXTERNAL" "#define SPDLOG_FMT_EXTERNAL" _tweakme_contents "${_tweakme_contents}")
    utils_write_file_if_different("${_tweakme}" "${_tweakme_contents}")
  endif()

  file(REMOVE_RECURSE "${_third_party_dir}/spdlog/fmt/bundled")
  file(REMOVE "${_third_party_dir}/task_thread_pool.hpp")

  file(COPY "${UTILS_BS_THREAD_POOL_ROOT}/include/BS_thread_pool.hpp" DESTINATION "${_third_party_dir}")

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

utils_resolve_dependency(
  utils_eigen
  "${UTILS_EIGEN_LOCAL_DIR}"
  "Eigen/Core"
  "https://gitlab.com/libeigen/eigen/-/archive/${UTILS_EIGEN_VERSION}/eigen-${UTILS_EIGEN_VERSION}.tar.gz"
  UTILS_EIGEN_ROOT
  UTILS_EIGEN_SOURCE
)
set(UTILS_EIGEN_INCLUDE_DIR "${UTILS_EIGEN_ROOT}")

utils_resolve_dependency(
  utils_cli11
  "${UTILS_CLI11_LOCAL_DIR}"
  "include/CLI/CLI.hpp"
  "https://github.com/CLIUtils/CLI11/archive/refs/tags/v${UTILS_CLI11_VERSION}.tar.gz"
  UTILS_CLI11_ROOT
  UTILS_CLI11_SOURCE
)
set(UTILS_CLI11_INCLUDE_DIR "${UTILS_CLI11_ROOT}/include")

utils_resolve_dependency(
  utils_fmt
  "${UTILS_FMT_LOCAL_DIR}"
  "include/fmt/format.h"
  "https://github.com/fmtlib/fmt/archive/refs/tags/${UTILS_FMT_VERSION}.tar.gz"
  UTILS_FMT_ROOT
  UTILS_FMT_SOURCE
)
set(UTILS_FMT_INCLUDE_DIR "${UTILS_FMT_ROOT}/include")

utils_resolve_dependency(
  utils_spdlog
  "${UTILS_SPDLOG_LOCAL_DIR}"
  "include/spdlog/spdlog.h"
  "https://github.com/gabime/spdlog/archive/refs/tags/v${UTILS_SPDLOG_VERSION}.tar.gz"
  UTILS_SPDLOG_ROOT
  UTILS_SPDLOG_SOURCE
)
set(UTILS_SPDLOG_INCLUDE_DIR "${UTILS_SPDLOG_ROOT}/include")

utils_resolve_dependency(
  utils_bs_thread_pool
  "${UTILS_BS_THREAD_POOL_LOCAL_DIR}"
  "include/BS_thread_pool.hpp"
  "https://github.com/bshoshany/thread-pool/archive/refs/tags/v${UTILS_BS_THREAD_POOL_VERSION}.tar.gz"
  UTILS_BS_THREAD_POOL_ROOT
  UTILS_BS_THREAD_POOL_SOURCE
)

utils_resolve_dependency(
  utils_autodiff
  "${UTILS_AUTODIFF_LOCAL_DIR}"
  "autodiff/forward/dual.hpp"
  "https://github.com/ebertolazzi/autodiff/archive/refs/heads/${UTILS_AUTODIFF_VERSION}.tar.gz"
  UTILS_AUTODIFF_ROOT
  UTILS_AUTODIFF_SOURCE
)

utils_sync_resolved_3rdparty()

# Sources: ONLY .cc
file(
  GLOB_RECURSE SOURCES
  CONFIGURE_DEPENDS
  "${SOURCE_DIR}/*.cc"
)

# Project headers and vendored 3rd-party headers are installed separately.
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

foreach(H IN LISTS PROJECT_HEADERS_FULL)
  set_source_files_properties("${H}" PROPERTIES HEADER_FILE_ONLY TRUE)
endforeach()

set(INCLUDE_DIRS
  "${SOURCE_DIR}/Utils/3rd"
  "${SOURCE_DIR}"
  "${SOURCE_DIR}/Utils"
)

file(GLOB_RECURSE UTILS_SUBDIRS CONFIGURE_DEPENDS "${SOURCE_DIR}/Utils/*/")
foreach(DIR IN LISTS UTILS_SUBDIRS)
  if(IS_DIRECTORY "${DIR}")
    list(APPEND INCLUDE_DIRS "${DIR}")
  endif()
endforeach()
list(REMOVE_DUPLICATES INCLUDE_DIRS)

function(utils_configure_target target_name)
  target_compile_features(${target_name} PUBLIC cxx_std_17)
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
  target_compile_definitions(
    ${target_name}
    PUBLIC
      FMT_HEADER_ONLY
  )

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    target_compile_options(${target_name} PRIVATE -Wall -Wextra)
  endif()

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

function(utils_add_platform_library_alias target_name base_name library_kind)
  if(library_kind STREQUAL "STATIC")
    set(target_filename "${CMAKE_STATIC_LIBRARY_PREFIX}${base_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    if(WIN32)
      set(alias_filename "lib${base_name}_${UTILS_PLATFORM_SUFFIX}_static${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
      set(alias_filename "${CMAKE_STATIC_LIBRARY_PREFIX}${base_name}_${UTILS_PLATFORM_SUFFIX}_static${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endif()
  else()
    set(target_filename "${CMAKE_SHARED_LIBRARY_PREFIX}${base_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(WIN32)
      set(alias_filename "lib${base_name}_${UTILS_PLATFORM_SUFFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    else()
      set(alias_filename "${CMAKE_SHARED_LIBRARY_PREFIX}${base_name}_${UTILS_PLATFORM_SUFFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()
  endif()

  if(WIN32)
    add_custom_command(
      TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "$<TARGET_FILE:${target_name}>"
              "$<TARGET_FILE_DIR:${target_name}>/${alias_filename}"
      VERBATIM
    )

    if(UTILS_INSTALL_LOCAL)
      install(CODE
        "file(MAKE_DIRECTORY \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
"
        "execute_process(
"
        "  COMMAND \"${CMAKE_COMMAND}\" -E copy_if_different
"
        "          \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${target_filename}\"
"
        "          \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${alias_filename}\"
"
        ")"
      )
    endif()
  else()
    add_custom_command(
      TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E rm -f
              "$<TARGET_FILE_DIR:${target_name}>/${alias_filename}"
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              "${target_filename}"
              "$<TARGET_FILE_DIR:${target_name}>/${alias_filename}"
      VERBATIM
    )

    if(UTILS_INSTALL_LOCAL)
      install(CODE
        "file(MAKE_DIRECTORY \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
"
        "execute_process(
"
        "  COMMAND \"${CMAKE_COMMAND}\" -E rm -f
"
        "          \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${alias_filename}\"
"
        ")
"
        "execute_process(
"
        "  COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
"
        "          \"${target_filename}\"
"
        "          \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${alias_filename}\"
"
        ")"
      )
    endif()
  endif()
endfunction()

# Static target: keeps compatibility with previous cmake_utils naming.
add_library(${UTILS_NAMESPACE}_${PROJECT_NAME}_Static STATIC ${SOURCES} ${PROJECT_HEADERS_FULL})
add_library(${UTILS_NAMESPACE}::${PROJECT_NAME}_Static ALIAS ${UTILS_NAMESPACE}_${PROJECT_NAME}_Static)
set_target_properties(${UTILS_NAMESPACE}_${PROJECT_NAME}_Static PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
utils_configure_target(${UTILS_NAMESPACE}_${PROJECT_NAME}_Static)
utils_add_platform_library_alias(${UTILS_NAMESPACE}_${PROJECT_NAME}_Static ${PROJECT_NAME} STATIC)

# Shared target: optional, same naming scheme as before.
if(UTILS_BUILD_SHARED)
  add_library(${UTILS_NAMESPACE}_${PROJECT_NAME} SHARED ${SOURCES} ${PROJECT_HEADERS_FULL})
  add_library(${UTILS_NAMESPACE}::${PROJECT_NAME} ALIAS ${UTILS_NAMESPACE}_${PROJECT_NAME})
  set_target_properties(${UTILS_NAMESPACE}_${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
  utils_configure_target(${UTILS_NAMESPACE}_${PROJECT_NAME})
  utils_add_platform_library_alias(${UTILS_NAMESPACE}_${PROJECT_NAME} ${PROJECT_NAME} SHARED)
endif()

# Tests
if(UTILS_ENABLE_TESTS)
  enable_testing()

  set(LIBS ${UTILS_NAMESPACE}_${PROJECT_NAME}_Static ${CMAKE_DL_LIBS})

  if(UNIX AND NOT APPLE)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    list(APPEND LIBS Threads::Threads)
  elseif(WIN32)
    list(APPEND LIBS ws2_32 iphlpapi kernel32 pdh psapi)
  endif()

  file(
    GLOB_RECURSE TEST_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src_tests/*.cc"
  )

  if(TEST_SOURCES)
    add_custom_target("${PROJECT_NAME}_all_tests" ALL)
  endif()

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
    target_link_libraries(${TEST_NAME} PRIVATE ${LIBS})
    target_include_directories(${TEST_NAME} PRIVATE ${INCLUDE_DIRS})
    add_test(NAME ${TEST_NAME} COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/bin/${TEST_NAME}.exe")
    add_dependencies("${PROJECT_NAME}_all_tests" ${TEST_NAME})
  endforeach()
endif()

# Install
if(UTILS_INSTALL_LOCAL)
  set(_utils_install_targets ${UTILS_NAMESPACE}_${PROJECT_NAME}_Static)
  if(UTILS_BUILD_SHARED)
    list(APPEND _utils_install_targets ${UTILS_NAMESPACE}_${PROJECT_NAME})
  endif()

  install(
    TARGETS ${_utils_install_targets}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

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

  install(
    DIRECTORY "${SOURCE_DIR}/Utils/3rd/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Utils/3rd"
    PATTERN ".DS_Store" EXCLUDE
    PATTERN ".gitkeep" EXCLUDE
    PATTERN "*.swp" EXCLUDE
    PATTERN "*.swo" EXCLUDE
  )

  install(
    EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${UTILS_NAMESPACE}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
endif()

set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "UtilsLite C++ utility library")
set(CPACK_GENERATOR "TGZ;ZIP")
include(CPack)

message(STATUS "")
message(STATUS "${PROJECT_NAME} configuration summary:")
message(STATUS "  Version          : ${PROJECT_VERSION}")
message(STATUS "  Compiler         : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Platform suffix  : ${UTILS_PLATFORM_SUFFIX}")
message(STATUS "  Build shared     : ${UTILS_BUILD_SHARED}")
message(STATUS "  Build tests      : ${UTILS_ENABLE_TESTS}")
message(STATUS "  Install rules    : ${UTILS_INSTALL_LOCAL}")
message(STATUS "  Headers install  : ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "  Libraries install: ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "  3rd-party include: ${CMAKE_INSTALL_INCLUDEDIR}/Utils/3rd")
message(STATUS "  Eigen            : ${UTILS_EIGEN_SOURCE}")
message(STATUS "  CLI11            : ${UTILS_CLI11_SOURCE}")
message(STATUS "  fmt              : ${UTILS_FMT_SOURCE}")
message(STATUS "  spdlog           : ${UTILS_SPDLOG_SOURCE}")
message(STATUS "  BS thread pool   : ${UTILS_BS_THREAD_POOL_SOURCE}")
message(STATUS "  autodiff         : ${UTILS_AUTODIFF_SOURCE}")
message(STATUS "")
