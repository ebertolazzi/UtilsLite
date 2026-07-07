############################################################################
#                                                                          #
#  file: cmake/Update3rdParties.cmake                                      #
#                                                                          #
#  Regenerate the vendored third-party headers under src/Utils/3rd.        #
#                                                                          #
#  This is a *maintainer-only* replacement for the old ThirdParties/*      #
#  Rakefiles. It downloads each dependency (via FetchContent), rewrites    #
#  its #include directives so the headers work when flattened under        #
#  src/Utils/3rd, applies the autodiff tanh-derivative patch, and copies   #
#  the result into the source tree. The normal build never runs this: it   #
#  simply consumes the already-committed headers.                          #
#                                                                          #
#  Usage (from the project top level):                                     #
#     cmake -B build -DUTILS_UPDATE_3RDPARTY=ON                            #
#  The headers are regenerated at configure time; inspect `git diff`.      #
#                                                                          #
############################################################################

include_guard(GLOBAL)
include(FetchContent)

# ----------------------------------------------------------------------------
# Pinned versions (match the previous ThirdParties/*/Rakefile values)
# ----------------------------------------------------------------------------
set(UTILS_3RD_EIGEN_VERSION            "5.0.0"  CACHE STRING "Eigen version to vendor")
set(UTILS_3RD_SPDLOG_VERSION           "1.16.0" CACHE STRING "spdlog version to vendor")
set(UTILS_3RD_BS_THREAD_POOL_VERSION   "5.0.0"  CACHE STRING "BS::thread_pool version to vendor")
set(UTILS_3RD_AUTODIFF_VERSION         "1.1.2"  CACHE STRING "autodiff version to vendor")
set(UTILS_3RD_TASK_THREAD_POOL_VERSION "1.0.10" CACHE STRING "task-thread-pool version to vendor")

# ----------------------------------------------------------------------------
# Helper: rewrite #include directives in a single file.
# Extra arguments are (regex replacement) pairs applied in order.
# Only writes the file back when its content actually changes (like the
# original Ruby, which reported "Updated: <file>").
# ----------------------------------------------------------------------------
function(_utils_rewrite_file FILE)
  if(NOT EXISTS "${FILE}")
    return()
  endif()
  file(READ "${FILE}" _content)
  set(_orig "${_content}")

  set(_pairs ${ARGN})
  list(LENGTH _pairs _n)
  math(EXPR _last "${_n} - 1")
  set(_i 0)
  while(_i LESS _n)
    math(EXPR _j "${_i} + 1")
    list(GET _pairs ${_i} _regex)
    list(GET _pairs ${_j} _replace)
    string(REGEX REPLACE "${_regex}" "${_replace}" _content "${_content}")
    math(EXPR _i "${_i} + 2")
  endwhile()

  if(NOT _content STREQUAL _orig)
    file(WRITE "${FILE}" "${_content}")
    message(STATUS "  updated: ${FILE}")
  endif()
endfunction()

# Apply a rewrite to every *file* matching a GLOB pattern.
function(_utils_rewrite_glob PATTERN)
  file(GLOB _files "${PATTERN}")
  foreach(_f ${_files})
    if(NOT IS_DIRECTORY "${_f}")
      _utils_rewrite_file("${_f}" ${ARGN})
    endif()
  endforeach()
endfunction()

# Replace destination directory with a fresh copy of a source directory.
function(_utils_replace_dir SRC DST)
  file(REMOVE_RECURSE "${DST}")
  get_filename_component(_parent "${DST}" DIRECTORY)
  file(MAKE_DIRECTORY "${_parent}")
  file(COPY "${SRC}/" DESTINATION "${DST}")
endfunction()

# ----------------------------------------------------------------------------
# Main entry point.
# ----------------------------------------------------------------------------
function(utils_update_3rdparties)
  set(_root  "${CMAKE_CURRENT_SOURCE_DIR}")
  set(_dst   "${_root}/src/Utils/3rd")
  set(_stage "${CMAKE_BINARY_DIR}/_3rd_stage")

  file(MAKE_DIRECTORY "${_dst}")
  file(REMOVE_RECURSE "${_stage}")
  file(MAKE_DIRECTORY "${_stage}")

  message(STATUS "==============================================================")
  message(STATUS "Regenerating vendored third-party headers in ${_dst}")
  message(STATUS "==============================================================")

  # Download (extract only, never add_subdirectory) --------------------------
  # The bogus SOURCE_SUBDIR keeps FetchContent from configuring each project.
  FetchContent_Declare( eigen_src
    URL "https://gitlab.com/libeigen/eigen/-/archive/${UTILS_3RD_EIGEN_VERSION}/eigen-${UTILS_3RD_EIGEN_VERSION}.zip"
    SOURCE_SUBDIR _do_not_configure )
  FetchContent_Declare( spdlog_src
    URL "https://github.com/gabime/spdlog/archive/refs/tags/v${UTILS_3RD_SPDLOG_VERSION}.tar.gz"
    SOURCE_SUBDIR _do_not_configure )
  FetchContent_Declare( bs_thread_pool_src
    URL "https://github.com/bshoshany/thread-pool/archive/refs/tags/v${UTILS_3RD_BS_THREAD_POOL_VERSION}.tar.gz"
    SOURCE_SUBDIR _do_not_configure )
  FetchContent_Declare( autodiff_src
    URL "https://github.com/autodiff/autodiff/archive/refs/tags/v${UTILS_3RD_AUTODIFF_VERSION}.tar.gz"
    SOURCE_SUBDIR _do_not_configure )
  FetchContent_Declare( task_thread_pool_src
    URL "https://github.com/alugowski/task-thread-pool/archive/refs/tags/v${UTILS_3RD_TASK_THREAD_POOL_VERSION}.tar.gz"
    SOURCE_SUBDIR _do_not_configure )

  FetchContent_MakeAvailable(
    eigen_src spdlog_src bs_thread_pool_src autodiff_src task_thread_pool_src )

  # --- Eigen ----------------------------------------------------------------
  # Copy the Eigen/ header tree verbatim (no include rewriting needed).
  message(STATUS "Vendoring Eigen ${UTILS_3RD_EIGEN_VERSION}")
  _utils_replace_dir("${eigen_src_SOURCE_DIR}/Eigen" "${_dst}/Eigen")

  # --- spdlog ---------------------------------------------------------------
  # Flatten <spdlog/...> includes to path-relative ones.
  message(STATUS "Vendoring spdlog ${UTILS_3RD_SPDLOG_VERSION}")
  set(_spdlog_stage "${_stage}/spdlog")
  file(COPY "${spdlog_src_SOURCE_DIR}/include/spdlog/" DESTINATION "${_spdlog_stage}")
  # top-level headers:  #include <spdlog/foo> -> #include "foo"
  _utils_rewrite_glob("${_spdlog_stage}/*"
    "#include <spdlog/([^>]*)>" "#include \"\\1\"")
  # one level deep:     #include "spdlog/foo" / <spdlog/foo> -> #include "../foo"
  _utils_rewrite_glob("${_spdlog_stage}/*/*"
    "#include \"spdlog/([^\"]*)\"" "#include \"../\\1\""
    "#include <spdlog/([^>]*)>"    "#include \"../\\1\"")
  _utils_replace_dir("${_spdlog_stage}" "${_dst}/spdlog")

  # --- BS::thread_pool (single header) --------------------------------------
  message(STATUS "Vendoring BS::thread_pool ${UTILS_3RD_BS_THREAD_POOL_VERSION}")
  file(COPY "${bs_thread_pool_src_SOURCE_DIR}/include/BS_thread_pool.hpp"
       DESTINATION "${_dst}")

  # --- autodiff -------------------------------------------------------------
  # Flatten <autodiff/...> and redirect <Eigen/...> to the vendored copy,
  # then patch the tanh derivative (matches the original Rakefile).
  message(STATUS "Vendoring autodiff ${UTILS_3RD_AUTODIFF_VERSION}")
  set(_autodiff_stage "${_stage}/autodiff")
  file(COPY "${autodiff_src_SOURCE_DIR}/autodiff/" DESTINATION "${_autodiff_stage}")
  # Upstream ships a Bazel BUILD file in the headers dir; the vendored tree
  # does not carry it. Drop it so the result matches the committed layout.
  file(REMOVE "${_autodiff_stage}/BUILD")
  # two levels deep (autodiff/*/*)
  _utils_rewrite_glob("${_autodiff_stage}/*/*"
    "#include <eigen3/Eigen/([^>]*)>" "#include \"../../../Eigen/\\1\""
    "#include <Eigen/([^>]*)>"        "#include \"../../../Eigen/\\1\""
    "#include <autodiff/([^>]*)>"     "#include \"../\\1\"")
  # three levels deep (autodiff/*/*/*)
  _utils_rewrite_glob("${_autodiff_stage}/*/*/*"
    "#include <eigen3/Eigen/([^>]*)>" "#include \"../../../Eigen/\\1\""
    "#include <Eigen/([^>]*)>"        "#include \"../../../Eigen/\\1\""
    "#include <autodiff/([^>]*)>"     "#include \"../../\\1\"")
  _utils_patch_autodiff("${_autodiff_stage}")
  _utils_replace_dir("${_autodiff_stage}" "${_dst}/autodiff")

  # --- task-thread-pool (single header) -------------------------------------
  message(STATUS "Vendoring task-thread-pool ${UTILS_3RD_TASK_THREAD_POOL_VERSION}")
  file(COPY "${task_thread_pool_src_SOURCE_DIR}/include/task_thread_pool.hpp"
       DESTINATION "${_dst}")

  message(STATUS "==============================================================")
  message(STATUS "Third-party headers regenerated. Review with `git diff`.")
  message(STATUS "==============================================================")
endfunction()

# ----------------------------------------------------------------------------
# Local source patches applied on top of upstream autodiff to reproduce the
# committed src/Utils/3rd/autodiff tree exactly. These are edits the project
# carries that are NOT plain include rewrites:
#
#  1. forward/dual/dual.hpp -- tanh() forward-mode derivative. Upstream writes
#         const T aux = One<T>() / cosh(self.val);
#         self.val    = tanh(self.val);
#         self.grad  *= aux * aux;
#     which is numerically poorer than the equivalent 1 - tanh^2. Rewrite the
#     TanhOp block ONLY (the identically-shaped TanOp block just above must be
#     left untouched) to:
#         self.val   = tanh(self.val);
#         self.grad *=  1 - self.val * self.val;
#
#  2. forward/utils/derivative.hpp -- upstream loops with `for(auto i = 0; ...)`
#     where `i` compares against a size_t `len`, triggering signed/unsigned
#     comparison warnings. Use `size_t` instead.
# ----------------------------------------------------------------------------
function(_utils_patch_autodiff STAGE_DIR)
  # --- 1. tanh derivative (TanhOp block only) ---
  set(_dual "${STAGE_DIR}/forward/dual/dual.hpp")
  if(NOT EXISTS "${_dual}")
    message(FATAL_ERROR "autodiff patch: ${_dual} not found")
  endif()
  file(READ "${_dual}" _c)
  set(_orig "${_c}")
  # drop the `const T aux = ... cosh(self.val);` line (cosh is unique to TanhOp)
  string(REGEX REPLACE
    "[ \t]*const T aux = One<T>\\(\\) / cosh\\(self\\.val\\);[ \t]*\r?\n"
    ""
    _c "${_c}")
  # replace the gradient update, anchored to the preceding `tanh(...)` line so
  # the visually identical TanOp block (which uses `tan`) is not affected. The
  # captured whitespace (\1) preserves the original newline + indentation.
  string(REGEX REPLACE
    "(self\\.val = tanh\\(self\\.val\\);[ \t\r\n]+)self\\.grad \\*= aux \\* aux;"
    "\\1self.grad *=  1 - self.val * self.val;"
    _c "${_c}")
  if(_c STREQUAL _orig)
    message(FATAL_ERROR "autodiff tanh patch: pattern not found in ${_dual}")
  endif()
  file(WRITE "${_dual}" "${_c}")
  message(STATUS "  patched tanh derivative in ${_dual}")

  # --- 2. size_t loop index in derivative() ---
  set(_deriv "${STAGE_DIR}/forward/utils/derivative.hpp")
  if(NOT EXISTS "${_deriv}")
    message(FATAL_ERROR "autodiff patch: ${_deriv} not found")
  endif()
  file(READ "${_deriv}" _c)
  set(_orig "${_c}")
  string(REPLACE
    "for(auto i = 0; i < len; ++i)"
    "for(size_t i = 0; i < len; ++i)"
    _c "${_c}")
  if(_c STREQUAL _orig)
    message(FATAL_ERROR "autodiff derivative patch: pattern not found in ${_deriv}")
  endif()
  file(WRITE "${_deriv}" "${_c}")
  message(STATUS "  patched size_t loop index in ${_deriv}")
endfunction()
