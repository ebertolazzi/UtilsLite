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

function(_utils_rewrite_glob_recursive PATTERN)
  file(GLOB_RECURSE _files "${PATTERN}")
  foreach(_f ${_files})
    if(NOT IS_DIRECTORY "${_f}")
      _utils_rewrite_file("${_f}" ${ARGN})
    endif()
  endforeach()
endfunction()

function(_utils_rewrite_prefixed_includes_file FILE ROOT PREFIX)
  if(NOT EXISTS "${FILE}")
    return()
  endif()

  file(READ "${FILE}" _content)
  set(_orig "${_content}")
  get_filename_component(_file_dir "${FILE}" DIRECTORY)

  string(REGEX MATCHALL "#[ \t]*include[ \t]*[<\"]${PREFIX}/[^>\"]+[>\"]" _matches "${_content}")
  foreach(_match ${_matches})
    string(REGEX REPLACE ".*[<\"]${PREFIX}/([^>\"]+)[>\"]" "\\1" _suffix "${_match}")
    file(RELATIVE_PATH _relative "${_file_dir}" "${ROOT}/${_suffix}")
    string(REPLACE "${_match}" "#include \"${_relative}\"" _content "${_content}")
  endforeach()

  if(NOT _content STREQUAL _orig)
    file(WRITE "${FILE}" "${_content}")
    message(STATUS "  updated: ${FILE}")
  endif()
endfunction()

function(_utils_rewrite_prefixed_includes_recursive ROOT PREFIX)
  file(GLOB_RECURSE _files "${ROOT}/*")
  foreach(_f ${_files})
    if(NOT IS_DIRECTORY "${_f}")
      _utils_rewrite_prefixed_includes_file("${_f}" "${ROOT}" "${PREFIX}")
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
# Local source patches applied on top of the autodiff source tree to reproduce the
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
  string(REGEX MATCH
    "self\\.val = tanh\\(self\\.val\\);[ \t\r\n]+self\\.grad \\*=  *1 - self\\.val \\* self\\.val;"
    _tanh_already_patched "${_c}")
  if(_tanh_already_patched)
    message(STATUS "  autodiff tanh patch not needed in ${_dual}")
  else()
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
  endif()

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
    message(STATUS "  autodiff derivative patch not needed in ${_deriv}")
    return()
  endif()
  file(WRITE "${_deriv}" "${_c}")
  message(STATUS "  patched size_t loop index in ${_deriv}")
endfunction()
