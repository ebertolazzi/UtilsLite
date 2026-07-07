# UTILS

UtilsLite is a collection of C++ utilities for numerical and systems code.
It includes containers, timing helpers, memory helpers, formatting helpers,
and thread-pool abstractions built on top of vendored third-party headers.

Third-party components currently used by the project include:

- [rang](https://github.com/agauniyal/rang) for terminal coloring
- [fmt](https://fmt.dev) for formatting
- [terminal-table](https://github.com/Bornageek/terminal-table) for table output
- [Eigen](https://eigen.tuxfamily.org) for linear algebra
- [CLI11](https://github.com/CLIUtils/CLI11) for command-line parsing
- [spdlog](https://github.com/gabime/spdlog) for logging
- [BS::thread_pool](https://github.com/bshoshany/thread-pool) for thread-pool support
- [autodiff](https://github.com/ebertolazzi/autodiff) for automatic differentiation

Online documentation is available [here](https://ebertolazzi.github.io/UtilsLite).

## BUILD, TEST, INSTALL

The project is built with plain CMake. Ninja is recommended, but any generator
works.

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

To build the tests:

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUTILS_ENABLE_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

When configured as the top-level project, the default install prefix is the
repository root and artifacts are installed under:

- headers: `lib/include`
- libraries: `lib/lib`
- executables: `lib/bin`

Install with:

```sh
cmake --install build
```

On Unix-like systems the build also creates platform-specific library aliases
as symbolic links. For example, the static library is installed as both
`libUtilsLite.a` and `libUtilsLite_osx_static.a` on macOS.

## THIRD-PARTY DEPENDENCIES

At configure time, UtilsLite first looks for sibling checkouts and only falls
back to `FetchContent` if they are missing:

- `../eigen`
- `../CLI11`
- `../fmt`
- `../spdlog`
- `../BS_thread_pool`
- `../autodiff` for a sibling checkout, otherwise `FetchContent` from the `main` branch of `ebertolazzi/autodiff`

Resolved headers are synchronized into `src/Utils/3rd`, which is the include
tree used by the library and installed under `lib/include/Utils/3rd`.
`spdlog` is normalized to use the vendored `fmt` headers shipped by UtilsLite
instead of its bundled copy.

## USE AS A DEPENDENCY

UtilsLite exports the following CMake targets:

- `utils::UtilsLite_Static`
- `utils::UtilsLite` when `UTILS_BUILD_SHARED=ON`

Example with `FetchContent`:

```cmake
include(FetchContent)

FetchContent_Declare(
  UtilsLite
  GIT_REPOSITORY https://github.com/ebertolazzi/UtilsLite.git
  GIT_TAG        main
)

FetchContent_MakeAvailable(UtilsLite)

target_link_libraries(my_target PRIVATE utils::UtilsLite_Static)
```

## MAINTAINER NOTES

To regenerate the vendored third-party headers:

```sh
cmake -B build -G Ninja -DUTILS_UPDATE_3RDPARTY=ON
```

This refreshes `src/Utils/3rd` from the pinned upstream versions, rewrites
include paths where needed, removes spdlog's bundled fmt copy, applies the
local autodiff patching, and updates the committed vendor tree. Review the
result with `git diff` before committing.

Third-party license files are collected under `licenses3rd/`.
