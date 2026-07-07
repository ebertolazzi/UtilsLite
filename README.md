# UTILS

A collection of useful code for C++ applications:

- Terminal coloring use code from [here](https://github.com/agauniyal/rang) by Abhinav Gauniyal ([license](http://unlicense.org))

- Stream compression use code from [here](https://github.com/geromueller/zstream-cpp) by Jonathan de Halleux and Gero Müller

- Stream formatting use code from [here](https://fmt.dev) by Victor Zverovich (MIT license)

- Table formatting use code from [here](https://github.com/Bornageek/terminal-table) by Andreas Wilhelm (Apache License, Version 2.0) partially rewritten.

- [Eigen3](https://eigen.tuxfamily.org) a C++ template library for linear algebra.

in addition a TreadPool class, TicToc class for timing, Malloc
class for easy allocation with traking of allocated memory.

- Online doc [here](https://ebertolazzi.github.io/UtilsLite)

## COMPILE AND TEST

The build system is pure CMake (Ninja recommended). No Ruby/rake is needed:
the third-party headers are vendored under `src/Utils/3rd` and committed to the
repository.

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

To also build and run the tests:

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUTILS_ENABLE_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

To compile with MinGW, open an MSYS2 shell and build as in a unix environment.
Any CMake generator (Make, Xcode, Visual Studio, ...) works too.

## USE AS A DEPENDENCY (FetchContent)

UtilsLite can be consumed directly by another CMake project. No external tools
or network downloads are required for the third-party dependencies since they
are vendored in the repository.

```cmake
include(FetchContent)
FetchContent_Declare(
  UtilsLite
  GIT_REPOSITORY https://github.com/ebertolazzi/UtilsLite.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(UtilsLite)

target_link_libraries(my_target PRIVATE Utils::UtilsLite_Static)
```

## UPDATING THE VENDORED THIRD-PARTY HEADERS (maintainers)

The headers under `src/Utils/3rd` (Eigen, spdlog, autodiff, BS::thread_pool,
task-thread-pool) are regenerated from upstream by CMake via `FetchContent`.
This replaces the old `ThirdParties/*` rake tasks. To refresh them:

```
cmake -B build -DUTILS_UPDATE_3RDPARTY=ON
```

This downloads each pinned release, rewrites its `#include` directives so the
headers work flattened under `3rd/`, applies the local autodiff patches, and
copies the result into `src/Utils/3rd`. Review the result with `git diff` and
commit it. Versions are pinned in `cmake/Update3rdParties.cmake`.