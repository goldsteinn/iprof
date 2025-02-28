cmake_minimum_required(VERSION 3.5.0)
project(
  IPROF
  VERSION 0.0.1
  LANGUAGES CXX
)

# Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Search for external programs
set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

# Language requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Baseline compiler flags
list(APPEND IPROF_COMPILE_FLAGS "-Wall")
list(APPEND IPROF_COMPILE_FLAGS "-Wextra")
list(APPEND IPROF_COMPILE_FLAGS "-Wno-unused-parameter")
list(APPEND IPROF_COMPILE_FLAGS "-Wno-unused-function")
list(APPEND IPROF_COMPILE_FLAGS "-Werror")
list(APPEND IPROF_COMPILE_FLAGS "-fno-semantic-interposition")
list(APPEND IPROF_COMPILE_FLAGS "-fvisibility=hidden")
list(APPEND IPROF_COMPILE_FLAGS "-fno-exceptions")
list(APPEND IPROF_COMPILE_FLAGS "-fno-unwind-tables")
list(APPEND IPROF_COMPILE_FLAGS "-fno-asynchronous-unwind-tables")
list(APPEND IPROF_COMPILE_FLAGS "-fno-rtti")

# Convert lists to strings for cmake
string(
  REPLACE ";"
          " "
          IPROF_COMPILE_FLAGS
          " ${IPROF_COMPILE_FLAGS}"
)

# Add iprof executable
add_library(iprof SHARED src/iprof.cc)
# Add compiler options
set_property(
  TARGET iprof
  APPEND_STRING
  PROPERTY COMPILE_FLAGS "${IPROF_COMPILE_FLAGS}"
)

# Find LLVM
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include(AddLLVM)

# Include LLVM headers
include_directories(${LLVM_INCLUDE_DIRS})

# Include LLVM components
llvm_map_components_to_libnames(
  IPROF_LLVM_LIBS
  AllTargetsAsmParsers
  AllTargetsMCAs
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MCA
  MC
  MCParser
  Support
  TargetParser
)

# Find QEMU
find_path(IPROF_QEMU_PLUGINS_HEADER NAMES "qemu-plugin.h")
message(STATUS "Found QEMU ${IPROF_QEMU_PLUGINS_HEADER}")

# Include QEMU Plugin Header
include_directories(${IPROF_QEMU_PLUGINS_HEADER})

# Find GLIB (QEMU Dependency)
find_path(
  IPROF_GLIB_HEADER
  NAMES "glib.h"
  PATH_SUFFIXES "glib-2.0"
)
message(STATUS "Found GLIB ${IPROF_GLIB_HEADER}")
include_directories(${IPROF_GLIB_HEADER})

find_path(
  IPROF_GLIBCONF_HEADER
  NAMES "glibconfig.h"
#  PATH_SUFFIXES "lib/x86_64-linux-gnu/glib-2.0/include"
)
message(STATUS "Found GLIBCONF ${IPROF_GLIBCONF_HEADER}")
include_directories(${IPROF_GLIBCONF_HEADER})

target_link_libraries(iprof ${IPROF_LLVM_LIBS})
