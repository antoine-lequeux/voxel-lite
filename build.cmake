cmake_minimum_required(VERSION 3.24)

# Build options
if(NOT DEFINED COMP)
    if(WIN32)
        set(COMP "msvc")
    else()
        set(COMP "clang")
    endif()
endif()

if(NOT DEFINED ARCH)
    set(ARCH "native")
endif()

if(NOT DEFINED CONFIG)
    set(CONFIG "Release")
endif()

# Compiler config
set(GEN_ARGS "")
if(COMP STREQUAL "clang")
    set(CXX_BIN "clang++")
    set(C_BIN "clang")
elseif(COMP STREQUAL "gcc")
    set(CXX_BIN "g++")
    set(C_BIN "gcc")
elseif(COMP STREQUAL "msvc")
    set(CXX_BIN "cl")
    set(C_BIN "cl")
else()
    message(FATAL_ERROR "COMP must be 'clang', 'gcc', or 'msvc'")
endif()

# Use Ninja if available
find_program(NINJA_BIN ninja)
if(NINJA_BIN AND NOT COMP STREQUAL "msvc")
    set(GEN_ARGS "-G" "Ninja")
endif()

set(BUILD_DIR "${CMAKE_CURRENT_LIST_DIR}/build/${COMP}-${ARCH}-${CONFIG}")

set(TARGET_NAME "voxel_engine")
set(EXE_SUFFIX "")
if(WIN32)
    set(EXE_SUFFIX ".exe")
endif()
set(EXE "${BUILD_DIR}/${TARGET_NAME}${EXE_SUFFIX}")

# Build
message(STATUS "==> Config: [${COMP} | ${ARCH} | ${CONFIG}]...")

if(COMP STREQUAL "msvc")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -B ${BUILD_DIR}
            -DCMAKE_BUILD_TYPE=${CONFIG}
            -DARCH=${ARCH}
        COMMAND_ERROR_IS_FATAL ANY
    )
else()
    execute_process(
        COMMAND ${CMAKE_COMMAND} -B ${BUILD_DIR} ${GEN_ARGS}
            -DCMAKE_BUILD_TYPE=${CONFIG}
            -DCMAKE_CXX_COMPILER=${CXX_BIN}
            -DCMAKE_C_COMPILER=${C_BIN}
            -DARCH=${ARCH}
        COMMAND_ERROR_IS_FATAL ANY
    )
endif()

if(EXISTS "${BUILD_DIR}/compile_commands.json")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${BUILD_DIR}/compile_commands.json"
            "${CMAKE_CURRENT_LIST_DIR}/compile_commands.json"
    )
endif()

message(STATUS "==> Compiling VoxelLite engine...")
execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${BUILD_DIR} --config ${CONFIG} -j
    COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "==> Binary was successfully generated: ${EXE}")