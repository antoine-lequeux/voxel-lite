cmake_minimum_required(VERSION 3.24)

set(CMAKE_POLICY_VERSION_MINIMUM 3.10)

project(voxel-lite VERSION 0.1.0 LANGUAGES CXX C)

# ---------------------------------------------------------
# Global C++23 & build setup
# ---------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type" FORCE)
endif()

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
endif()

# macOS specific RPATH setup
if(APPLE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH "@executable_path")
endif()

# ---------------------------------------------------------
# Target architecture option
# ---------------------------------------------------------
set(ARCH "native" CACHE STRING "Target architecture: native, avx2, avx512, apple-silicon")

set(ARCH_FLAGS "")
if(MSVC)
    if(ARCH STREQUAL "avx2")
        set(ARCH_FLAGS /arch:AVX2)
    elseif(ARCH STREQUAL "avx512")
        set(ARCH_FLAGS /arch:AVX512)
    endif()
else()
    if(ARCH STREQUAL "native")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
            set(ARCH_FLAGS -mcpu=native)
        else()
            set(ARCH_FLAGS -march=native)
        endif()
    elseif(ARCH STREQUAL "avx2")
        set(ARCH_FLAGS -march=x86-64-v3)
    elseif(ARCH STREQUAL "avx512")
        set(ARCH_FLAGS -march=x86-64-v4)
    elseif(ARCH STREQUAL "apple-silicon")
        set(ARCH_FLAGS -mcpu=apple-m1)
    else()
        message(FATAL_ERROR "Unsupported ARCH option: ${ARCH}")
    endif()
endif()

# ---------------------------------------------------------
# Third party dependencies
# ---------------------------------------------------------
include(FetchContent)

# Disable build options for dependencies
set(SDL_TESTS                OFF CACHE BOOL "" FORCE)
set(ENKITS_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(ENKITS_BUILD_C_INTERFACE OFF CACHE BOOL "" FORCE)
set(ENKITS_INSTALL           OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_TESTS          OFF CACHE BOOL "" FORCE)

FetchContent_Declare(vulkan_headers GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git                    GIT_TAG vulkan-sdk-1.4.304.0 SYSTEM)
FetchContent_Declare(sdl3           GIT_REPOSITORY https://github.com/libsdl-org/SDL.git                                 GIT_TAG release-3.2.8        SYSTEM)
FetchContent_Declare(enkits         GIT_REPOSITORY https://github.com/dougbinks/enkiTS.git                               GIT_TAG v1.12                SYSTEM)
FetchContent_Declare(imgui          GIT_REPOSITORY https://github.com/ocornut/imgui.git                                  GIT_TAG v1.91.9b-docking     SYSTEM)
FetchContent_Declare(glm            GIT_REPOSITORY https://github.com/g-truc/glm.git                                     GIT_TAG 1.0.1                SYSTEM)
FetchContent_Declare(volk           GIT_REPOSITORY https://github.com/zeux/volk.git                                      GIT_TAG vulkan-sdk-1.4.304.0 SYSTEM)
FetchContent_Declare(vma            GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git GIT_TAG v3.2.1               SYSTEM)
FetchContent_Declare(vkbootstrap    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git                    GIT_TAG v1.3.299             SYSTEM)

# Silence warnings during third-party library compilation
set(BACKUP_C_FLAGS "${CMAKE_C_FLAGS}")
set(BACKUP_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
endif()

FetchContent_MakeAvailable(vulkan_headers sdl3 enkits imgui glm volk vma vkbootstrap)

target_link_libraries(volk PRIVATE Vulkan::Headers)
target_link_libraries(volk_headers INTERFACE Vulkan::Headers)

set(CMAKE_C_FLAGS "${BACKUP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${BACKUP_CXX_FLAGS}")

# ---------------------------------------------------------
# Slang shader compiler download
# ---------------------------------------------------------
set(SLANG_VERSION "2026.14.1")
if(WIN32)
    set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-windows-x86_64.zip")
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-macos-aarch64.zip")
elseif(APPLE)
    set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-macos-x86_64.zip")
else()
    set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.zip")
endif()

FetchContent_Declare(slang_compiler URL ${SLANG_URL})
FetchContent_MakeAvailable(slang_compiler)

if(WIN32)
    set(SLANGC_EXECUTABLE "${slang_compiler_SOURCE_DIR}/bin/slangc.exe")
else()
    set(SLANGC_EXECUTABLE "${slang_compiler_SOURCE_DIR}/bin/slangc")
endif()

# ---------------------------------------------------------
# ImGui target construction
# ---------------------------------------------------------
set(IMGUI_DIR ${imgui_SOURCE_DIR})
add_library(ImGui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(ImGui SYSTEM PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_compile_definitions(ImGui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK VK_NO_PROTOTYPES)
target_link_libraries(ImGui PRIVATE SDL3::SDL3 volk::volk_headers Vulkan::Headers)

# ---------------------------------------------------------
# Executable & source configuration
# ---------------------------------------------------------
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/engine/engine.cpp
    src/renderer/vulkancontext.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${vma_SOURCE_DIR}/include
)

# Global defines (Vulkan + GLM configuration)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    VK_NO_PROTOTYPES
    VMA_STATIC_VULKAN_FUNCTIONS=0
    VMA_DYNAMIC_VULKAN_FUNCTIONS=1
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

# Link core libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    SDL3::SDL3
    ImGui
    enkiTS
    glm::glm
    volk::volk
    vk-bootstrap
    Vulkan::Headers
)

# ---------------------------------------------------------
# Compiler flags & options
# ---------------------------------------------------------
if(ARCH_FLAGS)
    target_compile_options(${PROJECT_NAME} PRIVATE ${ARCH_FLAGS})
    target_link_options(${PROJECT_NAME} PRIVATE ${ARCH_FLAGS})
endif()

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W4
        /permissive-
        /utf-8
        $<$<CONFIG:Release>:/O2 /Ob2 /DNDEBUG>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
    target_compile_options(${PROJECT_NAME} PRIVATE /EHa- /GR-)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
        -Wnon-virtual-dtor -Wno-missing-field-initializers -Wno-missing-braces
        -Wold-style-cast -Wdouble-promotion -Wcast-align -Wimplicit-fallthrough
        -Wunreachable-code -Wnull-dereference -Wundef
        -fno-exceptions -fno-rtti
        $<$<CONFIG:Release>:-O3 -DNDEBUG -fomit-frame-pointer>
        $<$<CONFIG:Debug>:-O0 -g3>
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE AND NOT WIN32)
        target_link_options(${PROJECT_NAME} PRIVATE "-fuse-ld=lld")
    endif()
endif()

# Link-Time Optimization (LTO)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok)
if(ipo_ok)
    set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# ---------------------------------------------------------
# Shader compilation pipeline
# ---------------------------------------------------------
file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang")

set(ALL_SHADER_SPVS "")

foreach(SHADER_SRC IN LISTS SHADER_SOURCES)
    get_filename_component(SHADER_NAME ${SHADER_SRC} NAME_WE)
    get_filename_component(SHADER_FILENAME ${SHADER_SRC} NAME)
    
    set(SHADER_SPV "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spirv")

    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SRC} -target spirv -profile glsl_460 -o ${SHADER_SPV}
        DEPENDS ${SHADER_SRC}
        COMMENT "Compiling ${SHADER_FILENAME} to ${SHADER_NAME}.spirv..."
        VERBATIM
    )

    list(APPEND ALL_SHADER_SPVS ${SHADER_SPV})
endforeach()

add_custom_target(CompileShaders DEPENDS ${ALL_SHADER_SPVS})
add_dependencies(${PROJECT_NAME} CompileShaders)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:SDL3::SDL3> $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

if(ALL_SHADER_SPVS)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ALL_SHADER_SPVS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()