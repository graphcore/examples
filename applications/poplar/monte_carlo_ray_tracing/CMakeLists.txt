cmake_minimum_required(VERSION 3.10)

project(ipu_trace)

# Build the CPU MC ray tracer
# sub-project for reference:
add_subdirectory("light")

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -Werror -Wno-unused-parameter -Wno-ignored-qualifiers)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})

file(GLOB IPU_TRACE_SRC ${CMAKE_SOURCE_DIR}/src/*.hpp ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(ipu_trace ${IPU_TRACE_SRC} codelets.gp)
target_link_libraries(ipu_trace Boost::program_options ${OpenCV_LIBS} exr -lpopops -lpoputil -lpoprand -lpoplar OpenMP::OpenMP_CXX -lpthread -lpvti)

file(GLOB LIGHT_SRC ${PROJECT_SOURCE_DIR}/light/src/*.hpp ${PROJECT_SOURCE_DIR}light/src/*.cpp)
add_custom_command(
    PRE_BUILD
    MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/src/codelets/codelets.cpp
    DEPENDS ${LIGHT_SRC}
    COMMAND popc -O3 -Wdouble-promotion -I${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/codelets/codelets.cpp -o codelets.gp
    OUTPUT codelets.gp
    WORKING_DIRECTORY ${CMAKE_BUILD_DIR}
)

file(GLOB EXRTOOL_SRC ${CMAKE_SOURCE_DIR}/src/exr/*.hpp ${CMAKE_SOURCE_DIR}/src/exr/*.cpp)
add_executable(exrtool ${EXRTOOL_SRC})
target_link_libraries(exrtool Boost::program_options ${OpenCV_LIBS} exr OpenMP::OpenMP_CXX -lpthread)
