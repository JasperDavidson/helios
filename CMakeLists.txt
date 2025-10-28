cmake_minimum_required(VERSION 3.19)
project(helios LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(Helios_Core STATIC src/DataManager.cpp src/Tasks.cpp)
target_include_directories(Helios_Core PUBLIC inc)

add_library(Helios_ThreadPool STATIC src/ThreadPool/ThreadPool.cpp)
target_include_directories(Helios_ThreadPool PUBLIC inc)
target_link_libraries(Helios_ThreadPool PUBLIC Helios_Core)

if(APPLE)
	set(SHADER_SOURCE ${CMAKE_SOURCE_DIR}/src/MetalTest/test_kernel.metal)
	set(SHADER_IR ${CMAKE_SOURCE_DIR}/src/MetalTest/test_kernel.air)
	set(METALLIB_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_kernel.metallib)

	# Converting metal shaders to IR
	add_custom_command(
		OUTPUT ${SHADER_IR}
		COMMAND xcrun -sdk macosx metal -c ${SHADER_SOURCE} -o ${SHADER_IR}
		DEPENDS ${SHADER_IR}
		COMMENT "Converting from kernel code to IR"
	)
	add_custom_target(ir_shaders DEPENDS ${SHADER_IR})

	# Create the metal executable and link it with the .metallib dependency
	add_custom_command(
		OUTPUT ${METALLIB_OUTPUT}
		COMMAND xcrun -sdk macosx metallib -c ${SHADER_IR} -o ${METALLIB_OUTPUT}
		DEPENDS ${SHADER_SOURCE}
		COMMENT "Compiling metal shaders"
	)
	add_custom_target(compile_shaders DEPENDS ${METALLIB_OUTPUT})

	# Define the metal library
	add_library(Helios_MetalExecutor STATIC src/Executors/MetalExecutor.mm)
	target_include_directories(Helios_MetalExecutor PUBLIC inc)
	target_link_libraries(Helios_MetalExecutor PUBLIC Helios_Core)
	add_dependencies(Helios_MetalExecutor compile_shaders)

	# Link the Metal and Foundation libraries
	find_library(METAL_FRAMEWORK Metal REQUIRED)
	find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

	target_link_libraries(
	    Helios_MetalExecutor
	    PRIVATE
	    ${METAL_FRAMEWORK}
	    ${FOUNDATION_FRAMEWORK}
	)
endif()

add_library(Helios_Engine STATIC src/Scheduler.cpp src/Runtime.cpp)
target_include_directories(Helios_Engine PUBLIC inc)
target_link_libraries(Helios_Engine PUBLIC Helios_Core Helios_ThreadPool)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(pool_test tests/thread_pool_tests.cpp)
target_include_directories(pool_test PUBLIC inc)

target_link_libraries(
  pool_test
  Helios_ThreadPool
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(pool_test)
