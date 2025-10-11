cmake_minimum_required(VERSION 3.19)
project(helios LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
	set(SHADER_NAME test_kernel)
	set(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/MetalTest)
	set(SHADER_FILE ${WORKING_DIRECTORY}/${SHADER_NAME}.metal)
	set(AIR_FILE ${WORKING_DIRECTORY}/${SHADER_NAME}.air)
	set(METALLIB_FILE ${WORKING_DIRECTORY}/${SHADER_NAME}.metallib)

	# Compiling .metal shader to .air
	add_custom_command(
		OUTPUT ${AIR_FILE}
		DEPENDS ${SHADER_FILE}
		COMMAND xcrun -sdk macosx metal
			-c ${SHADER_FILE}
			-o ${AIR_FILE}
		WORKING_DIRECTORY ${WORKING_DIRECTORY}
	)

	# Linking .air into .metallib
	add_custom_target(
		compile_shaders
		DEPENDS ${AIR_FILE}
		COMMAND xcrun -sdk macosx metallib
			${AIR_FILE}
			-o ${METALLIB_FILE}
		WORKING_DIRECTORY ${WORKING_DIRECTORY}
	)
	
	# Create the metal executable and link it with the .metallib dependency
	add_executable(metal_exec src/Executors/MetalExecutor.mm)
	set_property(TARGET metal_exec PROPERTY LANGUAGE OBJCXX)
	target_compile_options(metal_exec PRIVATE -F${CMAKE_OSX_SYSROOT}/System/Library/Frameworks)
	target_include_directories(metal_exec PUBLIC inc)
	add_dependencies(metal_exec compile_shaders)

	# Link the Metal and Foundation libraries
	find_library(METAL_FRAMEWORK Metal)
	find_library(FOUNDATION_FRAMEWORK Foundation)

	target_link_libraries(metal_exec
	    ${METAL_FRAMEWORK}
	    ${FOUNDATION_FRAMEWORK}
	    objc
	)

	# Copying ${SHADER_NAME}.metallib to executable directory
	add_custom_command(
		TARGET metal_exec POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy
			${METALLIB_FILE}
			$<TARGET_FILE_DIR:metal_exec>
	)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(pool_test src/ThreadPool/ThreadPool.cpp tests/thread_pool_tests.cpp)
target_include_directories(pool_test PUBLIC inc)

target_link_libraries(
  pool_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(pool_test)
