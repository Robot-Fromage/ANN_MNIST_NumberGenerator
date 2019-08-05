cmake_minimum_required( VERSION 3.13.2 )

file(
    GLOB_RECURSE
    source_list
    RELATIVE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "Basic_MNIST_Test/Include/*"
    "Basic_MNIST_Test/Source/*"
)

add_executable( Basic_MNIST_Test ${source_list} )

target_include_directories(
    Basic_MNIST_Test
    PUBLIC
    ${ANNMNISTNG_TINYDNN_INCLUDE_PATH}
    "Basic_MNIST_Test/Include/"
    PRIVATE
    "Basic_MNIST_Test/Source/"
)

# For MSVC parse pretty filter names
foreach(source IN LISTS source_list)
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach()
