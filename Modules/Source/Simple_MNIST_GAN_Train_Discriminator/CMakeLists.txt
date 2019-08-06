cmake_minimum_required( VERSION 3.13.2 )

file(
    GLOB_RECURSE
    source_list
    RELATIVE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "Simple_MNIST_GAN_Train_Discriminator/Include/*"
    "Simple_MNIST_GAN_Train_Discriminator/Source/*"
)

add_executable( Simple_MNIST_GAN_Train_Discriminator ${source_list} )

target_include_directories(
    Simple_MNIST_GAN_Train_Discriminator
    PUBLIC
    ${ANNMNISTNG_TINYDNN_INCLUDE_PATH}
    "Simple_MNIST_GAN_Train_Discriminator/Include/"
    PRIVATE
    "Simple_MNIST_GAN_Train_Discriminator/Source/"
)

# For MSVC parse pretty filter names
foreach(source IN LISTS source_list)
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach()
