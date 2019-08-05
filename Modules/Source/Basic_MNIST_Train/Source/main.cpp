/*************************************************************************
*
*   ANN_MNIST_NumberGenerator
*__________________
*
* Basic_MNIST_Train/main.cpp
* Layl
* Please refer to LICENSE.md
*/


#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

#ifndef NDEBUG
#define CONFIG_SUFFIX d
#else
#define CONFIG_SUFFIX
#endif
#define STRINGIZE( i ) #i

int main()
{
    // Uniform random vec_t:
    // tiny_dnn::vec_t in( 100 );
    // tiny_dnn::uniform_rand(in.begin(), in.end(), 0, 1);

    // load MNIST dataset
    std::vector<tiny_dnn::label_t> train_labels, test_labels;
    std::vector<tiny_dnn::vec_t> train_images, test_images;

    std::string data_dir_path = "Resources/Datasets/MNIST";
    tiny_dnn::parse_mnist_labels( data_dir_path + "/train-labels.idx1-ubyte", &train_labels );
    tiny_dnn::parse_mnist_images( data_dir_path + "/train-images.idx3-ubyte", &train_images, 0.0, 1.0, 0, 0 );
    tiny_dnn::parse_mnist_labels( data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels );
    tiny_dnn::parse_mnist_images( data_dir_path + "/t10k-images.idx3-ubyte", &test_images, 0.0, 1.0, 0, 0 );

    // specify loss-function and learning strategy
    tiny_dnn::network<tiny_dnn::sequential> nn;
    tiny_dnn::gradient_descent optimizer;
    tiny_dnn::core::backend_t backend = tiny_dnn::core::backend_t::internal;

    nn  << tiny_dnn::layers::fc( 784, 15, true, backend )
        << tiny_dnn::activation::sigmoid()
        << tiny_dnn::layers::fc( 15, 10, true, backend )
        << tiny_dnn::activation::sigmoid();

    std::cout << "start training" << std::endl;

    tiny_dnn::progress_display disp(train_images.size());
    tiny_dnn::timer t;

    int epoch = 1;
    double learning_rate = 3.0;
    int n_train_epochs = 30;
    int n_minibatch = 10;

    // create callback
    auto on_enumerate_epoch = [&]() {
        std::cout   << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
                    << t.elapsed() << "s elapsed." << std::endl;
        ++epoch;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

    // training
    nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, n_minibatch,
                            n_train_epochs, on_enumerate_minibatch,
                            on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);
    // save network model & trained weights
    std::string suffix = STRINGIZE( CONFIG_SUFFIX );
    nn.save( "BasicModel" + suffix );

    return  0;
}