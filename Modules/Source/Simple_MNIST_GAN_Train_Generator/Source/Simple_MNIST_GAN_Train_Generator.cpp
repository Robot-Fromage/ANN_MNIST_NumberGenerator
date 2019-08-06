/*************************************************************************
*
*   ANN_MNIST_NumberGenerator
*__________________
*
* Simple_MNIST_GAN_Train_Generator/main.cpp
* Layl
* Please refer to LICENSE.md
*/


#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

int main()
{
    // This approach seems clumsy but is quite simple to implement

    // Uniform random vec_t:
    // tiny_dnn::vec_t in( 100 );
    // tiny_dnn::uniform_rand(in.begin(), in.end(), 0, 1);

    // Traverse layers:
    // for (layer* l : net) {
    //     std::cout << l->layer_type() << std::endl;
    // }

    // Freeze layer:
    // net[1]->set_trainable(false);

    // load MNIST dataset
    std::vector<tiny_dnn::label_t> train_labels, test_labels;
    std::vector<tiny_dnn::vec_t> train_images, test_images;

    //std::string data_dir_path = "Resources/Datasets/MNIST";
    //tiny_dnn::parse_mnist_labels( data_dir_path + "/train-labels.idx1-ubyte", &train_labels );
    //tiny_dnn::parse_mnist_images( data_dir_path + "/train-images.idx3-ubyte", &train_images, 0.0, 1.0, 0, 0 );
    //tiny_dnn::parse_mnist_labels( data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels );
    //tiny_dnn::parse_mnist_images( data_dir_path + "/t10k-images.idx3-ubyte", &test_images, 0.0, 1.0, 0, 0 );

    // Load Discriminator from Basic MNIST model
    tiny_dnn::network<tiny_dnn::sequential> discriminator;

    std::string model_dir_path = "Resources/TrainedModels";
    discriminator.load( model_dir_path + "/BasicModel" );

    // Build Generator model
    // specify loss-function and learning strategy
    tiny_dnn::network<tiny_dnn::sequential> generator;
    tiny_dnn::adam optimizer;
    tiny_dnn::core::backend_t backend = tiny_dnn::core::backend_t::internal;

    generator   << tiny_dnn::layers::fc( 100, 6272, true, backend )     // F1, 100-in, 6272-out
                << tiny_dnn::activation::leaky_relu()

                << tiny_dnn::layers::deconv( 7, 7, 5, 128, 128, tiny_dnn::padding::same, true, 1, 1, backend ) // D2, 6272-in, 128@7x7-out
                << tiny_dnn::activation::leaky_relu()
                << tiny_dnn::layers::ave_unpool( 7, 7, 128, 2 ) // U3, 128@7x7-in, 128@14x14-out
                << tiny_dnn::activation::leaky_relu()

                << tiny_dnn::layers::deconv( 14, 14, 5, 128, 128, tiny_dnn::padding::same, true, 1, 1, backend ) // D4, 128@14x14-in, 128@14x14-out
                << tiny_dnn::activation::leaky_relu()
                << tiny_dnn::layers::ave_unpool( 14, 14, 128, 2 ) // U5, 128@14x14-in, 128@28x28-out
                << tiny_dnn::activation::leaky_relu()

                << tiny_dnn::layers::conv( 28, 28, 5, 128, 1, tiny_dnn::padding::same, true, 1, 1, 1, 1, backend ); // D6, 128@28x28-in, 784-out

    std::cout << "start training" << std::endl;

    // Compose networks
    // Freeze discriminator
    for( tiny_dnn::layer* l : discriminator )
        l->set_trainable( false );

    tiny_dnn::network<tiny_dnn::sequential> GAN;
    for( tiny_dnn::layer* l : generator )
        GAN << *l;

    for( tiny_dnn::layer* l : discriminator )
        GAN << *l;

    tiny_dnn::progress_display disp(train_images.size());
    tiny_dnn::timer t;

    int epoch = 1;
    double learning_rate = 1.0;
    int n_train_epochs = 50;
    int n_minibatch = 200;

    // create callback
    auto on_enumerate_epoch = [&]() {
        std::cout   << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
                    << t.elapsed() << "s elapsed." << std::endl;
        ++epoch;
        tiny_dnn::result res = GAN.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };


    for( tiny_dnn::layer* l : GAN  )
    {
        std::cout << "Type: " << l->layer_type() << ", In: " << l->in_size() << ", Out: " << l->out_size() <<  ", Frozen: " << ( l->trainable() ? "No" : "Yes" ) << std::endl;
    }
    // training
    /*
    GAN.train<tiny_dnn::mse>( optimizer, train_images, train_labels, n_minibatch,
                              n_train_epochs, on_enumerate_minibatch,
                              on_enumerate_epoch);
    */

    std::cout << "end training." << std::endl;

    // test and show results
    //GAN.test(test_images, test_labels).print_detail(std::cout);
    // save network model & trained weights
    //GAN.save( "BasicModel" );

    return  0;
}