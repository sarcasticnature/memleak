#include <iostream>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

#include "memleak/mnist_util.hpp"

int main()
{
    std::cout << "Hello MNIST" << std::endl;
    xt::xarray<uint8_t> raw_train = memleak::read_mnist_images("data/train-images.idx3-ubyte");
    xt::xarray<uint8_t> raw_labels = memleak::read_mnist_labels("data/train-labels.idx1-ubyte");

    xt::xarray<double> train = memleak::normalize_mnist(raw_train);
    xt::xarray<double> train_labels = memleak::onehot_encode(raw_labels);

    std::cout << train << std::endl;
    std::cout << train_labels << std::endl;

    return 0;
}
