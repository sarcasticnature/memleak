#include <iostream>

#include "xtensor/xio.hpp"

#include "memleak/mnist_util.hpp"

int main()
{
    std::cout << "Hello MNIST" << std::endl;
    auto imgs = memleak::read_mnist_images("data/train-images.idx3-ubyte");
    auto labels = memleak::read_mnist_labels("data/train-labels.idx1-ubyte");

    std::cout << imgs << std::endl;
    std::cout << labels << std::endl;

    return 0;
}
