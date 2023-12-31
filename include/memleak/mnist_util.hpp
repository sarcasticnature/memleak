#include <fstream>
#include <cstdint>
#include <vector>

#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"

namespace memleak
{

// Modified from https://stackoverflow.com/a/33384846

uint32_t swap_endian(uint32_t x)
{
    return (x << 24) | (x << 8 & 0x00FF0000) | (x >> 8 & 0x0000FF00) | (x >> 24 & 0x000000FF);
}

xt::xarray<uint8_t> read_mnist_images(const std::string& path)
{
    xt::xarray<uint8_t> output;
    if (std::ifstream file(path, std::ios::binary); file) {
        uint32_t magic, image_cnt, rows, cols;

        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (swap_endian(magic) != 2051) {
            return output;
        }

        file.read(reinterpret_cast<char*>(&image_cnt), sizeof(image_cnt));
        image_cnt = swap_endian(image_cnt);
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        rows = swap_endian(rows);
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        cols = swap_endian(cols);

        uint32_t image_size = rows * cols;
        uint8_t* image;
        std::vector<size_t> shape = {image_cnt, image_size};
        output.resize(shape);
        shape = {image_size};

        for (size_t i = 0; i < image_cnt; ++i) {
            image = new uint8_t[image_size];
            file.read(reinterpret_cast<char*>(image), image_size);
            auto row = xt::row(output, i);
            row = xt::adapt(image, image_size, xt::acquire_ownership(), shape);
        }
    }

    return output;
}

xt::xarray<uint8_t> read_mnist_labels(const std::string& path)
{
    xt::xarray<uint8_t> output;
    if (std::ifstream file(path, std::ios::binary); file) {
        uint32_t magic, image_cnt;

        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (swap_endian(magic) != 2049) {
            return output;
        }

        file.read(reinterpret_cast<char*>(&image_cnt), sizeof(image_cnt));
        image_cnt = swap_endian(image_cnt);

        std::vector<size_t> shape = {image_cnt, 1};
        output.resize(shape);
        shape = {image_cnt};

        uint8_t* labels = new uint8_t[image_cnt];
        file.read(reinterpret_cast<char*>(labels), image_cnt);
        xt::col(output, 0) = xt::adapt(labels, image_cnt, xt::acquire_ownership(), shape);

        //for (size_t i = 0; i < image_cnt; ++i) {
        //    image = new uint8_t[image_size];
        //    file.read(reinterpret_cast<char*>(image), image_size);
        //    auto row = xt::row(output, i);
        //    row = xt::adapt(image, image_size, xt::acquire_ownership(), shape);
        //}
    }

    return output;
}

} // namespace memleak
