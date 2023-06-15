#pragma once
#include "Digits.h"

bool isLittleEndian() {
    unsigned int x = 1;
    char* cptr = (char*)&x;
    if (*cptr) {
        return true;
    }
    return false;
}

uint32_t swapBytes(uint32_t num) {
    uint32_t n1 = num << 24;
    uint32_t n2 = num << 8;
    n2 = n2 & 0x00FF0000;
    uint32_t n3 = num >> 8;
    n3 = n3 & 0x0000FF00;
    uint32_t n4 = num >> 24;
    return n1 | n2 | n3 | n4;
}

class LabelFile {
public:
    uint32_t magicNumber;
    uint32_t numberOfItems;
    vector<unsigned char> labels;
    LabelFile()
    {
        magicNumber = 0;
        numberOfItems = 0;
        labels = vector<unsigned char>();
    };
};

class ImageFile {
public:
    uint32_t magicNumber;
    uint32_t numberOfImages;
    uint32_t numberOfRows;
    uint32_t numberOfCols;
    vector<unsigned char> images;
    ImageFile()
    {
        magicNumber = 0;
        numberOfImages = 0;
        numberOfRows = 0;
        numberOfCols = 0;
        images = vector<unsigned char>();
    }
};

LabelFile readLabelFile(string filename) {
    LabelFile file = LabelFile();
    ifstream labelFile(filename, ios_base::in | ios_base::binary);
    if (!labelFile.is_open()) {
        cout << "Could not open label file:" << filename.c_str() << '\n';
        return file;
    }
    labelFile.seekg(0, ios::beg);
    char magic[sizeof(uint32_t)];
    labelFile.read(magic, sizeof(uint32_t));
    memcpy(&file.magicNumber, &magic, sizeof(magic));
    if (isLittleEndian()) {
        file.magicNumber = swapBytes(file.magicNumber);
    }

    char numItems[sizeof(uint32_t)];
    labelFile.read(numItems, sizeof(uint32_t));
    memcpy(&file.numberOfItems, &numItems, sizeof(numItems));
    if (isLittleEndian()) {
        file.numberOfItems = swapBytes(file.numberOfItems);
    }

    for (size_t i = 0; i < file.numberOfItems; i++) {
        char label[sizeof(char)];
        labelFile.read(label, sizeof(char));
        file.labels.push_back(label[0]);
    }

    labelFile.close();
    return file;
}

ImageFile readImageFile(string filename) {
    ImageFile file = ImageFile();
    ifstream imageFile(filename, ios_base::in | ios_base::binary);
    if (!imageFile.is_open()) {
        cout << "Could not open label file:" << filename.c_str() << '\n';
        return file;
    }
    imageFile.seekg(0, ios::beg);

    char magic[sizeof(uint32_t)];
    imageFile.read(magic, sizeof(uint32_t));
    memcpy(&file.magicNumber, &magic, sizeof(uint32_t));
    if (isLittleEndian()) {
        file.magicNumber = swapBytes(file.magicNumber);
    }

    char numImages[sizeof(uint32_t)];
    imageFile.read(numImages, sizeof(uint32_t));
    memcpy(&file.numberOfImages, &numImages, sizeof(uint32_t));
    if (isLittleEndian()) {
        file.numberOfImages = swapBytes(file.numberOfImages);
    }

    char rows[sizeof(uint32_t)];
    imageFile.read(rows, sizeof(uint32_t));
    memcpy(&file.numberOfRows, &rows, sizeof(uint32_t));
    if (isLittleEndian()) {
        file.numberOfRows = swapBytes(file.numberOfRows);
    }

    char cols[sizeof(uint32_t)];
    imageFile.read(cols, sizeof(uint32_t));
    memcpy(&file.numberOfCols, &rows, sizeof(uint32_t));
    if (isLittleEndian()) {
        file.numberOfCols = swapBytes(file.numberOfCols);
    }

    size_t buffSize = file.numberOfImages * file.numberOfCols * file.numberOfRows;
    file.images.resize(buffSize);
    imageFile.read((char*)file.images.data(), buffSize);
    imageFile.close();
    return file;
}