// Digits.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Digits.h"
#include "FileReader.h"
#include "Net.h"
#include "SDL.h"

const int SCREEN_WIDTH = 784;
const int SCREEN_HEIGHT = 784;

using namespace std;

vector<vector<double>> collectInputData(vector<unsigned char> data, size_t items, size_t size) {
    vector<vector<double>> v;
    for (size_t i = 0; i < items; i++) {
        vector<double> w;
        for (size_t j = 0; j < size; j++) {
            double value = data[(i * size) + j];
            value /= 255.0f;
            w.push_back(value);
        }
        v.push_back(w);
    }
    return v;
}

vector<vector<double>> collectOutputData(vector<unsigned char> data, size_t items) {
    vector<vector<double>> v;
    for (size_t i = 0; i < items; i++) {
        vector<double> w;
        w.resize(10);
        size_t label = size_t(data[i]);
        w[label] = 1.0f;
        v.push_back(w);
    }
    return v;
}

unsigned outputToDigit(vector<double> netOutput) {
    pair<double, unsigned> bestGuess = { 0.0f, 0 };
    for (size_t i = 0; i < netOutput.size(); i++) {
        if (netOutput[i] > bestGuess.first) {
            bestGuess = { netOutput[i], (unsigned)i };
        }
    }
    return bestGuess.second;
}

vector<double> gridToInput(vector<vector<pair<SDL_Rect, double>>> grid) {
    vector<double> r;
    for (size_t i = 0; i < grid.size(); i++) {
        for (size_t j = 0; j < grid[i].size(); j++) {
            r.push_back(grid[j][i].second);
        }
    }
    return r;
}

Net findBestNet(Net initialNet, double targetGrade, vector<vector<double>> trainInputs, vector<vector<double>> trainOutputs, vector<vector<double>> testInputs, vector<vector<double>> testOutputs) {
    
    cout << "Training New Net...\n";
    Net currentNet = initialNet;

    //get node stats
    size_t inputSize = currentNet.getInputSize();
    size_t outputSize = currentNet.getOutputSize();
    vector<size_t> hiddenSizes = currentNet.getHiddenSizes();
    double bias = currentNet.getBiasVal();

    cout << "\tInputSize: " + to_string(inputSize) + '\n';
    cout << "\tOutputSize: " + to_string(outputSize) + '\n';
    cout << "\tHiddenSizes: ";
    for(size_t i = 0; i < hiddenSizes.size(); i++){
        cout << to_string(hiddenSizes[i]);
        if (i != hiddenSizes.size() - 1) {
            cout << ", ";
        }
    }
    cout << '\n';
    cout << "\tLearning Rate: " + to_string(currentNet.getLearningRate()) + '\n';
    cout << "\tBias: " + to_string(bias) + '\n';
    while (true) {
        
        double bestMSE = INFINITY;
        double worstMSE = 0.0f;
        double avgMSE = 0.0f;
        bool nanError = false;
        for (size_t i = 0; i < trainInputs.size(); i++) {
            cout << '\r';
            currentNet.train(trainInputs[i], trainOutputs[i]);
            double mSE = currentNet.meanSquaredError(trainOutputs[i]);
            if (mSE < bestMSE) {
                bestMSE = mSE;
            }
            if (mSE > worstMSE) {
                worstMSE = mSE;
            }
            if (mSE != mSE) {
                nanError = true;
                break;
            }
            avgMSE += (mSE / trainInputs.size());
            cout << "Trained: " + to_string(i + 1) + "/" + to_string(trainInputs.size()) + " MSE: " + to_string(mSE) + " Best MSE: " + to_string(bestMSE) + " Worst MSE: " + to_string(worstMSE) + " Avg MSE: " + to_string(avgMSE);;
        }
        cout << '\n';

        if (nanError) {
            //-nan check
            //lr too high
            cout << "-nan Error! LR is too high! Lowering LR and Restarting...\n";
            double newLR = currentNet.getLearningRate() * 0.5;
            cout << "New LR: " + to_string(newLR) + '\n';
            currentNet.setLearningRate(newLR);
            currentNet.randomizeWeights();
            continue;
        }

        unsigned correct = 0;
        double grade = 0.0f;
        for (size_t i = 0; i < testInputs.size(); i++) {
            cout << '\r';
            currentNet.test(testInputs[i]);
            unsigned guess = outputToDigit(currentNet.getOutputs());
            unsigned actual = outputToDigit(testOutputs[i]);

            if (guess == actual) {
                correct++;
            }
            grade = (double)correct / (double)testInputs.size();
            cout << "Tested: " + to_string(i + 1) + "/" + to_string(testInputs.size()) + " Grade: " + to_string(grade);
        }
        if (grade >= targetGrade) {
            cout << "\nFound a good Net! Returning...\n";
            return currentNet;
        }
        else {
            //grade too low!
            cout << "\nFailed to reach grade! Doing more Training...\n";
        }
    }
}

int main(int argc, char* args[])
{
    srand(unsigned(time(NULL)));
    Net N;

    unsigned option = 0;

    while ((option != 1) && (option != 2)) {
        cout << "Choose an option below and press Enter:\n\t1: Train a new Net\n\t2: Load a Net from a file\n";
        cin >> option;
    }
    if (option == 1) {
        
        cout << "Loading Training Data From Files...\n";
        ImageFile trainImageFile = readImageFile("train-images.idx3-ubyte");
        LabelFile trainLabelFile = readLabelFile("train-labels.idx1-ubyte");

        size_t inputSize = trainImageFile.numberOfCols * trainImageFile.numberOfRows;
        size_t outputSize = 10;

        vector<vector<double>> trainInputs;
        vector<vector<double>> trainOutputs;

        trainInputs = collectInputData(trainImageFile.images, trainImageFile.numberOfImages, trainImageFile.numberOfCols * trainImageFile.numberOfRows);
        trainOutputs = collectOutputData(trainLabelFile.labels, trainLabelFile.numberOfItems);

        ImageFile testImageFile = readImageFile("t10k-images.idx3-ubyte");
        LabelFile testLabelFile = readLabelFile("t10k-labels.idx1-ubyte");

        vector<vector<double>> testInputs;
        vector<vector<double>> testOutputs;

        testInputs = collectInputData(testImageFile.images, testImageFile.numberOfImages, testImageFile.numberOfCols * testImageFile.numberOfRows);
        testOutputs = collectOutputData(testLabelFile.labels, testLabelFile.numberOfItems);

        N = Net(inputSize, outputSize, {}, 1.0f, 1.0f);

        cout << "Specify a target accuracy for the Net in decimal form (0.75 = %75 accuracy)\n(Be aware that Nets with accuracies greater than 0.9 may take a long time to train, if they are even possible)\n";

        double targetGrade;
        cin >> targetGrade;
        N = findBestNet(N, targetGrade, trainInputs, trainOutputs, testInputs, testOutputs);

        string filename;
        cout << "Specify a filename for the new net: \n";
        cin >> filename;
        N.writeFile(filename);
        cout << "Writing Done!\n";
        string answer;
        while (answer != "y" && answer != "n") {
            answer = "";
            cout << "Would you Like to use the new net now? (y/n)\n";
            cin >> answer;
        }
        if (answer == "n") {
            cout << "Goodbye!";
            return 0;
        }
        
    }
    else if (option == 2) {
        string filename;
        cout << "Enter the filename, including extension, of the file you would like to load: \n";
        cin >> filename;
        N = Net(filename);
        cout << "Load Successful!\n";
    }
    

    

    //The window we'll be rendering to
    SDL_Window* window = NULL;

    //The surface contained by the window
    SDL_Surface* screenSurface = NULL;

    //Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }
    //Create window
    window = SDL_CreateWindow("Draw a Digit Between 0 and 9 and Press Enter! (Press C to Clear, Esc to Close)", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (window == NULL)
    {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }
    //Get window surface
    screenSurface = SDL_GetWindowSurface(window);

    const int TILE_WIDTH = SCREEN_WIDTH / 28;
    const int TILE_HEIGHT = SCREEN_HEIGHT / 28;
    vector<vector<pair<SDL_Rect, double>>> drawTiles;

    for (size_t i = 0; i < 28; i++) {
        vector<pair<SDL_Rect, double>> v;
        for (size_t j = 0; j < 28; j++) {
            pair<SDL_Rect, double> p;
            SDL_Rect r;
            r.x = i * TILE_WIDTH;
            r.y = j * TILE_HEIGHT;
            r.w = TILE_WIDTH;
            r.h = TILE_HEIGHT;
            p.first = r;
            p.second = 0.0f;
            v.push_back(p);
        }
        drawTiles.push_back(v);
    }

    SDL_Event e;
    bool quit = false;
    pair<int, int> mousePos;

    unsigned fps = 0;
    time_t secTimer = time(nullptr) + 1;
    while (!quit){
        auto frameTimer = chrono::system_clock::now();

        Uint32 mouseButtons = SDL_GetMouseState(&mousePos.first, &mousePos.second);
        int tileX = mousePos.first / TILE_WIDTH;
        int tileY = mousePos.second / TILE_HEIGHT;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
            if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_KP_ENTER:
                        {
                        vector<double> input = gridToInput(drawTiles);
                        N.test(input);
                        unsigned guess = outputToDigit(N.getOutputs());
                        SDL_SetWindowTitle(window, to_string(guess).c_str());
                        }
                        break;
                    case SDLK_ESCAPE:
                        quit = true;
                        break;
                    case SDLK_c:
                        for (size_t i = 0; i < drawTiles.size(); i++) {
                            for (size_t j = 0; j < drawTiles[i].size(); j++) {
                                drawTiles[i][j].second = 0.0f;
                            }
                        }
                }
            }
        }
        if (mouseButtons & SDL_BUTTON_LMASK) {
            drawTiles[tileX][tileY].second = 1.0f;
            vector<pair<int, int>> brush;
            brush.push_back(pair<int, int>({ tileX - 1, tileY }));
            brush.push_back(pair<int, int>({ tileX + 1, tileY }));
            brush.push_back(pair<int, int>({ tileX, tileY - 1 }));
            brush.push_back(pair<int, int>({ tileX, tileY + 1 }));
            for (size_t i = 0; i < brush.size(); i++) {
                auto tile = brush[i];
                drawTiles[tile.first][tile.second].second += 0.1;
                if (drawTiles[tile.first][tile.second].second > 1.0f) {
                    drawTiles[tile.first][tile.second].second = 1.0f;
                }
            }
        }

        SDL_FillRect(screenSurface, NULL, SDL_MapRGB(screenSurface->format, 0x00, 0x00, 0x00));

        for (size_t i = 0; i < drawTiles.size(); i++) {
            for (size_t j = 0; j < drawTiles[i].size(); j++) {
                int intensity = int(drawTiles[i][j].second * 255.0f);
                SDL_FillRect(screenSurface, &(drawTiles[i][j]).first, SDL_MapRGB(screenSurface->format, intensity, intensity, intensity));
            }
        }


        SDL_UpdateWindowSurface(window);
        fps++;
        if (time(nullptr) >= secTimer) {
            cout << fps << '\n';
            secTimer = time(nullptr) + 1;
            fps = 0;
        }

        auto frameDur = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - frameTimer);
        float millis = frameDur.count();
        float target = (1.0f / 60.0f) * 1000.0f;
        //cout << milli << "," << target << '\n';
        if (millis < target) {
            this_thread::sleep_for(chrono::milliseconds(int(target - millis)));
        }
    }


    //Destroy window
    SDL_DestroyWindow(window);

    //Quit SDL subsystems
    SDL_Quit();
    
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
