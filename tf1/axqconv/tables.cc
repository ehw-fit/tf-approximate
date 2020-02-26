/***
 * This program generates a big JSON file with all outputs for visualisation of multipliers
 **/
#include <cstdio>
#include <iostream>

#include "approximate_selector.h"
using namespace tensorflow;


void PrintOut(ApproximateSelector & ax) {

    std::cout << "[";
    for(int i = 0; i < 256; i++)
        for(int w = 0; w < 256; w++) {
            if(i || w)
                std::cout << ", ";
            std::cout << ax.multiplicate(i, w);

        }
    std::cout << "]";

}

int main(int argc, char ** argv) {
    std::cout << "{" << std::endl;
    for (std::string mult; std::getline(std::cin, mult);) {
        std::cout << "  \"" << mult << "\": { "  << std::endl;

        ApproximateSelector untuned;
        untuned.init(mult.c_str(), false);
        std::cout << "    \"untuned\": ";
        PrintOut(untuned);
        std::cout << "," << std::endl;


        ApproximateSelector tunedsw;
        tunedsw.init(mult.c_str(), false, true);
        std::cout << "    \"tunedsw\": ";
        PrintOut(tunedsw);
        std::cout << "," << std::endl;

        ApproximateSelector tuned;
        tuned.init(mult.c_str(), true);
        std::cout << "    \"tuned\": ";
        PrintOut(tuned);
        std::cout << std::endl;

        std::cout << "  }," << std::endl;
    }

    std::cout << "\"none\" : []}" << std::endl;

    return 0;
}