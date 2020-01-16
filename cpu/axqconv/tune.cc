/** 
 * This tool illustrates the resulting mapping function for a given multiplier
 * */
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <sstream>
#include <string>

#include "axmult.h"



int main(int argc, char ** argv) {
    
    int axtable[256 * 256];
    for(int a = 0; a < 256; a++) {
        for(int b = 0; b < 256; b++) {
            //axtable[a * 256 + b] = mul8u_Y48(a, b);
            axtable[a * 256 + b] = mul8u_7C1(a, b); // MAE = 20%
        }
    }

    std::cout << "Table prefiled" << std::endl;

    int naxtable[256 * 256];

    std::stringstream maplist;
    int prev_map = -1;
    for(int w = 0; w < 256; w++) {
        int wpos = -1;
        int werr = 0;
        int origerr = 0;
        
        for(int w2 = 0; w2 < 256; w2++) {
            int sumerr = 0;
            for(int i = 0; i < 256; i++) {
                sumerr += abs(i * w - axtable[i * 256 + w2]);
            }

            if(sumerr < werr || (werr == sumerr && w2 == w) || wpos < 0) {
                werr = sumerr;
                wpos = w2;
            }

            if(w == w2) {
                origerr = sumerr;
            }
        }

        // copy to new table
        
        for(int i = 0; i < 256; i++) {
            naxtable[i * 256 + w] = axtable[i * 256 + wpos];
        }

        

        if(wpos != w) {
            if (prev_map == wpos) {
                std::cout << ", " << w; // << " ( err = " << (werr / 256.0) << " )" << " ( orig = " << (origerr / 256.0) << " )"<< std::endl;
            }
            else {

                std::cout << "->" << prev_map << std::endl <<w; // << " ( err = " << (werr / 256.0) << " )" << " ( orig = " << (origerr / 256.0) << " )"<< std::endl;

            }
            prev_map = wpos;
        }
    }
        std::cout << "->" << prev_map << std::endl; // << " ( err = " << (werr / 256.0) << " )" << " ( orig = " << (origerr / 256.0) << " )"<< std::endl;
        std::cout << std::endl;

    return 0;
}
