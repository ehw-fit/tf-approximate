#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>

#include "axmult.h"


namespace tensorflow {
class ApproximateSelector {

public:
    ApproximateSelector() {
        bw_ = 0;
        ax_table_ = NULL;
    }

    void init(const char * operation_name, bool tune, bool swap = false ) {
        bw_ = 8;
        ax_table_ = new int32_t[1 <<(2*bw_)];

        int32_t * tmpaxtable;
        int maxval = 1 << bw_;
        
        if(tune)
            tmpaxtable = new int32_t[1<<(2*bw_)];
        else
            tmpaxtable = ax_table_;
        

        int mulid = AxFindId(operation_name);

        if(mulid < 0) {
            char buff[512];
            snprintf(buff, 512, "Unknown multiplier \"%s\"", operation_name);
            throw std::invalid_argument(buff);
        }

        for(int a = 0; a < maxval; a++) {
            for(int b = 0; b < maxval; b++) {
                //tmpaxtable[a * 256 + b] = mul8u_Y48(a, b);
                if(swap) 
                    tmpaxtable[a * maxval + b] = AxDo(mulid, b, a);
                else
                    tmpaxtable[a * maxval + b] = AxDo(mulid, a, b);

            }

        }

        if(tune) {

            for(int w = 0; w < maxval; w++) {
                int wpos = -1;
                int werr = 0;
                int origerr = 0;
                
                for(int w2 = 0; w2 < maxval; w2++) {
                    int sumerr = 0;
                    for(int i = 0; i < maxval; i++) {
                        sumerr += abs(i * w - tmpaxtable[i * maxval + w2]);
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
                for(int i = 0; i < maxval; i++) {
                    ax_table_[i * maxval + w] = tmpaxtable[i * maxval + wpos];
                }
        
        #if 0
                if(wpos != w)
                    std::cerr << "mapping " << w << " -> " << wpos << " ( err = " << (werr / (float)maxval) << " )" << " ( orig = " << (origerr / (float)maxval) << " )"<< std::endl;
        #endif
            }


            delete[] tmpaxtable;
        }
    }

    ~ApproximateSelector()
    {
        if(ax_table_)
            delete[] ax_table_;
        ax_table_ = NULL;
    }

    int32_t multiplicate(int32_t data, int32_t weight) {
        return ax_table_[data * (1 << bw_) + weight];
    }

    private:
    int32_t * ax_table_;
    int bw_;
};

}