#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "data_structure.h"
#include "load.h"
#include "kernel.h"
#include "config.h"

static DDR4_format DDR4;               
static RAB_format RAB;
static registers_format REG;
 
int main()
{
    
    srand(1);
    for (int t = 0; t < N_TOKEN; t++)
        for (int i = 0; i < N_HC; i++)
            for (int j = 0; j < D_MODEL; j++)
                DDR4.i_x[t][i][j] = fp32_to_bf16((float)(rand() % 100) / 10.0f);
 
    for (int i = 0; i < N_TOKEN; i += RAB_TOKENS) {
        load_gl_x_i(&DDR4, &RAB, i);                      
        memcmp(RAB.i_x, DDR4.i_x + i, sizeof RAB.i_x);
 
        for (int t = 0; t < RAB_TOKENS; t++) {
            int tok = i + t;
            load_lc_x_i(&RAB, &REG, t);                     
            flatten(&REG);                                  
            rmsnorm(&REG);                             
 
            memcmp(REG.x_i, DDR4.i_x[tok], sizeof REG.x_i);
            for (int i = 0; i < N_HC; i++)
                for (int j = 0; j < D_MODEL; j++)
                    REG.x_i_flatten[i * D_MODEL + j] == bf16_to_fp32(DDR4.i_x[tok][i][j]);
 
            double ss = 0.0;                                 
            for (int k = 0; k < N_FLAT; k++) ss += (double)REG.y_i[k] * REG.y_i[k];
            double mean = ss / N_FLAT;
            double rms = sqrt(mean);
        }
    }
    for (int i=0; i<N_HC * D_MODEL; i++) {
        printf ("%.6f ", REG.y_i[i]);
    }
    printf("\n");
    return 0;
}