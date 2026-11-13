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
 
int main(void)
{
    /* gan gia tri truc tiep vao DDR4.i_x */
    srand(1);
    for (int t = 0; t < N_TOKEN; t++)
        for (int i = 0; i < N_HC; i++)
            for (int j = 0; j < D_MODEL; j++)
                DDR4.i_x[t][i][j] = fp32_to_bf16((float)(rand() % 100) / 10.0f);
 
    for (int base = 0; base < N_TOKEN; base += RAB_TOKENS) {
        load_gl_x_i(&DDR4, &RAB, base);                      
        assert(memcmp(RAB.i_x, DDR4.i_x + base, sizeof RAB.i_x) == 0);
 
        for (int t = 0; t < RAB_TOKENS; t++) {
            int tok = base + t;
            load_lc_x_i(&RAB, &REG, t);                     
            flatten(&REG);                                  
            rmsnorm(&REG);                             
 
            assert(memcmp(REG.x_i, DDR4.i_x[tok], sizeof REG.x_i) == 0);
            for (int i = 0; i < N_HC; i++)
                for (int j = 0; j < D_MODEL; j++)
                    assert(REG.x_i_flatten[i * D_MODEL + j] == bf16_to_fp32(DDR4.i_x[tok][i][j]));
 
            double ss = 0.0;                                 /* gamma = 1 -> mean(y^2) ~ 1 */
            for (int k = 0; k < N_FLAT; k++) ss += (double)REG.y_i[k] * REG.y_i[k];
            assert(fabs(ss / N_FLAT - 1.0) < 1e-4);
        }
    }
 
    assert(bf16_to_fp32(fp32_to_bf16(16.45f)) == 16.375f);   /* 16.45 -> 16.375 do cat bit */
    for (int i=0; i<N_HC * D_MODEL; i++) {
        printf ("%.6f ", REG.y_i[i]);
    }
    //printf("y_i[0..3] cua token cuoi = %f %f %f %f\n", REG.y_i[0], REG.y_i[1], REG.y_i[2], REG.y_i[3]);
    return 0;
}