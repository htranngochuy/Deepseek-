#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
 
        for (int t = 0; t < RAB_TOKENS; t++) {
            load_lc_x_i(&RAB, &REG, t);                     
            flatten(&REG);                                  
            rmsnorm(&REG);                             
        }
    }
    for (int i=0; i<4; i++) {
        printf ("%.6f ", REG.y_i[i]);
    }
    printf("\n");
    return 0;
}