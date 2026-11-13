#ifndef DATA_STRUCTURE_H
#define DATA_STRUCTURE_H

#include <stdint.h>
#include "data_type.h" 
#include "config.h"

typedef struct {
    bf16 i_x[N_TOKEN][N_HC][D_MODEL];
} DDR4_format;

typedef struct {
    bf16 i_x[RAB_TOKENS][N_HC][D_MODEL];
} RAB_format;

typedef struct {
    bf16 x_i[N_HC][D_MODEL]; 
    float x_i_flatten[N_HC * D_MODEL];
    float sum_xi_squared;
    float y_i[N_HC * D_MODEL];
} registers_format;

#endif 
