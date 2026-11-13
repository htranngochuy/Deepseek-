#include "kernel.h"

void flatten(registers_format *reg) {
    for (int i = 0; i < N_HC; i++)
        for (int j = 0; j < D_MODEL; j++)
            reg->x_i_flatten[i * D_MODEL + j] = bf16_to_fp32(reg->x_i[i][j]);
}

void rmsnorm(registers_format *reg) {
    float sum = 0.0f;
    for (int i = 0; i < N_HC * D_MODEL; i++) {
        float val = reg->x_i_flatten[i];
        sum += val * val;
    }
    reg->sum_xi_squared = sum; 
    float mean = sum / (N_HC * D_MODEL) + EPS;
    float rms = sqrt(mean); 
    for (int i = 0; i < N_HC * D_MODEL; i++) {
        float val = reg->x_i_flatten[i];
        reg->y_i[i] = val / rms; 
    }
}