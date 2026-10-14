#include <stdio.h> 
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define N_hc 4 
#define EPS 1e-6f 
#define D_MODEL 4096 


typedef struct {
    uint16_t bits; 
} bf16; 


float bf16_to_fp32(bf16 b) {
    uint32_t sign     = (b.bits >> 15);
    uint32_t exponent = (b.bits >> 7) & 0xFF;
    uint32_t mantissa = b.bits & 0x7F;
 
    uint32_t mant = mantissa << 16;  
    uint32_t fp32_bits = (sign << 31) | (exponent << 23) | mant;
    float result;
    memcpy(&result, &fp32_bits, sizeof(result));
    return result;
}
 

bf16 fp32_to_bf16(float f) {
    uint32_t fp32_bits;
    memcpy(&fp32_bits, &f, sizeof(fp32_bits));
 
    uint32_t sign     = (fp32_bits >> 31);
    uint32_t exponent = (fp32_bits >> 23) & 0xFF;
    uint32_t mantissa = fp32_bits & 0x7FFFFF;
 
    uint32_t mant = mantissa >> 16;  
    bf16 b;
    b.bits = (uint16_t)((sign << 15) | (exponent << 7) | mant);
    return b;
}


void rmsnorm (bf16 *input, bf16 *output, int size) {
    float sum = 0.0f; 
    for (int i = 0; i < size; i++) {
        float val = bf16_to_fp32(input[i]); 
        sum += val * val; 
    }
    float mean = sum / size; 
    float rms = sqrtf(mean); 
    for (int i = 0; i < size; i++) {
        float val = bf16_to_fp32(input[i]); 
        output[i] = fp32_to_bf16(val / (rms + EPS)); 
    }
}


void flatten (bf16 *input, bf16 *output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = input[i * cols + j]; 
        }
    }
}


int main () {
    //bf16 b = fp32_to_bf16(16.45f); 
    bf16 X_l [N_hc][D_MODEL]; 
    for (int i = 0; i < N_hc; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            X_l[i][j] = fp32_to_bf16((float)(rand() % 100) / 10.0f); 
        }
    }
    //int d = 0;
    flatten ((bf16*) X_l, (bf16*) X_l, N_hc, D_MODEL); 
    rmsnorm ((bf16*) X_l, (bf16*) X_l, N_hc * D_MODEL);
    for (int i = 0; i < N_hc; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            printf("%f ", bf16_to_fp32(X_l[i][j])); 
            //d++; 
        }
        printf("\n"); 
    }
    return 0; 
}