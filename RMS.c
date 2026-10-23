#include <stdio.h> 
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define N_HC 4 
#define EPS 1e-6f 
#define D_MODEL 4096 


typedef struct {
    uint16_t bits; 
} bf16; 


float bf16_to_fp32(bf16 b) {
    uint32_t sign = ((uint32_t)(b.bits & 0x8000)) << 16;
    uint32_t exponent = ((uint32_t)(b.bits & 0x7F80)) << 16;
    uint32_t mantissa = ((uint32_t)(b.bits & 0x007F)) << 16;
    uint32_t fp32_bits = sign | exponent | mantissa;
    float result;
    memcpy(&result, &fp32_bits, sizeof(result));
    return result;
}
 

bf16 fp32_to_bf16(float f) {
    uint32_t fp32_bits;
    memcpy(&fp32_bits, &f, sizeof(fp32_bits));
    uint16_t sign = (fp32_bits & 0x80000000u) >> 16;      
    uint16_t expo = (fp32_bits & 0x7F800000u) >> 16;      
    uint16_t mant = (fp32_bits & 0x007F0000u) >> 16;      
    bf16 b = { (uint16_t)(sign | expo | mant) };
    return b;
}


void rmsnorm (bf16 *input, float *output, int size, float *gamma) {
    float sum  = 0.0f;
    for (int i=0; i<size; i++) {
        float val = bf16_to_fp32(input[i]);
        sum += val * val;
    }
    float mean = sum / size + EPS;
    float rms = sqrt (mean); 
    for (int i=0; i<size; i++) {
        float val = bf16_to_fp32(input[i]);
        output[i] = gamma[i] * val / rms; 
    }
}


void flatten (int rows, int cols, bf16 input[rows][cols], bf16 *output) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = input[i][j]; 
        }
    }
}

void check_bit () 
{ 
    float f = 16.45f; 
    uint32_t check; 
    memcpy (&check, &f, sizeof(check)); 

    bf16 b = fp32_to_bf16(f);
    float b16 = bf16_to_fp32(b);
    uint32_t check16;
    memcpy(&check16, &b16, sizeof(check16));

    printf("Gia tri goc (Float)    : %.4f\n", f);
    printf("Gia tri phuc hoi (BF16): %.4f\n", b16);
    printf("Hex goc (FP32)    : 0x%08X\n", check);
    printf("Hex phuc hoi (FP32)    : 0x%08X\n", check16);
}
int main () {
    //bf16 b = fp32_to_bf16(16.45f); 
    bf16 X_l [N_HC][D_MODEL]; 
    bf16 x_flat [N_HC * D_MODEL]; 
    float y_l [N_HC * D_MODEL];
    // Khoi tao ngau nhien cho ma tran X_l
    for (int i = 0; i < N_HC; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            X_l[i][j] = fp32_to_bf16((float)(rand() % 100) / 10.0f); 
        }
    }
    //int d = 0;
    flatten (N_HC, D_MODEL, X_l, x_flat);
    // Kiem tra flatten 
    for (int i=0; i<N_HC; i++) {
        for (int j=0; j<D_MODEL; j++) {
            if (x_flat[i * D_MODEL + j].bits != X_l[i][j].bits) {
                printf ("Lam phang khong dung\n");
                return 1; 
            }
        }
    }
    printf ("Lam phang dung \n");
    // Khoi tao gamma 
    float gamma [N_HC * D_MODEL];
    for (int i=0; i<N_HC*D_MODEL; i++) {
        gamma[i] = 1.0f;
    }
    rmsnorm (x_flat, y_l, N_HC * D_MODEL, gamma);   
    //int d = 0; 
    for (int i=0; i<N_HC * D_MODEL; i++) {
        printf ("%.6f ", y_l[i]);
        break; 
        //return 0; 
        //d++; 
    }
    check_bit();
    //bf16 b = fp32_to_bf16(16.45f); 
    //printf ("kich thuoc cua bf16: %zu bytes\n", sizeof(bf16));
    return 0; 
}