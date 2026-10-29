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

typedef struct {
    bf16 i_x[N_HC][D_MODEL];
    //float i_y[N_HC * D_MODEL];
} RAB_format; 

typedef struct {
    float gamma; 
    bf16 x_i_flatten[N_HC * D_MODEL];
    float sum_xi_squared[N_HC * D_MODEL];
    float rms;
    float y_i[N_HC * D_MODEL];
} registers_format; 

void load_i_x (RAB_format *RAB, bf16 i_x[N_HC][D_MODEL]) {
    for (int i=0; i<N_HC; i++) {
        for (int j=0; j<D_MODEL; j++) {
            RAB->i_x[i][j] = i_x[i][j]; 
        }
    }
}

void rmsnorm (registers_format *regs, float gamma, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float val = bf16_to_fp32(regs->x_i_flatten[i]);
        sum += val * val;
    }
    regs->sum_xi_squared[0] = sum; 
    float mean = sum / size + EPS;
    regs->rms = sqrt(mean); 
    for (int i = 0; i < size; i++) {
        float val = bf16_to_fp32(regs->x_i_flatten[i]);
        regs->y_i[i] = gamma * val / regs->rms; 
    }
}


void flatten (RAB_format *rab, registers_format *regs) {
    for (int i = 0; i < N_HC; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            regs->x_i_flatten[i * D_MODEL + j] = rab->i_x[i][j]; 
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
    //bf16 x_flat [N_HC * D_MODEL]; 
    float y_l [N_HC * D_MODEL];
    // Khoi tao ngau nhien cho ma tran X_l
    for (int i = 0; i < N_HC; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            X_l[i][j] = fp32_to_bf16((float)(rand() % 100) / 10.0f); 
        }
    }
    //int d = 0;
    RAB_format rab;
    registers_format regs;
    load_i_x(&rab, X_l);
    flatten(&rab, &regs);
    // Kiem tra flatten 
    for (int i=0; i<N_HC; i++) {
        for (int j=0; j<D_MODEL; j++) {
            if (regs.x_i_flatten[i * D_MODEL + j].bits != X_l[i][j].bits) {
                printf ("Lam phang khong dung\n");
                return 0; 
            }
        }
    }
    printf ("Lam phang dung \n");
    // Khoi tao gamma 
    float gamma  = 1.0f;
    rmsnorm (&regs, gamma, N_HC * D_MODEL);   
    //int d = 0; 
    for (int i=0; i<N_HC * D_MODEL; i++) {
        printf ("%.6f ", regs.y_i[i]);
        break; 
        //return 0; 
        //d++; 
    }
    check_bit();
    //bf16 b = fp32_to_bf16(16.45f); 
    //printf ("kich thuoc cua bf16: %zu bytes\n", sizeof(bf16));
    return 0; 
}