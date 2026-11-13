#ifndef DATA_TYPE_H 
#define DATA_TYPE_H

#include <stdint.h>
#include <string.h>

typedef struct {
    uint16_t bits; 
} bf16; 

float bf16_to_fp32(bf16 b);
 
bf16 fp32_to_bf16(float f);

#endif 