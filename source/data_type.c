#include "data_type.h"

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
