#ifndef KERNEL_H
#define KERNEL_H
 
#include <math.h>
#include "data_structure.h"
#include "data_type.h"
 
/* Convert the register's bf16 input to a flat fp32 array. */
void flatten(
    registers_format *reg
);
 
/* y_i = gamma * x / sqrt(mean(x^2) + EPS), stored as fp32. */
void rmsnorm(
    registers_format *reg
);

#endif