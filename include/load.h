#ifndef LOAD_H
#define LOAD_H

#include <stdint.h>
#include "data_structure.h" 

void load_gl_x_i (
    DDR4_format *DDR4, 
    RAB_format *RAB, 
    int base_token);

void load_lc_x_i (
    RAB_format *RAB, 
    registers_format *regs, 
    int token_id); 

#endif 
