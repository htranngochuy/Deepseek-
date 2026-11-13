#include "load.h"

void load_gl_x_i (DDR4_format *DDR4, RAB_format *RAB, int base_token) {
    memcpy(RAB->i_x, DDR4->i_x[base_token], sizeof RAB->i_x);
}

void load_lc_x_i (RAB_format *RAB, registers_format *regs, int token_id) {
    memcpy(regs->x_i, RAB->i_x[token_id], sizeof(bf16) * N_HC * D_MODEL);
}