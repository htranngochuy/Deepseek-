#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define N_HC 4
#define EPS 1e-6f
#define D_MODEL 4096
#define D_EXPERT 256
#define TOPK_EXPERT 6
#define SIZE_LUT 129280
#define SCALED 1.5f
#define HASH_LAYER 3

typedef struct
{
    uint16_t bits;
} bf16;

float bf16_to_fp32(bf16 b)
{
    uint32_t sign = ((uint32_t)(b.bits & 0x8000)) << 16;
    uint32_t exponent = ((uint32_t)(b.bits & 0x7F80)) << 16;
    uint32_t mantissa = ((uint32_t)(b.bits & 0x007F)) << 16;
    uint32_t fp32_bits = sign | exponent | mantissa;
    float result;
    memcpy(&result, &fp32_bits, sizeof(result));
    return result;
}

bf16 fp32_to_bf16(float f)
{
    uint32_t fp32_bits;
    memcpy(&fp32_bits, &f, sizeof(fp32_bits));
    uint16_t sign = (fp32_bits & 0x80000000u) >> 16;
    uint16_t expo = (fp32_bits & 0x7F800000u) >> 16;
    uint16_t mant = (fp32_bits & 0x007F0000u) >> 16;
    bf16 b = {(uint16_t)(sign | expo | mant)};
    return b;
}

float sqrt_softplus(float x)
{
    float sp = (x > 20.0f) ? x : log1pf(expf(x));
    return sqrtf(sp);
}
void g_score(int N_token, bf16 x[N_token][D_MODEL], bf16 WR[D_MODEL][D_EXPERT], bf16 g[N_token][D_EXPERT])
{
    for (int i = 0; i < N_token; i++)
    {
        float mul[D_EXPERT] = {0.0f};
        for (int j = 0; j < D_MODEL; j++)
        {
            float t = bf16_to_fp32(x[i][j]);
            for (int k = 0; k < D_EXPERT; k++)
                mul[k] += t * bf16_to_fp32(WR[j][k]);
        }
        for (int k = 0; k < D_EXPERT; k++)
            g[i][k] = fp32_to_bf16(mul[k]);
    }
}

void g_activate(int N_token, bf16 g[N_token][D_EXPERT], bf16 gp[N_token][D_EXPERT])
{
    for (int i = 0; i < N_token; i++)
        for (int k = 0; k < D_EXPERT; k++)
            gp[i][k] = fp32_to_bf16(sqrt_softplus(bf16_to_fp32(g[i][k])));
}

void hash_lookup(int N_token, int token_ids[N_token], int hash_id[SIZE_LUT][TOPK_EXPERT], int indice[N_token][TOPK_EXPERT])
{
    for (int i = 0; i < N_token; i++) {
        for (int j = 0; j < TOPK_EXPERT; j++)
            indice[i][j] = hash_id[token_ids[i]][j];
    }

}
void g_normalize(int N_token, bf16 gp[N_token][D_EXPERT], int indice[N_token][TOPK_EXPERT], bf16 weight[N_token][TOPK_EXPERT])
{
    for (int i = 0; i < N_token; i++)
    {
        float gf[TOPK_EXPERT];
        float sum = 0.0f;
        for (int j = 0; j < TOPK_EXPERT; j++)
        {
            gf[j] = bf16_to_fp32(gp[i][indice[i][j]]);
            sum += gf[j];
        }
        for (int j = 0; j < TOPK_EXPERT; j++)
            weight[i][j] = fp32_to_bf16(gf[j] / sum * SCALED);
    }
}

void arg_topk(int N_token, bf16 g[N_token][D_EXPERT], float bias[D_EXPERT], int indice[N_token][TOPK_EXPERT])
{
    for (int i = 0; i < N_token; i++)
    {
        int mark[D_EXPERT] = {0};
        for (int j = 0; j < TOPK_EXPERT; j++)
        {
            float max_val = -INFINITY;
            int max_idx = -1;
            for (int k = 0; k < D_EXPERT; k++)
            {
                float val = bf16_to_fp32(g[i][k]) + bias[k];
                if (mark[k] == 0 && val > max_val)
                {
                    max_val = val;
                    max_idx = k;
                }
            }
            indice[i][j] = max_idx;
            mark[max_idx] = 1;
        }
    }
}

void router1_3(int N_token, int token_ids[N_token], int hash_id[SIZE_LUT][TOPK_EXPERT], bf16 x[N_token][D_MODEL], bf16 WR[D_MODEL][D_EXPERT], int indice[N_token][TOPK_EXPERT], bf16 weight[N_token][TOPK_EXPERT])
{
    bf16(*g)[D_EXPERT] = malloc(sizeof(bf16[N_token][D_EXPERT]));
    bf16(*gp)[D_EXPERT] = malloc(sizeof(bf16[N_token][D_EXPERT]));
    g_score(N_token, x, WR, g);
    g_activate(N_token, g, gp);
    hash_lookup(N_token, token_ids, hash_id, indice);
    g_normalize(N_token, gp, indice, weight);
    free(g);
    free(gp);
}

void router4(int N_token, bf16 x[N_token][D_MODEL], bf16 WR[D_MODEL][D_EXPERT], float bias[D_EXPERT], bf16 weight[N_token][TOPK_EXPERT], int indice[N_token][TOPK_EXPERT])
{
    bf16(*g)[D_EXPERT] = malloc(sizeof(bf16[N_token][D_EXPERT]));
    bf16(*gp)[D_EXPERT] = malloc(sizeof(bf16[N_token][D_EXPERT]));
    g_score(N_token, x, WR, g);
    g_activate(N_token, g, gp);
    arg_topk(N_token, gp, bias, indice);
    g_normalize(N_token, gp, indice, weight);
    free(g);
    free(gp);
}

int router_layer(int layer, int N_token, bf16 x[N_token][D_MODEL], bf16 WR[D_MODEL][D_EXPERT], float bias[D_EXPERT], int token_ids[N_token], int hash_id[SIZE_LUT][TOPK_EXPERT], bf16 weight[N_token][TOPK_EXPERT], int indice[N_token][TOPK_EXPERT])
{
    if (layer <= HASH_LAYER)
        router1_3(N_token, token_ids, hash_id, x, WR, indice, weight);
    else
        router4(N_token, x, WR, bias, weight, indice);
    return 0;
}

int main()
{
    const int N = 4;
    bf16(*x)[D_MODEL] = malloc(sizeof(bf16[N][D_MODEL]));
    bf16(*WR)[D_EXPERT] = malloc(sizeof(bf16[D_MODEL][D_EXPERT]));
    float *bias = malloc(sizeof(float[D_EXPERT]));
    int (*hash_id)[TOPK_EXPERT] = malloc(sizeof(int[SIZE_LUT][TOPK_EXPERT]));
    int *token_ids = malloc(sizeof(int[N]));
    bf16(*weight)[TOPK_EXPERT] = malloc(sizeof(bf16[N][TOPK_EXPERT]));
    int (*indice)[TOPK_EXPERT] = malloc(sizeof(int[N][TOPK_EXPERT]));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < D_MODEL; j++)
            x[i][j] = fp32_to_bf16((rand() % 100) / 10.0f);
    for (int j = 0; j < D_MODEL; j++)
        for (int k = 0; k < D_EXPERT; k++)
            WR[j][k] = fp32_to_bf16((rand() % 100) / 10.0f * 0.05f);
    for (int k = 0; k < D_EXPERT; k++)
        bias[k] = (rand() % 100) / 10.0f * 0.1f;

    for (int t = 0; t < SIZE_LUT; t++)
    {
        int used[D_EXPERT] = {0};
        for (int j = 0; j < TOPK_EXPERT; j++)
        {
            int e;
            do
            {
                e = rand() % D_EXPERT;
            } while (used[e]);
            used[e] = 1;
            hash_id[t][j] = e;
        }
    }
    token_ids[0] = 100;
    token_ids[1] = 2024;
    token_ids[2] = 31337;
    token_ids[3] = SIZE_LUT - 1;

    const int layers[] = {1, 3, 4};
    for (int li = 0; li < 3; li++)
    {
        int layer = layers[li];
        router_layer(layer, N, x, WR, bias, token_ids, hash_id, weight, indice);

        printf("layer %d \n", layer);
        for (int i = 0; i < N; i++)
        {
            float sum = 0.0f;
            printf("  token %-6d  ", token_ids[i]);
            for (int j = 0; j < TOPK_EXPERT; j++)
                printf(" %3d", indice[i][j]);

            for (int j = 0; j < TOPK_EXPERT; j++)
            {
                printf(" %.4f", bf16_to_fp32(weight[i][j]));
                sum += bf16_to_fp32(weight[i][j]);
            }
            printf("\n");
        }
    }

    free(x);
    free(WR);
    free(bias);
    free(hash_id);
    free(token_ids);
    free(weight);
    free(indice);
    return 0;
}