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

float sqrt_softplus(float x)
{
    float t;
    t = sqrt(logf(1.0f + expf(x)));
    return t;
}

void g_score(int N_token, float input[N_token][D_MODEL], float WR[D_MODEL][D_EXPERT], float output[N_token][D_EXPERT])
{
    for (int i = 0; i < N_token; i++)
    {
        for (int j = 0; j < D_MODEL; j++)
        {
            for (int k = 0; k < D_EXPERT; k++)
            {
                output[i][k] += input[i][j] * WR[j][k];
                output[i][k] = sqrt_softplus(output[i][k]);
            }
        }
    }
}

void hash_lookup(int N_token, int token_ids[N_token], int hash_id[SIZE_LUT][TOPK_EXPERT], int indice[N_token][TOPK_EXPERT])
{
    for (int i = 0; i < N_token; i++)
    {
        for (int j = 0; j < TOPK_EXPERT; j++)
        {
            indice[i][j] = hash_id[token_ids[i]][j];
        }
    }
}

void g_normalize(int N_token, float g[N_token][D_EXPERT], int indice[N_token][TOPK_EXPERT], float weight[N_token][TOPK_EXPERT])
{
    for (int i = 0; i < N_token; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < TOPK_EXPERT; j++)
        {
            weight[i][j] = g[i][indice[i][j]];
            sum += g[i][indice[i][j]];
        }
        for (int j = 0; j < TOPK_EXPERT; j++)
        {
            weight[i][j] = (weight[i][j] / sum) * SCALED;
        }
    }
}

void router1_3(int N_token, int token_ids[N_token], int hash_id[SIZE_LUT][TOPK_EXPERT], float x[N_token][D_MODEL], float WR[D_MODEL][D_EXPERT], int indice[N_token][TOPK_EXPERT], float weight[N_token][TOPK_EXPERT])
{
    float g_score_out[N_token][D_EXPERT];
    memset(g_score_out, 0, sizeof(g_score_out));
    g_score(N_token, x, WR, g_score_out);
    hash_lookup(N_token, token_ids, hash_id, indice);
    g_normalize(N_token, g_score_out, indice, weight);
}

void arg_topk(int N_token, float g[N_token][D_EXPERT], float bias[D_EXPERT], int indice[N_token][TOPK_EXPERT])
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
                float val = g[i][k] + bias[k];
                if (val > max_val && mark[k] == 0)
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

void router4(int N_token, float x[N_token][D_MODEL], float WR[D_MODEL][D_EXPERT], float bias[D_EXPERT], float weight[N_token][TOPK_EXPERT], int indice[N_token][TOPK_EXPERT])
{
    float g_score_out[N_token][D_EXPERT];
    memset(g_score_out, 0, sizeof(g_score_out));
    g_score(N_token, x, WR, g_score_out);
    arg_topk(N_token, g_score_out, bias, indice);
    g_normalize(N_token, g_score_out, indice, weight);
}

int router_layer(int layer, int N_token, float x[N_token][D_MODEL], float WR[D_MODEL][D_EXPERT], float bias[D_EXPERT], int token_ids[N_token], int hash_id[SIZE_LUT][TOPK_EXPERT], float weight[N_token][TOPK_EXPERT], int indice[N_token][TOPK_EXPERT])
{
    if (layer < HASH_LAYER)
        router1_3(N_token, token_ids, hash_id, x, WR, indice, weight);
    else
        router4(N_token, x, WR, bias, weight, indice);
    return 0;
}

int main()
{
    const int N = 4;
    float (*x)[D_MODEL] = malloc(sizeof(float[N][D_MODEL]));                  
    float (*WR)[D_EXPERT] = malloc(sizeof(float[D_MODEL][D_EXPERT]));         
    float *bias = malloc(sizeof(float[D_EXPERT]));                           
    int (*hash_id)[TOPK_EXPERT] = malloc(sizeof(int[SIZE_LUT][TOPK_EXPERT]));
    int *token_ids = malloc(sizeof(int[N]));                                  
    float (*weight)[TOPK_EXPERT] = malloc(sizeof(float[N][TOPK_EXPERT]));     
    int (*indice)[TOPK_EXPERT] = malloc(sizeof(int[N][TOPK_EXPERT]));         
    if (!x || !WR || !bias || !hash_id || !token_ids || !weight || !indice)
    {
        fprintf(stderr, "het bo nho\n");
        return 1;
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < D_MODEL; j++)
            x[i][j] = (rand() % 100) / 10.0f;
    for (int j = 0; j < D_MODEL; j++)
        for (int k = 0; k < D_EXPERT; k++)
            WR[j][k] = (rand() % 100) / 10.0f * 0.05f;
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
            
            /*for (int j = 0; j < TOPK_EXPERT; j++)
            {
                printf(" %.4f", weight[i][j]);
                sum += weight[i][j];
            }*/
            printf ("\n");
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