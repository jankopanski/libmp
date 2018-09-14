/****
 * Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ****/

#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mp.h>
#include "test_utils.h"
// #include "TextTable.h"

#define MIN_SIZE 1
#define MAX_SIZE 64*1024
#define ITER_COUNT_SMALL 40
#define ITER_COUNT_LARGE 20

constexpr int intlog2(int x) {
    int y = 0;
    while (x >>= 1) ++y;
    return y;
}

constexpr int times_num = intlog2(MAX_SIZE);
const int measure_op_num = 4;
int comm_size, my_rank, peer;
double times[times_num][measure_op_num];

void print(MPI_Comm comm) {
    if (!my_rank) {
        printf("\nSender\n");
        // printf("Size\tmp_isend_tag_on_stream\tmp_wait_tag_on_stream\tmp_wait_all\n");
        printf("%16s%25s%25s%25s%25s\n", "Size[B]\\Time[ms]", 
                "mp_isend_tag_on_stream", "mp_wait_tag_on_stream", "mp_wait_all", "wait event");
        for (int i = 0, size = MIN_SIZE; size <= MAX_SIZE; ++i, size *= 2) {
            printf("%16d%25f%25f%25f%25f\n", size, 
                    times[i][0], times[i][1], times[i][2], times[i][3]);
        }
    }
    fflush(stdout);
    MPI_CHECK(MPI_Barrier(comm));
    if (my_rank) {
        printf("\nReceiver\n");
        printf("%16s%25s%25s%25s%25s\n", "Size[B]\\Time[ms]", 
                "mp_irecv_tag", "mp_wait_tag_on_stream", "mp_wait_all", "wait event");
        for (int i = 0, size = MIN_SIZE; size <= MAX_SIZE; ++i, size *= 2) {
            printf("%16d%25f%25f%25f%25f\n", size, 
                    times[i][0], times[i][1], times[i][2], times[i][3]);
        }
    }
}

int sr_exchange (MPI_Comm comm, int index, int size, int iter_count)
{
    double wtime;
    size_t buf_size; 
    cudaStream_t stream;

    /*application and pack buffers*/
    void *buf_d = NULL;

    /*mp specific objects*/
    mp_request_t *req = NULL;
    mp_reg_t reg;

    cudaEvent_t start[iter_count], stop[iter_count];

    buf_size = size * iter_count;

    /*allocating requests*/
    req = (mp_request_t *) malloc(iter_count*sizeof(mp_request_t));
    CUDA_CHECK(cudaMalloc((void **)&buf_d, buf_size));
    CUDA_CHECK(cudaStreamCreate(&stream));	
    MP_CHECK(mp_register(buf_d, buf_size, &reg));
    for (int i = 0; i < iter_count; i++) {
        CUDA_CHECK(cudaEventCreate(&start[i]));
        CUDA_CHECK(cudaEventCreate(&stop[i]));
    }

    for (int j = 0; j < iter_count; j++) {
        if (!my_rank) {
            MPI_CHECK(MPI_Barrier(comm));
            times[index][0] -= MPI_Wtime();
            MP_CHECK(mp_isend_tag_on_stream ((void *)((uintptr_t)buf_d + size*j), 
                                            size, peer, j, &reg, &req[j], stream));
            times[index][0] += MPI_Wtime();
            CUDA_CHECK(cudaEventRecord(start[j]));
            times[index][1] -= MPI_Wtime();
            MP_CHECK(mp_wait_tag_on_stream(&req[j], stream));
            times[index][1] += MPI_Wtime();
            CUDA_CHECK(cudaEventRecord(stop[j]));
        } else {
            times[index][0] -= MPI_Wtime();
            MP_CHECK(mp_irecv_tag ((void *)((uintptr_t)buf_d + size*j), 
                                    size, peer, j, &reg, &req[j]));
            times[index][0] += MPI_Wtime();
            MPI_CHECK(MPI_Barrier(comm));
            CUDA_CHECK(cudaEventRecord(start[j]));
            times[index][1] -= MPI_Wtime();
            MP_CHECK(mp_wait_tag_on_stream(&req[j], stream));
            times[index][1] += MPI_Wtime();
            CUDA_CHECK(cudaEventRecord(stop[j]));
        }
    }
    times[index][2] -= MPI_Wtime();
    MP_CHECK(mp_wait_all_tag(iter_count, req));
    times[index][2] += MPI_Wtime();

    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < iter_count; i++) {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start[i], stop[i]));
        times[index][3] += (double) ms;
    }

    times[index][0] *= 1000 / iter_count;
    times[index][1] *= 1000 / iter_count;
    times[index][2] *= 1000;
    times[index][3] /= iter_count;

    // all ops in the stream should have been completed 
    usleep(1000);
    CUDA_CHECK(cudaStreamQuery(stream));
    MPI_CHECK(MPI_Barrier(comm));
    CUDA_CHECK(cudaDeviceSynchronize());
    mp_deregister(&reg);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(buf_d));
    free(req);

    return 0;
}

int main (int c, char *v[])
{
    int iter_count;
    int validate = 1;

    MPI_CHECK(MPI_Init(&c, &v));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));

    if (comm_size != 2) { 
	fprintf(stderr, "this test requires exactly two processes \n");
        exit(-1);
    }

    if (gpu_init(-1)) {
        fprintf(stderr, "got error while initializing GPU\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    peer = !my_rank;
    //Need to set CUDA_VISIBLE_DEVICES
    MP_CHECK(mp_init(MPI_COMM_WORLD, &peer, 1, MP_INIT_DEFAULT, 0));

    iter_count = ITER_COUNT_SMALL;

    if (!my_rank) {
        printf("Number of messages of size\n <= 1024 B: %d\n  > 1024 B: %d\n\n", 
                ITER_COUNT_SMALL, ITER_COUNT_LARGE);
        printf("Average time per single call\n\n");
    }

    for (int i = 0, size = MIN_SIZE; size <= MAX_SIZE; i++, size *= 2) 
    {
        if (size > 1024) {
            iter_count = ITER_COUNT_LARGE;
        }

        sr_exchange(MPI_COMM_WORLD, i, size, iter_count);
        fprintf(stderr, "Iteration: %d, size: %d\n", i, size);
    }

    print(MPI_COMM_WORLD);

    mp_finalize();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
