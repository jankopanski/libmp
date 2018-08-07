#include <mpi.h>
#include <mp.h>
#include "test_utils.h"

const int sleep_time = 20;

int main(int argc, char const *argv[])
{
    // sleep(sleep_time);
    int comm_size, my_rank, peer;
    int tag = 42;
    MPI_Init(&argc, (char ***) &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    gpu_init(-1);
    peer = !my_rank;
    MP_CHECK(mp_init(MPI_COMM_WORLD, &peer, 1, MP_INIT_DEFAULT, 0));

    int buf_size = 9;
    char *buf = NULL;
    void *buf_d = NULL;
    cudaStream_t stream;
    mp_request_t req;
    mp_reg_t reg;
    buf = (char *) malloc(buf_size + 1);
    CUDA_CHECK(cudaMalloc(&buf_d, buf_size));
    CUDA_CHECK(cudaStreamCreate(&stream));
    MP_CHECK(mp_register(buf_d, buf_size, &reg));

    fprintf(stderr, "begin %d\n", my_rank);
    if (!my_rank) {
        buf = (char *) "SEND_TEST";
        CUDA_CHECK(cudaMemcpy(buf_d, buf, buf_size, cudaMemcpyHostToDevice));
        // MPI_Barrier(MPI_COMM_WORLD);
        MP_CHECK(mp_isend_tag_on_stream(buf_d, buf_size, peer, tag, 
                                        &reg, &req, stream));
        fprintf(stderr, "send %d\n", my_rank);
    }
    else {
        CUDA_CHECK(cudaMemset(buf_d, 0, buf_size));
        MP_CHECK(mp_irecv_tag(buf_d, buf_size, peer, tag, &reg, &req));
        // MPI_Barrier(MPI_COMM_WORLD);
        fprintf(stderr, "recv %d\n", my_rank);
    }
    // sleep(sleep_time);
    // MP_CHECK(mp_wait_on_stream(&req, stream));
    MP_CHECK(mp_wait_tag_on_stream(&req, stream));
    fprintf(stderr, "wait on stream %d\n", my_rank);
    MP_CHECK(mp_wait(&req));
    fprintf(stderr, "wait %d\n", my_rank);
    if (my_rank) {
        CUDA_CHECK(cudaMemcpy(buf, buf_d, buf_size, cudaMemcpyDeviceToHost));
        buf[buf_size] = 0;
        printf("%s\n", buf);
    }
    fprintf(stderr, "end %d\n", my_rank);
    
    MPI_Barrier(MPI_COMM_WORLD);
    mp_finalize();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
