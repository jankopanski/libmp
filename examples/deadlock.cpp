#include <mpi.h>
#include <mp.h>
#include "test_utils.h"

const int sleep_time = 0;
const int num = 256;
const int size = 32;
const int iter = 5;

int main(int argc, char const *argv[])
{
    sleep(sleep_time);
    int comm_size, my_rank, peer;
    MPI_Init(&argc, (char ***) &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    gpu_init(-1);
    peer = !my_rank;
    MP_CHECK(mp_init(MPI_COMM_WORLD, &peer, 1, MP_INIT_DEFAULT, 0));

    char *sbuf_d = NULL, *rbuf_d = NULL;
    cudaStream_t stream;
    mp_request_t sreq[num];
    mp_request_t rreq[num];
    mp_reg_t sreg, rreg;
    CUDA_CHECK(cudaMalloc(&sbuf_d, size * num));
    CUDA_CHECK(cudaMalloc(&rbuf_d, size * num));
    CUDA_CHECK(cudaStreamCreate(&stream));
    MP_CHECK(mp_register(sbuf_d, size * num, &sreg));
    MP_CHECK(mp_register(rbuf_d, size * num, &rreg));

    fprintf(stderr, "begin %d\n", my_rank);

    for (int j = 0; j < iter; j++) {
        MPI_Barrier(MPI_COMM_WORLD);
        fprintf(stderr, "iteration %d %d\n", my_rank, j);

        if (!my_rank) {
            // MPI_Barrier(MPI_COMM_WORLD);
            for (int i = 0; i < num; i++) {
                fprintf(stderr, "send %d %d\n", my_rank, i);
                // MP_CHECK(mp_isend_on_stream(sbuf_d + i * size, size, peer, &sreg, &sreq[i], stream));
                // MP_CHECK(mp_wait_on_stream(&sreq[i], stream));
                MPI_Barrier(MPI_COMM_WORLD);
                MP_CHECK(mp_isend_tag_on_stream(sbuf_d + i * size, size, peer, i, &sreg, &sreq[i], stream));
                MP_CHECK(mp_wait_tag_on_stream(&sreq[i], stream));
                // MP_CHECK(mp_irecv_tag(rbuf_d + i * size, size, peer, i, &rreg, &rreq[i]));
                // MP_CHECK(mp_wait_tag_on_stream(&rreq[i], stream));
            }
        }
        else {
            for (int i = 0; i < num; i++) {
                fprintf(stderr, "recv %d %d\n", my_rank, i);
                // MP_CHECK(mp_irecv(rbuf_d + i * size, size, peer, &rreg, &rreq[i]));
                // MP_CHECK(mp_wait_on_stream(&rreq[i], stream));
                MP_CHECK(mp_irecv_tag(rbuf_d + i * size, size, peer, i, &rreg, &rreq[i]));
                MPI_Barrier(MPI_COMM_WORLD);
                MP_CHECK(mp_wait_tag_on_stream(&rreq[i], stream));
                // MP_CHECK(mp_isend_tag_on_stream(sbuf_d + i * size, size, peer, i, &sreg, &sreq[i], stream));
                // MP_CHECK(mp_wait_tag_on_stream(&sreq[i], stream));
            }
            // MPI_Barrier(MPI_COMM_WORLD);
        }
        // sleep(sleep_time);
        // fprintf(stderr, "wait on stream %d\n", my_rank);
        if (!my_rank) {
            // MP_CHECK(mp_wait_all(num, sreq));
            MP_CHECK(mp_wait_all_tag(num, sreq));
        }
        else {
            // MP_CHECK(mp_wait_all(num, rreq));
            MP_CHECK(mp_wait_all_tag(num, rreq));
        }
        // MP_CHECK(mp_wait_all_tag(num, rreq));
        // MP_CHECK(mp_wait_all_tag(num, sreq));
        // fprintf(stderr, "wait %d\n", my_rank);
    }

    fprintf(stderr, "end %d\n", my_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    mp_finalize();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
