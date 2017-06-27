#include "mp_common_examples.hpp"


#define BUF_SIZE 20

int main(int argc, char *argv[])
{
	int ret=0, i=0, myId=0, peersNum=0, libmp_version, oob_type, tl_type;
	int use_gpu_buffers=0;
	char *envVar = NULL;
	char ** sBuf, ** rBuf;
	char ** hostBuf;
	int device_id=MP_DEFAULT;

	mp_region_t * mp_regs_recv, * mp_regs_send;
	mp_request_t * mp_reqs_recv, * mp_reqs_send;
	
	//GPUDirect Async
	envVar = getenv("MP_USE_GPU");
	if (envVar != NULL) {
		device_id = atoi(envVar);
	}

	ret = mp_init(argc, argv, device_id);
	if(ret)
		exit(EXIT_FAILURE);

	mp_query_param(MP_PARAM_VERSION, &libmp_version);
	mp_query_param(MP_OOB_TYPE, &oob_type);
	mp_query_param(MP_TL_TYPE, &tl_type);
	mp_query_param(MP_MY_RANK, &myId);
	mp_query_param(MP_NUM_RANKS, &peersNum);
	if(!myId) printf("*************\nNum Peers: %d My Id: %d\nLibMP version: %x, OOB Type: %d, TL Type: %d\n*************\n", 
					peersNum, myId, libmp_version, oob_type, tl_type);


	//GPUDirect RDMA
	envVar = getenv("MP_GPU_BUFFERS"); 
	if (envVar != NULL) {
		use_gpu_buffers = atoi(envVar);
	}

	printf("Rank %d, Using GPU buffers: %d\n", myId, use_gpu_buffers);

	// ===== Create mem objs
	if(!myId) printf("\n==== Create mem objs ====\n");
	rBuf 			= (char **) calloc(peersNum, sizeof(char *));
	sBuf 			= (char **) calloc(peersNum, sizeof(char *));
	hostBuf 		= (char **) calloc(peersNum, sizeof(char *));
	mp_regs_recv 	= mp_create_regions(peersNum);
	mp_regs_send 	= mp_create_regions(peersNum);
	mp_reqs_recv 	= mp_create_request(peersNum);
	mp_reqs_send 	= mp_create_request(peersNum);

	for(i=0; i<peersNum; i++)
	{
		if(i != myId)
		{
			if(use_gpu_buffers == 1)
			{
				CUDA_CHECK(cudaMalloc((void **)&rBuf[i], BUF_SIZE));
				CUDA_CHECK(cudaMemset(rBuf[i], 0, BUF_SIZE)); 

				CUDA_CHECK(cudaMalloc((void **)&sBuf[i], BUF_SIZE));
				CUDA_CHECK(cudaMemset(sBuf[i], ('a'+myId), BUF_SIZE)); 

				hostBuf[i] = (char *) calloc(BUF_SIZE, sizeof(char));
			}
			else
			{
				CUDA_CHECK(cudaMallocHost((void **)&rBuf[i], BUF_SIZE*sizeof(char)));
				memset(rBuf[i], 0, BUF_SIZE);
				CUDA_CHECK(cudaMallocHost((void **)&sBuf[i], BUF_SIZE*sizeof(char)));
				memset(sBuf[i], ('a'+myId), BUF_SIZE);
			}

			MP_CHECK(mp_register_region_buffer(rBuf[i], BUF_SIZE, &mp_regs_recv[i]));
			if(!myId) printf("[%d] mp_register_region_buffer recv\n", myId);

			MP_CHECK(mp_irecv(rBuf[i], BUF_SIZE, i, &mp_regs_recv[i], &mp_reqs_recv[i]));
			if(!myId) printf("[%d] Recv Client %d, request=%p\n", myId, i, &mp_reqs_recv[i]);

			MP_CHECK(mp_register_region_buffer(sBuf[i], BUF_SIZE, &mp_regs_send[i]));
			if(!myId) printf("[%d] mp_register_region_buffer send\n", myId);
		}	
	}
	
	//Ensure all recvs have been posted
	mp_barrier();

	if(!myId)
		printf("\n==== Send Msg ====\n");

	for(i=0; i<peersNum; i++) {
		if(i != myId)
		{
			MP_CHECK(mp_isend_async(sBuf[i], BUF_SIZE, i, &mp_regs_send[i], &mp_reqs_send[i], NULL));
			if(!myId) printf("[%d] Send Client %d, request=%p\n", myId, i, &mp_reqs_send[i]);
		}
	}
	
	if(!myId) printf("\n==== Wait ====\n");

	for(i=0; i<peersNum; i++) {
		if(i != myId)
		{
			MP_CHECK(mp_wait_async(&mp_reqs_recv[i], NULL));
			if(use_gpu_buffers == 1)
				CUDA_CHECK(cudaMemcpyAsync(hostBuf[i], rBuf[i], BUF_SIZE, cudaMemcpyDeviceToHost, NULL));

			if(!myId) printf("[%d] Wait Async for Recv %d, req=%p\n", myId, i, &mp_reqs_recv[i]);
		}
	}

	for(i=0; i<peersNum; i++) {
		if(i != myId)
		{
			MP_CHECK(mp_wait_async(&mp_reqs_send[i], NULL));
			if(!myId) printf("[%d] Wait Async for Send %d, req=%p\n", myId, i, &mp_reqs_send[i]);
		}
	}

	//wait recv
	for(i=0; i<peersNum; i++) {
		if(i != myId)
		{
			MP_CHECK(mp_wait(&mp_reqs_recv[i]));
			if(!myId) printf("[%d] Wait for Recv %d, req=%p\n", myId, i, &mp_reqs_recv[i]);
			MP_CHECK(mp_wait(&mp_reqs_send[i]));
			if(!myId) printf("[%d] Wait for Send %d, req=%p\n", myId, i, &mp_reqs_send[i]);
		}
	}

	for(i=0; i<peersNum; i++) {
		if(i != myId)
		{
			if(use_gpu_buffers == 1)
			{
				CUDA_CHECK(cudaMemcpy(hostBuf[i], rBuf[i], BUF_SIZE, cudaMemcpyDeviceToHost));
				printf("[%d] Received from [%d]: %s\n", myId, i, hostBuf[i]);
			}
			else
				printf("[%d] Received from [%d]: %s\n", myId, i, rBuf[i]);
		}
	}

	// ===== Cleanup
	for(i=0; i<peersNum; i++)
	{
		if(i != myId)
		{
			if(use_gpu_buffers == 1)
			{
				cudaFree(rBuf[i]);
				cudaFree(sBuf[i]);
				free(hostBuf[i]);
			}
			else
			{				
				cudaFreeHost(rBuf[i]);
				cudaFreeHost(sBuf[i]);
			}
		}
	}

	if(use_gpu_buffers == 1)
		free(hostBuf);
	
	MP_CHECK(mp_unregister_regions(peersNum, mp_regs_recv));
	MP_CHECK(mp_unregister_regions(peersNum, mp_regs_send));

	free(rBuf);
	free(sBuf);
	free(mp_regs_recv);
	free(mp_regs_send);
	free(mp_reqs_recv);
	free(mp_reqs_send);
	
	// ===== Finalize
	mp_finalize();

	return 0;
}