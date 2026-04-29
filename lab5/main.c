/* noncontiguous access with a single collective I/O function */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


//#define FILESIZE      1048576
#define FILESIZE 1024
#define INTS_PER_BLK  1

int main(int argc, char **argv)
{
  int *buf, *readbuf, rank, nprocs, nints, bufsize;
  int nblocks, local_errors, total_errors;
  MPI_File fh;
  MPI_Datatype filetype;

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Get rank
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (FILESIZE % nprocs != 0 || (FILESIZE / nprocs) % sizeof(int) != 0) {
    if (rank == 0) {
      fprintf(stderr, "FILESIZE must divide evenly into integer buffers for all processes.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  bufsize = FILESIZE/nprocs;
  buf = (int *) malloc(bufsize);
  readbuf = (int *) malloc(bufsize);
  if (buf == NULL || readbuf == NULL) {
    fprintf(stderr, "Rank %d failed to allocate buffers.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  nints = bufsize/sizeof(int);
  if (nints % INTS_PER_BLK != 0) {
    if (rank == 0) {
      fprintf(stderr, "Each process must have a whole number of file blocks.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  memset(buf, 'A'+rank, nints * sizeof(int));
  memset(readbuf, 0, nints * sizeof(int));

  // Create datatype
  nblocks = nints / INTS_PER_BLK;
  MPI_Type_vector(nblocks, INTS_PER_BLK, INTS_PER_BLK * nprocs, MPI_INT, &filetype);
  MPI_Type_commit(&filetype);

  // Setup file view
  MPI_File_open(MPI_COMM_WORLD, "noncontig.out",
                MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fh);
  MPI_File_set_size(fh, 0);
  MPI_File_set_view(fh, rank * INTS_PER_BLK * sizeof(int),
                    MPI_INT, filetype, "native", MPI_INFO_NULL);

  // Collective MPI-IO write
  MPI_File_write_all(fh, buf, nints, MPI_INT, MPI_STATUS_IGNORE);

  // Close file
  MPI_File_close(&fh);
  
  //Implement collective file reading as an exercise.
  MPI_File_open(MPI_COMM_WORLD, "noncontig.out",
                MPI_MODE_RDONLY,
                MPI_INFO_NULL, &fh);
  MPI_File_set_view(fh, rank * INTS_PER_BLK * sizeof(int),
                    MPI_INT, filetype, "native", MPI_INFO_NULL);
  MPI_File_read_all(fh, readbuf, nints, MPI_INT, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  local_errors = memcmp(buf, readbuf, nints * sizeof(int)) != 0;
  MPI_Reduce(&local_errors, &total_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    if (total_errors == 0) {
      printf("Collective MPI-IO write/read verification passed.\n");
    } else {
      printf("Collective MPI-IO write/read verification failed on %d rank(s).\n",
             total_errors);
    }
  }

  MPI_Type_free(&filetype);
  free(readbuf);

  free(buf);
  
  //Finalize
  MPI_Finalize();

  return 0; 
}

