#include "mmul.hh"

#include <mpi.h>
// #include <mkl_spblas.h>


void column_multiply(
    sparse_elt* a_mat_slice,
    double* b_mat_slice,
    double* c_mat_slice
)
{
    MPI_Comm group_comm;
    MPI_Comm_split(MPI_COMM_WORLD, id/P.c, id, &group_comm);

    sparse_elt* temp_a_1 = new sparse_elt[P.k*P.c*P.q];
    sparse_elt* temp_a_2 = new sparse_elt[P.k*P.c*P.q];

    MPI_Allgather(
        a_mat_slice,
        sizeof(sparse_elt)*P.k*P.q,
        MPI_BYTE,
        temp_a_1,
        sizeof(sparse_elt)*P.k*P.q,
        MPI_BYTE,
        group_comm
    );


    for (int i=0; i<P.n*P.k; i++)
        c_mat_slice[i] = 0;

    for (int i=0; i<P.p/P.c; i++) // rounds
    {
        // communicate
        MPI_Request requests[2];
        MPI_Status statuses[2];

        MPI_Isend(
            temp_a_1,
            sizeof(sparse_elt)*P.k*P.c*P.q,
            MPI_BYTE,
            (P.p+id-P.c)%P.p,
            0,
            MPI_COMM_WORLD,
            &requests[0]
        );
        MPI_Irecv(
            temp_a_2,
            sizeof(sparse_elt)*P.k*P.c*P.q,
            MPI_BYTE,
            (id+P.c)%P.p,
            0,
            MPI_COMM_WORLD,
            &requests[1]
        );

        // multiply
        if (P.mkl)
        {
            // TODO
            // struct matrix_descr desc;
            // desc.type = SPARSE_MATRIX_GENERAL;

            // sparse_matrix_t mkl_a;
            // mkl_sparse_d_mm(
            //     SPARSE_OPERATION_TRANSPOSE,
            //     1.0,
            //     mkl_a,
            //     SPARSE_LAYOUT_COLUMN_MAJOR,

            // );
        }
        else
        {
            #pragma omp parallel for
            for (int dx=0; dx<P.k; dx++)
            {
                // int x = dx + id*P.k;
                for (int dz = 0; dz < P.k*P.c; dz++)
                {
                    int z = dz + (((i+id/P.c)*P.c)%P.p)*P.k;
                    for (int j=0; j<P.q; j++)
                    {
                        auto elt = temp_a_1[dz*P.q + j];
                        if (elt.pos == -1)
                            continue;
                        int y = elt.pos;
                        c_mat_slice[dx*P.n + y] += elt.val * b_mat_slice[dx*P.n + z];
                    }
                }
            }
        }

        // sync
        MPI_Waitall(2, requests, statuses);

        std::swap(temp_a_1, temp_a_2);
    }

    delete[] temp_a_1;
    delete[] temp_a_2;
    MPI_Comm_free(&group_comm);
}
