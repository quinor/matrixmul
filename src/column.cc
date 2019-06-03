#include "mmul.hh"

#include <mpi.h>
#include <mkl.h>


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

    // csr
    int* rows_start = new int[P.n];
    int* rows_end = new int[P.n];
    int* poses = new int[P.k*P.c*P.q];
    double* vals = new double[P.k*P.c*P.q];

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
            struct matrix_descr desc;
            desc.type = SPARSE_MATRIX_TYPE_GENERAL;

            for (int j=0; j<P.n; j++)
                rows_start[j] = rows_end[j] = 0;
            int cur = 0;

            // convert to CSR
            for (int dz=0; dz < P.k*P.c; dz++)
            {
                int z = dz + (((i+id/P.c)*P.c)%P.p)*P.k;

                rows_start[z] = cur;
                for (int j=0; j<P.q; j++)
                {
                    auto elt = temp_a_1[dz*P.q + j];
                    if (elt.pos == -1)
                        continue;
                    poses[cur] = elt.pos;
                    vals[cur] = elt.val;
                    cur++;
                }
                rows_end[z] = cur;
            }
            sparse_matrix_t mkl_a;
            mkl_sparse_d_create_csr(
                &mkl_a,
                SPARSE_INDEX_BASE_ZERO,
                P.n,
                P.n,
                rows_start,
                rows_end,
                poses,
                vals
            );

            //run multiplication
            mkl_sparse_d_mm(
                SPARSE_OPERATION_TRANSPOSE,
                1.0,
                mkl_a,
                desc,
                SPARSE_LAYOUT_COLUMN_MAJOR,
                b_mat_slice,
                P.k,
                P.n,
                1.0,
                c_mat_slice,
                P.n
            );
            mkl_sparse_destroy(mkl_a);
        }
        else
        {
            // #pragma omp parallel for
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

    delete[] rows_start;
    delete[] rows_end;
    delete[] poses;
    delete[] vals;

    delete[] temp_a_1;
    delete[] temp_a_2;
    MPI_Comm_free(&group_comm);
}
