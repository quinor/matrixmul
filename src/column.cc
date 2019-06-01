#include "mmul.hh"

#include <mpi.h>


static void multiply(
    sparse_elt* a_mat_slice,
    double* b_mat_slice,
    double* c_mat_slice
)
{
    sparse_elt* temp_a_1 = new sparse_elt[P.k*P.q];
    sparse_elt* temp_a_2 = new sparse_elt[P.k*P.q];

    for (int i=0; i<P.k*P.q; i++)
        temp_a_1[i] = a_mat_slice[i];


    for (int i=0; i<P.n*P.k; i++)
        c_mat_slice[i] = 0;

    for (int i=0; i<P.p; i++) // rounds
    {
        // communicate
        MPI_Request requests[2];
        MPI_Status statuses[2];

        MPI_Isend(
            temp_a_1,
            sizeof(sparse_elt)*P.k*P.q,
            MPI_BYTE,
            (P.p+id-1)%P.p,
            0,
            MPI_COMM_WORLD,
            &requests[0]
        );
        MPI_Irecv(
            temp_a_2,
            sizeof(sparse_elt)*P.k*P.q,
            MPI_BYTE,
            (id+1)%P.p,
            0,
            MPI_COMM_WORLD,
            &requests[1]
        );

        // multiply
        // #pragma omp parallel for
        for (int dx=0; dx<P.k; dx++)
        {
            // int x = dx + id*P.k;
            for (int dz = 0; dz < P.k; dz++)
            {
                int z = dz + ((i+id)%P.p)*P.k;
                for (int j=0; j<P.q; j++)
                {
                    auto elt = temp_a_1[dz*P.q + j];
                    if (elt.pos == -1)
                        continue;
                    int y = elt.pos;
                    c_mat_slice[y*P.k + dx] += elt.val * b_mat_slice[z*P.k + dx];
                }
            }
        }

        // sync
        MPI_Waitall(2, requests, statuses);

        std::swap(temp_a_1, temp_a_2);
    }

    delete[] temp_a_1;
    delete[] temp_a_2;
}

void column(sparse_elt* a_mat_slice, double* b_mat_slice)
{
    double* c_mat_slice = new double[P.n*P.k];

    for (int i=0; i<P.e; i++)
    {
        multiply(a_mat_slice, b_mat_slice, c_mat_slice);
        std::swap(b_mat_slice, c_mat_slice);
    }

    if (P.verbose)
    {
        // print the result (in b)

        double* result;
        if (id == 0)
            result = new double[P.n*P.n];

        MPI_Gather(
            b_mat_slice,
            sizeof(double)*P.n*P.k,
            MPI_BYTE,
            result,
            sizeof(double)*P.n*P.k,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD
        );

        if (id == 0)
        {
            printf("%d %d\n", P.real_n, P.real_n);
            for (int y=0; y<P.real_n; y++)
            {
                for (int x=0; x<P.real_n; x++)
                    printf("% 12.5lf ", result[(x/P.k)*P.n*P.k + y*P.k + x%P.k]);
                printf("\n");
            }

            delete[] result;
        }
    }

    delete[] b_mat_slice;
    delete[] c_mat_slice;
}
