#include "mmul.hh"

#include <mpi.h>


void inner_multiply(
    sparse_elt* a_mat_slice,
    double* b_mat_slice,
    double* c_mat_slice
)
{
    int gid = id/P.c;
    int tid = id%P.c;
    // this worker (gid, tid) computes c/p x 1/c sized block (gid, tid)
    MPI_Comm group_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gid, tid, &group_comm);

    sparse_elt* temp_a_1 = new sparse_elt[P.k*P.c*P.q];
    sparse_elt* temp_a_2 = new sparse_elt[P.k*P.c*P.q];

    // the format for gathered b, c is awkward (3d) but heh, happens...
    double* b_mat_gathered = new double[P.k*P.c*P.n];
    double* c_mat_gathered = new double[P.k*P.n];

    MPI_Request* requests = new MPI_Request[P.c+1];
    MPI_Status* statuses = new MPI_Status[P.c+1];

    // gather A, B and C into c-sized blocks
    MPI_Iallgather(
        a_mat_slice,
        sizeof(sparse_elt)*P.k*P.q,
        MPI_BYTE,
        temp_a_1,
        sizeof(sparse_elt)*P.k*P.q,
        MPI_BYTE,
        group_comm,
        &requests[0]
    );
    MPI_Iallgather(
        b_mat_slice,
        sizeof(double)*P.k*P.n,
        MPI_BYTE,
        b_mat_gathered,
        sizeof(double)*P.k*P.n,
        MPI_BYTE,
        group_comm,
        &requests[1]
    );

    for (int i=0; i<P.n*P.k; i++)
        c_mat_gathered[i] = 0;

    MPI_Waitall(2, requests, statuses);


    // rotate A (actually "transpose" switching tid and gid%c)
    int q = P.p/(P.c*P.c);

    MPI_Isend(
        temp_a_1,
        sizeof(sparse_elt)*P.k*P.c*P.q,
        MPI_BYTE,
        P.c * (tid*q + gid%q) + gid/q,
        0,
        MPI_COMM_WORLD,
        &requests[0]
    );
    MPI_Irecv(
        temp_a_2,
        sizeof(sparse_elt)*P.k*P.c*P.q,
        MPI_BYTE,
        P.c * (tid*q + gid%q) + gid/q,
        0,
        MPI_COMM_WORLD,
        &requests[1]
    );

    MPI_Waitall(2, requests, statuses);

    std::swap(temp_a_1, temp_a_2);


    for (int i=0; i<q; i++) // rounds
    {
        // communicate
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
        #pragma omp parallel for
        for (int dy = 0; dy < P.k*P.c; dy++)
        {
            int dy2 = dy + ((i+gid)%q) * P.k*P.c;
            int y = dy2 + tid * P.n/P.c;
            for (int j=0; j<P.q; j++)
            {
                auto elt = temp_a_1[dy*P.q + j];
                if (elt.pos == -1)
                    continue;
                int z = elt.pos;
                for (int dx=0; dx<P.k*P.c; dx++)
                {
                    // x = dx + gid*P.k*P.c;
                    c_mat_gathered[(dx/P.k) * P.k*P.n/P.c + dy2 * P.k + dx%P.k] += elt.val * b_mat_gathered[(dx/P.k) * P.k*P.n + z * P.k + dx%P.k];
                }
            }
        }

        // sync
        MPI_Waitall(2, requests, statuses);

        std::swap(temp_a_1, temp_a_2);
    }

    for (int i=0; i<P.c; i++)
        MPI_Igather(
            c_mat_gathered + i*(P.k*P.n/P.c),
            sizeof(double)*P.k*P.n/P.c,
            MPI_BYTE,
            c_mat_slice,
            sizeof(double)*P.k*P.n/P.c,
            MPI_BYTE,
            i,
            group_comm,
            &requests[i]
        );
    MPI_Waitall(P.c, requests, statuses);

    delete[] temp_a_1;
    delete[] temp_a_2;
    delete[] b_mat_gathered;
    delete[] c_mat_gathered;
    delete[] requests;
    delete[] statuses;

    MPI_Comm_free(&group_comm);
}
