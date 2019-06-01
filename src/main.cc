#include <mpi.h>
#include <cstdio>
#include "mmul.hh"
#include "densematgen.h"

params P;
int id;
std::string filename;

int main (int argc, char** argv)
{
    MPI_Init(&argc,&argv); /* intialize the library with parameters caught by the runtime */

    MPI_Comm_size(MPI_COMM_WORLD, &P.p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // std::vector<double> elts;
    // std::vector<int> offsets;
    // std::vector<int> begs;

    std::vector<sparse_elt> sparse_mat;

    if (id == 0)
    {
        parse_cli(argc, argv);
        sparse_mat = load_file(filename);//, elts, offsets, begs);
    }

    // distribute params
    MPI_Bcast(&P, sizeof(params), MPI_BYTE, 0, MPI_COMM_WORLD);

    // allocate A matrix slice
    sparse_elt* a_mat_slice = new sparse_elt[P.k*P.q];

    // scatter/receive A matrix slice (horizontal or vertical)
    MPI_Scatter(
        sparse_mat.data(),
        sizeof(sparse_elt)*P.k*P.q,
        MPI_BYTE,
        a_mat_slice,
        sizeof(sparse_elt)*P.k*P.q,
        MPI_BYTE,
        0,
        MPI_COMM_WORLD
    );

    // destroy the loaded sparse matrix
    std::vector<sparse_elt>().swap(sparse_mat);

    // generate columns of dense B matrix

    double* b_mat_slice = new double[P.n*P.k];

    for (int y=0; y<P.n; y++)
        for (int x=0; x<P.k; x++)
            if (y < P.real_n && x+id*P.k < P.real_n)
                b_mat_slice[y*P.k + x] = generate_double(P.seed, y, x+id*P.k);
            else
                b_mat_slice[y*P.k + x] = 0;

    // if (P.inner)
    //     inner(a_mat_slice, b_mat_slice);
    // else
        column(a_mat_slice, b_mat_slice);

    // fprintf(stderr, "I am %d/%d and know that n is %d\n", id, P.p, P.n);

    delete[] a_mat_slice;
    // delete[] b_mat_slice;

    MPI_Finalize();
    return 0;
}