#include "mmul.hh"

#include <cstdio>
#include <getopt.h>


std::vector<sparse_elt> load_file(std::string fname)
{
    FILE* f;

    f = fopen(fname.c_str(), "r");
    if (f == nullptr)
    {
        fprintf(stderr, "failed to open file %s\n", fname.c_str());
        exit(-1);
    }

    int w, h, k, q;
    fscanf(f, "%d%d%d%d", &w, &h, &k, &q);

    if (w != h)
    {
        fprintf(stderr, "Number of rows and cols (%d and %d) does not match\n", w, h);
        exit(-1);
    }

    P.real_n = w;
    P.k = (P.real_n+P.p-1)/P.p;
    P.n = P.k*P.p;
    P.q = q;

    fprintf(stderr, "Matrix of size %d (aligned to %d with %d processes)\n", P.real_n, P.n, P.p);

    std::vector<double> elts;
    std::vector<int> offsets;
    std::vector<int> begs;

    elts.resize(k);
    offsets.resize(k);
    begs.resize(P.real_n+1);

    for (int i=0; i<k; i++)
        fscanf(f, "%lf", &elts[i]);

    for (int i=0; i<=P.real_n; i++)
        fscanf(f, "%d", &begs[i]);

    for (int i=0; i<k; i++)
        fscanf(f, "%d", &offsets[i]);

    std::vector<sparse_elt> ret;

    if (P.inner)
    // no need to transpose
    {
        ret.resize(P.n*P.q, {-1, 0});

        for (int y=0; y<P.real_n; y++)
            for (int x=0, i=begs[y]; i<begs[y+1]; x++, i++)
                ret[y*P.q + x] = {offsets[i], elts[i]};
    }
    else
    // transpose the sparse matrix
    {
        std::vector<int> off_copy = offsets;
        int top = 0, cur = 0, last = -1;
        std::sort(off_copy.begin(), off_copy.end());
        for (int i=0; i<off_copy.size(); i++)
        {
            if (off_copy[i] != last)
            {
                top = std::max(top, cur);
                last = off_copy[i];
                cur = 1;
            }
            else
                cur += 1;
        }
        top = std::max(top, cur);
        P.q = top;
        ret.resize(P.n*P.q, {-1, 0});

        std::vector<int> counts;
        counts.resize(P.n, 0);

        for (int y=0; y<P.real_n; y++)
            for (int i=begs[y]; i<begs[y+1]; i++)
            {
                int x = offsets[i];
                ret[x*P.q + (counts[x]++)] = {y, elts[i]};
            }
    }

    return ret;
}


void parse_cli(int argc, char** argv)
{
    P.seed = -1;
    int c;
    while ((c = getopt(argc, argv, "f:s:ic:e:g:vm")) != -1)
    {
        switch(c)
        {
            case 'f':
                filename = optarg;
                break;
            case 's':
                P.seed = atoi(optarg);
                break;
            case 'i':
                P.inner = 1;
                break;
            case 'v':
                P.verbose = 1;
                break;
            case 'c':
                P.c = atoi(optarg);
                break;
            case 'e':
                P.e = atoi(optarg);
                break;
            case 'g':
                P.ge_flag = 1;
                P.ge_value = atof(optarg);
                break;
        }
    }

    if (filename.size() == 0 || P.seed == -1 || P.c == 0 || P.e == 0)
    {
        fprintf(stderr, "commandline options missing.\n");
        exit(-2);
    }

    if (P.p % P.c != 0)
    {
        fprintf(stderr, "group size does not divide process count\n");
        exit(-3);
    }

    if (P.inner && P.p % (P.c*P.c) != 0)
    {
        fprintf(stderr, "group size squared does not divide process count\n");
        exit(-3);
    }
}
