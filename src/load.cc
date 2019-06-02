#include "mmul.hh"

#include <cstdio>
#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;


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
    po::options_description desc("Allowed options");

    desc.add_options()
        ("help,h", "produce help message")
        ("input-file,f", po::value<std::string>(), "input file")
        ("seed,s", po::value<int>(), "seed for the matrix generation algorithm")
        ("inner,i", "turn on inner algorithm (by default: column)")
        ("verbose,v", "print the C matrix")
        ("replication,c", po::value<int>(), "c parameter of the algorithm")
        ("exponent,e", po::value<int>(), "number of matrix multiplications")
        ("greater-equal,g", po::value<double>(), "print number of elements of C greater or equal than the value")
        ("mkl,m", "turn on the MKL for matrix multiplication")
    ;

    po::variables_map vm;

    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))
    {
        std::cerr << desc << "\n";
        exit(-1);
    }

    if (
        vm.count("input-file") == 0 ||
        vm.count("seed") == 0 ||
        vm.count("replication") == 0 ||
        vm.count("exponent") == 0
    )
    {
        std::cerr<<"commandline options missing.\n"<<desc<<"\n";
        exit(-2);
    }

    P.seed = vm["seed"].as<int>();
    P.c = vm["replication"].as<int>();
    P.e = vm["exponent"].as<int>();

    P.inner = (vm.count("inner") != 0);
    P.verbose = (vm.count("verbose") != 0);
    P.mkl = (vm.count("mkl") != 0);
    P.ge_flag = (vm.count("greater-equal") != 0);
    if (P.ge_flag)
        P.ge_value = vm["greater-equal"].as<double>();

    filename = vm["input-file"].as<std::string>();

    if (P.p % P.c != 0)
    {
        std::cerr<<"group size does not divide process count\n";
        exit(-3);
    }

    if (P.inner && P.p % (P.c*P.c) != 0)
    {
        std::cerr<<"group size squared does not divide process count\n";
        exit(-3);
    }
}
