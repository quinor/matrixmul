#pragma once

#include <string>
#include <vector>

struct params {
    int real_n;             // matrix size
    int n;                  // adjusted matrix size
    int q;                  // max elts per sparse vector
    int p;                  // processes
    int k;                  // == n/p
    int c;                  // replication size
    int seed;               // seed
    int e;                  // exponent

    bool inner;             // turn on inner algorithm
    bool verbose;           // print result matrix
    bool mkl;               // use mkl
    bool ge_flag;           // use greater-equal output
    double ge_value;        // greater-equal threshold
};

struct sparse_elt {
    int pos;
    double val;
};

extern params P;
extern int id;
extern std::string filename;


std::vector<sparse_elt> load_file(std::string fname);

void parse_cli(int argc, char** argv);

void inner(sparse_elt* a_mat_slice, double* b_mat_slice);
void column(sparse_elt* a_mat_slice, double* b_mat_slice);
