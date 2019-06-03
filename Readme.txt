The project is still in a very crude, unrefined state. I will send a better version after the deadline.
1) It currently compiles on Okeanos only with GCC 5.3;  `module swap PrgEnv-cray PrgEnv-gnu` and `module swap gcc/4.9.3 gcc/5.3.0` are required
2) The report is non-existent
3) The performance is untested, though the algorithm should be correct.
4) The code uses OpenMP, so please use `--tasks-per-node 2`.
