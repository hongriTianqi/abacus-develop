#ifndef PARALLEL_K2D_H
#define PARALLEL_K2D_H

#include "module_cell/parallel_kpoints.h"
#include "module_basis/module_ao/parallel_2d.h"
#include "module_hamilt_general/matrixblock.h"
#ifdef __MPI
#include "mpi.h"
#endif

/***
 * This is a class to realize k-points parallelism in LCAO code.
 * It is now designed only to work with 2D Eigenvalue solver parallelism.
 */

template <typename TK>
class Parallel_K2D
{
public:
    /**
     * Define Parallel_K2D as a signleton class
     */
    static Parallel_K2D& get_instance()
    {
        static Parallel_K2D instance;
        return instance;
    }
    /// delete copy constructor
    Parallel_K2D(const Parallel_K2D&) = delete;
    /// delete copy assignment
    Parallel_K2D& operator=(const Parallel_K2D&) = delete;

    /**
     * Public member functions
     */
    /// set the number of k-points
    void set_kpar(int kpar) { kpar_ = kpar; }
    /// get the number of k-points
    int get_kpar() const { return kpar_; }
    /// set the total number of k-points
    void set_nkstot(int nkstot) { nkstot_ = nkstot; }
    /// get the total number of k-points
    int get_nkstot() const { return nkstot_; }

    /**
     * the pointer to Parallel_Kpoints
     */
    Parallel_Kpoints* Pkpoints = nullptr;
    Parallel_2D* P2D_global = nullptr;
    Parallel_2D* P2D_local = nullptr;

    /**
     * the global Hk, Sk, and dmk matrices in large Parallel_2D format
     */
    std::vector<hamilt::MatrixBlock<TK>> HK_global;
    std::vector<hamilt::MatrixBlock<TK>> SK_global;

    /**
     * the local Hk, Sk, and dmk matrices in small Parallel_2D format
     */
    std::vector<hamilt::MatrixBlock<TK>> HK_local;
    std::vector<hamilt::MatrixBlock<TK>> SK_local;
    /**
     * public mpi info
     */
    int NPROC_IN_POOL;
    int MY_POOL;
    int RANK_IN_POOL;

#ifdef __MPI
    MPI_Comm POOL_WORLD_K2D; // mohan add 2012-01-13
#endif

private:
    /**
     * Public member functions
     */
    /// private constructor
    Parallel_K2D()
    {
    }
    /// private destructor
    ~Parallel_K2D()
    {
    }

private:
    /**
     * Private member variables
     */
    int kpar_ = 1;
    int nkstot_ = 0;
private:
    /**
     * Private variables to deal with input
     */
    static bool initialized;

public:
    /**
     * Public member functions for inputs
     */
    /// initialize the Parallel_K2D class
    static void set_initialized(bool initialized_in) { Parallel_K2D<TK>::initialized = initialized_in; }
    /// check if the Parallel_K2D class is initialized
    static bool get_initialized() { return Parallel_K2D<TK>::initialized; }

};

#endif