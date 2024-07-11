#ifndef PARALLEL_K2D_H
#define PARALLEL_K2D_H

#include "module_basis/module_ao/parallel_2d.h"
#include "module_cell/parallel_kpoints.h"
#include "module_hamilt_general/matrixblock.h"
#ifdef __MPI
#include "mpi.h"
#endif
#include "module_hamilt_general/hamilt.h"

/***
 * This is a class to realize k-points parallelism in LCAO code.
 * It is now designed only to work with 2D Eigenvalue solver parallelism.
 */

template <typename TK>
class Parallel_K2D {
  public:
    /**
     * Define Parallel_K2D as a singleton class
     */
    static Parallel_K2D& get_instance() {
        static Parallel_K2D instance;
        return instance;
    }
    /// delete copy constructor
    Parallel_K2D(const Parallel_K2D&) = delete;
    /// delete copy assignment
    Parallel_K2D& operator=(const Parallel_K2D&) = delete;

    /**
     * the pointer to Parallel_Kpoints
     */
    Parallel_Kpoints* Pkpoints = nullptr;
    Parallel_2D* P2D_global = nullptr;
    Parallel_2D* P2D_pool = nullptr;

    /**
     * the local Hk, Sk matrices in POOL_WORLD_K2D
     */
    std::vector<TK> hk_pool;
    std::vector<TK> sk_pool;

#ifdef __MPI
    MPI_Comm POOL_WORLD_K2D;
#endif

  public:
    /**
     * Public member functions
     */
    /// this function sets the parallel environment for k-points parallelism
    /// including the glabal and pool 2D parallel distribution
    void set_para_env(int nks,
                    const int& nw,
                    const int& nb2d,
                    const int& nproc,
                    const int& my_rank,
                    const int& nspin);

    /// this function distributes the Hk and Sk matrices to hk_pool and sk_pool
    void distribute_hsk(hamilt::Hamilt<TK>* pHamilt,
                        const std::vector<int>& ik_kpar,
                        const int& nw);

    /// this function unsets the parallel environment for k-points parallelism
    /// including the glabal and pool 2D parallel distribution
    void unset_para_env();
    /// set the number of k-points
    void set_kpar(int kpar);
    /// get the number of k-points
    int get_kpar() { return this->kpar_; }
    /// initialize the Parallel_K2D class
    void set_initialized(bool initialized) { this->initialized_ = initialized; }
    /// check if the Parallel_K2D class is initialized
    bool get_initialized() { return this->initialized_; }
    /// get my pool
    int get_my_pool() { return this->MY_POOL; }

  private:
    /**
     * Private member variables
     */
    int kpar_ = 0;
    bool initialized_ = false;
    /// private constructor
    Parallel_K2D() {}
    /// private destructor
    ~Parallel_K2D() {}

    /**
     * public mpi info
     */
    int NPROC_IN_POOL;
    int MY_POOL;
    int RANK_IN_POOL;
};

#endif