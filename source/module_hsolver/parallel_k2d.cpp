#include "parallel_k2d.h"

#include "module_base/parallel_global.h"
#include "module_base/scalapack_connector.h"

template <typename TK>
void Parallel_K2D<TK>::set_para_env(int nks,
                                    const int& nw,
                                    const int& nb2d,
                                    const int& nproc,
                                    const int& my_rank,
                                    const int& nspin) {
    int kpar = this->get_kpar();
    Parallel_Global::divide_mpi_groups(nproc,
                                       kpar,
                                       my_rank,
                                       this->NPROC_IN_POOL,
                                       this->MY_POOL,
                                       this->RANK_IN_POOL);
    MPI_Comm_split(MPI_COMM_WORLD,
                   this->MY_POOL,
                   this->RANK_IN_POOL,
                   &this->POOL_WORLD_K2D);
    this->Pkpoints = new Parallel_Kpoints;
    this->P2D_global = new Parallel_2D;
    this->P2D_pool = new Parallel_2D;
    this->Pkpoints
        ->kinfo(nks, kpar, this->MY_POOL, this->RANK_IN_POOL, nproc, nspin);
    this->P2D_global->init(nw, nw, nb2d, MPI_COMM_WORLD);
    this->P2D_pool->init(nw, nw, nb2d, this->POOL_WORLD_K2D);
}

template <typename TK>
void Parallel_K2D<TK>::distribute_hsk(hamilt::Hamilt<TK>* pHamilt,
                                      const std::vector<int>& ik_kpar,
                                      const int& nw) {
    this->set_initialized(false);
    for (int ipool = 0; ipool < ik_kpar.size(); ++ipool)
    {
        pHamilt->updateHk(ik_kpar[ipool]);
        hamilt::MatrixBlock<TK> HK_global, SK_global;
        pHamilt->matrix(HK_global, SK_global);
        if (this->MY_POOL == this->Pkpoints->whichpool[ik_kpar[ipool]]) {
            this->hk_pool.resize(this->P2D_pool->get_local_size(), 0.0);
            this->sk_pool.resize(this->P2D_pool->get_local_size(), 0.0);
        }
        int desc_pool[9];
        std::copy(this->P2D_pool->desc, this->P2D_pool->desc + 9, desc_pool);
        if (this->MY_POOL != this->Pkpoints->whichpool[ik_kpar[ipool]]) {
            desc_pool[1] = -1;
        }
        Cpxgemr2d(nw,
                  nw,
                  HK_global.p,
                  1,
                  1,
                  this->P2D_global->desc,
                  hk_pool.data(),
                  1,
                  1,
                  desc_pool,
                  this->P2D_global->blacs_ctxt);
        Cpxgemr2d(nw,
                  nw,
                  SK_global.p,
                  1,
                  1,
                  this->P2D_global->desc,
                  sk_pool.data(),
                  1,
                  1,
                  desc_pool,
                  this->P2D_global->blacs_ctxt);
    }
    this->set_initialized(true);
}

template <typename TK>
void Parallel_K2D<TK>::unset_para_env() {
    if (this->Pkpoints != nullptr) {
        delete this->Pkpoints;
        this->Pkpoints = nullptr;
    }
    if (this->P2D_global != nullptr) {
        delete this->P2D_global;
        this->P2D_global = nullptr;
    }
    if (this->P2D_pool != nullptr) {
        delete this->P2D_pool;
        this->P2D_pool = nullptr;
    }
    MPI_Comm_free(&this->POOL_WORLD_K2D);
}

template <typename TK>
void Parallel_K2D<TK>::set_kpar(int kpar) {
    if (kpar < 1) {
        ModuleBase::WARNING_QUIT("Parallel_K2D::set_kpar",
                                 "kpar must be greater than 0.");
    }
    this->kpar_ = kpar;
}

template <typename TK>
int Parallel_K2D<TK>::cal_ncol_bands(int nbands, Parallel_2D* p2d) {
    // for psi
    int end_id = 0;
    int block = nbands / p2d->nb;
    if (block * p2d->nb < nbands) {
        block++;
    }
    if (p2d->dim1 > block) {
        std::cout << " cpu 2D distribution : " << p2d->dim0 << "*" << p2d->dim1
                  << std::endl;
        std::cout << " but, the number of bands-row-block is " << block
                  << std::endl;
        ModuleBase::WARNING_QUIT("Parallel_K2D::cal_ncol_bands",
                                 "some processor has no bands-row-blocks.");
    }
    int col_b_bands = block / p2d->dim1;
    if (p2d->coord[1] < block % p2d->dim1) {
        col_b_bands++;
    }
    if (block % p2d->dim1 == 0) {
        end_id = p2d->dim1 - 1;
    } else {
        end_id = block % p2d->dim1 - 1;
    }
    int ncol_bands = 0;
    if (p2d->coord[1] == end_id) {
        ncol_bands
            = (col_b_bands - 1) * p2d->nb + (nbands - (block - 1) * p2d->nb);
    } else {
        ncol_bands = col_b_bands * p2d->nb;
    }
    return ncol_bands;
}

template class Parallel_K2D<double>;
template class Parallel_K2D<std::complex<double>>;