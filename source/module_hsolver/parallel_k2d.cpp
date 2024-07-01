#include "parallel_k2d.h"
#include "module_base/parallel_global.h"
#include "module_base/scalapack_connector.h"

template <typename TK>
void Parallel_K2D<TK>::set_para_env(hamilt::Hamilt<TK>* pHamilt,
    int nks,
    const int& nw,
    const int& nb2d,
    const int& nproc,
    const int& my_rank,
    const int& nspin)
{
    this->set_initialized(false);
    int kpar = this->get_kpar();
    std::cout << "nkstot = " << nks << std::endl;
    std::cout << "kpar = " << kpar << std::endl;
    Parallel_Global::divide_mpi_groups(nproc,
                                           kpar,
                                           my_rank,
                                           this->NPROC_IN_POOL,
                                           this->MY_POOL,
                                           this->RANK_IN_POOL);
    MPI_Comm_split(MPI_COMM_WORLD, this->MY_POOL, this->RANK_IN_POOL,&this->POOL_WORLD_K2D);
    this->Pkpoints = new Parallel_Kpoints;
    this->P2D_global = new Parallel_2D;
    this->P2D_local = new Parallel_2D;
    this->Pkpoints->kinfo(nks, kpar, this->MY_POOL, this->RANK_IN_POOL, nproc, nspin);
    /*
    for (int ipool = 0; ipool < k2d.get_kpar(); ipool++)
    {
        std::cout << "nks_pool[" << ipool << "] = " << k2d.Pkpoints->nks_pool[ipool] << std::endl;
        std::cout << "startk_pool[" << ipool << "] = " << k2d.Pkpoints->startk_pool[ipool] << std::endl;
        std::cout << "startpro_pool[" << ipool << "] = " << k2d.Pkpoints->get_startpro_pool(ipool) << std::endl;
    }
    */
    this->P2D_global->init(nw, nw, nb2d, MPI_COMM_WORLD);
    this->P2D_local->init(nw, nw, nb2d, this->POOL_WORLD_K2D);
    std::vector<TK> hk_full(nks * nw*nw);
    std::vector<TK> sk_full(nks * nw*nw);
    Parallel_2D pv_helper;
    pv_helper.set(nw, nw, nw, this->P2D_global->comm_2D, this->P2D_global->blacs_ctxt);

    int nks_pool = this->Pkpoints->nks_pool[this->MY_POOL];
    hk_local.resize(nks_pool);
    sk_local.resize(nks_pool);
    for (int ik = 0; ik < nks_pool; ik++)
    {
        hk_local[ik].resize(this->P2D_local->get_local_size(), 0.0);
        sk_local[ik].resize(this->P2D_local->get_local_size(), 0.0);
    }
    /// distribute Hk and Sk to hk_local and sk_local
    for (int ik = 0; ik < nks; ++ik)
    {
        pHamilt->updateHk(ik);
        hamilt::MatrixBlock<TK> HK_global, SK_global;
        pHamilt->matrix(HK_global, SK_global);
        //
        int desc_pool[9];
        std::copy(this->P2D_local->desc, this->P2D_local->desc + 9, desc_pool);
        if (this->MY_POOL != this->Pkpoints->whichpool[ik]) {
            desc_pool[1] = -1;
        }
        int ik_pool = ik - this->Pkpoints->startk_pool[this->MY_POOL];
        Cpxgemr2d(nw, nw, HK_global.p, 1, 1, this->P2D_global->desc,
            hk_local[ik_pool].data(), 1, 1, desc_pool, this->P2D_global->blacs_ctxt);
        Cpxgemr2d(nw, nw, SK_global.p, 1, 1, this->P2D_global->desc,
            sk_local[ik_pool].data(), 1, 1, desc_pool, this->P2D_global->blacs_ctxt);
    }
    this->set_initialized(true);
}

template <typename TK>
void Parallel_K2D<TK>::unset_para_env()
{
    if (this->Pkpoints != nullptr) delete this->Pkpoints;
    if (this->P2D_global != nullptr) delete this->P2D_global;
    if (this->P2D_local != nullptr) delete this->P2D_local;
    MPI_Comm_free(&this->POOL_WORLD_K2D);
}

template <typename TK>
int Parallel_K2D<TK>::ik = 0;

template class Parallel_K2D<double>;
template class Parallel_K2D<std::complex<double>>;