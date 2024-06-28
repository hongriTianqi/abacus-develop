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
    for (int ik = 0; ik < nks; ++ik)
    {
        pHamilt->updateHk(ik);
        hamilt::MatrixBlock<TK> HK_global, SK_global;
        pHamilt->matrix(HK_global, SK_global);
        // collect Hk to hk_full
        Cpxgemr2d(nw, nw, HK_global.p, 1, 1, HK_global.desc,
            hk_full.data() + ik * nw * nw, 1, 1, pv_helper.desc, pv_helper.blacs_ctxt);
        // collect Sk to sk_full
        Cpxgemr2d(nw, nw, SK_global.p, 1, 1, SK_global.desc,
            sk_full.data() + ik * nw * nw, 1, 1, pv_helper.desc, pv_helper.blacs_ctxt);
    }
    pv_helper.set(nw, nw, nw, this->P2D_local->comm_2D, this->P2D_local->blacs_ctxt);
    this->hk_local.resize(this->P2D_local->get_row_size() * this->P2D_local->get_col_size());
    this->sk_local.resize(this->P2D_local->get_row_size() * this->P2D_local->get_col_size());
    for (int ik = 0; ik < this->Pkpoints->nks_pool[this->MY_POOL]; ik++)
    {
        int ik_global = ik + this->Pkpoints->startk_pool[this->MY_POOL];
        // distribute Hk to hk_local
        Cpxgemr2d(nw, nw, hk_full.data() + ik_global * nw * nw, 1, 1, pv_helper.desc,
            this->hk_local.data(), 1, 1, this->P2D_local->desc, pv_helper.blacs_ctxt);
        // distribute Sk to sk_local
        Cpxgemr2d(nw, nw, sk_full.data() + ik_global * nw * nw, 1, 1, pv_helper.desc,
            this->sk_local.data(), 1, 1, this->P2D_local->desc, pv_helper.blacs_ctxt);
    }
}

template <typename TK>
void Parallel_K2D<TK>::unset_para_env()
{
    if (this->Pkpoints != nullptr) delete this->Pkpoints;
    if (this->P2D_global != nullptr) delete this->P2D_global;
    if (this->P2D_local != nullptr) delete this->P2D_local;
    MPI_Comm_free(&this->POOL_WORLD_K2D);
}

template class Parallel_K2D<double>;
template class Parallel_K2D<std::complex<double>>;