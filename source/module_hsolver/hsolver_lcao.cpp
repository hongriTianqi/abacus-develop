#include "hsolver_lcao.h"

#include "diago_cg.h"
#include "diago_scalapack.h"
#include "module_base/timer.h"
#include "module_hsolver/diago_iter_assist.h"
#include "module_hsolver/kernels/math_kernel_op.h"
#include "module_io/write_HS.h"

#include <ATen/core/tensor.h>
#include <ATen/core/tensor_map.h>
#include <ATen/core/tensor_types.h>
#ifdef __CUSOLVERMP
#include "diago_cusolvermp.h"
#endif // __CUSOLVERMP
#ifdef __ELPA
#include "diago_elpa.h"
#endif
#ifdef __CUDA
#include "diago_cusolver.h"
#endif
#ifdef __PEXSI
#include "diago_pexsi.h"
#include "module_elecstate/elecstate_lcao.h"
#endif

#include "module_hsolver/parallel_k2d.h"
#include "module_base/scalapack_connector.h"
#include <unistd.h>

namespace hsolver
{

template <typename T, typename Device>
void HSolverLCAO<T, Device>::solveTemplate(hamilt::Hamilt<T>* pHamilt,
                                           psi::Psi<T>& psi,
                                           elecstate::ElecState* pes,
                                           const std::string method_in,
                                           const bool skip_charge)
{
    ModuleBase::TITLE("HSolverLCAO", "solve");
    ModuleBase::timer::tick("HSolverLCAO", "solve");
    // select the method of diagonalization
    this->method = method_in;

    // init
    if (this->method == "scalapack_gvx")
    {
        if (this->pdiagh != nullptr)
        {
            if (this->pdiagh->method != this->method)
            {
                delete[] this->pdiagh;
                this->pdiagh = nullptr;
            }
        }
        if (this->pdiagh == nullptr)
        {
            this->pdiagh = new DiagoScalapack<T>();
            this->pdiagh->method = this->method;
        }
    }
#ifdef __ELPA
    else if (this->method == "genelpa")
    {
        if (this->pdiagh != nullptr)
        {
            if (this->pdiagh->method != this->method)
            {
                delete[] this->pdiagh;
                this->pdiagh = nullptr;
            }
        }
        if (this->pdiagh == nullptr)
        {
            this->pdiagh = new DiagoElpa<T>();
            this->pdiagh->method = this->method;
        }
    }
#endif
#ifdef __CUDA
    else if (this->method == "cusolver")
    {
        if (this->pdiagh != nullptr)
        {
            if (this->pdiagh->method != this->method)
            {
                delete[] this->pdiagh;
                this->pdiagh = nullptr;
            }
        }
        if (this->pdiagh == nullptr)
        {
            this->pdiagh = new DiagoCusolver<T>(this->ParaV);
            this->pdiagh->method = this->method;
        }
    }
    else if (this->method == "cusolvermp")
    {
#ifdef __CUSOLVERMP
        if (this->pdiagh != nullptr)
        {
            if (this->pdiagh->method != this->method)
            {
                delete[] this->pdiagh;
                this->pdiagh = nullptr;
            }
        }
        if (this->pdiagh == nullptr)
        {
            this->pdiagh = new DiagoCusolverMP<T>();
            this->pdiagh->method = this->method;
        }
#else
        ModuleBase::WARNING_QUIT("HSolverLCAO", "CUSOLVERMP did not compiled!");
#endif
    }
#endif
    else if (this->method == "lapack")
    {
        ModuleBase::WARNING_QUIT("hsolver_lcao", "please fix lapack solver!!!");
        // We are not supporting diagonalization with lapack
        // until the obsolete globalc::hm is removed from
        // diago_lapack.cpp
        /*
        if (this->pdiagh != nullptr)
        {
            if (this->pdiagh->method != this->method)
            {
                delete[] this->pdiagh;
                this->pdiagh = nullptr;
            }
        }
        if (this->pdiagh == nullptr)
        {
            this->pdiagh = new DiagoLapack();
            this->pdiagh->method = this->method;
        }
        */
        ModuleBase::WARNING_QUIT("HSolverLCAO::solve", "This method of DiagH is not supported!");
    }
    else if (this->method == "cg_in_lcao")
    {

        if (this->pdiagh != nullptr)
        {
            if (this->pdiagh->method != this->method)
            {
                delete reinterpret_cast<DiagoCG<T>*>(this->pdiagh);
                this->pdiagh = nullptr;
            }
        }
        if (this->pdiagh == nullptr)
        {
            auto subspace_func = [](const ct::Tensor& psi_in, ct::Tensor& psi_out) {
                // psi_in should be a 2D tensor:
                // psi_in.shape() = [nbands, nbasis]
                const auto ndim = psi_in.shape().ndim();
                REQUIRES_OK(ndim == 2, "dims of psi_in should be less than or equal to 2");
            };
            this->pdiagh = new DiagoCG<T>(GlobalV::BASIS_TYPE,
                                          GlobalV::CALCULATION,
                                          false,
                                          subspace_func,
                                          DiagoIterAssist<T>::LCAO_DIAG_THR,
                                          DiagoIterAssist<T>::LCAO_DIAG_NMAX,
                                          GlobalV::NPROC_IN_POOL);
            this->pdiagh->method = this->method;
        }
    }
#ifdef __PEXSI
    else if (this->method == "pexsi")
    {
        if (this->pdiagh != nullptr)
        {
            if (this->pdiagh->method != this->method)
            {
                delete[] this->pdiagh;
                this->pdiagh = nullptr;
            }
            auto tem = dynamic_cast<DiagoPexsi<T>*>(this->pdiagh);
        }
        if (this->pdiagh == nullptr)
        {
            DiagoPexsi<T>* tem = new DiagoPexsi<T>(this->ParaV);
            this->pdiagh = tem;
            // this->pdiagh = dynamic_cast<DiagoPexsi<T>*>(tem);
            this->pdiagh->method = this->method;
        }
    }
#endif
    else
    {
        ModuleBase::WARNING_QUIT("HSolverLCAO::solve", "This method of DiagH is not supported!");
    }

    if (this->method == "cg_in_lcao")
    {
        this->precondition_lcao.resize(psi.get_nbasis());

        using Real = typename GetTypeReal<T>::type;
        // set precondition
        for (size_t i = 0; i < precondition_lcao.size(); i++)
        {
            precondition_lcao[i] = 1.0;
        }
    }

    if (Parallel_K2D<double>::get_instance().get_kpar() > 1)
    {
        auto &k2d = Parallel_K2D<T>::get_instance();
        k2d.set_para_env(pHamilt,
                        psi.get_nk(),
                        GlobalV::NLOCAL,
                        GlobalV::NB2D,
                        GlobalV::NPROC,
                        GlobalV::MY_RANK,
                        GlobalV::NSPIN);
        /// set psi_pool
        auto psi_pool = psi::Psi<T>(psi.get_nk(), psi.get_nbands(), k2d.P2D_local->nrow, nullptr);
        for (int ik = 0; ik < psi.get_nk(); ++ik)
        {
            for (int ib = 0; ib < psi_pool.get_nbands(); ++ib)
            {
                for (int i = 0; i < psi_pool.get_nbasis(); ++i)
                {
                    psi_pool(ik, ib, i) = 0.0;
                }
            }
        }
        /// Loop over k points for solve Hamiltonian to charge density
        for (int ik = 0; ik < k2d.Pkpoints->nks_pool[k2d.MY_POOL]; ++ik)
        {
            k2d.ik = ik;
            /// global index of k point
            int ik_global = ik + k2d.Pkpoints->startk_pool[k2d.MY_POOL];
            /// local psi in pool
            psi_pool.fix_k(ik_global);
            /// solve eigenvector and eigenvalue for H(k)
            this->hamiltSolvePsiK(pHamilt, psi_pool, &(pes->ekb(ik_global, 0)));
        }

        for (int ik = 0; ik < 1; ++ik)
        {
            psi_pool.fix_k(ik);
            /*
            for (int irank = 0; irank < GlobalV::NPROC; irank++)
            {
                if (GlobalV::MY_RANK == irank)
                {
                    std::cout << " psi_pool MY_RANK = " << GlobalV::MY_RANK << " ik = " << ik
                    << " nbands = " << psi_pool.get_nbands()
                    << " nbasis = " << psi_pool.get_nbasis() << " ";
                    for (int ib = 0; ib < psi_pool.get_nbands(); ib++)
                    {
                        for (int i = 0; i < psi_pool.get_nbasis(); i++)
                        {
                            std::cout << psi_pool(ib, i) << " ";
                        }
                    }
                    std::cout << std::endl;
                    std::cout << std::endl;
                }
                usleep(10000);
            }
            */
            psi.fix_k(ik);
            int desc_pool[9];
            std::copy(k2d.P2D_local->desc, k2d.P2D_local->desc + 9, desc_pool);
            if (k2d.MY_POOL != k2d.Pkpoints->whichpool[ik])
            {
                desc_pool[1] = -1;
            }
            Cpxgemr2d(k2d.P2D_global->get_row_size(), k2d.P2D_global->get_col_size(), psi_pool.get_pointer(), 1, 1, desc_pool,
                    psi.get_pointer(), 1, 1, k2d.P2D_global->desc, k2d.P2D_global->blacs_ctxt);
            /*
            for (int irank = 0; irank < GlobalV::NPROC; irank++)
            {
                if (GlobalV::MY_RANK == irank)
                {
                    std::cout << " psi MY_RANK = " << GlobalV::MY_RANK << " ik_global = " << ik
                    << " nbands = " << psi.get_nbands()
                    << " nbasis = " << psi.get_nbasis() << " ";
                    //std::cout << "ik = " << ik << " MY_RANK = " << GlobalV::MY_RANK << " MY_POOL " << k2d.MY_POOL << " desc_pool[ik][1] = " << desc_pool[1] << std::endl;
                    for (int ib = 0; ib < psi.get_nbands(); ib++)
                    {
                        for (int i = 0; i < psi.get_nbasis(); i++)
                        {
                            std::cout << psi(ib, i) << " ";
                        }
                    }
                    std::cout << std::endl;
                    std::cout << std::endl;
                }
                usleep(10000);
            }
            */
        //
        }
        k2d.unset_para_env();
        MPI_Barrier(MPI_COMM_WORLD);
        //std::cout << __FILE__ << " " << __LINE__ << std::endl;
        //exit(0);
        //exit(0);
    }
    else
    {
        /// Loop over k points for solve Hamiltonian to charge density
        for (int ik = 0; ik < psi.get_nk(); ++ik)
        {
            /// update H(k) for each k point
            pHamilt->updateHk(ik);

            psi.fix_k(ik);

            // solve eigenvector and eigenvalue for H(k)
            this->hamiltSolvePsiK(pHamilt, psi, &(pes->ekb(ik, 0)));
            //
            for (int irank = 0; irank < GlobalV::NPROC; irank++)
            {
                if (GlobalV::MY_RANK == irank)
                {
                    std::cout << " MY_RANK = " << GlobalV::MY_RANK << " ik_global = " << ik
                    << " nbands = " << psi.get_nbands()
                    << " nbasis = " << psi.get_nbasis() << " ";
                    for (int ib = 0; ib < psi.get_nbands(); ib++)
                    {
                        for (int i = 0; i < psi.get_nbasis(); i++)
                        {
                            std::cout << psi(ib, i) << " ";
                        }
                    }
                    std::cout << std::endl;
                    std::cout << std::endl;
                }
            }
            //
        }
    }

    if (this->method == "cg_in_lcao")
    {
        this->is_first_scf = false;
    }

    if (this->method != "genelpa" && this->method != "scalapack_gvx" && this->method != "lapack"
        && this->method != "cusolver" && this->method != "cusolvermp" && this->method != "cg_in_lcao"
        && this->method != "pexsi")
    {
        delete this->pdiagh;
        this->pdiagh = nullptr;
    }

    // used in nscf calculation
    if (skip_charge)
    {
        ModuleBase::timer::tick("HSolverLCAO", "solve");
        return;
    }

    // calculate charge by psi
    // called in scf calculation
#ifdef __PEXSI
    if (this->method == "pexsi")
    {
        DiagoPexsi<T>* tem = dynamic_cast<DiagoPexsi<T>*>(this->pdiagh);
        if (tem == nullptr)
            ModuleBase::WARNING_QUIT("HSolverLCAO", "pexsi need debug!");
        elecstate::ElecStateLCAO<T>* _pes = dynamic_cast<elecstate::ElecStateLCAO<T>*>(pes);
        pes->f_en.eband = tem->totalFreeEnergy;
        // maybe eferm could be dealt with in the future
        _pes->dmToRho(tem->DM, tem->EDM);
    }
    else
#endif
    {
        pes->psiToRho(psi);
    }
    ModuleBase::timer::tick("HSolverLCAO", "solve");
}

template <typename T, typename Device>
void HSolverLCAO<T, Device>::solve(hamilt::Hamilt<T>* pHamilt,
                                   psi::Psi<T>& psi,
                                   elecstate::ElecState* pes,
                                   const std::string method_in,
                                   const bool skip_charge)
{
    this->solveTemplate(pHamilt, psi, pes, this->method, skip_charge);
}

template <typename T, typename Device>
void HSolverLCAO<T, Device>::hamiltSolvePsiK(hamilt::Hamilt<T>* hm, psi::Psi<T>& psi, double* eigenvalue)
{
    ModuleBase::TITLE("HSolverLCAO", "hamiltSolvePsiK");
    ModuleBase::timer::tick("HSolverLCAO", "hamiltSolvePsiK");

    if (this->method != "cg_in_lcao")
    {
        this->pdiagh->diag(hm, psi, eigenvalue);
    }
    else
    {

        using ct_Device = typename ct::PsiToContainer<base_device::DEVICE_CPU>::type;
        auto cg = reinterpret_cast<DiagoCG<T>*>(this->pdiagh);

        hamilt::MatrixBlock<T> h_mat, s_mat;
        hm->matrix(h_mat, s_mat);

        // set h_mat & s_mat
        for (int i = 0; i < h_mat.row; i++)
        {
            for (int j = i; j < h_mat.col; j++)
            {
                h_mat.p[h_mat.row * j + i] = hsolver::my_conj(h_mat.p[h_mat.row * i + j]);
                s_mat.p[s_mat.row * j + i] = hsolver::my_conj(s_mat.p[s_mat.row * i + j]);
            }
        }

        const T *one_ = nullptr, *zero_ = nullptr;
        one_ = new T(static_cast<T>(1.0));
        zero_ = new T(static_cast<T>(0.0));

        auto hpsi_func = [h_mat, one_, zero_](const ct::Tensor& psi_in, ct::Tensor& hpsi_out) {
            ModuleBase::timer::tick("DiagoCG_New", "hpsi_func");
            // psi_in should be a 2D tensor:
            // psi_in.shape() = [nbands, nbasis]
            const auto ndim = psi_in.shape().ndim();
            REQUIRES_OK(ndim <= 2, "dims of psi_in should be less than or equal to 2");

            Device* ctx = {};

            gemv_op<T, Device>()(ctx,
                                 'N',
                                 h_mat.row,
                                 h_mat.col,
                                 one_,
                                 h_mat.p,
                                 h_mat.row,
                                 psi_in.data<T>(),
                                 1,
                                 zero_,
                                 hpsi_out.data<T>(),
                                 1);

            ModuleBase::timer::tick("DiagoCG_New", "hpsi_func");
        };

        auto spsi_func = [s_mat, one_, zero_](const ct::Tensor& psi_in, ct::Tensor& spsi_out) {
            ModuleBase::timer::tick("DiagoCG_New", "spsi_func");
            // psi_in should be a 2D tensor:
            // psi_in.shape() = [nbands, nbasis]
            const auto ndim = psi_in.shape().ndim();
            REQUIRES_OK(ndim <= 2, "dims of psi_in should be less than or equal to 2");

            Device* ctx = {};

            gemv_op<T, Device>()(ctx,
                                 'N',
                                 s_mat.row,
                                 s_mat.col,
                                 one_,
                                 s_mat.p,
                                 s_mat.row,
                                 psi_in.data<T>(),
                                 1,
                                 zero_,
                                 spsi_out.data<T>(),
                                 1);

            ModuleBase::timer::tick("DiagoCG_New", "spsi_func");
        };

        if (this->is_first_scf)
        {
            for (size_t i = 0; i < psi.get_nbands(); i++)
            {
                for (size_t j = 0; j < psi.get_nbasis(); j++)
                {
                    psi(i, j) = *zero_;
                }
                psi(i, i) = *one_;
            }
        }

        auto psi_tensor = ct::TensorMap(psi.get_pointer(),
                                        ct::DataTypeToEnum<T>::value,
                                        ct::DeviceTypeToEnum<ct_Device>::value,
                                        ct::TensorShape({psi.get_nbands(), psi.get_nbasis()}))
                              .slice({0, 0}, {psi.get_nbands(), psi.get_current_nbas()});

        auto eigen_tensor = ct::TensorMap(eigenvalue,
                                          ct::DataTypeToEnum<Real>::value,
                                          ct::DeviceTypeToEnum<ct::DEVICE_CPU>::value,
                                          ct::TensorShape({psi.get_nbands()}));

        auto prec_tensor = ct::TensorMap(this->precondition_lcao.data(),
                                         ct::DataTypeToEnum<Real>::value,
                                         ct::DeviceTypeToEnum<ct::DEVICE_CPU>::value,
                                         ct::TensorShape({static_cast<int>(this->precondition_lcao.size())}))
                               .slice({0}, {psi.get_current_nbas()});

        cg->diag(hpsi_func, spsi_func, psi_tensor, eigen_tensor, prec_tensor);

        // TODO: Double check tensormap's potential problem
        ct::TensorMap(psi.get_pointer(), psi_tensor, {psi.get_nbands(), psi.get_nbasis()}).sync(psi_tensor);
    }

    ModuleBase::timer::tick("HSolverLCAO", "hamiltSolvePsiK");
}

template class HSolverLCAO<double>;
template class HSolverLCAO<std::complex<double>>;

} // namespace hsolver