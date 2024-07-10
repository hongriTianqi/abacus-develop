#include "hsolver_lcao.h"

#include "diago_cg.h"

#ifdef __MPI
#include "diago_scalapack.h"
#include "mpi.h"
#else
#include "diago_lapack.h"
#endif

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
#include "module_base/scalapack_connector.h"
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h"

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
#ifdef __MPI
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
#else
        ModuleBase::WARNING_QUIT("HSolverLCAO", "Scalapack not supported in SERIAL VERSION");
#endif
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
#ifndef __MPI
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
            this->pdiagh = new DiagoLapack<T>();
            this->pdiagh->method = this->method;
        }
#else
        ModuleBase::WARNING_QUIT("HSolverLCAO::solve", "This method of DiagH is not supported!");
#endif
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

    auto pHamiltLCAO_complex = dynamic_cast<hamilt::HamiltLCAO<T, std::complex<double>>*>(pHamilt);
    auto pHamiltLCAO_real = dynamic_cast<hamilt::HamiltLCAO<T, double>*>(pHamilt);
    if (pHamiltLCAO_complex != nullptr)
    {
        if (pHamiltLCAO_complex->get_kpar() > 1)
        {
            this->set_parak_env();
            this->init_psi_pool(pHamiltLCAO_complex, psi.get_nk());
            pHamiltLCAO_complex->set_parak_init(true);
            for (int ik = 0; ik < pHamiltLCAO_complex->get_Pkpoints()->get_max_nks_pool(); ++ik)
            {
                pHamiltLCAO_complex->updateHk(ik);
                int ik_global = ik + pHamiltLCAO_complex->get_Pkpoints()->startk_pool[pHamiltLCAO_complex->get_my_pool()];
                if (ik_global < psi.get_nk())
                {
                    this->psi_pool.fix_k(ik_global);
                    this->hamiltSolvePsiK(pHamiltLCAO_complex, this->psi_pool, &(pes->ekb(ik_global, 0)));
                }
            }
            this->collect_psi_pool(pHamiltLCAO_complex, pes, psi);
            pHamiltLCAO_complex->set_parak_init(false);
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
            }
        }
    }
    else if (pHamiltLCAO_real != nullptr)
    {
        if (pHamiltLCAO_real->get_kpar() > 1)
        {
            this->init_psi_pool(pHamiltLCAO_real, psi.get_nk());
            pHamiltLCAO_real->set_parak_init(true);
            for (int ik = 0; ik < pHamiltLCAO_real->get_Pkpoints()->get_max_nks_pool(); ++ik)
            {
                pHamiltLCAO_real->updateHk(ik);
                int ik_global = ik + pHamiltLCAO_real->get_Pkpoints()->startk_pool[pHamiltLCAO_real->get_my_pool()];
                if (ik_global < psi.get_nk())
                {
                    this->psi_pool.fix_k(ik_global);
                    this->hamiltSolvePsiK(pHamiltLCAO_real, this->psi_pool, &(pes->ekb(ik_global, 0)));
                }
            }
            this->collect_psi_pool(pHamiltLCAO_real, pes, psi);
            pHamiltLCAO_real->set_parak_init(false);
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
            }
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

template <typename T, typename Device>
void HSolverLCAO<T, Device>::init_psi_pool(hamilt::Hamilt<T>* pHamilt, int nks)
{
    const int zero = 0;
    int nbands = this->ParaV->nbands;
    int nb2d = this->ParaV->get_block_size();
    int ncol_bands_pool = numroc_(&nbands, &nb2d, &(pHamilt->P2D_pool->coord[1]), &zero, &(pHamilt->P2D_pool->dim1));
    this->psi_pool = psi::Psi<T>(nks, ncol_bands_pool, pHamilt->P2D_pool->nrow, nullptr);
}


template <typename T, typename Device>
void HSolverLCAO<T, Device>::collect_psi_pool(hamilt::Hamilt<T>* pHamilt, elecstate::ElecState* pes, psi::Psi<T>& psi)
{
    int nbands = this->ParaV->nbands;
    int nrow = this->ParaV->get_global_row_size();
    for (int ik = 0; ik < psi->get_nk(); ++ik)
    {
        /// bcast ekb
        int source
            = pHamilt->Pkpoints->get_startpro_pool(pHamilt->Pkpoints->whichpool[ik]);
        MPI_Bcast(&(pes->ekb(ik, 0)),
                  nbands,
                  MPI_DOUBLE,
                  source,
                  MPI_COMM_WORLD);
        /// bcast psi
        int desc_pool[9];
        std::copy(pHamilt->P2D_pool->desc, pHamilt->P2D_pool->desc + 9, desc_pool);
        if (pHamilt->get_my_pool() != pHamilt->Pkpoints->whichpool[ik]) {
            desc_pool[1] = -1;
        }
        psi_pool.fix_k(ik);
        psi.fix_k(ik);
        Cpxgemr2d(nrow,
                  nbands,
                  psi_pool.get_pointer(),
                  1,
                  1,
                  desc_pool,
                  psi.get_pointer(),
                  1,
                  1,
                  this->ParaV->desc,
                  this->ParaV->blacs_ctxt);
    }
    pHamilt->set_parak_init(false);
}

template class HSolverLCAO<double>;
template class HSolverLCAO<std::complex<double>>;

} // namespace hsolver