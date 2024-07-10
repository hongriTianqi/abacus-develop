#include "hamilt_lcao.h"

#include "module_base/global_variable.h"
#include "module_base/memory.h"
#include "module_base/timer.h"
#include "module_hamilt_lcao/module_dftu/dftu.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#ifdef __DEEPKS
#include "module_hamilt_lcao/module_deepks/LCAO_deepks.h"
#include "operator_lcao/deepks_lcao.h"
#endif
#ifdef __EXX
#include "module_ri/Exx_LRI_interface.h"
#include "operator_lcao/op_exx_lcao.h"
#endif
#ifdef __ELPA
#include "module_hsolver/diago_elpa.h"
#endif
#include "module_elecstate/potentials/H_TDDFT_pw.h"
#include "module_hamilt_general/module_xc/xc_functional.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer_funcs.h"
#include "module_hsolver/hsolver_lcao.h"
#include "operator_lcao/dftu_lcao.h"
#include "operator_lcao/ekinetic_new.h"
#include "operator_lcao/meta_lcao.h"
#include "operator_lcao/nonlocal_new.h"
#include "operator_lcao/op_dftu_lcao.h"
#include "operator_lcao/op_exx_lcao.h"
#include "operator_lcao/overlap_new.h"
#include "operator_lcao/sc_lambda_lcao.h"
#include "operator_lcao/td_ekinetic_lcao.h"
#include "operator_lcao/td_nonlocal_lcao.h"
#include "operator_lcao/veff_lcao.h"

namespace hamilt
{

template <typename TK, typename TR>
HamiltLCAO<TK, TR>::HamiltLCAO(const Parallel_Orbitals* paraV, const K_Vectors& kv_in, const TwoCenterIntegrator& intor_overlap_orb)
{
    this->classname = "HamiltLCAO";

    this->kv = &kv_in;

    // Real space Hamiltonian is inited with template TR
    this->hR = new HContainer<TR>(paraV);
    this->sR = new HContainer<TR>(paraV);
    this->hsk = new HS_Matrix_K<TK>(paraV);

    this->getOperator() = new OverlapNew<OperatorLCAO<TK, TR>>(this->hsk,
                                                               this->kv->kvec_d,
                                                               this->hR,
                                                               this->sR,
                                                               &GlobalC::ucell,
                                                               &GlobalC::GridD,
                                                               &intor_overlap_orb);
}

template <typename TK, typename TR>
HamiltLCAO<TK, TR>::HamiltLCAO(Gint_Gamma* GG_in,
                               Gint_k* GK_in,
                               LCAO_Matrix* LM_in,
                               const Parallel_Orbitals* paraV,
                               elecstate::Potential* pot_in,
                               const K_Vectors& kv_in,
                               const TwoCenterBundle& two_center_bundle,
                               const int kpar_in,
                               elecstate::DensityMatrix<TK, double>* DM_in,
                               int* exx_two_level_step)
{
    this->kv = &kv_in;
    this->classname = "HamiltLCAO";

    // Real space Hamiltonian is inited with template TR
    this->hR = new HContainer<TR>(paraV);
    this->sR = new HContainer<TR>(paraV);
    this->hsk = new HS_Matrix_K<TK>(paraV);

    // Effective potential term (\sum_r <psi(r)|Veff(r)|psi(r)>) is registered without template
    std::vector<std::string> pot_register_in;
    if (GlobalV::VL_IN_H)
    {
        if (GlobalV::VION_IN_H)
        {
            pot_register_in.push_back("local");
        }
        if (GlobalV::VH_IN_H)
        {
            pot_register_in.push_back("hartree");
        }
        pot_register_in.push_back("xc");
        if (GlobalV::imp_sol)
        {
            pot_register_in.push_back("surchem");
        }
        if (GlobalV::EFIELD_FLAG)
        {
            pot_register_in.push_back("efield");
        }
        if (GlobalV::GATE_FLAG)
        {
            pot_register_in.push_back("gatefield");
        }
        if (GlobalV::ESOLVER_TYPE == "tddft")
        {
            pot_register_in.push_back("tddft");
        }
    }

    // Gamma_only case to initialize HamiltLCAO
    //
    // code block to construct Operator Chains
    if (std::is_same<TK, double>::value)
    {
        // fix HR to gamma case, where SR will be fixed in Overlap Operator
        this->hR->fix_gamma();
        // initial operator for Gamma_only case
        // overlap term (<psi|psi>) is indispensable
        // in Gamma_only case, target SK is this->hsk->get_sk(), the target SR is this->sR
        this->getOperator() = new OverlapNew<OperatorLCAO<TK, TR>>(this->hsk,
                                                                   this->kv->kvec_d,
                                                                   this->hR,
                                                                   this->sR,
                                                                   &GlobalC::ucell,
                                                                   &GlobalC::GridD,
                                                                   two_center_bundle.overlap_orb.get());

        // kinetic term (<psi|T|psi>)
        if (GlobalV::T_IN_H)
        {
            Operator<TK>* ekinetic = new EkineticNew<OperatorLCAO<TK, TR>>(this->hsk,
                                                                           this->kv->kvec_d,
                                                                           this->hR,
                                                                           &GlobalC::ucell,
                                                                           &GlobalC::GridD,
                                                                           two_center_bundle.kinetic_orb.get());
            this->getOperator()->add(ekinetic);
        }

        // nonlocal term (<psi|beta>D<beta|psi>)
        // in general case, target HR is this->hR, while target HK is this->hsk->get_hk()
        if (GlobalV::VNL_IN_H)
        {
            Operator<TK>* nonlocal = new NonlocalNew<OperatorLCAO<TK, TR>>(this->hsk,
                                                                           this->kv->kvec_d,
                                                                           this->hR,
                                                                           &GlobalC::ucell,
                                                                           &GlobalC::GridD,
                                                                           two_center_bundle.overlap_orb_beta.get());
            this->getOperator()->add(nonlocal);
        }

        // Effective potential term (\sum_r <psi(r)|Veff(r)|psi(r)>)
        // in general case, target HR is Gint::hRGint, while target HK is this->hsk->get_hk()
        if (GlobalV::VL_IN_H)
        {
            // only Potential is not empty, Veff and Meta are available
            if (pot_register_in.size() > 0)
            {
                // register Potential by gathered operator
                pot_in->pot_register(pot_register_in);
                // effective potential term
                Operator<TK>* veff = new Veff<OperatorLCAO<TK, TR>>(GG_in,
                                                                    this->hsk,
                                                                    this->kv->kvec_d,
                                                                    pot_in,
                                                                    this->hR, // no explicit call yet
                                                                    &GlobalC::ucell,
                                                                    &GlobalC::GridD
                );
                this->getOperator()->add(veff);
            }
        }

#ifdef __DEEPKS
        if (GlobalV::deepks_scf)
        {
            Operator<TK>* deepks = new DeePKS<OperatorLCAO<TK, TR>>(this->hsk,
                                                                    this->kv->kvec_d,
                                                                    this->hR, // no explicit call yet
                                                                    &GlobalC::ucell,
                                                                    &GlobalC::GridD,
                                                                    two_center_bundle.overlap_orb_alpha.get(),
                                                                    this->kv->get_nks(),
                                                                    DM_in);
            this->getOperator()->add(deepks);
        }
#endif

        // end node should be OperatorDFTU
        if (GlobalV::dft_plus_u)
        {
            Operator<TK>* dftu = nullptr;
            if (GlobalV::dft_plus_u == 2)
            {
                dftu = new OperatorDFTU<OperatorLCAO<TK, TR>>(this->hsk,
                                                              kv->kvec_d,
                                                              this->hR, // no explicit call yet
                                                              this->kv->isk);
            }
            else
            {
                dftu = new DFTU<OperatorLCAO<TK, TR>>(this->hsk,
                                                      this->kv->kvec_d,
                                                      this->hR,
                                                      GlobalC::ucell,
                                                      &GlobalC::GridD,
                                                      two_center_bundle.overlap_orb_onsite.get(),
                                                      &GlobalC::dftu);
            }
            this->getOperator()->add(dftu);
        }
    }
    // multi-k-points case to initialize HamiltLCAO, ops will be used
    else if (std::is_same<TK, std::complex<double>>::value)
    {
        // Effective potential term (\sum_r <psi(r)|Veff(r)|psi(r)>)
        // Meta potential term (\sum_r <psi(r)|tau(r)|psi(r)>)
        // in general case, target HR is Gint::pvpR_reduced, while target HK is this->hsk->get_hk()
        if (GlobalV::VL_IN_H)
        {
            // only Potential is not empty, Veff and Meta are available
            if (pot_register_in.size() > 0)
            {
                // register Potential by gathered operator
                pot_in->pot_register(pot_register_in);
                // Veff term
                this->getOperator() = new Veff<OperatorLCAO<TK, TR>>(GK_in,
                                                                     this->hsk,
                                                                     kv->kvec_d,
                                                                     pot_in,
                                                                     this->hR,
                                                                     &GlobalC::ucell,
                                                                     &GlobalC::GridD);
                // reset spin index and real space Hamiltonian matrix
                int start_spin = -1;
                GK_in->reset_spin(start_spin);
                GK_in->destroy_pvpR();
                GK_in->allocate_pvpR();
            }
        }

        // initial operator for multi-k case
        // overlap term is indispensable
        Operator<TK>* overlap = new OverlapNew<OperatorLCAO<TK, TR>>(this->hsk,
                                                                     this->kv->kvec_d,
                                                                     this->hR,
                                                                     this->sR,
                                                                     &GlobalC::ucell,
                                                                     &GlobalC::GridD,
                                                                     two_center_bundle.overlap_orb.get());
        if (this->getOperator() == nullptr)
        {
            this->getOperator() = overlap;
        }
        else
        {
            this->getOperator()->add(overlap);
        }

        // kinetic term (<psi|T|psi>),
        // in general case, target HR is this->hR, while target HK is this->hsk->get_hk()
        if (GlobalV::T_IN_H)
        {
            Operator<TK>* ekinetic = new EkineticNew<OperatorLCAO<TK, TR>>(this->hsk,
                                                                           this->kv->kvec_d,
                                                                           this->hR,
                                                                           &GlobalC::ucell,
                                                                           &GlobalC::GridD,
                                                                           two_center_bundle.kinetic_orb.get());
            this->getOperator()->add(ekinetic);
        }

        // nonlocal term (<psi|beta>D<beta|psi>)
        // in general case, target HR is this->hR, while target HK is this->hsk->get_hk()
        if (GlobalV::VNL_IN_H)
        {
            Operator<TK>* nonlocal = new NonlocalNew<OperatorLCAO<TK, TR>>(this->hsk,
                                                                           this->kv->kvec_d,
                                                                           this->hR,
                                                                           &GlobalC::ucell,
                                                                           &GlobalC::GridD,
                                                                           two_center_bundle.overlap_orb_beta.get());
            // TDDFT velocity gague will calculate full non-local potential including the original one and the
            // correction on its own. So the original non-local potential term should be skipped
            if (GlobalV::ESOLVER_TYPE != "tddft" || elecstate::H_TDDFT_pw::stype != 1)
            {
                this->getOperator()->add(nonlocal);
            }
            else
            {
                delete nonlocal;
            }
        }

#ifdef __DEEPKS
        if (GlobalV::deepks_scf)
        {
            Operator<TK>* deepks = new DeePKS<OperatorLCAO<TK, TR>>(this->hsk,
                                                                    this->kv->kvec_d,
                                                                    hR,
                                                                    &GlobalC::ucell,
                                                                    &GlobalC::GridD,
                                                                    two_center_bundle.overlap_orb_alpha.get(),
                                                                    this->kv->get_nks(),
                                                                    DM_in);
            this->getOperator()->add(deepks);
        }
#endif
        // TDDFT_velocity_gague
        if (TD_Velocity::tddft_velocity)
        {
            if(!TD_Velocity::init_vecpot_file) elecstate::H_TDDFT_pw::update_At();
            Operator<TK>* td_ekinetic = new TDEkinetic<OperatorLCAO<TK, TR>>(this->hsk,
                                                                             this->hR,
                                                                             kv,
                                                                             &GlobalC::ucell,
                                                                             &GlobalC::GridD,
                                                                             two_center_bundle.overlap_orb.get());
            this->getOperator()->add(td_ekinetic);

            Operator<TK>* td_nonlocal = new TDNonlocal<OperatorLCAO<TK, TR>>(this->hsk,
                                                                             this->kv->kvec_d,
                                                                             this->hR,
                                                                             &GlobalC::ucell,
                                                                             &GlobalC::GridD);
            this->getOperator()->add(td_nonlocal);
        }
        if (GlobalV::dft_plus_u)
        {
            Operator<TK>* dftu = nullptr;
            if (GlobalV::dft_plus_u == 2)
            {
                dftu = new OperatorDFTU<OperatorLCAO<TK, TR>>(this->hsk,
                                                              kv->kvec_d,
                                                              this->hR, // no explicit call yet
                                                              this->kv->isk);
            }
            else
            {
                dftu = new DFTU<OperatorLCAO<TK, TR>>(this->hsk,
                                                      this->kv->kvec_d,
                                                      this->hR,
                                                      GlobalC::ucell,
                                                      &GlobalC::GridD,
                                                      two_center_bundle.overlap_orb_onsite.get(),
                                                      &GlobalC::dftu);
            }
            this->getOperator()->add(dftu);
        }
        if (GlobalV::sc_mag_switch)
        {
            Operator<TK>* sc_lambda = new OperatorScLambda<OperatorLCAO<TK, TR>>(this->hsk,
                                                                                 kv->kvec_d,
                                                                                 this->hR, // no explicit call yet
                                                                                 this->kv->isk);
            this->getOperator()->add(sc_lambda);
        }
    }

#ifdef __EXX
    if (GlobalC::exx_info.info_global.cal_exx)
    {
    Operator<TK>*exx = new OperatorEXX<OperatorLCAO<TK, TR>>(this->hsk,
            LM_in,
            this->hR,
            *this->kv,
            LM_in->Hexxd,
            LM_in->Hexxc,
            Add_Hexx_Type::R,
            exx_two_level_step,
            !GlobalC::restart.info_load.restart_exx
            && GlobalC::restart.info_load.load_H);
            this->getOperator()->add(exx);
    }
#endif
    // if NSPIN==2, HR should be separated into two parts, save HR into this->hRS2
    int memory_fold = 1;
    if (GlobalV::NSPIN == 2)
    {
        this->hRS2.resize(this->hR->get_nnr() * 2);
        this->hR->allocate(this->hRS2.data(), 0);
        memory_fold = 2;
    }

    this->kpar = kpar_in;
    if (this->kpar > 1)
    {
        this->Pkpoints = new Parallel_Kpoints;
        this->P2D_pool = new Parallel_Orbitals;
        int nks = this->kv->get_nks();
        int nproc = paraV->dim0*paraV->dim1;
        int nrow = paraV->get_global_row_size();
        int ncol = paraV->get_global_col_size();
        int nb2d = paraV->get_block_size();
        this->Pkpoints->kinfo(nks, this->kpar, this->MY_POOL, this->RANK_IN_POOL, nproc, GlobalV::NSPIN);
        this->P2D_pool->init(nrow, ncol, nb2d, this->POOL_WORLD_K2D);
    }

    ModuleBase::Memory::record("HamiltLCAO::hR", this->hR->get_memory_size() * memory_fold);
    ModuleBase::Memory::record("HamiltLCAO::sR", this->sR->get_memory_size());

    return;
}

// case for multi-k-points
template <typename TK, typename TR>
void HamiltLCAO<TK, TR>::matrix(MatrixBlock<TK>& hk_in, MatrixBlock<TK>& sk_in)
{
    auto op = dynamic_cast<OperatorLCAO<TK, TR>*>(this->getOperator());
    assert(op != nullptr);
    op->matrixHk(hk_in, sk_in);
}

template <typename TK, typename TR>
void HamiltLCAO<TK, TR>::updateHk(const int ik)
{
    ModuleBase::TITLE("HamiltLCAO", "updateHk");
    ModuleBase::timer::tick("HamiltLCAO", "updateHk");
    if (this->get_parak_init() == true)
    {
        this->set_parak_init(false);
        int nks = this->kv->get_nks();
        std::vector<int> ik_kpar;
        int ik_avail = 0;
        for (int ipool = 0; ipool < this->kpar; ++ipool) {
            if (ik + this->Pkpoints->startk_pool[ipool] < nks) {
                ik_avail++;
            }
        }
        if (ik_avail == 0) {
            ModuleBase::WARNING_QUIT("HSolverLCAO::solve", "ik_avail is 0!");
        } else {
            ik_kpar.resize(ik_avail);
            for (int ipool = 0; ipool < ik_avail; ++ipool) {
                ik_kpar[ipool] = ik + this->Pkpoints->startk_pool[ipool];
            }
        }
        int nrow = this->P2D_pool->get_global_row_size();
        int ncol = this->P2D_pool->get_global_col_size();
        for (int ipool = 0; ipool < ik_kpar.size(); ++ipool)
        {
            this->updateHk(ik_kpar[ipool]);
            hamilt::MatrixBlock<TK> HK_global, SK_global;
            this->matrix(HK_global, SK_global);
            if (this->MY_POOL == this->Pkpoints->whichpool[ik_kpar[ipool]]) {
                this->hsk_pool = new HS_Matrix_K<TK>(this->P2D_pool);
            }
            int desc_pool[9];
            std::copy(this->P2D_pool->desc, this->P2D_pool->desc + 9, desc_pool);
            if (this->MY_POOL != this->Pkpoints->whichpool[ik_kpar[ipool]]) {
                desc_pool[1] = -1;
            }
            Cpxgemr2d(nrow,
                      ncol,
                      HK_global.p,
                      1,
                      1,
                      this->paraV->desc,
                      this->hsk_pool->get_hk(),
                      1,
                      1,
                      desc_pool,
                      this->P2D_global->blacs_ctxt);
            Cpxgemr2d(nrow,
                      ncol,
                      SK_global.p,
                      1,
                      1,
                      this->paraV->desc,
                      this->hsk_pool->get_sk(),
                      1,
                      1,
                      desc_pool,
                      this->P2D_global->blacs_ctxt);
        }
        this->set_parak_init(true);
    }
    else if (this->get_parak_init() == false)
    {
        // update global spin index
        if (GlobalV::NSPIN == 2)
        {
            // if Veff is added and current_spin is changed, refresh HR
            if (GlobalV::VL_IN_H && this->kv->isk[ik] != this->current_spin)
            {
                // change data pointer of HR
                this->hR->allocate(this->hRS2.data() + this->hRS2.size() / 2 * this->kv->isk[ik], 0);
                if (this->refresh_times > 0)
                {
                    this->refresh_times--;
                    dynamic_cast<hamilt::OperatorLCAO<TK, TR>*>(this->ops)->set_hr_done(false);
                }
            }
            this->current_spin = this->kv->isk[ik];
        }
        this->getOperator()->init(ik);
    }
    ModuleBase::timer::tick("HamiltLCAO", "updateHk");
}

/// set parak init
template <typename TK, typename TR>
void HamiltLCAO<TK, TR>::set_parak_init(const bool parak_init_in)
{
    if (parak_init_in)
    {
        this->parak_init = true;
        this->getOperator()->set_hsk_pool(this->hsk_pool);
    }
    else
    {
        this->parak_init = false;
        this->getOperator()->set_hsk_pool(nullptr);
    }
}

template <typename TK, typename TR>
void HamiltLCAO<TK, TR>::refresh()
{
    ModuleBase::TITLE("HamiltLCAO", "refresh");
    dynamic_cast<hamilt::OperatorLCAO<TK, TR>*>(this->ops)->set_hr_done(false);
    if (GlobalV::NSPIN == 2)
    {
        this->refresh_times = 1;
        this->current_spin = 0;
        if (this->hR->get_nnr() != this->hRS2.size() / 2)
        {
            // operator has changed, resize hRS2
            this->hRS2.resize(this->hR->get_nnr() * 2);
        }
        this->hR->allocate(this->hRS2.data(), 0);
    }
}

// get Operator base class pointer
template <typename TK, typename TR>
Operator<TK>*& HamiltLCAO<TK, TR>::getOperator()
{
    return this->ops;
}

template <typename TK, typename TR>
void HamiltLCAO<TK, TR>::updateSk(const int ik, const int hk_type)
{
    ModuleBase::TITLE("HamiltLCAO", "updateSk");
    ModuleBase::timer::tick("HamiltLCAO", "updateSk");
    ModuleBase::GlobalFunc::ZEROS(this->getSk(), this->get_size_hsk());
    if (hk_type == 1) // collumn-major matrix for SK
    {
        const int nrow = this->hsk->get_pv()->get_row_size();
        hamilt::folding_HR(*this->sR, this->getSk(), this->kv->kvec_d[ik], nrow, 1);
    }
    else if (hk_type == 0) // row-major matrix for SK
    {
        const int ncol = this->hsk->get_pv()->get_col_size();
        hamilt::folding_HR(*this->sR, this->getSk(), this->kv->kvec_d[ik], ncol, 0);
    }
    ModuleBase::timer::tick("HamiltLCAO", "updateSk");
}

// case for nspin<4, gamma-only k-point
template class HamiltLCAO<double, double>;
// case for nspin<4, multi-k-points
template class HamiltLCAO<std::complex<double>, double>;
// case for nspin == 4, non-collinear spin case
template class HamiltLCAO<std::complex<double>, std::complex<double>>;
} // namespace hamilt
