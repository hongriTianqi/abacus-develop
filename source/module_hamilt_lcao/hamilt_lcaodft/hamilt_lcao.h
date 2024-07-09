#ifndef W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_HAMILT_LCAO_HAMILT_LCAODFT_HAMILT_LCAO_H
#define W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_HAMILT_LCAO_HAMILT_LCAODFT_HAMILT_LCAO_H

#include "module_basis/module_nao/two_center_bundle.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_elecstate/potentials/potential_new.h"
#include "module_hamilt_general/hamilt.h"
#include "module_hamilt_lcao/hamilt_lcaodft/LCAO_matrix.h"
#include "module_hamilt_lcao/hamilt_lcaodft/local_orbital_charge.h"
#include "module_hamilt_lcao/module_gint/gint_gamma.h"
#include "module_hamilt_lcao/module_gint/gint_k.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer.h"
#include "module_hamilt_lcao/hamilt_lcaodft/hs_matrix_k.hpp"
#include "module_cell/parallel_kpoints.h"
namespace hamilt
{

// template first for type of k space H matrix elements
// template second for type of temporary matrix, gamma_only fix-gamma-matrix + S-gamma, multi-k fix-Real + S-Real
template <typename TK, typename TR>
class HamiltLCAO : public Hamilt<TK>
{
  public:
    /**
     * @brief Constructor of Hamiltonian for LCAO base
     * HR and SR will be allocated with Operators
     */
    HamiltLCAO(Gint_Gamma* GG_in,
               Gint_k* GK_in,
               LCAO_Matrix* LM_in,
               const Parallel_Orbitals* paraV,
               elecstate::Potential* pot_in,
               const K_Vectors& kv_in,
               const TwoCenterBundle& two_center_bundle,
               const int kpar_in,
               elecstate::DensityMatrix<TK, double>* DM_in,
               int* exx_two_level_step = nullptr);
    /**
     * @brief Constructor of vacuum Operators, only HR and SR will be initialed as empty HContainer
     */
    HamiltLCAO(const Parallel_Orbitals* paraV, const K_Vectors& kv_in, const TwoCenterIntegrator& intor_overlap_orb);

    ~HamiltLCAO()
    {
        if (this->ops != nullptr)
        {
            delete this->ops;
        }
        delete this->hR;
        delete this->sR;
        delete this->hsk;
    }

    /// get pointer of Operator<TK> ops
    Operator<TK>*& getOperator();
    /// get hk-pointer
    TK* getHk() const{return this->hsk->get_hk();}
    /// get sk-pointer
    TK* getSk() const{return this->hsk->get_sk();}
    int get_size_hsk() const{return this->hsk->get_size();}
    /// get HR pointer of *this->hR, which is a HContainer<TR> and contains H(R)
    HContainer<TR>*& getHR(){return this->hR;}
    /// get SR pointer of *this->sR, which is a HContainer<TR> and contains S(R)
    HContainer<TR>*& getSR(){return this->sR;}
    /// refresh the status of HR
    void refresh() override;

    // for target K point, update consequence of hPsi() and matrix()
    virtual void updateHk(const int ik) override;

    /**
     * @brief special for LCAO, update SK only
     *
     * @param ik target K point
     * @param hk_type 0: SK is row-major, 1: SK is collumn-major
     * @return void
     */
    void updateSk(const int ik, const int hk_type = 0);

    // core function: return H(k) and S(k) matrixs for direct solving eigenvalues.
    // not used in PW base
    // void matrix(MatrixBlock<std::complex<double>> &hk_in, MatrixBlock<std::complex<double>> &sk_in) override;
    void matrix(MatrixBlock<TK>& hk_in, MatrixBlock<TK>& sk_in) override;

  private:
    const K_Vectors* kv = nullptr;

    // Real space Hamiltonian
    HContainer<TR>* hR = nullptr;
    HContainer<TR>* sR = nullptr;

    HS_Matrix_K<TK>* hsk = nullptr;
    // special case for NSPIN=2 , data of HR should be separated into two parts
    // save them in this->hRS2;
    std::vector<TR> hRS2;
    int refresh_times = 1;

    /// current_spin for NSPIN=2, 0: hamiltonian for spin up, 1: hamiltonian for spin down
    int current_spin = 0;

    /// kpar for parallel calculation
    int kpar = 1;
    /// the pointer to Parallel_Kpoints
    Parallel_Kpoints* Pkpoints = nullptr;
    /// the pointer to manage 2D parallel in k pool
    Parallel_2D* P2D_pool = nullptr;
    /// the pointer to hsk_pool
    HS_Matrix_K<TK>* hsk_pool = nullptr;
};

} // namespace hamilt

#endif
