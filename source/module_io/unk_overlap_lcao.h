#ifndef UNKOVERLAP_LCAO
#define UNKOVERLAP_LCAO

#include "module_base/sph_bessel_recursive.h"
#include "module_base/vector3.h"
#include "module_base/ylm.h"
#include "module_basis/module_ao/ORB_atomic_lm.h"
#include "module_basis/module_ao/ORB_gaunt_table.h"
#include "module_basis/module_ao/ORB_read.h"
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_cell/klist.h"
#include "module_hamilt_lcao/hamilt_lcaodft/center2_orb-orb11.h"
#include "module_hamilt_lcao/hamilt_lcaodft/center2_orb-orb21.h"
#include "module_hamilt_lcao/hamilt_lcaodft/center2_orb.h"
#include "module_hamilt_lcao/module_gint/grid_technique.h"

#include <map>
#include <set>
#include <vector>

class unkOverlap_lcao
{
  public:
    ModuleBase::Sph_Bessel_Recursive::D2* psb_ = nullptr;
    ORB_gaunt_table MGT;
    Numerical_Orbital_Lm orb_r; // New r vector, exists in atomic orbital form, expanded in solid spherical function

    std::vector<std::vector<std::vector<ModuleBase::Vector3<double>>>> orb1_orb2_R;
    std::vector<std::vector<std::vector<double>>> psi_psi;
    std::vector<std::vector<std::vector<ModuleBase::Vector3<double>>>> psi_r_psi;
    bool allocate_flag;                      // translate: Used to initialize the array
    int** cal_tag;                           // Used for parallel scheme

    int kpoints_number;

    std::map<
        size_t,
        std::map<size_t, std::map<size_t, std::map<size_t, std::map<size_t, std::map<size_t, Center2_Orb::Orb11>>>>>>
        center2_orb11;

    std::map<
        size_t,
        std::map<size_t, std::map<size_t, std::map<size_t, std::map<size_t, std::map<size_t, Center2_Orb::Orb21>>>>>>
        center2_orb21_r;

    unkOverlap_lcao();
    ~unkOverlap_lcao();

    void init(const Grid_Technique& gt, const int nkstot);
    int iw2it(int iw);
    int iw2ia(int iw);
    int iw2iL(int iw);
    int iw2iN(int iw);
    int iw2im(int iw);
    void cal_R_number();
    void cal_orb_overlap();
    void prepare_midmatrix_pblas(const int ik_L,
                                 const int ik_R,
                                 const ModuleBase::Vector3<double> dk,
                                 std::complex<double>*& midmatrix,
                                 const Parallel_Orbitals& pv,
                                 const K_Vectors& kv);
    std::complex<double> det_berryphase(const int ik_L,
                                        const int ik_R,
                                        const ModuleBase::Vector3<double> dk,
                                        const int occ_bands,
                                        const Parallel_Orbitals& para_orb,
                                        const psi::Psi<std::complex<double>>* psi_in,
                                        const K_Vectors& kv);
};

#endif
