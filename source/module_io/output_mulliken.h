#ifndef OUTPUT_MULLIKEN_H
#define OUTPUT_MULLIKEN_H
#include "module_io/output_interface.h"
#include <map>
#include <vector>
#include "module_base/matrix.h"
#include "module_base/complexmatrix.h"
#include "module_hamilt_lcao/hamilt_lcaodft/LCAO_matrix.h"
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h"
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_cell/unitcell.h"
#include "module_cell/cell_index.h"

namespace ModuleIO
{

/// @brief the output interface to write the Mulliken population charges
template <typename TK>
class Output_Mulliken : public Output_Interface
{
public:
    /// constructor of Output_Mulliken
    Output_Mulliken(LCAO_Matrix* LM,
        hamilt::Hamilt<TK>* p_hamilt,
        Parallel_Orbitals *ParaV,
        const CellIndex& cell_index,
        const std::vector<std::vector<TK>>& dm,
        const K_Vectors& kv,
        int nspin);
    /// the outer interface to write the Mulliken population charges
    void write(int istep, std::string out_dir);
    /// print atom mag to running log file
    void print_atom_mag(const std::vector<std::vector<double>>& atom_chg, std::ostream& os);
    /// get total charge
    std::vector<double> get_tot_chg();
    /// get atom charge
    std::vector<std::vector<double>> get_atom_chg();
    /// get orbital charge
    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> get_orb_chg();
    /// write orbital info
    void write_orb_info(std::string out_dir);
    /// an empty write function
    void write() override;
    /// returun atom_mulliken for updateing STRU file
    std::vector<std::vector<double>> get_atom_mulliken(std::vector<std::vector<double>>& atom_chg);

private:
    /******************************************************************
     * private functions
    *******************************************************************/
    /// write mulliken.txt for the case of nspin=1
    void write_mulliken_nspin1(int istep,
            const std::vector<double>& tot_chg,
            const std::vector<std::vector<double>>& atom_chg,
            const std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>& orb_chg,
            std::ofstream& os);
    /// write mulliken.txt for the case of nspin=2
    void write_mulliken_nspin2(int istep,
            const std::vector<double>& tot_chg,
            const std::vector<std::vector<double>>& atom_chg,
            const std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>& orb_chg,
            std::ofstream& os);
    /// write mulliken.txt for the case of nspin=4
    void write_mulliken_nspin4(int istep,
            const std::vector<double>& tot_chg,
            const std::vector<std::vector<double>>& atom_chg,
            const std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>& orb_chg,
            std::ofstream& os);
    /// set nspin
    void set_nspin(int nspin_in);
    /// set orbital parallel info
    void set_ParaV(Parallel_Orbitals* ParaV_in);
    /// collect_mw from matrix multiplication result
    void collect_MW(ModuleBase::matrix& MecMulP, const ModuleBase::ComplexMatrix& mud, int nw, int isk);
    /// mulliken population = trace(dm*overlap)
    void cal_orbMulP(LCAO_Matrix* LM, const std::vector<std::vector<TK>>& dm);

private:
    /******************************************************************
     * private variables
    *******************************************************************/
    LCAO_Matrix* LM_ = nullptr;
    hamilt::Hamilt<TK>* p_hamilt_ = nullptr;
    Parallel_Orbitals *ParaV_ = nullptr;
    const std::vector<std::vector<TK>> dm_;
    const K_Vectors& kv_;
    int nspin_;
    CellIndex cell_index_;
    ModuleBase::matrix orbMulP_;
};

} // namespace ModuleIO

#endif // OUTPUT_MULLIKEN_H
