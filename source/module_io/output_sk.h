#ifndef OUTPUT_SK_H
#define OUTPUT_SK_H

#include "module_io/output_interface.h"
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h"

namespace ModuleIO
{

template <typename TK>
class Output_Sk : public Output_Interface
{
public:
    /// constructur of Output_Sk 
    Output_Sk(LCAO_Matrix* LM,
        hamilt::Hamilt<TK>* p_hamilt,
        Parallel_Orbitals *ParaV,
        int nspin,
        int nks);
    /// @brief the function to get Sk for a given k-point
    TK* get_Sk(int ik);
    /// @brief an empty write function
    void write() override;
private:
    LCAO_Matrix* LM_ = nullptr;
    hamilt::Hamilt<TK>* p_hamilt_ = nullptr;
    Parallel_Orbitals *ParaV_ = nullptr;
    int nks_;
    int nspin_;
    std::vector<TK> SK;
};

}

#endif // OUTPUT_SK_H