#include "module_io/output_dmk.h"
#include "module_io/output_sk.h"

namespace ModuleIO
{

template <typename TK>
Output_DMK<TK>::Output_DMK(elecstate::DensityMatrix<TK,double>* p_DM)
    : p_DM_(p_DM)
{
}

template <typename TK>
TK* Output_DMK<TK>::get_DMK(int ik)
{
    TK* p_DMK = nullptr;
    return p_DMK;
}

template <typename TK>
void Output_DMK<TK>::write()
{
}

template class Output_DMK<double>;
template class Output_DMK<std::complex<double>>;

template <typename TK>
Output_Sk<TK>::Output_Sk(LCAO_Matrix* LM,
    hamilt::Hamilt<TK>* p_hamilt,
    Parallel_Orbitals *ParaV,
    int nspin,
    int nks)
    : LM_(LM), p_hamilt_(p_hamilt), ParaV_(ParaV), nspin_(nspin), nks_(nks)
{
}

template <typename TK>
TK* Output_Sk<TK>::get_Sk(int ik)
{
    TK* p_Sk = nullptr;
    return p_Sk;
    
}

template <typename TK>
void Output_Sk<TK>::write()
{
}

template class Output_Sk<double>;
template class Output_Sk<std::complex<double>>;



} // namespace ModuleIO