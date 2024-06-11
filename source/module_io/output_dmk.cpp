#include "module_io/output_dmk.h"

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
    return p_DM_->get_DMK_vector()[ik].data();
}

template <typename TK>
void Output_DMK<TK>::write()
{
}

} // namespace ModuleIO