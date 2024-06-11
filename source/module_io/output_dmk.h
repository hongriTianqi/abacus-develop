#include "module_io/output_interface.h"
#include "module_elecstate/module_dm/density_matrix.h"

namespace ModuleIO
{

template <typename TK>
class Output_DMK : public Output_Interface
{
public:
    Output_DMK(elecstate::DensityMatrix<TK,double>* p_DM);
    TK* get_DMK(int ik);
    void write() override;
private:
    elecstate::DensityMatrix<TK,double>* p_DM_ = nullptr;
};
    
} // namespace ModuleIO