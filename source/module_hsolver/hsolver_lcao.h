#ifndef HSOLVERLCAO_H
#define HSOLVERLCAO_H

#include "hsolver.h"
#include "module_basis/module_ao/parallel_orbitals.h"

namespace hsolver
{

template <typename T, typename Device = base_device::DEVICE_CPU>
class HSolverLCAO : public HSolver<T, Device>
{
  public:
    HSolverLCAO(const Parallel_Orbitals* ParaV_in, const K_Vectors* kv)
    {
      this->classname = "HSolverPW"; 
      this->ParaV = ParaV_in;
      }
    /*void init(
        const Basis* pbas
        //const Input &in,
    ) override;
    void update(//Input &in
    ) override;*/

    void solve(hamilt::Hamilt<T>* pHamilt, psi::Psi<T>& psi, elecstate::ElecState* pes, const std::string method_in, const bool skip_charge) override;

    static std::vector<int> out_mat_hs; // mohan add 2010-09-02
    static int out_mat_hsR; // LiuXh add 2019-07-16
    static int out_mat_t;
    static int out_mat_dh;

  private:
      void hamiltSolvePsiK(hamilt::Hamilt<T>* hm, psi::Psi<T>& psi, double* eigenvalue);

      void solveTemplate(hamilt::Hamilt<T>* pHamilt, psi::Psi<T>& psi, elecstate::ElecState* pes, const std::string method_in, const bool skip_charge);
    /*void solveTemplate(
        hamilt::Hamilt* pHamilt,
        psi::Psi<std::complex<double>>& psi,
        elecstate::ElecState* pes
    );*/

    const Parallel_Orbitals* ParaV;


    bool is_first_scf = true;

    using Real = typename GetTypeReal<T>::type;
    std::vector<Real> precondition_lcao;

    /// psi in pool
    psi::Psi<T> psi_pool;
    /// @brief initialize psi_pool in k parallel case
    /// @param nks
    void init_psi_pool(hamilt::Hamilt<T>* pHamilt, int nks);
    /// collect psi_pool from each pool to psi
    void collect_psi_pool(hamilt::Hamilt<T>* pHamilt, elecstate::ElecState* pes, psi::Psi<T>& psi);
};

template <typename T, typename Device>
std::vector<int> HSolverLCAO<T, Device>::out_mat_hs = {0, 8};
template <typename T, typename Device>
int HSolverLCAO<T, Device>::out_mat_hsR = 0;
template <typename T, typename Device>
int HSolverLCAO<T, Device>::out_mat_t = 0;
template <typename T, typename Device>
int HSolverLCAO<T, Device>::out_mat_dh = 0;

template <typename T>
inline  T my_conj(T value)
{
    return value;
}

template <>
inline  std::complex<double> my_conj(std::complex<double> value)
{
    return std::conj(value);
}

} // namespace hsolver

#endif