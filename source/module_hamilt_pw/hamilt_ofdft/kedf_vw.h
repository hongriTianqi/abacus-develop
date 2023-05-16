#ifndef KEDF_vW_H
#define KEDF_vW_H
#include <math.h>
#include <stdio.h>

#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "module_base/matrix.h"
#include "module_base/timer.h"
#include "module_basis/module_pw/pw_basis.h"

/**
 * @brief A class which calculates kinetic energy, potential, and stress with von Weizsäcker (vW) KEDF.
 * See Weizsäcker C F. Zeitschrift für Physik, 1935, 96(7): 431-458.
 * @author sunliang on 2022-06
 */
class KEDF_vW
{
  public:
    KEDF_vW()
    {
        this->stress.create(3, 3);
    }
    ~KEDF_vW()
    {
    }

    void set_para(int nx, double dV, double vw_weight);

    double get_energy(double **pphi, ModulePW::PW_Basis *pw_rho);
    double get_energy_density(double **pphi, int is, int ir, ModulePW::PW_Basis *pw_rho);
    void vW_potential(const double *const *pphi, ModulePW::PW_Basis *pw_rho, ModuleBase::matrix &rpotential);
    void get_stress(const double *const *pphi, ModulePW::PW_Basis *pw_rho, double inpt_vWenergy = -1);

    double vWenergy = 0.;
    ModuleBase::matrix stress;

  private:
    void laplacianPhi(const double *const *pphi, double **rLapPhi, ModulePW::PW_Basis *pw_rho);

    int nx = 0;
    double dV = 0.;
    double vw_weight = 1.;
};
#endif