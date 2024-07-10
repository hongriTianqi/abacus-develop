#include "esolver.h"
#include "module_base/module_device/device.h"
#include "esolver_ks_pw.h"
#include "esolver_sdft_pw.h"
#ifdef __LCAO
#include "esolver_ks_lcaopw.h"
#include "esolver_ks_lcao.h"
#include "esolver_ks_lcao_tddft.h"
#endif
#include "esolver_dp.h"
#include "esolver_lj.h"
#include "esolver_of.h"
#include "module_md/md_para.h"
#include <stdexcept>

namespace ModuleESolver
{

void ESolver::printname()
{
	std::cout << classname << std::endl;
}

std::string determine_type()
{
	std::string esolver_type = "none";
	if (GlobalV::BASIS_TYPE == "pw")
	{
		if(GlobalV::ESOLVER_TYPE == "sdft")
		{
			esolver_type = "sdft_pw";
		}
		else if(GlobalV::ESOLVER_TYPE == "ofdft")
		{
			esolver_type = "ofdft";
		}
		else if(GlobalV::ESOLVER_TYPE == "ksdft")
		{
			esolver_type = "ksdft_pw";
		}
	}
	else if (GlobalV::BASIS_TYPE == "lcao_in_pw")
	{
#ifdef __LCAO
		if(GlobalV::ESOLVER_TYPE == "sdft")
		{
			esolver_type = "sdft_pw";
		}
		else if(GlobalV::ESOLVER_TYPE == "ksdft")
		{
            esolver_type = "ksdft_lip";
		}
#else
		ModuleBase::WARNING_QUIT("ESolver", "Calculation involving numerical orbitals must be compiled with __LCAO");
#endif
	}
	else if (GlobalV::BASIS_TYPE == "lcao")
	{
#ifdef __LCAO
		if(GlobalV::ESOLVER_TYPE == "tddft")
		{
			esolver_type = "ksdft_lcao_tddft";
		}
		else if(GlobalV::ESOLVER_TYPE == "ksdft")
		{
			esolver_type = "ksdft_lcao";
		}
#else
		ModuleBase::WARNING_QUIT("ESolver", "Calculation involving numerical orbitals must be compiled with __LCAO");
#endif
	}

	if(GlobalV::ESOLVER_TYPE == "lj")
	{
		esolver_type = "lj_pot";
	}
	else if(GlobalV::ESOLVER_TYPE == "dp")
	{
		esolver_type = "dp_pot";
	}
	else if(esolver_type == "none")
	{
		ModuleBase::WARNING_QUIT("ESolver", "No such esolver_type combined with basis_type");
	}

	GlobalV::ofs_running << " The esolver type has been set to : " << esolver_type << std::endl;

	auto device_info = GlobalV::device_flag;

	for (char &c : device_info)
	{
		if (std::islower(c))
		{
			c = std::toupper(c);
		}
	}
	if (GlobalV::MY_RANK == 0)
	{
		std::cout << " RUNNING WITH DEVICE  : " << device_info << " / "
			<< base_device::information::get_device_info(GlobalV::device_flag) << std::endl;
	}

	GlobalV::ofs_running << "\n RUNNING WITH DEVICE  : " << device_info << " / "
		<< base_device::information::get_device_info(GlobalV::device_flag) << std::endl;

	return esolver_type;
}


//Some API to operate E_Solver
ESolver* init_esolver()
{
	//determine type of esolver based on INPUT information
	const std::string esolver_type = determine_type();

	//initialize the corresponding Esolver child class
	if (esolver_type == "ksdft_pw")
	{
#if ((defined __CUDA) || (defined __ROCM))
		if (GlobalV::device_flag == "gpu")
		{
			if (GlobalV::precision_flag == "single")
			{
				return new ESolver_KS_PW<std::complex<float>, base_device::DEVICE_GPU>();
			}
			else
			{
				return new ESolver_KS_PW<std::complex<double>, base_device::DEVICE_GPU>();
			}
		}
#endif
		if (GlobalV::precision_flag == "single")
		{
			return new ESolver_KS_PW<std::complex<float>, base_device::DEVICE_CPU>();
		}
		else
		{
			return new ESolver_KS_PW<std::complex<double>, base_device::DEVICE_CPU>();
		}
	}
#ifdef __LCAO
    else if (esolver_type == "ksdft_lip")
    {
        if (GlobalV::precision_flag == "single")
        {
            return new ESolver_KS_LIP<std::complex<float>>();
        }
        else
        {
            return new ESolver_KS_LIP<std::complex<double>>();
        }
    }
    else if (esolver_type == "ksdft_lcao")
	{
		if (GlobalV::GAMMA_ONLY_LOCAL)
		{
			return new ESolver_KS_LCAO<double, double>();
		}
		else if (GlobalV::NSPIN < 4)
		{
			return new ESolver_KS_LCAO<std::complex<double>, double>();
		}
		else
		{
			return new ESolver_KS_LCAO<std::complex<double>, std::complex<double>>();
		}
	}
	else if (esolver_type == "ksdft_lcao_tddft")
	{
		return new ESolver_KS_LCAO_TDDFT();
	}
#endif
	else if (esolver_type == "sdft_pw")
	{
		return new ESolver_SDFT_PW();
	}
	else if(esolver_type == "ofdft")
	{
		return new ESolver_OF();
	}
	else if (esolver_type == "lj_pot")
	{
		return new ESolver_LJ();
	}
	else if (esolver_type == "dp_pot")
	{
		return new ESolver_DP(INPUT.mdp.pot_file);
	}
	throw std::invalid_argument("esolver_type = "+std::string(esolver_type)+". Wrong in "+std::string(__FILE__)+" line "+std::to_string(__LINE__));
}


void clean_esolver(ESolver*& pesolver)
{
// Zhang Xiaoyang modified in 2024/7/6:
// Note: because of the init method of serial lcao hsolver
// it needs no release step for it, or this [delete] will cause Segmentation Fault
// Probably it will be modified later.
#ifdef __MPI
	delete pesolver;
#endif
}

}
