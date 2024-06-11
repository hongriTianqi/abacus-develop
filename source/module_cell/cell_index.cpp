#include "cell_index.h"
#include "module_base/tool_quit.h"

CellIndex::CellIndex(const UnitCell& ucell, const int& nspin)
{
    this->atomLabels = ucell.get_atomLabels();
    this->atomCounts = ucell.get_atomCounts();
    this->orbitalCounts = ucell.get_orbitalCounts();
    this->lnchiCounts = ucell.get_lnchiCounts();
    this->npol_ = (nspin == 4) ? 2 : 1;
    this->check_atomCounts();
}

int CellIndex::get_ntype()
{
    return this->atomCounts.size();
}

int CellIndex::get_nat()
{
    int nat = 0;
    for (int it = 0; it < this->atomCounts.size(); ++it) {
        nat += this->atomCounts[it];
    }
    return nat;
}

int CellIndex::get_nat(int it)
{
    return this->atomCounts[it];
}

// get iat
int CellIndex::get_iat(int itype, int atom_index)
{
    if (itype < 0 || itype >= this->get_ntype())
    {
        ModuleBase::WARNING_QUIT("CellIndex::get_iat","itype out of range [0, ntype)");
    }
    if (atom_index < 0 || atom_index >= this->atomCounts[itype])
    {
        ModuleBase::WARNING_QUIT("CellIndex::get_iat","atom index out of range [0, nat)");
    }
    int iat = 0;
    for (int it = 0; it < this->atomCounts.size(); ++it) {
        if (it == itype)
        {
            break;
        }
        iat += this->atomCounts[it];
    }
    iat += atom_index;
    return iat;
}

int CellIndex::get_nw()
{
    int nw = 0;
    for (int it = 0; it < this->orbitalCounts.size(); ++it) {
        nw += this->orbitalCounts[it]*this->atomCounts[it]*this->npol_;
    }
    return nw;
}

int CellIndex::get_nw(int iat)
{
    int it = this->iat2it(iat);
    return this->orbitalCounts[it];
}

int CellIndex::get_iwt(int iat, int orbital_index)
{
    if (iat < 0 || iat >= this->get_nat())
    {
        ModuleBase::WARNING_QUIT("SpinConstrain::get_iwt","iat out of range [0, nat)");
    }
    int it = this->iat2it(iat);
    int ia = this->iat2ia(iat);
    if (orbital_index < 0 || orbital_index >= this->orbitalCounts[it]*this->npol_)
    {
        ModuleBase::WARNING_QUIT("SpinConstrain::get_iwt","orbital index out of range [0, atom_nw*npol)");
    }
    int iwt = 0;
    for (int it0 = 0; it0 < this->orbitalCounts.size(); ++it0) {
        if (it0 == it)
        {
            break;
        }
        iwt += this->orbitalCounts[it0]*this->atomCounts[it0]*this->npol_;
    }
    for (int i = 0; i < ia; ++i) {
        iwt += this->orbitalCounts[it]*this->npol_;
    }
    iwt += orbital_index;
    return iwt;
}

int CellIndex::get_maxL(int iat)
{
    int it = this->iat2it(iat);
    return this->lnchiCounts[it].size()-1;
}

/// @brief  get nchi
int CellIndex::get_nchi(int iat, int L)
{
    int it = this->iat2it(iat);
    if (L < 0 || L >= this->lnchiCounts[it].size())
    {
        ModuleBase::WARNING_QUIT("CellIndex::get_nchi","L out of range [0, maxL]");
    }
    return this->lnchiCounts[it][L];
}

void CellIndex::check_atomCounts()
{
    if (!this->atomCounts.size())
    {
        ModuleBase::WARNING_QUIT("CellIndex::check_atomCounts","atomCounts is not set");
    }
    if (this->get_nat() <= 0)
    {
        ModuleBase::WARNING_QUIT("CellIndex::check_atomCounts","nat <= 0");
    }
    for (int it = 0; it < this->atomCounts.size(); ++it) {
        if (this->atomCounts[it] <= 0)
        {
            ModuleBase::WARNING_QUIT("CellIndex::check_atomCounts","number of atoms <= 0 for some element");
        }
    }
}

std::string CellIndex::get_atom_label(int iat, bool order)
{
    int it = this->iat2it(iat);
    int ia = this->iat2ia(iat);
    std::string atomType = atomLabels[it];
    if (order) return atomType + std::to_string(ia+1);
    return atomType;
}

int CellIndex::iat2it(int iat)
{
    int running_iat = 0;
    int it = -1; // Tracks the index of the atom in atomLabels

    // Find the type of atom associated with the total order
    for (int i = 0; i < this->atomCounts.size(); ++i) {
        if (running_iat + atomCounts[i] > iat) {
            it = i;
            break;
        }
        running_iat += atomCounts[i];
    }
    if (it == -1) {
        ModuleBase::WARNING_QUIT("CellIndex::get_atom_label", "iat out of range [0, nat)");
    }
    return it;
}

int CellIndex::iat2ia(int iat)
{
    int it = this->iat2it(iat);
    // sum of atoms of previous types
    int running_iat = 0;
    for (int i = 0; i < it; ++i) {
        running_iat += atomCounts[i];
    }
    return iat - running_iat;
}

int CellIndex::iw2l(int iat, int iw)
{
    int it = this->iat2it(iat);
    int maxL = this->lnchiCounts[it].size()-1;
    for (int L = 0; L <= maxL; ++L) {
        int nchi = this->lnchiCounts[it][L];
        int blockSize = nchi * (2 * L + 1);
        if (iw < blockSize) {
            return L;
        }
        iw -= blockSize;
        if (iw < 0) {
            ModuleBase::WARNING_QUIT("CellIndex::iw2l", "iw out of range [0, nw)");
        }
    }
    if (iw >= 0) {
        ModuleBase::WARNING_QUIT("CellIndex::iw2l", "iw out of range [0, nw)");
    }
}

int CellIndex::iw2z(int iat, int iw)
{
    int it = this->iat2it(iat);
    int maxL = this->lnchiCounts[it].size()-1;
    for (int L = 0; L <= maxL; ++L) {
        int nchi = this->lnchiCounts[it][L];
        int blockSize = nchi * (2 * L + 1);
        if (iw < blockSize) {
            return iw / (2 * L + 1);
        }
        iw -= blockSize;
        if (iw < 0) {
            ModuleBase::WARNING_QUIT("CellIndex::iw2l", "iw out of range [0, nw)");
        }
    }
    if (iw >= 0) {
        ModuleBase::WARNING_QUIT("CellIndex::iw2l", "iw out of range [0, nw)");
    }
}

int CellIndex::iw2m(int iat, int iw)
{
    int it = this->iat2it(iat);
    int maxL = this->lnchiCounts[it].size()-1;
    for (int L = 0; L <= maxL; ++L) {
        int nchi = this->lnchiCounts[it][L];
        int blockSize = nchi * (2 * L + 1);
        if (iw < blockSize) {
            return iw % (2 * L + 1);
        }
        iw -= blockSize;
        if (iw < 0) {
            ModuleBase::WARNING_QUIT("CellIndex::iw2l", "iw out of range [0, nw)");
        }
    }
    if (iw >= 0) {
        ModuleBase::WARNING_QUIT("CellIndex::iw2l", "iw out of range [0, nw)");
    }
}