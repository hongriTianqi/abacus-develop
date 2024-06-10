#ifndef CELL_INDEX_H
#define CELL_INDEX_H
#include <map>
#include "module_cell/unitcell.h"

class CellIndex
{
public:
    CellIndex() = default;
    CellIndex(const UnitCell& ucell, int& npol);
public:
    /// @brief the total number of atoms
    int get_nat();
    /// @brief the total number of atoms of a given type
    int get_nat(int it);
    /// get ntype
    int get_ntype();
    /// get iat
    int get_iat(int itype, int atom_index);
    /// get nw
    int get_nw();
    /// get nw of a given type
    int get_nw(int iat);
    /// get iwt
    int get_iwt(int iat, int orbital_index);
    /// check atomCounts
    int get_maxL(int iat);
    /// @brief  get nchi
    int get_nchi(int iat, int L);
    /// get atom label
    std::string get_atom_label(int iat, bool order=false);
    /// get L from iw
    int iw2l(int iat, int iw);
    /// get Z from iw
    int iw2z(int iat, int iw);
    /// get m from iw
    int iw2m(int iat, int iw);
private:
    std::vector<int> atomCounts;
    std::vector<int> orbitalCounts;
    std::vector<std::vector<int>> lnchiCounts;
    std::vector<std::string> atomLabels;
    int npol_;
    void check_atomCounts();
    int iat2it(int iat);
    int iat2ia(int iat);
};


#endif // CELL_INDEX_H
