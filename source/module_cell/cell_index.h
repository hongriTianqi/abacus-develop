#ifndef CELL_INDEX_H
#define CELL_INDEX_H
#include <map>
#include "module_cell/unitcell.h"

class CellIndex
{
public:
    CellIndex() = default;
    CellIndex(const UnitCell& ucell, const int& nspin);
public:
    /// @brief the total number of atoms
    int get_nat();
    /// @brief the total number of atoms of a given type
    int get_nat(int it);
    /// @brief get ntype
    int get_ntype();
    /// @brief get nw
    int get_nw();
    /// @brief get nw of a given type
    int get_nw(int iat);
    /// @brief get iwt
    int get_iwt(int iat, int orbital_index);
    /// @brief get maximum L of a given atom
    int get_maxL(int iat);
    /// @brief  get nchi of a given atom and a give L
    int get_nchi(int iat, int L);
    /// @brief get atom label of a given atom
    std::string get_atom_label(int iat, bool order=false);
    /// @brief write orbital info into file
    void write_orb_info(std::string out_dir);
private:
    /// atomCounts is a vector used to store the number of atoms for each type
    std::vector<int> atomCounts;
    /// orbitalCounts is a vector used to store the number of orbitals for each type
    std::vector<int> orbitalCounts;
    /// lnchiCounts is a vector used to store the number of chi for each L
    std::vector<std::vector<int>> lnchiCounts;
    /// atomLabels is a vector used to store the atom labels
    std::vector<std::string> atomLabels;
    /// npol is determined by nspin and used in get_iwt and get_nw
    int npol_;
    /// check nspin
    int check_nspin(int nspin);
    /// check if atomCounts is set ok
    void check_atomCounts();
    /// get type of atom from total order
    int iat2it(int iat);
    /// get index of atom in the same type
    int iat2ia(int iat);
    /// get L from iw
    int iw2l(int iat, int iw);
    /// get Z from iw
    int iw2z(int iat, int iw);
    /// get m from iw
    int iw2m(int iat, int iw);
};


#endif // CELL_INDEX_H
