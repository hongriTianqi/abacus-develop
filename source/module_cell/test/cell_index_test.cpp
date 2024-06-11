#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "module_base/mathzone.h"
#include "module_cell/unitcell.h"
#include "prepare_unitcell.h"
#include "module_cell/cell_index.h"

#ifdef __LCAO
#include "module_basis/module_ao/ORB_read.h"
InfoNonlocal::InfoNonlocal(){}
InfoNonlocal::~InfoNonlocal(){}
LCAO_Orbitals::LCAO_Orbitals(){}
LCAO_Orbitals::~LCAO_Orbitals(){}
#endif
Magnetism::Magnetism()
{
	this->tot_magnetization = 0.0;
	this->abs_magnetization = 0.0;
	this->start_magnetization = nullptr;
}
Magnetism::~Magnetism()
{
	delete[] this->start_magnetization;
}

/************************************************
 *  unit test of class CellIndex
 ***********************************************/

/**
 * - Tested Functions:
 *   - Indices used in UnitCell
 **/

class CellIndexTest : public testing::Test
{
    protected:
	std::unique_ptr<UnitCell> ucell{new UnitCell};
	std::string output;
};

TEST_F(CellIndexTest, Index)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    auto cell_index = CellIndex(*ucell, 1);
    EXPECT_EQ(2, cell_index.get_ntype());
    EXPECT_EQ(3, cell_index.get_nat());
    EXPECT_EQ(1, cell_index.get_nat(0));
    EXPECT_EQ(2, cell_index.get_nat(1));
    EXPECT_EQ(27, cell_index.get_nw());
    EXPECT_EQ(9, cell_index.get_nw(0));
    EXPECT_EQ(9, cell_index.get_nw(1));
    EXPECT_EQ(0, cell_index.get_iwt(0, 0));
    EXPECT_EQ(9, cell_index.get_iwt(1, 0));
    EXPECT_EQ(18, cell_index.get_iwt(2, 0));
    EXPECT_EQ(2, cell_index.get_maxL(0));
    EXPECT_EQ(2, cell_index.get_maxL(1));
    EXPECT_EQ(2, cell_index.get_maxL(2));
    EXPECT_EQ(1, cell_index.get_nchi(0, 0));
    EXPECT_EQ("C", cell_index.get_atom_label(0));
    EXPECT_EQ("H", cell_index.get_atom_label(1));
}

TEST_F(CellIndexTest, WriteOrbInfo)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    auto cell_index = CellIndex(*ucell, 1);
    cell_index.write_orb_info("./");
    std::ifstream ifs("./Orbital");
    std::string str((std::istreambuf_iterator<char>(ifs)),std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("#io    spec    l    m    z  sym"));
    remove("./Orbital");
}