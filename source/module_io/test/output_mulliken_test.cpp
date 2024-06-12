#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "module_cell/cell_index.h"
#include "module_io/output_sk.h"
#include "module_io/output_dmk.h"
#include "prepare_unitcell.h"

// mock functions
#ifdef __LCAO
InfoNonlocal::InfoNonlocal()
{
}
InfoNonlocal::~InfoNonlocal()
{
}
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

#include "../output_mulliken.h"

template <typename TK>
class OutputMullikenTest : public testing::Test
{
  protected:
    UnitCell* ucell;
    UcellTestPrepare utp = UcellTestLib["Si"];
    void SetUp() override
    {
        ucell = utp.SetUcellInfo();
    }
};

using MyTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_SUITE(OutputMullikenTest, MyTypes);

TYPED_TEST(OutputMullikenTest, EmptyWrite)
{
    EXPECT_EQ(1, 1);
}
