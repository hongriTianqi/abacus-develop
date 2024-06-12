#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "module_cell/cell_index.h"
#include "module_io/output_sk.h"
#include "module_io/output_dmk.h"
#include "../output_mulliken.h"

template <typename TK>
class OutputMullikenTest : public testing::Test
{
  protected:
    std::vector<std::string> atomLabels = {"Si"};
    std::vector<int> atomCounts = {1};
    std::vector<std::vector<int>> lnchiCounts = {{2, 2, 1}};
};

using MyTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_SUITE(OutputMullikenTest, MyTypes);

TYPED_TEST(OutputMullikenTest, EmptyWrite)
{
    CellIndex cell_index = CellIndex(this->atomLabels, this->atomCounts, this->lnchiCounts, 1);
    cell_index.write_orb_info("./");
    EXPECT_EQ(1, 1);
}
