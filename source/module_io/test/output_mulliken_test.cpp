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
};

using MyTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_SUITE(OutputMullikenTest, MyTypes);

TYPED_TEST(OutputMullikenTest, EmptyWrite)
{
    EXPECT_EQ(1, 1);
}
