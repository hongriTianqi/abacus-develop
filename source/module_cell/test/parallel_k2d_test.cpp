#include "module_cell/parallel_k2d.h"
#include "gtest/gtest.h"

/************************************************
 *  unit test of class Parallel_K2D
 ***********************************************/

/**
 * Test fixture for class Parallel_K2D
 */

class Parallel_K2DTest : public ::testing::Test
{
protected:
    Parallel_K2D &k2d = Parallel_K2D::get_instance();
};

TEST_F(Parallel_K2DTest, set_kpar)
{
    k2d.set_kpar(2);
    EXPECT_EQ(k2d.get_kpar(), 2);
}

TEST_F(Parallel_K2DTest, set_nkstot)
{
    k2d.set_nkstot(10);
    EXPECT_EQ(k2d.get_nkstot(), 10);
}