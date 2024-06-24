#ifdef __MPI
#include "module_cell/parallel_k2d.h"
#include "gtest/gtest.h"

#include <mpi.h>
/************************************************
 *  unit test of class Parallel_K2D
 ***********************************************/

/**
 * Test fixture for class Parallel_K2D
 */

class MPIContext
{
  public:
    MPIContext()
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &_size);
    }

    int GetRank() const
    {
        return _rank;
    }
    int GetSize() const
    {
        return _size;
    }

    int KPAR;
    int NPROC_IN_POOL;
    int MY_POOL;
    int RANK_IN_POOL;

  private:
    int _rank;
    int _size;
};

class ParaPrepare
{
  public:
    ParaPrepare(int KPAR_in, int nkstot_in) : KPAR_(KPAR_in), nkstot_(nkstot_in)
    {
    }
    int KPAR_;
    int nkstot_;
};

class Parallel_K2DTest : public ::testing::TestWithParam<ParaPrepare>
{
protected:
    Parallel_K2D &k2d = Parallel_K2D::get_instance();
    MPIContext mpi;
    int NPROC;
    int MY_RANK;
    void SetUp() override
    {
        NPROC = mpi.GetSize();
        MY_RANK = mpi.GetRank();
    }
};

TEST_P(Parallel_K2DTest, set_kpar)
{
    ParaPrepare pp = GetParam();
    int kpar = pp.KPAR_;
    int nkstot = pp.nkstot_;
    std::cout << "MY_RANK = " << MY_RANK << " NPROC = " << NPROC << " kpar = " << kpar << " nkstot = " << nkstot << std::endl;
    k2d.set_kpar(kpar);
    EXPECT_EQ(k2d.get_kpar(), kpar);
    k2d.set_nkstot(nkstot);
    EXPECT_EQ(k2d.get_nkstot(), nkstot);
}

INSTANTIATE_TEST_SUITE_P(TESTPK,
                         Parallel_K2DTest,
                         ::testing::Values(
                             // KPAR, nkstot
                             ParaPrepare(2, 8)));

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
#endif