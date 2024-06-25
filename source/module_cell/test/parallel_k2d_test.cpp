#ifdef __MPI
#include "module_base/parallel_global.h"
#include "module_cell/parallel_k2d.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

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

class ParallelK2DTest : public ::testing::TestWithParam<ParaPrepare>
{
protected:
    Parallel_K2D<double> &k2d = Parallel_K2D<double>::get_instance();
    MPIContext mpi;
    int NPROC;
    int MY_RANK;
    void SetUp() override
    {
        NPROC = mpi.GetSize();
        MY_RANK = mpi.GetRank();
    }
    void TearDown() override
    {
    }
};

TEST_P(ParallelK2DTest, DividePools)
{
    ParaPrepare pp = GetParam();
    mpi.KPAR = pp.KPAR_;
    k2d.Pkpoints = new Parallel_Kpoints;
    k2d.P2D_local = new Parallel_2D;
    if (mpi.KPAR > NPROC)
    {
        std::string output;
        testing::internal::CaptureStdout();
        EXPECT_EXIT(Parallel_Global::divide_mpi_groups(this->NPROC,
                                                       mpi.KPAR,
                                                       this->MY_RANK,
                                                       mpi.NPROC_IN_POOL,
                                                       mpi.MY_POOL,
                                                       mpi.RANK_IN_POOL),
                    testing::ExitedWithCode(1),
                    "");
        output = testing::internal::GetCapturedStdout();
        EXPECT_THAT(output, testing::HasSubstr("must be greater than the number of groups"));
    }
    else
    {
        Parallel_Global::divide_mpi_groups(this->NPROC,
                                           mpi.KPAR,
                                           this->MY_RANK,
                                           mpi.NPROC_IN_POOL,
                                           mpi.MY_POOL,
                                           mpi.RANK_IN_POOL);
        MPI_Comm_split(MPI_COMM_WORLD, mpi.MY_POOL, mpi.RANK_IN_POOL, &POOL_WORLD);
        k2d.Pkpoints->kinfo(pp.nkstot_, mpi.KPAR, mpi.MY_POOL, mpi.RANK_IN_POOL, this->NPROC, 1);
        /*
        for (int ik = 0; ik < pp.nkstot_; ik++)
        {
            std::cout << "whichpool[" << ik << "] = " << k2d.Pkpoints->whichpool[ik] << std::endl;
        }
        */
        for (int ipool = 0; ipool < mpi.KPAR; ipool++)
        {
            std::cout << "nks_pool[" << ipool << "] = " << k2d.Pkpoints->nks_pool[ipool] << std::endl;
            std::cout << "startk_pool[" << ipool << "] = " << k2d.Pkpoints->startk_pool[ipool] << std::endl;
            std::cout << "startpro_pool[" << ipool << "] = " << k2d.Pkpoints->get_startpro_pool(ipool) << std::endl;
        }
        k2d.P2D_local->init(10, 10, 1, POOL_WORLD, 0);
        std::cout << k2d.P2D_local->dim0 << " " << k2d.P2D_local->dim1 << std::endl;
    }
    delete k2d.Pkpoints;
    delete k2d.P2D_local;

}


INSTANTIATE_TEST_SUITE_P(TESTPK,
                         ParallelK2DTest,
                         ::testing::Values(
                             // KPAR, nkstot
                             ParaPrepare(2, 16)));

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
#endif