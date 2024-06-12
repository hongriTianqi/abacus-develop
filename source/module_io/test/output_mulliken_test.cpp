
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
    Parallel_Orbitals paraV;
    int nrow;
    int ncol;
};

using MyTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_SUITE(OutputMullikenTest, MyTypes);

TYPED_TEST(OutputMullikenTest, OrbInfo)
{
    CellIndex cell_index = CellIndex(this->atomLabels, this->atomCounts, this->lnchiCounts, 1);
    cell_index.write_orb_info("./");
    std::ifstream ifs("./Orbital");
    std::string str((std::istreambuf_iterator<char>(ifs)),std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("#io    spec    l    m    z  sym"));
    EXPECT_THAT(str, testing::HasSubstr("0      Si    2    3    1       dx^2-y^2"));
    remove("./Orbital");
}

#ifdef __MPI
TYPED_TEST(OutputMullikenTest, nspin1)
{
    this->nrow = 13;
    this->ncol = 13;
    this->paraV.init(this->nrow, this->ncol, 1, MPI_COMM_WORLD, 0);
    auto cell_index = CellIndex(this->atomLabels, this->atomCounts, this->lnchiCounts, 1);
    auto out_sk = ModuleIO::Output_Sk<TypeParam>(nullptr, nullptr, &this->paraV, 1, 1);
    auto out_dmk = ModuleIO::Output_DMK<TypeParam>(nullptr, &this->paraV, 1, 1);
    auto mulp = ModuleIO::Output_Mulliken<TypeParam>(&(out_sk), &(out_dmk), &(this->paraV), &(cell_index), {0}, 1);
    mulp.write(0, "./");
    std::vector<double> tot_chg = mulp.get_tot_chg();
    EXPECT_NEAR(tot_chg[0], 4.0, 1e-5);
    //std::cout << tot_chg[0] << std::endl;
}

TYPED_TEST(OutputMullikenTest, nspin2)
{
    this->nrow = 13;
    this->ncol = 13;
    this->paraV.init(this->nrow, this->ncol, 1, MPI_COMM_WORLD, 0);
    auto cell_index = CellIndex(this->atomLabels, this->atomCounts, this->lnchiCounts, 2);
    //auto out_sk = ModuleIO::Output_Sk<TypeParam>(nullptr, nullptr, &this->paraV, 2, 1);
    //auto out_dmk = ModuleIO::Output_DMK<TypeParam>(nullptr, &this->paraV, 2, 1);
    //auto mulp = ModuleIO::Output_Mulliken<TypeParam>(&(out_sk), &(out_dmk), &(this->paraV), &(cell_index), {0, 1}, 2);
    auto out_sk = ModuleIO::Output_Sk<double>(nullptr, nullptr, &this->paraV, 2, 1);
    auto out_dmk = ModuleIO::Output_DMK<double>(nullptr, &this->paraV, 2, 1);
    auto mulp = ModuleIO::Output_Mulliken<double>(&(out_sk), &(out_dmk), &(this->paraV), &(cell_index), {0, 1}, 2);
    mulp.write(0, "./");
    std::vector<double> tot_chg = mulp.get_tot_chg();
    EXPECT_NEAR(tot_chg[0], 3.0, 1e-5);
    EXPECT_NEAR(tot_chg[1], 1.0, 1e-5);
    //std::cout << tot_chg[0] << std::endl;
    //std::cout << tot_chg[1] << std::endl;
}

#include "mpi.h"
int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);

    int nprocs;
    int myrank;

    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();

    return result;
}
#endif