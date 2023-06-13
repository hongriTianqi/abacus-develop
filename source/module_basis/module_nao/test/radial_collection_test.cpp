#include "module_basis/module_nao/radial_collection.h"
#include "gtest/gtest.h"

#ifdef __MPI
#include <mpi.h>
#endif

/***********************************************************
 *      Unit test of class "RadialCollection"
 ***********************************************************/
/*!
 *  Tested functions:
 *
 *  - build
 *      - parse a series of orbital or pseudopotential files and
 *        initialize AtomicRadials/BetaRadials objects accordingly.
 *
 *  - all "getters"
 *      - Get access to private members.
 *
 *  - all "batch setters"
 *      - Set a property for all RadialSet objects at once
 *                                                                      */
class RadialCollectionTest : public ::testing::Test
{
  protected:
    void SetUp();
    void TearDown();

    RadialCollection orb;                                         //!< object under test
    int nfile = 0; // number of orbital/pseudopotential files
    std::string* file = nullptr; //!< orbitals file to read from
    std::string log_file = "./test_files/atomic_orbital.log";         //!< file for logging
};

void RadialCollectionTest::SetUp() {
    std::string dir = "../../../../../tests/PP_ORB/";
    nfile = 4;
    file = new std::string[nfile];
    file[0] = dir + "C_gga_8au_100Ry_2s2p1d.orb";
    file[1] = dir + "H_gga_8au_60Ry_2s1p.orb";
    file[2] = dir + "O_gga_10au_100Ry_2s2p1d.orb";
    file[3] = dir + "Fe_gga_9au_100Ry_4s2p2d1f.orb";
}

void RadialCollectionTest::TearDown() {
    delete[] file;
}

TEST_F(RadialCollectionTest, BuildAndGet) {
    orb.build(nfile, file, 'o');

    EXPECT_EQ(orb.symbol(0), "C");
    EXPECT_EQ(orb.symbol(1), "H");
    EXPECT_EQ(orb.symbol(2), "O");
    EXPECT_EQ(orb.symbol(3), "Fe");

    EXPECT_EQ(orb.ntype(), 4);
    EXPECT_EQ(orb.lmax(), 3);
    EXPECT_DOUBLE_EQ(orb.rcut_max(), 10.0);

    EXPECT_EQ(orb.nzeta(0,0), 2);
    EXPECT_EQ(orb.nzeta(0,1), 2);
    EXPECT_EQ(orb.nzeta(0,2), 1);

    EXPECT_EQ(orb.nzeta(1,0), 2);
    EXPECT_EQ(orb.nzeta(1,1), 1);

    EXPECT_EQ(orb.nzeta(2,0), 2);
    EXPECT_EQ(orb.nzeta(2,1), 2);
    EXPECT_EQ(orb.nzeta(2,2), 1);

    EXPECT_EQ(orb.nzeta(3,0), 4);
    EXPECT_EQ(orb.nzeta(3,1), 2);
    EXPECT_EQ(orb.nzeta(3,2), 2);
    EXPECT_EQ(orb.nzeta(3,3), 1);

    EXPECT_EQ(orb.nzeta_max(0), 2);
    EXPECT_EQ(orb.nzeta_max(1), 2);
    EXPECT_EQ(orb.nzeta_max(2), 2);
    EXPECT_EQ(orb.nzeta_max(3), 4);

    EXPECT_EQ(orb.nchi(0), 5);
    EXPECT_EQ(orb.nchi(1), 3);
    EXPECT_EQ(orb.nchi(2), 5);
    EXPECT_EQ(orb.nchi(3), 9);
    EXPECT_EQ(orb.nchi(), 22);

    for (int itype = 0; itype <= 3; ++itype) {
        EXPECT_EQ(orb(itype).itype(), itype);
    }

    for (int itype = 0; itype <= 3; ++itype) {
        for (int l = 0; l <= orb(itype).lmax(); ++l) {
            for (int izeta = 0; izeta != orb(itype).nzeta(l); ++izeta) {
                EXPECT_EQ(orb(itype, l, izeta).l(), l);
            }
        }
    }
}

TEST_F(RadialCollectionTest, BatchSet) {
    orb.build(nfile, file, 'o');

    ModuleBase::SphericalBesselTransformer sbt;
    orb.set_transformer(&sbt);
    orb.set_uniform_grid(true, 2001, 20.0);

    EXPECT_EQ(orb.rcut_max(), 20.0);
    for (int itype = 0; itype != orb.ntype(); ++itype) {
        for (int l = 0; l <= orb(itype).lmax(); ++l) {
            for (int izeta = 0; izeta != orb.nzeta(itype, l); ++izeta) {
                EXPECT_EQ(&sbt, orb(itype, l, izeta).ptr_sbt());
                EXPECT_DOUBLE_EQ(orb(itype, l, izeta).rcut(), 20.0);
            }
        }
    }

    double* grid = new double[3];
    grid[0] = 0.0;
    grid[1] = 1.0;
    grid[2] = 3.14;

    orb.set_grid(true, 3, grid, 'i');
    for (int itype = 0; itype != orb.ntype(); ++itype) {
        for (int l = 0; l <= orb(itype).lmax(); ++l) {
            for (int izeta = 0; izeta != orb.nzeta(itype, l); ++izeta) {
                EXPECT_DOUBLE_EQ(orb(itype, l, izeta).rcut(), 3.14);
            }
        }
    }
    delete[] grid;
}

int main(int argc, char** argv)
{

#ifdef __MPI
    MPI_Init(&argc, &argv);
#endif

    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef __MPI
    MPI_Finalize();
#endif

    return result;
}
