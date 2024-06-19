#include "parallel_common.h"

#ifdef __MPI
#include <mpi.h>
#endif

#include <cstring>

#ifdef __MPI
void Parallel_Common::bcast_string(const int &my_rank, std::string &object) // Peize Lin fix bug 2019-03-18
{
    int size = object.size();
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    char *swap = new char[size + 1];
    if (0 == my_rank)
        strcpy(swap, object.c_str());
    MPI_Bcast(swap, size + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (0 != my_rank)
        object = static_cast<std::string>(swap);
    delete[] swap;
    return;
}

void Parallel_Common::bcast_string(const int &my_rank, std::string *object, const int n) // Peize Lin fix bug 2019-03-18
{
    for (int i = 0; i < n; i++)
        bcast_string(my_rank, object[i]);
    return;
}

void Parallel_Common::bcast_complex_double(std::complex<double> &object)
{
    MPI_Bcast(&object, 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}

void Parallel_Common::bcast_complex_double(std::complex<double> *object, const int n)
{
    MPI_Bcast(object, n, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}

void Parallel_Common::bcast_double(double &object)
{
    MPI_Bcast(&object, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Parallel_Common::bcast_double(double *object, const int n)
{
    MPI_Bcast(object, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Parallel_Common::bcast_int(int &object)
{
    MPI_Bcast(&object, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void Parallel_Common::bcast_int(int *object, const int n)
{
    MPI_Bcast(object, n, MPI_INT, 0, MPI_COMM_WORLD);
}

void Parallel_Common::bcast_bool(const int &my_rank, bool &object)
{
    int swap = object;
    if (my_rank == 0)
        swap = object;
    MPI_Bcast(&swap, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank != 0)
        object = static_cast<bool>(swap);
}

void Parallel_Common::bcast_char(char *object, const int n)
{
    MPI_Bcast(object, n, MPI_CHAR, 0, MPI_COMM_WORLD);
}

#endif
