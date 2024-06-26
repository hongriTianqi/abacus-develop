#include "parallel_k2d.h"

template <typename TK>
bool Parallel_K2D<TK>::initialized = false;

template class Parallel_K2D<double>;
template class Parallel_K2D<std::complex<double>>;