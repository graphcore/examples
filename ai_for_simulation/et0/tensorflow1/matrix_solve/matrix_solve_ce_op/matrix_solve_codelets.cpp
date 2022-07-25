// Copyright (c) 2021 Graphcore Ltd. All rights reserved.


#include <poplar/Vertex.hpp>
#include <ipu_vector_math>
#include <print.h>

#define  USE_VECTOR_2

using namespace poplar;

static constexpr auto SPAN    = poplar::VectorLayout::SPAN;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

template<typename T>
struct FloatDef{
};

template<>
struct FloatDef<float>{
  static inline constexpr float2  kZeroV    = { 0.0f, 0.0f };
  static inline constexpr float2  kOneV     = { 1.0f, 1.0f };
  static inline const int         kSftBits  = 1;
  static inline const int         kPlus     = 1;
  static inline const int         kStep     = 2;
  typedef   float2                FVType;
  typedef   int2                  IVType;
  typedef   int                   IType;
};

template<>
struct FloatDef<half>{
  static inline constexpr half4   kZeroV    = { 0.0f, 0.0f, 0.0f, 0.0f };
  static inline constexpr half4   kOneV     = { 1.0f, 1.0f, 1.0f, 1.0f };
  static inline const int         kSftBits  = 2;
  static inline const int         kPlus     = 3;
  static inline const int         kStep     = 4;
  typedef   half4                 FVType;
  typedef   short4                IVType;
  typedef   short                 IType;
};

template <typename FloatType>
class [[poplar::constraint("elem(*A_) != elem(*buf_)")]] MatrixSolveVertex : public Vertex {
public:
  MatrixSolveVertex();
  Input<Vector<FloatType>>                               A_;
  Input<Vector<FloatType>>                               b_;
  InOut<Vector<FloatType>>                               x_;

  InOut<Vector<FloatType, ONE_PTR, 8, true>>             buf_;

  const int                                              dim_size_;

public:
  static void lu(FloatType const* src, int dim_size, int* P, FloatType* L, FloatType* U)
  {
    int    ele_cnt   = dim_size * dim_size;
    int    i         = 0;
    int    j         = 0;
    for(i = 0 ; i < ele_cnt ;  i ++){
      L[i] = 0.0f;
      U[i] = src[i];
    }
    for(i = 0 ; i < dim_size ; i ++){
        L[i * dim_size + i] = 1.0f;
        P[i]                = i;
    }
    for(j = 0; j < dim_size - 1; j ++)
    {
      i               = j;
      FloatType max   = ipu::fabs(U[i * dim_size + j]);
      int       max_i = i;
      for(i = j + 1; i < dim_size ; i ++)
      {
        FloatType  cur_val = ipu::fabs(U[i * dim_size + j]);
        int        new_i   = i;
        int        mask    = max < cur_val;
        max                = ipu::fmax(max, cur_val);
        mask               = -mask;
        max_i              = (new_i & mask) | (max_i & (~mask));
      }
      if(j != max_i)
      {
        int         len_U       = dim_size - j;
        int         start_pos_j = j * dim_size + j;
        int         start_pos_i = max_i * dim_size + j;
        FloatType*  ptr_j       = U + start_pos_j;
        FloatType*  ptr_i       = U + start_pos_i;
        FloatType   tmp_data    = 0.0f;
        for(int k = 0 ; k < len_U ; k ++){
          tmp_data    = ptr_j[k] ;
          ptr_j[k]    = ptr_i[k] ;
          ptr_i[k]    = tmp_data ;
        }

        start_pos_j =   j * dim_size;
        start_pos_i = max_i * dim_size;
        ptr_j       = L + start_pos_j;
        ptr_i       = L + start_pos_i;
        for(int k = 0 ; k < j ; k ++){
          tmp_data    = ptr_j[k] ;
          ptr_j[k]    = ptr_i[k] ;
          ptr_i[k]    = tmp_data ;
        }

        int tmp   = P[j];
        P[j]      = P[max_i];
        P[max_i]  = tmp;

      }
      for(i = j + 1; i < dim_size ; i ++)
      {
        int       start_pos_i  = i * dim_size;
        int       start_pos_j  = j * dim_size;
        FloatType cur_L        = U[start_pos_i + j]/U[start_pos_j + j];
        L[start_pos_i + j] = cur_L;
        for(int k = j ; k < dim_size ; k ++)
          U[start_pos_i + k] =  U[start_pos_i + k] - cur_L * U[start_pos_j + k];
      }
    }
  }

  static void matrix_solve(FloatType const* A, FloatType const* b, int dim_size, FloatType* x, FloatType* L, FloatType* U, int* P)
  {
    int     i = 0, j = 0;

    lu(A, dim_size, P, L, U);

    FloatType res     = 0.0f;
    //z=Pb
    for(i = 0 ; i < dim_size ; i ++)
      x[i] = b[P[i]];

    //Ly=z => y
    for(i = 0 ; i < dim_size; i ++)
    {
      res = 0.0f;
      for(j = 0 ; j < i; j ++)
        res += L[i * dim_size + j] * x[j];
      x[i] = (x[i] - res) / L[i * dim_size + j];
    }

    //Ux=y => x
    for(i = 0 ; i < dim_size; i ++)
    {
      int max_idx = dim_size - 1;
      int start_i = max_idx - i;
      res = 0.0f;
      for(j = 0 ; j < i; j ++){
        int cur_idx = max_idx - j;
        res += U[start_i * dim_size + cur_idx] * x[cur_idx];
      }
      x[start_i] = (x[start_i] - res) / U[start_i * dim_size + max_idx - j];
    }
  };

  template<typename FType, typename FVType, typename IVType, typename std::enable_if<std::is_same<FType, float>::value, void>::type* = nullptr>
  static void luV(FType const* src, int dim_size, IVType* P, FVType* L, FVType* U)
  {
    int     ele_cnt   = dim_size * dim_size;
    int     i         = 0;
    int     j         = 0;
    FVType  tmp_data ;
    for(i = 0 ; i < ele_cnt ;  i ++){
      L[i]    = FloatDef<FType>::kZeroV ;
      U[i][0] = src[i];
      U[i][1] = src[ele_cnt + i];
    }
    for(i = 0 ; i < dim_size ; i ++){
        L[i * dim_size + i] = FloatDef<FType>::kOneV ;
        P[i]                = {    i,    i } ;
    }
    for(j = 0; j < dim_size - 1; j ++)
    {
      FVType   max   = ipu::fabs(U[j * dim_size + j]);
      IVType   cur_j = { j, j };
      IVType   max_i = cur_j;
      for(i = j + 1; i < dim_size ; i ++)
      {
        FVType  cur_val = ipu::fabs(U[i * dim_size + j]);
        IVType  mask    = (IVType)(max < cur_val);
        max             = ipu::fmax(max, cur_val);
        IVType  new_i   = { i, i };
        max_i           = (mask & new_i) | ((~mask) & max_i);
      }
      int2   diff  = cur_j - max_i;
      if(0 != (diff[0] | diff[1]))
      {
        int      len_U       = dim_size - j;
        int      start_pos_j =    j  * dim_size + j;
        IVType   start_pos_i = max_i * dim_size + cur_j;
        FVType*  ptr_j       = U + start_pos_j;
        FVType*  ptr_i_0     = U + start_pos_i[0];
        FVType*  ptr_i_1     = U + start_pos_i[1];
        for(int k = 0 ; k < len_U ; k ++){
          tmp_data       = ptr_j[k] ;
          ptr_j[k][0]    = ptr_i_0[k][0] ;
          ptr_j[k][1]    = ptr_i_1[k][1] ;
          ptr_i_0[k][0]  = tmp_data[0] ;
          ptr_i_1[k][1]  = tmp_data[1] ;
        }

        start_pos_j = j * dim_size;
        start_pos_i = max_i * dim_size;
        ptr_j       = L + start_pos_j;
        ptr_i_0     = L + start_pos_i[0];
        ptr_i_1     = L + start_pos_i[1];
        for(int k = 0 ; k < j ; k ++){
          tmp_data       = ptr_j[k] ;
          ptr_j[k][0]    = ptr_i_0[k][0] ;
          ptr_j[k][1]    = ptr_i_1[k][1] ;
          ptr_i_0[k][0]  = tmp_data[0] ;
          ptr_i_1[k][1]  = tmp_data[1] ;
        }

        IVType tmp      = P[j];
        P[j]            = { P[max_i[0]][0], P[max_i[1]][1] };
        P[max_i[0]][0]  = tmp[0];
        P[max_i[1]][1]  = tmp[1];
      }

      for(i = j + 1; i < dim_size ; i ++)
      {
        int    start_pos_i  = i * dim_size;
        int    start_pos_j  = j * dim_size;
        FVType cur_L        = U[start_pos_i + j]/U[start_pos_j + j];
        L[start_pos_i + j]  = cur_L;
        for(int k = j ; k < dim_size ; k ++)
          U[start_pos_i + k] =  U[start_pos_i + k] - cur_L * U[start_pos_j + k];
      }
    }
  }

  template<typename FType, typename FVType, typename IVType, typename std::enable_if<std::is_same<FType, half>::value, void>::type* = nullptr>
  static void luV(FType const* src, int dim_size, IVType* P, FVType* L, FVType* U)
  {
    int     ele_cnt   = dim_size * dim_size;
    int     i         = 0;
    int     j         = 0;
    FVType  tmp_data ;
    for(i = 0 ; i < ele_cnt ;  i ++){
      L[i]    = FloatDef<FType>::kZeroV ;
      U[i][0] = src[i];
      U[i][1] = src[ele_cnt + i];
      U[i][2] = src[2 * ele_cnt + i];
      U[i][3] = src[3 * ele_cnt + i];
    }
    for(i = 0 ; i < dim_size ; i ++){
        L[i * dim_size + i] = FloatDef<FType>::kOneV ;
        P[i]                = { (short int)i, (short int)i, (short int)i, (short int)i } ;
    }

    for(j = 0; j < dim_size - 1; j ++)
    {
      FVType   max   = ipu::fabs(U[j * dim_size + j]);
      IVType   cur_j = { static_cast<short>(j), static_cast<short>(j), static_cast<short>(j), static_cast<short>(j) };
      IVType   max_i = cur_j;
      for(i = j + 1; i < dim_size ; i ++)
      {
        FVType  cur_val = ipu::fabs(U[i * dim_size + j]);
        IVType  mask    = (IVType)(max < cur_val);
        max             = ipu::fmax(max, cur_val);
        IVType  new_i   = { static_cast<short>(i), static_cast<short>(i), static_cast<short>(i), static_cast<short>(i) } ;
        max_i           = (mask & new_i) | ((~mask) & max_i);
      }
      
      IVType   diff       = cur_j - max_i;
      int2     max_i_low  = { max_i[0], max_i[1] };
      int2     max_i_high = { max_i[2], max_i[3] };
      int2     cur_j_2    = { j, j };
      if(0 != (diff[0] | diff[1] | diff[2] | diff[3]))
      {
        int      len_U            = dim_size - j;
        int      start_pos_j      =         j  * dim_size + j;
        int2     start_pos_i_low  = max_i_low  * dim_size + cur_j_2;
        int2     start_pos_i_high = max_i_high * dim_size + cur_j_2;
        FVType*  ptr_j            = U + start_pos_j;
        FVType*  ptr_i_0          = U + start_pos_i_low[0];
        FVType*  ptr_i_1          = U + start_pos_i_low[1];
        FVType*  ptr_i_2          = U + start_pos_i_high[0];
        FVType*  ptr_i_3          = U + start_pos_i_high[1];
        for(int k = 0 ; k < len_U ; k ++){
          tmp_data       = ptr_j[k] ;
          ptr_j[k][0]    = ptr_i_0[k][0] ;
          ptr_j[k][1]    = ptr_i_1[k][1] ;
          ptr_j[k][2]    = ptr_i_2[k][2] ;
          ptr_j[k][3]    = ptr_i_3[k][3] ;
          ptr_i_0[k][0]  = tmp_data[0] ;
          ptr_i_1[k][1]  = tmp_data[1] ;
          ptr_i_2[k][2]  = tmp_data[2] ;
          ptr_i_3[k][3]  = tmp_data[3] ;
        }

        start_pos_j      =         j  * dim_size;
        start_pos_i_low  = max_i_low  * dim_size;
        start_pos_i_high = max_i_high * dim_size;
        ptr_j       = L + start_pos_j;
        ptr_i_0     = L + start_pos_i_low[0];
        ptr_i_1     = L + start_pos_i_low[1];
        ptr_i_2     = L + start_pos_i_high[0];
        ptr_i_3     = L + start_pos_i_high[1];
        for(int k = 0 ; k < j ; k ++){
          tmp_data       = ptr_j[k] ;
          ptr_j[k][0]    = ptr_i_0[k][0] ;
          ptr_j[k][1]    = ptr_i_1[k][1] ;
          ptr_j[k][2]    = ptr_i_2[k][2] ;
          ptr_j[k][3]    = ptr_i_3[k][3] ;
          ptr_i_0[k][0]  = tmp_data[0] ;
          ptr_i_1[k][1]  = tmp_data[1] ;
          ptr_i_2[k][2]  = tmp_data[2] ;
          ptr_i_3[k][3]  = tmp_data[3] ;
        }

        IVType tmp      = P[j];
        P[j]            = { P[max_i[0]][0], P[max_i[1]][1], P[max_i[2]][2], P[max_i[3]][3] };
        P[max_i[0]][0]  = tmp[0];
        P[max_i[1]][1]  = tmp[1];
        P[max_i[2]][2]  = tmp[2];
        P[max_i[3]][3]  = tmp[3];
      }

      for(i = j + 1; i < dim_size ; i ++)
      {
        int    start_pos_i  = i * dim_size;
        int    start_pos_j  = j * dim_size;
        FVType cur_L        = U[start_pos_i + j]/U[start_pos_j + j];
        L[start_pos_i + j]  = cur_L;
        for(int k = j ; k < dim_size ; k ++)
          U[start_pos_i + k] =  U[start_pos_i + k] - cur_L * U[start_pos_j + k];
      }
    }
  }

  template<typename FType, typename FVType, typename IVType, typename std::enable_if<std::is_same<FType, float>::value, void>::type* = nullptr>
  static void matrix_solveV(FType const* A, FType const* b, int dim_size, FType* x, FVType* L, FVType* U, IVType* P, FVType* tmp_data)
  {
    int     i = 0, j = 0;

    luV<FType, FVType, IVType>(A, dim_size, P, L, U);

    FVType res = FloatDef<FType>::kZeroV;
    //z=Pb
    for(i = 0 ; i < dim_size ; i ++)
    {
      tmp_data[i]  = { b[P[i][0]], b[dim_size + P[i][1]] };
    }

    //Ly=z => y
    for(i = 0 ; i < dim_size; i ++)
    {
      res = FloatDef<FType>::kZeroV;
      for(j = 0 ; j < i; j ++)
        res += L[i * dim_size + j] * tmp_data[j];
      tmp_data[i] = (tmp_data[i] - res) / L[i * dim_size + j];
    }

    //Ux=y => x
    for(i = 0 ; i < dim_size; i ++)
    {
      int max_idx = dim_size - 1;
      int start_i = max_idx - i;
      res = FloatDef<FType>::kZeroV ;
      for(j = 0 ; j < i; j ++){
        int cur_idx = max_idx - j;
        res += U[start_i * dim_size + cur_idx] * tmp_data[cur_idx];
      }

      tmp_data[start_i] = (tmp_data[start_i] - res) / U[start_i * dim_size + max_idx - j];
    }

    for(i = 0 ; i < dim_size ; i ++){
      x[i]            = tmp_data[i][0];
      x[dim_size + i] = tmp_data[i][1];
    }
  };

  template<typename FType, typename FVType, typename IVType, typename std::enable_if<std::is_same<FType, half>::value, void>::type* = nullptr>
  static void matrix_solveV(FType const* A, FType const* b, int dim_size, FType* x, FVType* L, FVType* U, IVType* P, FVType* tmp_data)
  {
    int     i = 0, j = 0;

    luV<FType, FVType, IVType>(A, dim_size, P, L, U);

    FVType res = FloatDef<FType>::kZeroV;
    //z=Pb
    for(i = 0 ; i < dim_size ; i ++)
      tmp_data[i]  = { b[P[i][0]], b[dim_size + P[i][1]], b[2 * dim_size + P[i][2]], b[3 * dim_size + P[i][3]] };

    //Ly=z => y
    for(i = 0 ; i < dim_size; i ++)
    {
      res = FloatDef<FType>::kZeroV;
      for(j = 0 ; j < i; j ++)
        res += L[i * dim_size + j] * tmp_data[j];
      tmp_data[i] = (tmp_data[i] - res) / L[i * dim_size + j];
    }

    //Ux=y => x
    for(i = 0 ; i < dim_size; i ++)
    {
      int max_idx = dim_size - 1;
      int start_i = max_idx - i;
      res = FloatDef<FType>::kZeroV ;
      for(j = 0 ; j < i; j ++){
        int cur_idx = max_idx - j;
        res += U[start_i * dim_size + cur_idx] * tmp_data[cur_idx];
      }
      tmp_data[start_i] = (tmp_data[start_i] - res) / U[start_i * dim_size + max_idx - j];
    }

    for(i = 0 ; i < dim_size ; i ++){
      x[i]                = tmp_data[i][0];
      x[dim_size + i]     = tmp_data[i][1];
      x[2 * dim_size + i] = tmp_data[i][2];
      x[3 * dim_size + i] = tmp_data[i][3];
    }
  };

  template<typename T>
  static void run(T const*                       A,
                  T const*                       b,
                  T*                             x,
                  int                            loop,
                  int                            dim_size,
                  typename FloatDef<T>::FVType*  L,
                  typename FloatDef<T>::FVType*  U,
                  typename FloatDef<T>::FVType*  Assist,
                  typename FloatDef<T>::IVType*  P)
  {
    const int  ele_cnt  = dim_size * dim_size;
    auto       loopV    = (loop >> FloatDef<T>::kSftBits) << FloatDef<T>::kSftBits;
    T const*   Aptr     = A;
    T const*   bptr     = b;
    T*         xptr     = x;
    for (auto i = 0u; i < loopV; i += FloatDef<T>::kStep) {
      matrix_solveV<T, typename FloatDef<T>::FVType, typename FloatDef<T>::IVType>(Aptr, 
                                                                           bptr, 
                                                                           dim_size, 
                                                                           xptr, 
                                                                           L, 
                                                                           U, 
                                                                           P, 
                                                                           Assist);
      Aptr += FloatDef<T>::kStep * ele_cnt;
      bptr += FloatDef<T>::kStep * dim_size;
      xptr += FloatDef<T>::kStep * dim_size;
    }

    for(auto i = loopV ; i < loop ; i ++)
    {
      matrix_solve(Aptr, bptr, dim_size, xptr, (T*)L, (T*)U, (int*)P);
      Aptr += ele_cnt;
      bptr += dim_size;
      xptr += dim_size;
    }
  }

  bool compute() {
    int ele_cnt      = dim_size_ * dim_size_;
    int dim_size_v   = ((dim_size_ + FloatDef<FloatType>::kPlus) >> FloatDef<FloatType>::kSftBits) << FloatDef<FloatType>::kSftBits;
    int ele_cnt_v    = ((ele_cnt   + FloatDef<FloatType>::kPlus) >> FloatDef<FloatType>::kSftBits) << FloatDef<FloatType>::kSftBits;
    FloatType const*                       A       = &(A_[0]);
    FloatType const*                       b       = &(b_[0]);
    FloatType*                             x       = &(x_[0]);
    typename FloatDef<FloatType>::FVType*  buf_ptr = (typename FloatDef<FloatType>::FVType*)(&(buf_[0]));
    typename FloatDef<FloatType>::FVType*  L       = buf_ptr;
    typename FloatDef<FloatType>::FVType*  U       = L + ele_cnt_v;
    typename FloatDef<FloatType>::FVType*  Assist  = U + ele_cnt_v;
    typename FloatDef<FloatType>::IVType*  P       = (typename FloatDef<FloatType>::IVType*)(Assist + dim_size_v);
    int        loop     = b_.size();
    loop                = loop / dim_size_;
    run<FloatType>(A, b, x, loop, dim_size_, L, U, Assist, P);
    return true;
  };
};

template class MatrixSolveVertex<float>;
template class MatrixSolveVertex<half>;
