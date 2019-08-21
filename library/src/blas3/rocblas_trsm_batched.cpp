/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "gemm.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "trsm_host.hpp"
#include "trtri_trsm.hpp"
#include "utility.h"
#include <algorithm>
#include <cstdio>
#include <tuple>

#define A(ii, jj) (A + (ii) + (jj)*lda)
#define B(ii, jj) (B + (ii) + (jj)*ldb)
#define X(ii, jj) (X + (ii) + (jj)*m)
#define invA(ii) (invA + (ii)*BLOCK)

namespace
{
    // Shared memory usuage is (128/2)^2 * sizeof(float) = 32K. LDS is 64K per CU. Theoretically
    // you can use all 64K, but in practice no.
    constexpr rocblas_int STRSM_BLOCK = 128;
    constexpr rocblas_int DTRSM_BLOCK = 128;

    template <typename>
    constexpr char rocblas_trsm_name[] = "unknown";
    template <>
    constexpr char rocblas_trsm_name<float>[] = "rocblas_batched_strsm";
    template <>
    constexpr char rocblas_trsm_name<double>[] = "rocblas_batched_dtrsm";

    /* ============================================================================================ */

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsm_batched_ex_impl(rocblas_handle    handle,
                                                rocblas_side      side,
                                                rocblas_fill      uplo,
                                                rocblas_operation transA,
                                                rocblas_diagonal  diag,
                                                rocblas_int       m,
                                                rocblas_int       n,
                                                const T*          alpha,
                                                const T* const    A[],
                                                rocblas_int       lda,
                                                T*                B[],
                                                rocblas_int       ldb,
                                                rocblas_int       batch_count,
                                                const T*          supplied_invA      = nullptr,
                                                rocblas_int       supplied_invA_size = 0)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        /////////////
        // LOGGING //
        /////////////
        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto side_letter   = rocblas_side_letter(side);
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_trsm_name<T>,
                              side,
                              uplo,
                              transA,
                              diag,
                              m,
                              n,
                              *alpha,
                              A,
                              lda,
                              B,
                              ldb,
                              batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f trsm_batched -r",
                              rocblas_precision_string<T>,
                              "--side",
                              side_letter,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transA_letter,
                              "--diag",
                              diag_letter,
                              "-m",
                              m,
                              "-n",
                              n,
                              "--alpha",
                              *alpha,
                              "--lda",
                              lda,
                              "--ldb",
                              ldb,
                              "--batch",
                              batch_count);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_trsm_name<T>,
                              side,
                              uplo,
                              transA,
                              diag,
                              m,
                              n,
                              alpha,
                              A,
                              lda,
                              B,
                              ldb,
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                log_profile(handle,
                            rocblas_trsm_name<T>,
                            "side",
                            side_letter,
                            "uplo",
                            uplo_letter,
                            "transA",
                            transA_letter,
                            "diag",
                            diag_letter,
                            "m",
                            m,
                            "n",
                            n,
                            "lda",
                            lda,
                            "ldb",
                            ldb,
                            "batch",
                            batch_count);
            }
        }

        /////////////////////
        // ARGUMENT CHECKS //
        /////////////////////
        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(!alpha || !A)
            return rocblas_status_invalid_pointer;
        if(!B)
            return rocblas_status_invalid_pointer;

        // A is of size lda*k
        rocblas_int k = side == rocblas_side_left ? m : n;

        if(batch_count < 0)
            return rocblas_status_invalid_size;
        // TODO: Should these return invalid_size even if batch_count == 0?
        if(lda < k && batch_count > 0)
            return rocblas_status_invalid_size;
        if(ldb < m && batch_count > 0)
            return rocblas_status_invalid_size;
        if((m < 0 || n < 0) && batch_count > 0)
            return rocblas_status_invalid_size;


        //////////////////////
        // MEMORY MANAGEMENT//
        //////////////////////
        // quick return if possible.
        // return status_size_unchanged if device memory size query
        if(!m || !n)
            return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                         : rocblas_status_success;

        rocblas_status status;

        status = rocblas_trsm_batched_template<BLOCK, T>(handle,
                                                         side,
                                                         uplo,
                                                         transA,
                                                         diag,
                                                         m,
                                                         n,
                                                         alpha,
                                                         A,
                                                         lda,
                                                         B,
                                                         ldb,
                                                         batch_count,
                                                         supplied_invA,
                                                         supplied_invA_size);

        return status;

    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strsm_batched(rocblas_handle     handle,
                                     rocblas_side       side,
                                     rocblas_fill       uplo,
                                     rocblas_operation  transA,
                                     rocblas_diagonal   diag,
                                     rocblas_int        m,
                                     rocblas_int        n,
                                     const float*       alpha,
                                     const float* const A[],
                                     rocblas_int        lda,
                                     float*             B[],
                                     rocblas_int        ldb,
                                     rocblas_int        batch_count)
{
    return rocblas_trsm_batched_ex_impl<STRSM_BLOCK>(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
}

rocblas_status rocblas_dtrsm_batched(rocblas_handle      handle,
                                     rocblas_side        side,
                                     rocblas_fill        uplo,
                                     rocblas_operation   transA,
                                     rocblas_diagonal    diag,
                                     rocblas_int         m,
                                     rocblas_int         n,
                                     const double*       alpha,
                                     const double* const A[],
                                     rocblas_int         lda,
                                     double*             B[],
                                     rocblas_int         ldb,
                                     rocblas_int         batch_count)
{
    return rocblas_trsm_batched_ex_impl<DTRSM_BLOCK>(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
}

// rocblas_status rocblas_trsm_batched_ex(rocblas_handle    handle,
//                                                rocblas_side      side,
//                                                rocblas_fill      uplo,
//                                                rocblas_operation transA,
//                                                rocblas_diagonal  diag,
//                                                rocblas_int       m,
//                                                rocblas_int       n,
//                                                const void*       alpha,
//                                                const void* const A,
//                                                rocblas_int       lda,
//                                                void*             B,
//                                                rocblas_int       ldb,
//                                                rocblas_int       batch_count,
//                                                const void*       invA,
//                                                rocblas_int       invA_size,
//                                                rocblas_datatype  compute_type)
// {
//     switch(compute_type)
//     {
//     case rocblas_datatype_f64_r:
//         return rocblas_trsm_batched_ex_impl<DTRSM_BLOCK>(handle,
//                                                          side,
//                                                          uplo,
//                                                          transA,
//                                                          diag,
//                                                          m,
//                                                          n,
//                                                          static_cast<const double*>(alpha),
//                                                          static_cast<const double*>(A),
//                                                          lda,
//                                                          static_cast<double*>(B),
//                                                          ldb,
//                                                          batch_count,
//                                                          static_cast<const double*>(invA),
//                                                          invA_size);

//     case rocblas_datatype_f32_r:
//         return rocblas_trsm_batched_ex_impl<STRSM_BLOCK>(handle,
//                                                          side,
//                                                          uplo,
//                                                          transA,
//                                                          diag,
//                                                          m,
//                                                          n,
//                                                          static_cast<const float*>(alpha),
//                                                          static_cast<const float*>(A),
//                                                          lda,
//                                                          static_cast<float*>(B),
//                                                          ldb,
//                                                          batch_count,
//                                                          static_cast<const float*>(invA),
//                                                          invA_size);

//     default:
//         return rocblas_status_not_implemented;
//     }
// }

} // extern "C"