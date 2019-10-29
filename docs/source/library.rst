
.. toctree::
   :maxdepth: 4 
   :caption: Contents:

======
rocBLAS
======

Introduction
------------

Overview
********

A BLAS implementation on top of AMD’s Radeon Open Compute ROCm runtime and toolchains. 
rocBLAS is implemented in the HIP programming language and optimized for AMD’s latest 
discrete GPUs.

======== =========
Acronym  Expansion
======== =========
**BLAS**    **B**\ asic **L**\ inear **A**\ lgebra **S**\ ubprograms
**ROCm**    **R**\ adeon **O**\ pen **C**\ ompute Platfor\ **m**
**HIP**     **H**\ eterogeneous-Compute **I**\ nterface for **P**\ ortability
======== =========

Building and Installing
-----------------------
Installing pre-build packages
*****************************
rocBLAS can be installed on Ubuntu using

::

   sudo apt-get update
   sudo apt-get install rocblas

rocBLAS Debian packages can also be downloaded from the `rocBLAS releases tag <https://github.com/ROCmSoftwarePlatform/rocBLAS/releases>`_. These may be newer than the package from apt-get.

Building from Source
********************

Download rocBLAS
````````````````

Download the master branch of rocBLAS from github using:

::

   git clone -b master https://github.com/ROCmSoftwarePlatform/rocBLAS.git
   cd rocBLAS

Note if you want to contribute to rocBLAS, you will need the develop
branch, not the master branch, and you will need to read
.github/CONTRIBUTING.md.

Below are steps to build either (dependencies + library) or
(dependencies + library + client). You only need (dependencies +
library) if you call rocBLAS from your code, or if you need to install
rocBLAS for other users. The client contains the test code and examples.

It is recommended that the script install.sh be used to build rocBLAS.
If you need individual commands, they are also given.

Use install.sh to build (library dependencies + library)
````````````````````````````````````````````````````````

Common uses of install.sh to build (library dependencies + library) are
in the table below.

+-------------------------------------------+--------------------------+
| install.sh_command                        | description              |
+===========================================+==========================+
| ``./install.sh -h``                       | Help information.        |
+-------------------------------------------+--------------------------+
| ``./install.sh -d``                       | Build library            |
|                                           | dependencies and library |
|                                           | in your local directory. |
|                                           | The -d flag only needs   |
|                                           | to be used once. For     |
|                                           | subsequent invocations   |
|                                           | of install.sh it is not  |
|                                           | necessary to rebuild the |
|                                           | dependencies.            |
+-------------------------------------------+--------------------------+
| ``./install.sh``                          | Build library in your    |
|                                           | local directory. It is   |
|                                           | assumed dependencies     |
|                                           | have been built          |
+-------------------------------------------+--------------------------+
| ``./install.sh -i``                       | Build library, then      |
|                                           | build and install        |
|                                           | rocBLAS package in       |
|                                           | /opt/rocm/rocblas. You   |
|                                           | will be prompted for     |
|                                           | sudo access. This will   |
|                                           | install for all users.   |
|                                           | If you want to keep      |
|                                           | rocBLAS in your local    |
|                                           | directory, you do not    |
|                                           | need the -i flag.        |
+-------------------------------------------+--------------------------+

Use install.sh to build (library dependencies + client dependencies + library + client)
```````````````````````````````````````````````````````````````````````````````````````

The client contains executables in the table below.

=============== ====================================================
executable name description
=============== ====================================================
rocblas-test    runs Google Tests to test the library
rocblas-bench   executable to benchmark or test individual functions
example-sscal   example C code calling rocblas_sscal function
=============== ====================================================

Common uses of install.sh to build (dependencies + library + client) are
in the table below.

+-------------------------------------------+--------------------------+
| install.sh_command                        | description              |
+===========================================+==========================+
| ``./install.sh -h``                       | Help information.        |
+-------------------------------------------+--------------------------+
| ``./install.sh -dc``                      | Build library            |
|                                           | dependencies, client     |
|                                           | dependencies, library,   |
|                                           | and client in your local |
|                                           | directory. The -d flag   |
|                                           | only needs to be used    |
|                                           | once. For subsequent     |
|                                           | invocations of           |
|                                           | install.sh it is not     |
|                                           | necessary to rebuild the |
|                                           | dependencies.            |
+-------------------------------------------+--------------------------+
| ``./install.sh -c``                       | Build library and client |
|                                           | in your local directory. |
|                                           | It is assumed the        |
|                                           | dependencies have been   |
|                                           | built.                   |
+-------------------------------------------+--------------------------+
| ``./install.sh -idc``                     | Build library            |
|                                           | dependencies, client     |
|                                           | dependencies, library,   |
|                                           | client, then build and   |
|                                           | install the rocBLAS      |
|                                           | package. You will be     |
|                                           | prompted for sudo        |
|                                           | access. It is expected   |
|                                           | that if you want to      |
|                                           | install for all users    |
|                                           | you use the -i flag. If  |
|                                           | you want to keep rocBLAS |
|                                           | in your local directory, |
|                                           | you do not need the -i   |
|                                           | flag.                    |
+-------------------------------------------+--------------------------+
| ``./install.sh -ic``                      | Build and install        |
|                                           | rocBLAS package, and     |
|                                           | build the client. You    |
|                                           | will be prompted for     |
|                                           | sudo access. This will   |
|                                           | install for all users.   |
|                                           | If you want to keep      |
|                                           | rocBLAS in your local    |
|                                           | directory, you do not    |
|                                           | need the -i flag.        |
+-------------------------------------------+--------------------------+

Build (library dependencies + library) Using Individual Commands
````````````````````````````````````````````````````````````````

Before building the library please install the library dependencies
CMake, Python 2.7, and Python-yaml.

**CMake 3.5 or later**

The build infrastructure for rocBLAS is based on
`Cmake <https://cmake.org/>`__ v3.5. This is the version of cmake
available on ROCm supported platforms. If you are on a headless machine
without the x-windows system, we recommend using **ccmake**; if you have
access to X-windows, we recommend using **cmake-gui**.

Install one-liners cmake: \* Ubuntu: ``sudo apt install cmake-qt-gui``
\* Fedora: ``sudo dnf install cmake-gui``

**Python 2.7**

By default both python2 and python3 are on Ubuntu. You can check the
installation with ``python -V``. Python is used in Tensile, and Tensile
is part of rocBLAS. To build rocBLAS the default version of Python must
be Python 2.7, not Python 3.

**Python-yaml**

PyYAML files contain training information from Tensile that is used to
build gemm kernels in rocBLAS.

Install one-liners PyYAML: \* Ubuntu:
``sudo apt install python2.7 python-yaml`` \* Fedora:
``sudo dnf install python PyYAML``

**Build library**

The rocBLAS library contains both host and device code, so the HCC
compiler must be specified during cmake configuration to properly
initialize build tools. Example steps to build rocBLAS:

.. code:: bash

   # after downloading and changing to rocblas directory:
   mkdir -p build/release
   cd build/release
   # Default install path is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other install path
   # Default build config is 'Release', define -DCMAKE_BUILD_TYPE=Debug to specify Debug configuration
   CXX=/opt/rocm/bin/hcc cmake ../..
   make -j$(nproc)
   #if you want to install in /opt/rocm or the directory set in cmake with -DCMAKE_INSTALL_PREFIX
   sudo make install # sudo required if installing into system directory such as /opt/rocm

Build (library dependencies + client dependencies + library + client) using Individual Commands
```````````````````````````````````````````````````````````````````````````````````````````````

**Additional dependencies for the rocBLAS clients**

The unit tests and benchmarking applications in the client introduce the
following dependencies: 1. `boost <http://www.boost.org/>`__ 2.
`fortran <http://gcc.gnu.org/wiki/GFortran>`__ 2.
`lapack <https://github.com/Reference-LAPACK/lapack-release>`__ \*
lapack itself brings a dependency on a fortran compiler 3.
`googletest <https://github.com/google/googletest>`__

**boost**

Linux distros typically have an easy installation mechanism for boost
through the native package manager.

-  Ubuntu: ``sudo apt install libboost-program-options-dev``
-  Fedora: ``sudo dnf install boost-program-options``

Unfortunately, googletest and lapack are not as easy to install. Many
distros do not provide a googletest package with pre-compiled libraries,
and the lapack packages do not have the necessary cmake config files for
cmake to configure linking the cblas library. rocBLAS provide a cmake
script that builds the above dependencies from source. This is an
optional step; users can provide their own builds of these dependencies
and help cmake find them by setting the CMAKE_PREFIX_PATH definition.
The following is a sequence of steps to build dependencies and install
them to the cmake default /usr/local.

**gfortran and lapack**

LAPACK is used in the client to test rocBLAS. LAPACK is a Fortran
Library, so gfortran is required for building the client.

\*Ubuntu ``apt-get update``

``apt-get install gfortran``

\*Fedora ``yum install gcc-gfortran``

.. code:: bash

   mkdir -p build/release/deps
   cd build/release/deps
   cmake -DBUILD_BOOST=OFF ../../deps   # assuming boost is installed through package manager as above
   make -j$(nproc) install

Build Library and Client Using Individual Commands
``````````````````````````````````````````````````

Once dependencies are available on the system, it is possible to
configure the clients to build. This requires a few extra cmake flags to
the library cmake configure script. If the dependencies are not
installed into system defaults (like /usr/local ), you should pass the
CMAKE_PREFIX_PATH to cmake to help find them. \*
``-DCMAKE_PREFIX_PATH="<semicolon separated paths>"``

.. code:: bash

   # after downloading and changing to rocblas directory:
   mkdir -p build/release
   cd build/release
   # Default install location is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other
   CXX=/opt/rocm/bin/hcc cmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SAMPLES=ON ../..
   make -j$(nproc)
   sudo make install   # sudo required if installing into system directory such as /opt/rocm

Use of Tensile
``````````````

The rocBLAS library uses
`Tensile <https://github.com/ROCmSoftwarePlatform/Tensile>`__, which
supplies the high-performance implementation of xGEMM. Tensile is
downloaded by cmake during library configuration and automatically
configured as part of the build, so no further action is required by the
user to set it up.

CUDA build errata
`````````````````

rocBLAS is written with HiP kernels, so it should build and run on CUDA
platforms. However, currently the cmake infrastructure is broken with a
CUDA backend. However, a BLAS marshalling library that presents a common
interface for both ROCm and CUDA backends can be found with
`hipBLAS <https://github.com/ROCmSoftwarePlatform/hipBLAS>`__.

Common build problems
`````````````````````

-  **Issue:** Could not find a configuration file for package “LLVM”
   that is compatible with requested version “7.0”.

   **Solution:** You may have outdated rocBLAS dependencies in
   /usr/local. If you do not have anything other than rocBLAS
   dependencies in /usr/local, then rename /usr/local and re-build
   rocBLAS dependencies by running install.sh with the -d flag. If you
   have other software in /usr/local, then uninstall the rocBLAS
   dependencies, and re-install by running install.sh with the -d flag.

-  **Issue:** “Tensile could not be found because dependency Python
   Interp could not be found”.

   **Solution:** Due to a bug in Tensile, you may need cmake-gui 3.5 and
   above, though in the cmakefiles it requires 2.8.

-  **Issue:** HIP (/opt/rocm/hip) was built using hcc
   1.0.xxx-xxx-xxx-xxx, but you are using /opt/rocm/hcc/hcc with version
   1.0.yyy-yyy-yyy-yyy from hipcc. (version does not match) . Please
   rebuild HIP including cmake or update HCC_HOME variable.

   **Solution:** Download HIP from github and use hcc to `build from
   source <https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md>`__
   and then use the build HIP instead of /opt/rocm/hip one or singly
   overwrite the new build HIP to this location.

-  **Issue:** For Carrizo - HCC RUNTIME ERROR: Fail to find compatible
   kernel

   **Solution:** Add the following to the cmake command when
   configuring: -DCMAKE_CXX_FLAGS=“–amdgpu-target=gfx801”

-  **Issue:** For MI25 (Vega10 Server) - HCC RUNTIME ERROR: Fail to find
   compatible kernel

   **Solution:** export HCC_AMDGPU_TARGET=gfx900

-  **Issue:** Could not find a package configuration file provided by
   “ROCM” with any of the following names:

   ROCMConfig.cmake

   rocm-config.cmake

   **Solution:** Install `ROCm cmake
   modules <https://github.com/RadeonOpenCompute/rocm-cmake>`__

Example
-------
Following is a simple example for the :code:`rocblas_scal` function:

.. code:: c

   #include <stdlib.h>
   #include <stdio.h>
   #include <vector>
   #include <math.h>
   #include "rocblas.h"

   using namespace std;

   int main()
   {
       rocblas_int N = 10240;
       float alpha = 10.0;

       vector<float> hx(N);
       vector<float> hz(N);
       float* dx;
       float tolerance = 0, error;

       rocblas_handle handle;
       rocblas_create_handle(&handle);

       // allocate memory on device
       hipMalloc(&dx, N * sizeof(float));

       // Initial Data on CPU,
       srand(1);
       for( int i = 0; i < N; ++i )
       {
           hx[i] = rand() % 10 + 1;  //generate a integer number between [1, 10]
       }

       // save a copy in hz 
       hz = hx;

       hipMemcpy(dx, hx.data(), sizeof(float) * N, hipMemcpyHostToDevice);

       rocblas_sscal(handle, N, &alpha, dx, 1);

       // copy output from device memory to host memory
       hipMemcpy(hx.data(), dx, sizeof(float) * N, hipMemcpyDeviceToHost);

       // verify rocblas_scal result
       for(rocblas_int i=0;i<N;i++)
       {
           error = fabs(hz[i] * alpha - hx[i]);
           if(error > tolerance)
           {
             printf("error in element %d: CPU=%f, GPU=%f ", i, hz[i] * alpha, hx[i]);
             break;
           }
       }

       if(error > tolerance)
       {
           printf("SCAL Failed !\n");
       }
       else
       {
           printf("SCAL Success !\n");
       }

       hipFree(dx);
       rocblas_destroy_handle(handle);
       return 0;
   }

Paste the above code into the file rocblas_sscal_example.cpp

Use hipcc Compiler
*******************

The recommend host compiler is [hipcc]
(https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/). To use
hipcc you will need to add /opt/rocm/bin to your path with the
following:

::

   export PATH=$PATH:/opt/rocm/bin

The following makefile can be used to build the executable.

The Makefile assumes that rocBLAS is installed in the default location
/opt/rocm/rocblas. If you have rocBLAS installed in your home directory
in ~/rocBLAS/build/release/rocblas-install/rocblas then edit Makefile
and change /opt/rocm/rocblas to
~/rocBLAS/build/release/rocblas-install/rocblas.

You may need to give the location of the library with

::

   export LD_LIBRARY_PATH=/opt/rocm/rocblas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

Run the executable with the command

::

   ./rocblas_sscal_example

::

   # Makefile assumes rocBLAS is installed in /opt/rocm/rocblas

   ROCBLAS_INSTALL_DIR=/opt/rocm/rocblas
   ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
   ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
   ROCBLAS_LIB=rocblas
   HIP_INCLUDE=/opt/rocm/hip/include
   LDFLAGS=-L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB)
   LD=hipcc
   CFLAGS=-I$(ROCBLAS_INCLUDE) -I$(HIP_INCLUDE)
   CPP=hipcc
   OBJ=rocblas_sscal_example.o
   EXE=rocblas_sscal_example

   %.o: %.cpp
       $(CPP) -c -o $@ $< $(CFLAGS)

   $(EXE) : $(OBJ)
       $(LD) $(OBJ) $(LDFLAGS) -o $@ 

   clean:
       rm -f $(EXE) $(OBJ)

Use g++ Compiler
*****************

Use the Makefile below

::

   ROCBLAS_INSTALL_DIR=/opt/rocm/rocblas
   ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
   ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
   ROCBLAS_LIB=rocblas
   ROCM_INCLUDE=/opt/rocm/include
   LDFLAGS=-L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB) -L/opt/rocm/lib -lhip_hcc
   LD=g++
   CFLAGS=-I$(ROCBLAS_INCLUDE) -I$(ROCM_INCLUDE) -D__HIP_PLATFORM_HCC__
   CPP=g++
   OBJ=rocblas_sscal_example.o
   EXE=rocblas_sscal_example

   %.o: %.cpp
       $(CPP) -c -o $@ $< $(CFLAGS)

   $(EXE) : $(OBJ)
       $(LD) $(OBJ) $(LDFLAGS) -o $@

   clean:
       rm -f $(EXE) $(OBJ)

Running
-------

Notice
******

This secion describes running the examples, tests, and benchmarks in the
client. Before reading this Wiki, it is assumed rocBLAS (dependencies +
library + client) has been built as described in `1.Build <1.Build>`__

Examples
********

The default for [BUILD_DIR] is ~/rocblas/build.

::

   cd [BUILD_DIR]/release/clients/staging
   ./example-sscal
   ./example-scal-template
   ./example-sgemm
   ./example-sgemm-strided-batched

Code for the examples is at:
https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/develop/clients/samples

In addition see
`2.Example <https://github.com/ROCmSoftwarePlatform/rocBLAS/wiki/2.Example>`__.

Unit Tests
**********

Run tests with the following:

::

   cd [BUILD_DIR]/release/clients/staging
   ./rocblas-test

To run specific tests, use –gtest_filter=match where match is a
‘:’-separated list of wildcard patterns (called the positive patterns)
optionally followed by a ‘-’ and another ‘:’-separated pattern list
(called the negative patterns). For example, run gemv tests with the
following:

::

   cd [BUILD_DIR]/release/clients/staging
   ./rocblas-test --gtest_filter=*checkin*gemm*float*-*batched*:*NaN*

Benchmarks
**********

Run bencharmks with the following:

::

   cd [BUILD_DIR]/release/clients/staging
   ./rocblas-bench -h

The following are examples for running particular gemm and gemv
benchmark:

::

   ./rocblas-bench -f gemm -r s -m 1024 -n 1024 -k 1024 --transposeB T -v 1
   ./rocblas-bench -f gemv -m 9216 -n 9216 --lda 9216 --transposeA T

Exported BLAS functions
-----------------------

rocBLAS includes the following auxiliary functions:

+--------------------------+
| Function Name            |
+==========================+
| rocblas_create_handle    |
+--------------------------+
| rocblas_destroy_handle   |
+--------------------------+
| rocblas_add_stream       |
+--------------------------+
| rocblas_set_stream       |
+--------------------------+
| rocblas_get_stream       |
+--------------------------+
| rocblas_set_pointer_mode |
+--------------------------+
| rocblas_get_pointer_mode |
+--------------------------+
| rocblas_set_vector       |
+--------------------------+
| rocblas_get_vector       |
+--------------------------+
| rocblas_set_matrix       |
+--------------------------+
| rocblas_get_matrix       |
+--------------------------+


rocBLAS includes the following Level 1, 2, and 3 functions:

Level 1
*******

============== ====== ====== ============== ============== ====
Function       single double single complex double complex half
============== ====== ====== ============== ============== ====
rocblas_Xscal  x      x      x              x             
rocblas_Xcopy  x      x      x              x             
rocblas_Xdot   x      x      x              x              x
rocblas_Xswap  x      x      x              x             
rocblas_Xaxpy  x      x      x              x              x
rocblas_Xasum  x      x      x              x             
rocblas_Xnrm2  x      x      x              x             
rocblas_iXamax x      x      x              x             
rocblas_iXamin x      x      x              x             
============== ====== ====== ============== ============== ====

Level 2
*******

============= ====== ====== ============== ============== ====
Function      single double single complex double complex half
============= ====== ====== ============== ============== ====
rocblas_Xgemv x      x      x              x             
rocblas_Xger  x      x                                   
rocblas_Xsyr  x      x                                   
============= ====== ====== ============== ============== ====

Level 3
*******

============================= ====== ====== ============== ============== ====
Function                      single double single complex double complex half
============================= ====== ====== ============== ============== ====
rocblas_Xtrtri                x      x                                   
rocblas_Xtrtri_batched        x      x                                   
rocblas_Xtrsm                 x      x                                   
rocblas_Xgemm                 x      x      x              x              x
rocblas_Xgemm_strided_batched x      x      x              x              x
rocblas_Xgeam                 x      x                                   
============================= ====== ====== ============== ============== ====

Rules for obtaining the rocBLAS API from Legacy BLAS
****************************************************

1. The Legacy BLAS routine name is changed to lower case, and prefixed
   by rocblas_.

2. A first argument rocblas_handle handle is added to all rocBlas
   functions.

3. Input arguments are declared with the const modifier.

4. Character arguments are replaced with enumerated types defined in
   rocblas_types.h. They are passed by value on the host.

5. Array arguments are passed by reference on the device.

6. Scalar arguments are passed by value on the host with the following
   two exceptions:

-  Scalar values alpha and beta are passed by reference on either the
   host or the device. The rocBLAS functions will check to see it the
   value is on the device. If this is true, it is used, else the value
   on the host is used.

-  Where Legacy BLAS functions have return values, the return value is
   instead added as the last function argument. It is returned by
   reference on either the host or the device. The rocBLAS functions
   will check to see it the value is on the device. If this is true, it
   is used, else the value is returned on the host. This applies to the
   following functions: xDOT, xDOTU, xNRM2, xASUM, IxAMAX, IxAMIN.

7. The return value of all functions is rocblas_status, defined in
   rocblas_types.h. It is used to check for errors.

rocBLAS interface examples
**************************

In general, the rocBLAS interface is compatible with CPU oriented
[Netlib BLAS][] and the cuBLAS-v2 API, with the explicit exception that
traditional BLAS interfaces do not accept handles. The cuBLAS’
cublasHandle_t is replaced with rocblas_handle everywhere. Thus, porting
a CUDA application which originally calls the cuBLAS API to a HIP
application calling rocBLAS API should be relatively straightforward.
For example, the rocBLAS SGEMV interface is:

GEMV API
````````

.. code:: c

   rocblas_status
   rocblas_sgemv(rocblas_handle handle,
                 rocblas_operation trans,
                 rocblas_int m, rocblas_int n,
                 const float* alpha,
                 const float* A, rocblas_int lda,
                 const float* x, rocblas_int incx,
                 const float* beta,
                 float* y, rocblas_int incy);

LP64 interface
**************

The rocBLAS library is LP64, so rocblas_int arguments are 32 bit and
rocblas_long arguments are 64 bit.

Column-major storage and 1 based indexing
*****************************************

rocBLAS uses column-major storage for 2D arrays, and 1 based indexing
for the functions xMAX and xMIN. This is the same as Legacy BLAS and
cuBLAS.

If you need row-major and 0 based indexing (used in C language arrays)
download the `CBLAS <http://www.netlib.org/blas/#_cblas>`__ file
cblas.tgz. Look at the CBLAS functions that provide a thin interface to
Legacy BLAS. They convert from row-major, 0 based, to column-major, 1
based. This is done by swapping the order of function arguments. It is
not necessary to transpose matrices.

Pointer mode
************

The auxiliary functions rocblas_set_pointer and rocblas_get_pointer are
used to set and get the value of the state variable
rocblas_pointer_mode. If rocblas_pointer_mode ==
rocblas_pointer_mode_host then scalar parameters must be allocated on
the host. If rocblas_pointer_mode == rocblas_pointer_mode_device, then
scalar parameters must be allocated on the device.

There are two types of scalar parameter: 1. scaling parameters like
alpha and beta used in functions like axpy, gemv, gemm 2. scalar results
from functions amax, amin, asum, dot, nrm2

For scalar parameters like alpha and beta when rocblas_pointer_mode ==
rocblas_pointer_mode_host they can be allocated on the host heap or
stack. The kernel launch is asynchronous, and if they are on the heap
they can be freed after the return from the kernel launch. When
rocblas_pointer_mode == rocblas_pointer_mode_device they must not be
changed till the kernel completes.

For scalar results, when rocblas_pointer_mode ==
rocblas_pointer_mode_host then the function blocks the CPU till the GPU
has copied the result back to the host. When rocblas_pointer_mode ==
rocblas_pointer_mode_device the function will return after the
asynchronous launch. Similarly to vector and matrix results, the scalar
result is only available when the kernel has completed execution.

Asynchronous API
****************

Except a functions having memory allocation inside preventing
asynchronicity, most of the rocBLAS functions are configured to operate
in asynchronous fashion with respect to CPU, meaning these library
functions return immediately.


Logging
-------

Four environment variables can be set to control logging: \*
``ROCBLAS_LAYER`` \* ``ROCBLAS_LOG_TRACE_PATH`` \*
``ROCBLAS_LOG_BENCH_PATH`` \* ``ROCBLAS_LOG_PROFILE_PATH``

``ROCBLAS_LAYER`` is a bitwise OR of zero or more bit masks as follows:

-  If ``ROCBLAS_LAYER`` is not set, then there is no logging
-  If ``(ROCBLAS_LAYER & 1) != 0``, then there is trace logging
-  If ``(ROCBLAS_LAYER & 2) != 0``, then there is bench logging
-  If ``(ROCBLAS_LAYER & 4) != 0``, then there is profile logging

Trace logging outputs a line each time a rocBLAS function is called. The
line contains the function name and the values of arguments.

Bench logging outputs a line each time a rocBLAS function is called. The
line can be used with the executable ``rocblas-bench`` to call the
function with the same arguments.

Profile logging, at the end of program execution, outputs a YAML
description of each rocBLAS function called, the values of its
arguments, and the number of times it was called with those arguments.

The default stream for logging output is standard error. Three
environment variables can set the full path name for a log file: \*
``ROCBLAS_LOG_TRACE_PATH`` sets the full path name for trace logging \*
``ROCBLAS_LOG_BENCH_PATH`` sets the full path name for bench logging \*
``ROCBLAS_LOG_PROFILE_PATH`` sets the full path name for profile logging

If a path name cannot be opened, then the corresponding logging output
is streamed to standard error.

Note that performance will degrade when logging is enabled.

When profile logging is enabled, memory usage will increase. If the
program exits abnormally, then it is possible that profile logging will
not be outputted before the program exits.


Device and Stream Management
----------------------------

HIP Device management
*********************

hipSetDevice() & hipGetDevice() are HIP device management APIs. They are
NOT part of the rocBLAS API.

Before a HIP kernel invocation, users need to call hipSetDevice() to set
a device, e.g. device 1. If users do not explicitly call it, the system
by default sets it as device 0. Unless users explicitly call
hipSetDevice() to set to another device, their HIP kernels are always
launched on device 0.

The above is a HIP (and CUDA) device management approach and has nothing
to do with rocBLAS. rocBLAS honors the approach above and assumes users
have already set the device before a rocBLAS routine call.

Once users set the device, they create a handle with
``rocblas_status rocblas_create_handle(rocblas_handle *handle)``

Subsequent rocBLAS routines take this handle as an input parameter.
rocBLAS ONLY queries (by hipGetDevice) the user’s device; rocBLAS but
does NOT set the device for users. If rocBLAS does not see a valid
device, it returns an error message to users. It is the users’
responsibility to provide a valid device to rocBLAS and ensure the
device safety as explained soon.

Users CANNOT switch devices between rocblas_create_handle() and
rocblas_destroy_handle() (the same as cuBLAS requires). If users want to
change device, they must destroy the current handle, and create another
rocBLAS handle (context).

Stream management
*****************

HIP kernels are always launched in a queue (otherwise known as a stream,
they are the same thing).

If users do not explicitly specify a stream, the system provides a
default stream, maintained by the system. Users cannot create or destroy
the default stream. Howevers, users can freely create new streams (with
hipStreamCreate) and bind it to the rocBLAS handle:
``rocblas_set_stream(rocblas_handle handle, hipStream_t stream_id)`` HIP
kernels are invoked in rocBLAS routines. The rocBLAS handles are always
associated with a stream, and rocBLAS passes its stream to the kernels
inside the routine. One rocBLAS routine only takes one stream in a
single invocation. If users create a stream, they are responsible for
destroying it.

Multiple streams and multiple devices
*************************************

If the system under test has 4 HIP devices, users can run 4 rocBLAS
handles (also known as contexts) on 4 devices concurrently, but can NOT
span a single rocBLAS handle on 4 discrete devices. Each handle is
associated with a particular singular device, and a new handle should be
created for each additional device.


Contributing
------------

Contribution License Agreement
******************************

1. The code I am contributing is mine, and I have the right to license
   it.

2. By submitting a pull request for this project I am granting you a
   license to distribute said code under the MIT License for the
   project.

How to contribute
*****************

Our code contriubtion guidelines closely follows the model of `GitHub
pull-requests <https://help.github.com/articles/using-pull-requests/>`__.
This repository follows the `git
flow <http://nvie.com/posts/a-successful-git-branching-model/>`__
workflow, which dictates a /master branch where releases are cut, and a
/develop branch which serves as an integration branch for new code. \* A
`git extention <https://github.com/nvie/gitflow>`__ has been developed
to ease the use of the 'git flow' methodology, but requires manual
installation by the user. Refer to the projects wiki

Pull-request guidelines
***********************

-  target the **develop** branch for integration
-  ensure code builds successfully.
-  do not break existing test cases
-  new functionality will only be merged with new unit tests
-  new unit tests should integrate within the existing `googletest
   framework <https://github.com/google/googletest/blob/master/googletest/docs/primer.md>`__
-  tests must have good code coverage
-  code must also have benchmark tests, and performance must approach
   the compute bound limit or memory bound limit.

StyleGuide
**********

This project follows the `CPP Core
guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`__,
with few modifications or additions noted below. All pull-requests
should in good faith attempt to follow the guidelines stated therein,
but we recognize that the content is lengthy. Below we list our primary
concerns when reviewing pull-requests.

Interface
`````````

-  All public APIs are C89 compatible; all other library code should use
   c++14
-  Our minimum supported compiler is clang 3.6
-  Avoid CamelCase
-  This rule applies specifically to publicly visible APIs, but is also
   encouraged (not mandated) for internal code

Philosophy
``````````

-  `P.2 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-Cplusplus>`__:
   Write in ISO Standard C++14 (especially to support windows, linux and
   macos plaforms )
-  `P.5 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-compile-time>`__:
   Prefer compile-time checking to run-time checking

Implementation
``````````````

-  `SF.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-file-suffix>`__:
   Use a ``.cpp`` suffix for code files and an ``.h`` suffix for
   interface files if your project doesn't already follow another
   convention
-  We modify this rule:

   -  ``.h``: C header files
   -  ``.hpp``: C++ header files

-  `SF.5 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-consistency>`__:
   A ``.cpp`` file must include the ``.h`` file(s) that defines its
   interface
-  `SF.7 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using-directive>`__:
   Don't put a ``using``-directive in a header file
-  `SF.8 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-guards>`__:
   Use ``#include`` guards for all ``.h`` files
-  `SF.21 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-unnamed>`__:
   Don't use an unnamed (anonymous) ``namespace`` in a header
-  `SL.10 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rsl-arrays>`__:
   Prefer using ``std::array`` or ``std::vector`` instead of a C array
-  `C.9 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-private>`__:
   Minimize the exposure of class members
-  `F.3 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-single>`__:
   Keep functions short and simple
-  `F.21 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-out-multi>`__:
   To return multiple 'out' values, prefer returning a ``std::tuple``
-  `R.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rr-raii>`__:
   Manage resources automatically using RAII (this includes
   ``std::unique_ptr`` & ``std::shared_ptr``)
-  `ES.11 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-auto>`__:
   Use ``auto`` to avoid redundant repetition of type names
-  `ES.20 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-always>`__:
   Always initialize an object
-  `ES.23 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-list>`__:
   Prefer the ``{}`` initializer syntax
-  `CP.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#S-concurrency>`__:
   Assume that your code will run as part of a multi-threaded program
-  `I.2 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Ri-global>`__:
   Avoid global variables

Format
``````

C and C++ code is formatted using ``clang-format``. Use the clang-format
version for Clang 9, which is available in the ``/opt/rocm`` directory.
Please do not use your system's built-in ``clang-format``, as this is an
older version that will result in different results.

To format a file, use:

::

    /opt/rocm/hcc/bin/clang-format -style=file -i <path-to-source-file>

To format all files, run the following script in rocBLAS directory:

::

    #!/bin/bash
    git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/hcc/bin/clang-format  -style=file -i

Also, githooks can be installed to format the code per-commit:

::

    ./.githooks/install

Coding Guidelines
*****************

1.  With the `rocBLAS device memory allocation
    system <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/docs/Device_Memory_Allocation.pdf>`__,
    rocBLAS kernels should not call ``hipMalloc()`` or ``hipFree()`` in
    their own code, but should use the device memory manager.

    ``hipMalloc()`` and ``hipFree()`` are synchronizing operations which
    should be avoided as much as possible.

    The device memory allocation system provides:

    -  A ``device_malloc`` method for temporarily using device memory
       which has either been allocated before, or which is allocated on
       demand.
    -  A method to reuse device memory across rocBLAS calls, without
       allocating them and deallocating them at every call.
    -  A method for users to query how much device memory is needed for
       a particular kernel call, in order for it to perform optimally.
    -  A method for users to control how much device memory is
       allocated, or whether to leave it up to rocBLAS to allocate it on
       demand.

    **Extra pointers or size arguments for temporary storage should not
    be added to the end of public APIs, only private internal ones.**
    Instead, implementations of the public APIs should request and
    obtain device memory using the rocBLAS device memory manager.
    rocBLAS kernels in the public API must also detect and respond to
    *device memory size queries*.

    A kernel must allocate all of its device memory upfront, for use
    during the entirety of the kernel call. It must not allocate and
    deallocate device memory at different levels of kernel calls. This
    means that if a lower-level kernel needs device memory, it must be
    allocated by higher-level routines and passed down to the
    lower-level routines. When device memory can be shared between two
    or more operations, the maximum size needed by all them should be
    reported or allocated.

    Details are in the `Device Memory
    Allocation <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/docs/Device_Memory_Allocation.pdf>`__
    design document. Examples of how to use the device memory allocator
    are in
    `TRSV <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/src/blas2/rocblas_trsv.cpp>`__
    and
    `TRSM <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/src/blas3/rocblas_trsm.cpp>`__.

2.  Logging, argument error checking and device memory allocation should
    only occur at the top-level kernel routines. Therefore, if one
    rocBLAS routine calls another, the lower-level called routine(s)
    should not perform logging, argument checking, or device memory
    allocation. This can be accomplished in one of two ways:

    A. (Preferred.) Abstract out the computational part of the kernel
    into a separate template function (usually named
    ``rocblas_<kernel>_template``, and call it from a higher-level
    template routine (usually named ``rocblas_<kernel>_impl``) which
    does error-checking, device memory allocation, and logging, and
    which gets called by the C wrappers:

    .. code:: cpp

        template <...>
        rocblas_status rocblas_<kernel>_template(..., T* device_memory)
        {
            // Performs fast computation
            // No argument error checking
            // No logging
            // No device memory allocation -- any temporary device memory must be passed in through pointers
            // Can be called by other computational kernels
            // Called by rocblas_<kernel>_impl
            // Private internal API
        }

        template <...>
        rocblas_status rocblas_<kernel>_impl()
        {
            // Argument error checking
            // Logging
            // Responding to device memory size queries
            // Device memory allocation (through handle->device_malloc())
            // Temporarily switching to host pointer mode if scalar constants are used
            // Calls rocblas_<kernel>_template()
            // Private internal API
        }

        extern "C" rocblas_status rocblas_[hsdcz]<kernel>()
        {
            // C wrapper
            // Calls rocblas_<kernel>_impl()
            // Public API
        }

    B. Use a ``bool`` template argument to specify if the kernel
    template should perform full functionality or not. Pass device
    memory pointer(s) which will be used if full functionality is turned
    off:

    .. code:: cpp

        template <bool full_function, ...>
        rocblas_status rocblas_<kernel>_template(..., T* device_memory = nullptr)
        {
            if(full_function)
            {
                // Argument error checking
                // Logging
                // Responding to device memory size queries
                // Device memory allocation (memory pointer assumed already allocated otherwise)*
                // Temporarily switching to host pointer mode if scalar constants are used*
                return rocblas_<kernel>_template<false, ...>(...);
            }
            // Perform fast computation
            // Private internal API
        }

    \*Device memory allocation, and temporarily switching pointer mode,
    might be difficult to enclose in an ``if`` statement with the RAII
    design, so the code might have to use recursion to call the
    non-fully-functional version of itself after setting these things
    up. That's why method A above is preferred, but for some huge
    functions like GEMM, method B might be more practical to implement,
    since it disrupts existing code less.

3.  The pointer mode should be temporarily switched to host mode during
    kernels which pass constants to other kernels, so that host-side
    constants of ``-1.0``, ``0.0`` and ``1.0`` can be passed to kernels
    like ``GEMM``, without causing synchronizing host<->device memory
    copies. For example:

    .. code:: cpp

        // Temporarily switch to host pointer mode, saving current pointer mode, restored on return
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        // Get alpha
        T alpha_h;
        if(saved_pointer_mode == rocblas_pointer_mode_host)
            alpha_h = *alpha;
        else
            RETURN_IF_HIP_ERROR(hipMemcpy(&alpha_h, alpha, sizeof(T), hipMemcpyDeviceToHost));

    ``saved_pointer_mode`` can be read to get the old pointer mode. If
    the old pointer mode was host pointer mode, then the host pointer is
    dereferenced to get the value of alpha. If the old pointer mode was
    device pointer mode, then the value of ``alpha`` is copied from the
    device to the host.

    After the above code switches to host pointer mode, constant values
    can be passed to ``GEMM`` or other kernels by always assuming host
    mode:

    .. code:: cpp

        static constexpr T negative_one = -1;
        static constexpr T zero = 0;
        static constexpr T one = 1;

        rocblas_gemm_template( handle, transA, transB, jb, n, jb, alpha, invA, BLOCK, B, ldb, &zero, X, m);

    When ``saved_pointer_mode`` is destroyed, the handle's pointer mode
    returns to the previous pointer mode.

4.  When tests are added to ``rocblas-test`` and ``rocblas-bench``,
    refer to `this
    guide <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/gtest/README.md>`__.

    The test framework is templated, and uses
    `SFINAE <https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error>`__
    and ``std::enable_if<...>`` to enable and disable certain types for
    certain tests.

    YAML files are used to describe tests as combinations of arguments.
    ```rocblas_gentest.py`` <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/common/rocblas_gentest.py>`__
    is used to parse the YAML files and generate tests in the form of a
    binary file of
    ```Arguments`` <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/include/rocblas_arguments.hpp>`__
    records.

    The ``rocblas-test`` and ``rocblas-bench`` `type dispatch
    file <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/include/type_dispatch.hpp>`__
    is central to all tests. Basically, rather than duplicate:

    .. code:: cpp

        if(type == rocblas_datatype_f16_r)
            func1<rocblas_half>(args);
        else if(type == rocblas_datatype_f32_r)
            func<float>(args);
        else if(type == rocblas_datatype_f64_r)
            func<double>(args);

    etc. everywhere, it's done only in one place, and a ``template``
    template argument is passed to specify which action is actually
    taken. It's fairly abstract, but it is powerful. There are examples
    of using the type dispatch in
    ```clients/gtest/*_gtest.cpp`` <https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/develop/clients/gtest>`__
    and
    ```clients/benchmarks/client.cpp`` <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/benchmarks/client.cpp>`__.

5.  Code should not be copied-and pasted, but rather, templates, macros,
    `SFINAE <https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error>`__,
    `CRTP <https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern>`__,
    `lambdas <https://en.cppreference.com/w/cpp/language/lambda>`__,
    etc. should be used to factor out differences in similar code.

    A code should be made more generalized, rather than copied and
    modified, unless it is a completely different kernel function, and
    the old code is just being used as a start.

    If a new function is similar to an existing function, then the
    existing function should be generalized, or the new function and
    existing function should be refactored and based on a third
    templated function or class, rather than duplicating code.

6.  To differentiate between scalars located on either the host or
    device memory, a special function has been created, called
    ``load_scalar()``. If its argument is a pointer, it is dereferenced
    on the device. If the argument is a scalar, it is simply copied.
    This allows single HIP kernels to be written for both device and
    host memory:

    .. code:: cpp

        template <typename T, typename U>
        __global__ void axpy_kernel(rocblas_int n, U alpha_device_host, const T* x, rocblas_int incx, T* y, rocblas_int incy)
        {
            auto alpha = load_scalar(alpha_device_host);
            ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

           // bound
           if(tid < n)
               y[tid * incy] += alpha * x[tid * incx];
        }

    Here, ``alpha_device_host`` can either be a pointer to device
    memory, or a numeric value passed directly to the kernel from the
    host. The ``load_scalar()`` function dereferences it if it's a
    pointer to device memory, and simply returns its argument if it's
    numerical. The kernel is called from the host in one of two ways
    depending on the pointer mode:

    .. code:: cpp

        if(handle->pointer_mode == rocblas_pointer_mode_device)
            hipLaunchKernelGGL(axpy_kernel, blocks, threads, 0, rocblas_stream, n, alpha, x, incx, y, incy);
        else if(*alpha) // alpha is on host
            hipLaunchKernelGGL(axpy_kernel, blocks, threads, 0, rocblas_stream, n, *alpha, x, incx, y, incy);

    When the pointer mode indicates ``alpha`` is on the host, the
    ``alpha`` pointer is dereferenced on the host and the numeric value
    it points to is passed to the kernel. When the pointer mode
    indicates ``alpha`` is on the device, the ``alpha`` pointer is
    passed to the kernel and dereferenced by the kernel on the device.
    This allows a single kernel to handle both cases, eliminating
    duplicate code.

7.  If new arithmetic datatypes (like ``rocblas_bfloat16``) are created,
    then unless they correspond *exactly* to a predefined system type,
    they should be wrapped into a ``struct``, and not simply be a
    ``typedef`` to another type of the same size, so that their type is
    unique and can be differentiated from other types.

    Right now ``rocblas_half`` is ``typedef``\ ed to ``uint16_t``, which
    unfortunately prevents ``rocblas_half`` and ``uint16_t`` from being
    differentiable. If ``rocblas_half`` were simply a ``struct`` with a
    ``uint16_t`` member, then it would be a distinct type.

    It is legal to convert a pointer to a `standard-layout
    ``class``/``struct`` <https://en.cppreference.com/w/cpp/language/data_members#Standard_layout>`__
    to a pointer to its first element, and vice-versa, so the C API is
    unaffected by whether the type is enclosed in a ``struct`` or not.

8.  `RAII <https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization>`__
    classes should be used instead of explicit ``new``/``delete``,
    ``hipMalloc``/``hipFree``, ``malloc``/``free``, etc. RAII classes
    are automatically exception-safe because their destructor gets
    called during unwinding. They only have to be declared once to
    construct them, and they are automatically destroyed when they go
    out of scope. This is better than having to match ``new``/``delete``
    ``malloc``/``free`` calls in the code, especially when exceptions or
    early returns are possible.

    Even if an operation does not allocate and free memory, if it
    represents a change in state which must be undone when a function
    returns, then it belongs in an RAII class. For example,
    ``handle->push_pointer_mode()`` creates an RAII object which saves
    the pointer mode on construction, and restores it on destruction.

9.  When writing function templates, place any non-type parameters
    before type parameters, i.e., leave the type parameters at the end.
    For example:

    .. code:: cpp

        template <rocblas_int NB, typename T> // T is at end
        static rocblas_status rocblas_trtri_batched_template(rocblas_handle handle,
                                                             rocblas_fill uplo,
                                                             rocblas_diagonal diag,
                                                             rocblas_int n,
                                                             const T* A,
                                                             rocblas_int lda,
                                                             rocblas_int bsa,
                                                             T* invA,
                                                             rocblas_int ldinvA,
                                                             rocblas_int bsinvA,
                                                             rocblas_int batch_count,
                                                             T* C_tmp)
        {
            if(!n || !batch_count)
                return rocblas_status_success;

             if(n <= NB)
                 return rocblas_trtri_small_batched<NB>(  // T is automatically deduced
                     handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
             else
                 return rocblas_trtri_large_batched<NB>(  // T is automatically deduced
                     handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count, C_tmp);
        }

    The reason for this, is that the type template arguments can be
    automatically deduced from the actual function arguments, so that
    you don't have to pass the types in calls to the function, as shown
    in the example above when calling ``rocblas_trtri_small_batched``
    and ``rocblas_trtri_large_batched``. They have a ``typename T``
    parameter too, but it can be automatically deduced, so it doesn't
    need to be explicitly passed.

10. When writing functions like the above which are heavily dependent on
    block sizes, especially if they are in header files included by
    other files, template parameters for block sizes are strongly
    preferred to ``#define`` macros or ``constexpr`` variables. For
    ``.cpp`` files which are not included in other files, a
    ``static constexpr`` variable can be used. **Macros should never be
    used for constants.**

    Note: For constants inside of functions, ``static constexpr`` is
    preferred to just ``constexpr``, so that the variables do not need
    to be initialized at runtime.

    Note: C++14 variable templates can sometimes be used to provide
    constants. For example:

    .. code:: cpp

        template <typename T>
        static constexpr T negative_one = -1;

        template <typename T>
        static constexpr T zero = 0;

        template <typename T>
        static constexpr T one = 1;

11. static duration variables which aren't constants should usually be
    made function-local ``static`` variables, rather than namespace or
    class static variables. This is to avoid the `static initialization
    order
    fiasco <https://isocpp.org/wiki/faq/ctors#static-init-order>`__. For
    example:

    .. code:: cpp

        static auto& get_table()
        {
            // Placed inside function to avoid dependency on initialization order
            static std::unordered_map<std::string, size_t>* table = test_cleanup::allocate(&table);
            return *table;
        }

    This is sometimes called the *singleton* pattern. A ``static``
    variable is made local to a function rather than a namespace or
    class, and it gets initialized the first time the function is
    called. A reference to the ``static`` variable is returned from the
    function, and the function is used everywhere access to the variable
    is needed. In the case of multithreaded programs, the C++11 and
    later standards guarantee that there won't be any race conditions.
    It is also
    `faster <https://www.modernescpp.com/index.php/thread-safe-initialization-of-a-singleton>`__
    to initialize function-local ``static`` variables than it is to
    explicitly call ``std::call_once``. For example:

    .. code:: cpp

        void my_func()
        {
            static int dummy = (func_to_call_once(), 0);
        }

    This is much simpler and faster than explicitly calling
    ``std::call_once``, since the compiler has special ways of
    optimizing ``static`` initialization. The first time ``my_func()``
    is called, it will call ``func_to_call_once()`` once in a
    thread-safe way. After that, there is almost no overhead in later
    calls to ``my_func()``.

12. Functions are preferred to macros. Functions or functors inside of
    ``class`` / ``struct`` templates can be used when partial template
    specializations are needed.

    When C preprocessor macros are needed (such as if they contain a
    ``return`` statement to return from the calling function), if the
    macro's definition contains more than one simple expression, then
    `it should be wrapped in a
    ``do { } while(0)`` <https://stackoverflow.com/questions/154136/why-use-apparently-meaningless-do-while-and-if-else-statements-in-macros>`__,
    without a terminating semicolon. This is to allow them to be used
    inside ``if`` statements. For example:

    .. code:: cpp

        #define RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(h) \
            do                                               \
            {                                                \
                if((h)->is_device_memory_size_query())       \
                    return rocblas_status_size_unchanged;    \
            } while(0)

    The ``do { } while(0)`` allows the macro expansion to be a single
    statement which can be terminated with a semicolon, and which can be
    used anywhere a regular function call can be used.

13. For most template functions which are used in other compilation
    units, it is preferred that they be put in header files, rather than
    ``.cpp`` files, because putting them in ``.cpp`` files requires
    explicit instantiation of them for all possible arguments, and there
    are less opportunities for inlining and interprocedural
    optimization.

    The C++ standard explicitly says that unused templates can be
    omitted from the output, so including unused templates in a header
    file does not increase the size of the program, since only the used
    ones are in the final output.

    For template functions which are only used in one ``.cpp`` file,
    they can be placed in the ``.cpp`` file.

    Templates, like inline functions, are granted an exception to the
    one definition rule (ODR) as long as the sequence of tokens in each
    compilation unit is identical.

14. Functions and namespace-scope variables which are not a part of the
    public interface of rocBLAS, should either be marked static, be
    placed in an unnamed namespace, or be placed in
    ``namespace rocblas``. For example:

    .. code:: cpp

        namespace
        {
            // Private internal implementation
        } // namespace

        extern "C"
        {
            // Public C interfaces
        } // extern "C"

    However, unnamed namespaces should not be used in header files. If
    it is absolutely necessary to mark a function or variable as private
    to a compilation unit but defined in a header file, it should be
    declared ``static``, ``constexpr`` and/or ``inline`` (``constexpr``
    implies ``static`` for non-template variables and ``inline`` for
    functions).

    Even though rocBLAS goes into a shared library which exports a
    limited number of symbols, this is still a good idea, to decrease
    the chances of name collisions *inside* of rocBLAS.

15. ``std::string`` should only be used for strings which can grow, or
    which must be dynamically allocated as read-write strings. For
    simple static strings, strings returned from functions like
    ``getenv()``, or strings which are initialized once and then used
    read-only, ``const char*`` should be used to refer to the string or
    pass it as an argument.

    ``std::string`` involves dynamic memory allocation and copying of
    temporaries, which can be slow. ``std::string_view`` is supposed to
    help alleviate that, but it's not available until C++17, and we're
    using C++14 now. ``const char*`` should be used for read-only views
    of strings, in the interest of efficiency.

16. For code brevity and readability, when converting to *numeric*
    types, function-style casts are preferred to ``static_cast<>()`` or
    C-style casts. For example, ``T(x)`` is preferred to
    ``static_cast<T>(x)`` or ``(T)x``.

    When writing general containers or templates which can accept
    *arbitrary* types as parameters, not just *numeric* types, then the
    specific cast (``static_cast``, ``const_cast``,
    ``reinterpret_cast``) should be used, to avoid surprises.

    But when converting to *numeric* types, which have very
    well-understood behavior and are *side-effect free*, ``type(x)`` is
    more compact and clearer than ``static_cast<type>(x)``. For
    pointers, C-style casts are okay, such as ``(T*)A``.

17. For BLAS2 functions and BLAS1 functions with two vectors, the
    ``incx`` and/or ``incy`` arguments can be negative, which means the
    vector is treated backwards from the end. A simple trick to handle
    this, is to adjust the pointer to the end of the vector if the
    increment is negative, as in:

    .. code:: cpp

        if(incx < 0)
            x -= ptrdiff_t(incx) * (n - 1);
        if(incy < 0)
            y -= ptrdiff_t(incy) * (n - 1);

    After that adjustment, the code does not need to treat negative
    increments any differently than positive ones.

    Note: Some blocked matrix-vector algorithms which call other BLAS
    kernels may not work if this simple transformation is used; see
    `TRSV <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/src/blas2/rocblas_trsv.cpp>`__
    for an example, and how it's handled there.

18. For reduction operations, the file
    `reduction.h <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/src/blas1/reduction.h>`__
    has been created to systematize reductions and perform their device
    kernels in one place. This works for ``amax``, ``amin``, ``asum``,
    ``nrm2``, and (partially) ``dot`` and ``gemv``.
    ``rocblas_reduction_kernel`` is a generalized kernel which takes 3
    *functors* as template arguments:

    -  One to *fetch* values (such as fetching a complex value and
       taking the sum of the squares of its real and imaginary parts
       before reducing it)
    -  One to *reduce* values (such as to compute a sum or maximum)
    -  One to *finalize* the reduction (such as taking the square root
       of a sum of squares)

    There is a ``default_value()`` function which returns the default
    value for a reduction. The default value is the value of the
    reduction when the size is 0, and reducing a value with the
    ``default_value()`` does not change the value of the reduction.

19. When `type punning <https://en.wikipedia.org/wiki/Type_punning>`__
    is needed, ``union`` should be used instead of pointer-casting,
    which violates *strict aliasing*. For example:

    .. code:: cpp

        // zero extend lower 16 bits of bfloat16 to convert to IEEE float
        explicit __host__ __device__ operator float() const
        {
            union
            {
                uint32_t int32;
                float    fp32;
            } u = {uint32_t(data) << 16};
            return u.fp32; // Legal in C, nonstandard extension in C++
        }

    This violates the strict aliasing rule of
    `C <https://en.cppreference.com/w/c/language/object#Strict_aliasing>`__
    and
    `C++ <https://en.cppreference.com/w/cpp/language/reinterpret_cast#Type_aliasing>`__:

    .. code:: cpp

        // zero extend lower 16 bits of bfloat16 to convert to IEEE float
        explicit __host__ __device__ operator float() const
        {
            uint32_t int32 = uint32_t(data) << 16;
            return *(float *) &int32; // Violates strict aliasing rule in both C and C++
        }

    The only 100% standard C++ way to do it, is to use ``memcpy()``, but
    this should not be required as long as GCC or Clang are used:

    .. code:: cpp

        // zero extend lower 16 bits of bfloat16 to convert to IEEE float
        explicit __host__ __device__ operator float() const
        {
            uint32_t int32 = uint32_t(data) << 16;
            float fp32;
            static_assert(sizeof(int32) == sizeof(fp32), "Different sizes");
            memcpy(&fp32, &int32, sizeof(fp32));
            return fp32;
        }

20. ``<type_traits>`` classes which return Boolean values can be
    converted to ``bool`` in Boolean contexts. Hence many traits can be
    tested by simply creating an instance of them with ``{}``
    initialization syntax and using it in a Boolean context:

    .. code:: cpp

        template<typename T, typename = typename std::enable_if<std::is_same<T, float>{} ||
                                                                std::is_same<T, double>{}>::type>
        void function(T x)
        {
        }

    Here, instances of the ``std::is_same<...>`` traits class are
    created with the ``{}`` syntax. The resulting temporary objects can
    be explicitly converted to ``bool``, which is what occurs when an
    object appears in a conditional expression (``if``, ``while``,
    ``for``, ``&&``, ``||``, ``!``, ``? :``, etc.). This is a shorter
    syntax than using ``std::is_same<...>::value``.

