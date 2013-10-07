
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(OpenMP)

IF (OPENMP_FOUND)
  MESSAGE (STATUS "OpenMP Found with compiler flag : ${OpenMP_C_FLAGS}")
  SET (CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
ENDIF (OPENMP_FOUND)

SET(src init.cpp)
SET(luasrc init.lua decode.lua sample.lua infer.lua energies.lua examples.lua adjacency.lua X.t7)

ADD_TORCH_PACKAGE(gm "${src}" "${luasrc}" "Graphical Models")
TARGET_LINK_LIBRARIES(gm luaT TH)
