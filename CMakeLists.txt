
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

SET(src init.cpp)
SET(luasrc init.lua decode.lua infer.lua energies.lua examples.lua)

ADD_TORCH_PACKAGE(gm "${src}" "${luasrc}" "Graphical Models")
TARGET_LINK_LIBRARIES(gm luaT TH)
