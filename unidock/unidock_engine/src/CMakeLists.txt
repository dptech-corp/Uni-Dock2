set(sources
        cuda/bfgs.cuh
        cuda/bfgs_wrapper.cu
        cuda/cluster.cu
        cuda/common.cuh cuda/common.cu
        cuda/mc.cu
        cuda/model.cu
        cuda/mymath_warp.cuh
        cuda/optimize.cu
        cuda/task.cu
        cuda/vina.cu
        cuda/struct_array_manager_wrapper.cu cuda/struct_array_manager.cuh

        cluster/cluster.h

        constants/constants.h

        format/json.cpp         format/json.h
        format/rapidjson_parser.cpp format/rapidjson_parser.h

        geometry/quaternion.h
        geometry/rotation.h

        model/model.h

        myutils/common.h
        myutils/errors.h        myutils/errors.cpp
        myutils/matrix.h
        myutils/myio.h          myutils/myio.cpp
        myutils/mymath.h

        optimize/bfgs.h
        optimize/optimize.h

        screening/screening.cpp screening/screening.h
        screening/core.cpp screening/core.h

        search/mc.h
        search/bvh.cpp          search/bvh.h

        score/gaff2.cpp         score/gaff2.h
        score/score.h
        score/vina.h

        task/DockTask.cpp       task/DockTask.h
)

# compile lib for the bin to link to
add_library(${LIB_NAME} OBJECT ${sources})

# include src directory
target_include_directories(${LIB_NAME} SYSTEM PUBLIC # no warning from third-party
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_SOURCE_DIR}/include
        .
)
target_compile_definitions(${LIB_NAME} PUBLIC VERSION_NUMBER="${VERSION_NUMBER}")
target_link_libraries(${LIB_NAME} PUBLIC CUDA::cudart_static)
set_target_properties(${LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# install
install(TARGETS ${LIB_NAME}
        LIBRARY DESTINATION lib
)