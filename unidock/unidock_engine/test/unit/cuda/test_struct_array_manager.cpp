//
// Created for testing StructArrayManager template
//

#include <catch2/catch_amalgamated.hpp>
#include <vector>
#include "cuda/struct_array_manager.cuh"

//==========================================================
// Test structures and function declarations
//==========================================================

// Test structure
struct MyBox {
    int*    ids;      // Array of int per object
    float*  values;   // Array of float per object
    float   scale;    // Regular member
};

// Wrapper function declarations (implemented in wrapper.cu)
void scale_values_gpu(MyBox* objs_device, int n);
void modify_ids_gpu(MyBox* objs_device, int n);

//==========================================================
// Test cases
//==========================================================

TEST_CASE("StructArrayManager basic allocation and free", "[struct_array_manager]") {
    int nn = 4;
    std::vector<int> list_ids_size = {2, 3, 4, 5};
    std::vector<int> list_values_size = {3, 4, 5, 6};

    StructArrayManager<MyBox> list_mybox(nn);
    
    REQUIRE_NOTHROW(list_mybox.add_ptr_field<int*>({
        &MyBox::ids,
        sizeof(int),
        list_ids_size
    }));
    
    REQUIRE_NOTHROW(list_mybox.add_ptr_field<float*>({
        &MyBox::values,
        sizeof(float),
        list_values_size
    }));
    
    REQUIRE_NOTHROW(list_mybox.allocate_and_assign());
    
    // Verify memory is allocated
    REQUIRE(list_mybox.array_host != nullptr);
    REQUIRE(list_mybox.array_device != nullptr);
    REQUIRE(list_mybox.array_size == nn);
    
    // Verify pointer fields are correctly allocated
    int* h_ids = list_mybox.get_host_data<int*>(0);
    float* h_vals = list_mybox.get_host_data<float*>(1);
    REQUIRE(h_ids != nullptr);
    REQUIRE(h_vals != nullptr);
    
    // Verify pointers for each object are correctly set
    for (int i = 0; i < nn; i++) {
        REQUIRE(list_mybox.array_host[i].ids != nullptr);
        REQUIRE(list_mybox.array_host[i].values != nullptr);
    }
    
    // Cleanup
    REQUIRE_NOTHROW(list_mybox.free_all());
}

TEST_CASE("StructArrayManager add_ptr_field validation", "[struct_array_manager]") {
    int nn = 4;
    StructArrayManager<MyBox> list_mybox(nn);
    
    // Test that wrong size should throw exception
    std::vector<int> wrong_size = {2, 3, 4}; // Size is 3, should be 4
    REQUIRE_THROWS_AS(list_mybox.add_ptr_field<int*>({
        &MyBox::ids,
        sizeof(int),
        wrong_size
    }), std::runtime_error);
}

TEST_CASE("StructArrayManager data copy and GPU computation", "[struct_array_manager]") {
    int nn = 4;
    std::vector<int> list_ids_size = {2, 3, 4, 5};
    std::vector<int> list_values_size = {3, 4, 5, 6};

    StructArrayManager<MyBox> list_mybox(nn);
    list_mybox.add_ptr_field<int*>({
        &MyBox::ids,
        sizeof(int),
        list_ids_size
    });
    list_mybox.add_ptr_field<float*>({
        &MyBox::values,
        sizeof(float),
        list_values_size
    });
    list_mybox.allocate_and_assign();

    // Get CPU data pointers for writing
    int* h_ids = list_mybox.get_host_data<int*>(0);
    float* h_vals = list_mybox.get_host_data<float*>(1);

    // Fill host data
    int off_i = 0;
    int off_f = 0;
    std::vector<float> original_values;
    
    for (int i = 0; i < nn; i++) {
        MyBox& box = list_mybox.array_host[i];
        box.scale = 2.0f + i;
        
        for (int j = 0; j < list_ids_size[i]; j++) {
            h_ids[off_i + j] = i * 10 + j;
        }
        
        for (int j = 0; j < list_values_size[i]; j++) {
            float val = (float)(i + j);
            h_vals[off_f + j] = val;
            original_values.push_back(val);
        }
        
        off_i += list_ids_size[i];
        off_f += list_values_size[i];
    }

    // Copy to GPU
    REQUIRE_NOTHROW(list_mybox.copy_to_gpu());

    // Call wrapper function for GPU computation
    scale_values_gpu(list_mybox.array_device, nn);

    // Copy back to CPU
    REQUIRE_NOTHROW(list_mybox.copy_to_host());

    // Verify GPU computation results
    off_f = 0;
    for (int i = 0; i < nn; i++) {
        float expected_scale = 2.0f + i;
        for (int j = 0; j < list_values_size[i]; j++) {
            float expected = original_values[off_f + j] * expected_scale;
            REQUIRE_THAT(h_vals[off_f + j], Catch::Matchers::WithinAbs(expected, 1e-5f));
        }
        off_f += list_values_size[i];
    }

    // Cleanup
    list_mybox.free_all();
}

TEST_CASE("StructArrayManager multiple pointer fields modification", "[struct_array_manager]") {
    int nn = 3;
    std::vector<int> list_ids_size = {2, 3, 4};
    std::vector<int> list_values_size = {3, 4, 5};

    StructArrayManager<MyBox> list_mybox(nn);
    list_mybox.add_ptr_field<int*>({
        &MyBox::ids,
        sizeof(int),
        list_ids_size
    });
    list_mybox.add_ptr_field<float*>({
        &MyBox::values,
        sizeof(float),
        list_values_size
    });
    list_mybox.allocate_and_assign();

    // Get CPU data pointers
    int* h_ids = list_mybox.get_host_data<int*>(0);
    float* h_vals = list_mybox.get_host_data<float*>(1);

    // Fill initial data
    int off_i = 0;
    int off_f = 0;
    std::vector<int> original_ids;
    
    for (int i = 0; i < nn; i++) {
        MyBox& box = list_mybox.array_host[i];
        box.scale = 1.5f + i;
        
        for (int j = 0; j < list_ids_size[i]; j++) {
            int val = i * 10 + j;
            h_ids[off_i + j] = val;
            original_ids.push_back(val);
        }
        
        for (int j = 0; j < list_values_size[i]; j++) {
            h_vals[off_f + j] = (float)(i + j);
        }
        
        off_i += list_ids_size[i];
        off_f += list_values_size[i];
    }

    // Copy to GPU
    list_mybox.copy_to_gpu();

    // Modify ids
    modify_ids_gpu(list_mybox.array_device, nn);

    // Modify values
    scale_values_gpu(list_mybox.array_device, nn);

    // Copy back to CPU
    list_mybox.copy_to_host();

    // Verify ids modification results
    off_i = 0;
    for (int i = 0; i < nn; i++) {
        for (int j = 0; j < list_ids_size[i]; j++) {
            int expected = original_ids[off_i + j] + 100;
            REQUIRE(h_ids[off_i + j] == expected);
        }
        off_i += list_ids_size[i];
    }

    // Cleanup
    list_mybox.free_all();
}

// Note: Move semantics are disabled, so move constructor and move assignment are not tested

