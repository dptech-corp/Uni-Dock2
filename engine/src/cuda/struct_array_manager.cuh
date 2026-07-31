//
// Created by Congcong Liu on 2025/11/3.
//

#ifndef UD2_STRUCT_ARRAY_MANAGER_CUH
#define UD2_STRUCT_ARRAY_MANAGER_CUH

#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>

#include <spdlog/fmt/fmt.h>

#include "myutils/errors.h" 


//==========================================================
// ======== Template: Management of struct GPU Memory ======
//==========================================================

// Member Pointer Related Info
template<typename StructType, typename PtrType>
struct StructsMemberPtrField {
    PtrType StructType::* member_ptr;
    size_t type_size; //size_t
    std::vector<int> list_n_member;    // member lengths of all objects
};

/**
 * @brief Template class for managing GPU memory of struct array
 *
 * @tparam T struct type, which contains the pointer fields to be managed
 * This class is responsible for:
 * - For target GPU struct array, maintain a CPU and GPU memory
 * - Manage the memory allocation and release of the pointer fields in the struct
 * - Provide the data copy function between CPU and GPU
 *
 * Usage:
 * 1. Create a StructArrayManager object, specify the array size
 * 2. Call add_ptr_field() to register each pointer field in the struct
 * 3. Call allocate_and_assign() to allocate all memory
 * 4. Get the pointer to the data by get_host_data()
 * 5. Call copy_to_gpu() to copy the data to GPU
 * 6. Execute the computation on GPU
 * 7. Call copy_to_host() to copy the result back to CPU
 * 8. Call free_all() to release all memory
 *
 * @note Disable copy and move semantics
 */
template<typename T>
class StructArrayManager {
public:
    T* array_host = nullptr;   // pinned host memory
    T* array_device = nullptr; // device memory
    int array_size;

    // constructor
    StructArrayManager(int array_size) : array_size(array_size) {}

    // register a pointer type member_ptr field
    template<typename PtrType>
    void add_ptr_field(StructsMemberPtrField<T, PtrType> f) {
        if (f.list_n_member.size() != array_size) {
            throw std::runtime_error(
                fmt::format("list_n_member size ({}) must equal array_size ({})",
                           f.list_n_member.size(), array_size));
        }
        // new PtrField
        ptr_fields.push_back(new PtrField<PtrType>(f, array_size));
    }

    void allocate_and_assign() {
        //todo: memory pool? maybe alloc AoS + all_members once, then copy the data one by one
        // pros: we can free all memory once (a big block)

        // allocate memory on host and device for the whole array
        checkCUDA(cudaMallocHost(&array_host, array_size * sizeof(T)));
        checkCUDA(cudaMalloc(&array_device, array_size * sizeof(T)));

        // allocate memory on host and device for all registered members of all ptr_fields
        for (auto& field : ptr_fields){
            field->allocate();
        }
        // bind host-side struct member pointers to host_data
        for (auto& field : ptr_fields){
            field->assign_host_ptrs(array_host);
        }
    }

    void copy_to_gpu() {
        // 1) copy flattened payloads to device
        for (auto& field : ptr_fields){
            field->copy_to_gpu();
        }
        // 2) bake device pointers into a temporary host-side view, then memcpy to GPU
        std::vector<T> tmp(array_size);
        std::memcpy(tmp.data(), array_host, array_size * sizeof(T));
        for (auto& field : ptr_fields){
            field->assign_device_ptrs_into(tmp.data());
        }
        checkCUDA(cudaMemcpy(array_device, tmp.data(), array_size * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host() {
        // 1) copy device-side struct array (with device pointers) back to host array
        checkCUDA(cudaMemcpy(array_host, array_device, array_size * sizeof(T), cudaMemcpyDeviceToHost));
        // 2) copy flattened payloads back to host
        for (auto& field : ptr_fields){
            field->copy_to_host();
        }
        // 3) restore host-side struct member pointers to host_data
        for (auto& field : ptr_fields){
            field->assign_host_ptrs(array_host);
        }
    }

    void free_all() {
        // delete all device source
        for (auto& field : ptr_fields) {
            if (field){
                field->free_mem();
            }
        }
        if (array_device) {
            checkCUDA(cudaFree(array_device));
            array_device = nullptr;
        }
        if (array_host) {
            checkCUDA(cudaFreeHost(array_host));
            array_host = nullptr;
        }

        // delete all host source
        for (auto& field : ptr_fields) {
            delete field;
            field = nullptr;
        }
        ptr_fields.clear();
    }

    // get the host_data pointer of the i-th registered field
    template<typename PtrType>
    PtrType get_host_data(int field_idx) {
        UD2_REQUIRE(field_idx >= 0 && field_idx < ptr_fields.size(),
                    "field_idx out of range: %d", field_idx);
        auto p = dynamic_cast<PtrField<PtrType>*>(ptr_fields[field_idx]);
        UD2_REQUIRE(p != nullptr, "bad PtrType");
        return p->host_data;
    }

    // return the per-object lengths list for a given pointer field
    template<typename PtrType>
    const std::vector<int>& get_field_lengths(int field_idx) {
        UD2_REQUIRE(field_idx >= 0 && field_idx < ptr_fields.size(),
                    "field_idx out of range: %d", field_idx);

        auto p = dynamic_cast<PtrField<PtrType>*>(ptr_fields[field_idx]);
        UD2_REQUIRE(p != nullptr, "bad field_idx or PtrType");

        return p->field.list_n_member;
    }

    // return the host segment pointer for (field_idx, obj_idx)
    template<typename PtrType>
    PtrType get_segment_ptr(int field_idx, int obj_idx) {
        UD2_REQUIRE(field_idx >= 0 && field_idx < ptr_fields.size(),
                    "field_idx out of range: %d", field_idx);
        auto p = dynamic_cast<PtrField<PtrType>*>(ptr_fields[field_idx]);
        UD2_REQUIRE(p != nullptr, "bad PtrType");
        UD2_REQUIRE(obj_idx >= 0 && obj_idx < p->field.list_n_member.size(),
                    "obj_idx out of range: %d", obj_idx);
        return p->host_data + p->offsets[obj_idx];
    }

    // return the segment length for (field_idx, obj_idx)
    template<typename PtrType>
    int get_segment_len(int field_idx, int obj_idx) {
        UD2_REQUIRE(field_idx >= 0 && field_idx < ptr_fields.size(),
                    "field_idx out of range: %d", field_idx);
        auto p = dynamic_cast<PtrField<PtrType>*>(ptr_fields[field_idx]);
        UD2_REQUIRE(p != nullptr, "bad PtrType");
        UD2_REQUIRE(obj_idx >= 0 && obj_idx < p->field.list_n_member.size(),
                    "obj_idx out of range: %d", obj_idx);
        return p->field.list_n_member[obj_idx];
    }

private:
    struct PtrFieldBase {
        virtual void allocate() = 0;
        virtual void assign_host_ptrs(T* host_structs) = 0;
        virtual void  assign_device_ptrs_into(T* host_structs_view) = 0;
        virtual void copy_to_gpu() = 0;
        virtual void copy_to_host() = 0;
        virtual void free_mem() = 0;
        virtual ~PtrFieldBase() = default;
    };

    template<typename PtrType>
    struct PtrField : PtrFieldBase {
        StructsMemberPtrField<T, PtrType> field;
        PtrType device_data = nullptr;
        PtrType host_data = nullptr;
        size_t sum_n_member = 0;
        // per-object starting offsets into the flattened buffer
        std::vector<size_t> offsets;
        PtrField(StructsMemberPtrField<T, PtrType> f, int nobj) : field(f) {}

        void allocate() override {
            // allocate memory on GPU and host for the pointer field.
            sum_n_member = 0;
            offsets.resize(field.list_n_member.size());
            size_t off = 0;
            for (int i = 0; i < field.list_n_member.size(); ++i){
                offsets[i] = off;
                off += field.list_n_member[i];
            }
            sum_n_member = off;
            checkCUDA(cudaMalloc(&device_data, sum_n_member * field.type_size));
            checkCUDA(cudaMallocHost(&host_data, sum_n_member * field.type_size));
        }

        void assign_host_ptrs(T* host_structs) override {
            int offset = 0;
            for (int i = 0; i < field.list_n_member.size(); ++i) {
                host_structs[i].*field.member_ptr = host_data + offset;
                offset += field.list_n_member[i];
            }
        }

        void assign_device_ptrs_into(T* host_structs_view) override {
            int offset = 0;
            for (int i = 0; i < field.list_n_member.size(); ++i) {
                host_structs_view[i].*field.member_ptr = device_data + offset;
                offset += field.list_n_member[i];
            }
        }

        void copy_to_gpu() override {
            checkCUDA(cudaMemcpy(device_data, host_data, sum_n_member * field.type_size, cudaMemcpyHostToDevice));
        }

        void copy_to_host() override {
            checkCUDA(cudaMemcpy(host_data, device_data, sum_n_member * field.type_size, cudaMemcpyDeviceToHost));
        }

        void free_mem() override {
            if (device_data) {
                checkCUDA(cudaFree(device_data));
                device_data = nullptr;
            }
            if (host_data) {
                checkCUDA(cudaFreeHost(host_data));
                host_data = nullptr;
            }
        }
    };

    std::vector<PtrFieldBase*> ptr_fields;

public:
    // disable copy and move semantics
    StructArrayManager(const StructArrayManager&) = delete;
    StructArrayManager& operator=(const StructArrayManager&) = delete;
    StructArrayManager(StructArrayManager&&) = delete;
    StructArrayManager& operator=(StructArrayManager&&) = delete;
    
    ~StructArrayManager() {
        // Idempotent release
        free_all();
    }
};


#endif