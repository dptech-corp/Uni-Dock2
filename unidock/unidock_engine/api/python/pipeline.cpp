#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <spdlog/spdlog.h>
#include "model/model.h"
#include "format/json.h"
#include "screening/core.h"

namespace py = pybind11;

class DockingPipeline {
public:
    DockingPipeline(
        std::string output_dir,
        Real center_x, Real center_y, Real center_z,
        Real size_x, Real size_y, Real size_z,
        std::string task = "screen",
        std::string search_mode = "balance",
        int exhaustiveness = -1,
        bool randomize = true,
        int mc_steps = -1,
        int opt_steps = -1,
        int refine_steps = 5,
        int num_pose = 10,
        Real rmsd_limit = 1.0,
        Real energy_range = 5.0,
        int seed = 1234567,
        bool constraint_docking = false,
        bool use_tor_lib = false,
        int gpu_device_id = 0,
        std::string name_json = "from_python_obj"
    ) : _use_tor_lib(use_tor_lib){

        ipt.gpu_device_id = gpu_device_id;
        ipt.output_dir = output_dir;
        ipt.name_json = name_json;

        ipt.box.x_lo = center_x - size_x / 2;
        ipt.box.x_hi = center_x + size_x / 2;
        ipt.box.y_lo = center_y - size_y / 2;
        ipt.box.y_hi = center_y + size_y / 2;
        ipt.box.z_lo = center_z - size_z / 2;
        ipt.box.z_hi = center_z + size_z / 2;

        ipt.exhaustiveness = exhaustiveness;
        ipt.mc_steps = mc_steps;
        ipt.opt_steps = opt_steps;
        ipt.randomize = randomize;
        ipt.refine_steps = refine_steps;
        ipt.num_pose = num_pose;
        ipt.rmsd_limit = rmsd_limit;
        ipt.energy_range = energy_range;
        ipt.seed = seed;

        ipt.constraint_docking = constraint_docking;
        ipt.task = task;
        ipt.search_mode = search_mode;
    }

    void set_receptor(py::list receptor_info) {
        json_data["receptor"] = receptor_info;
        spdlog::info("Receptor data loaded into JSON cache");
    }

    void add_ligands(py::dict ligands_info) {
        for (const auto& item : ligands_info) {
            json_data[item.first] = item.second;
        }
        spdlog::info("Ligands data loaded into JSON cache");
    }

    void run() {
        // generate json string
        std::string json_str;
        {
            py::gil_scoped_acquire acquire;
            py::module_ json = py::module_::import("json");
            json_str = py::str(json.attr("dumps")(json_data));
        }
        py::gil_scoped_release release;

        // run screening
        ipt.fix_mol = UDFixMol();
        ipt.flex_mol_list.clear();
        ipt.fns_flex.clear();
        read_ud_from_json_string(json_str, ipt.box, ipt.fix_mol, ipt.flex_mol_list, ipt.fns_flex, _use_tor_lib);

        if (core_pipeline(ipt) != 0) {
            throw std::runtime_error("Core pipeline failed.");
        }

    }

private:
    DockParam _dock_param;
    UDFixMol _fix_mol;
    UDFlexMolList _flex_mol_list;
    std::vector<std::string> _fns_flex;

    bool _use_tor_lib = false;
    CoreInput ipt;
    py::dict json_data;
    bool use_tor_lib;
};


PYBIND11_MODULE(pipeline, m) {
    m.doc() = "Python bindings for the Uni-Dock2 molecular docking engine pipeline";

    py::class_<DockingPipeline>(m, "DockingPipeline")
        .def(py::init<
                std::string, Real, Real, Real, Real, Real, Real, std::string, std::string, 
                int, bool, int, int, int, int, Real, Real, int, bool, bool, int>(),
            py::arg("output_dir"),
            py::arg("center_x"), py::arg("center_y"), py::arg("center_z"),
            py::arg("size_x"), py::arg("size_y"), py::arg("size_z"),
            py::arg("task") = "screen",
            py::arg("search_mode") = "balance",
            py::arg("exhaustiveness") = -1,
            py::arg("randomize") = true,
            py::arg("mc_steps") = -1,
            py::arg("opt_steps") = -1,
            py::arg("refine_steps") = 5,
            py::arg("num_pose") = 10,
            py::arg("rmsd_limit") = 1.0,
            py::arg("energy_range") = 5.0,
            py::arg("seed") = 1234567,
            py::arg("constraint_docking") = false,
            py::arg("use_tor_lib") = false,
            py::arg("gpu_device_id") = 0
        )
        .def("set_receptor", &DockingPipeline::set_receptor, "Set the receptor molecule from a Python dictionary")
        .def("add_ligands", &DockingPipeline::add_ligands, "Add ligand molecules from a list of Python dictionaries")
        .def("run", &DockingPipeline::run, "Run the docking simulation");
}