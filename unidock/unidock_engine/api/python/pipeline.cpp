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
        std::string task,
        std::string search_mode,
        int exhaustiveness,
        bool randomize,
        int mc_steps,
        int opt_steps,
        int refine_steps,
        int num_pose,
        Real rmsd_limit,
        Real energy_range,
        int seed,
        bool constraint_docking,
        bool use_tor_lib,
        int gpu_device_id,
        std::string name_json,
        int max_gpu_mem
    ) : use_tor_lib(use_tor_lib) {

        ipt.gpu_device_id = gpu_device_id;
        ipt.max_gpu_memory = max_gpu_mem;
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
        read_ud_from_json_string(json_str, ipt.box, ipt.fix_mol, ipt.flex_mol_list, ipt.fns_flex, use_tor_lib);

        if (core_pipeline(ipt) != 0) {
            throw std::runtime_error("Core pipeline failed.");
        }
    }

private:
    bool use_tor_lib = false;
    CoreInput ipt;
    py::dict json_data;
};


PYBIND11_MODULE(pipeline, m) { // shared lib name: "pipeline.<py_version>-<platform>-<arch>.so"
    m.doc() = "Python bindings for the Uni-Dock2 molecular docking engine pipeline";

    py::class_<DockingPipeline>(m, "DockingPipeline",
        R"pbdoc(Uni-Dock2 molecular docking pipeline.)pbdoc")
        .def(py::init<
                std::string, Real, Real, Real, Real, Real, Real, std::string, std::string, 
                int, bool, int, int, int, int, Real, Real, int, bool, bool, int,
                std::string, int>(),
            py::kw_only(),  // force keyword-only 
            py::arg("output_dir") = CoreInputDefaults::output_dir,
            py::arg("center_x"), py::arg("center_y"), py::arg("center_z"),
            py::arg("size_x"), py::arg("size_y"), py::arg("size_z"),
            py::arg("task") = CoreInputDefaults::task,
            py::arg("search_mode") = CoreInputDefaults::search_mode,
            py::arg("exhaustiveness") = CoreInputDefaults::exhaustiveness,
            py::arg("randomize") = CoreInputDefaults::randomize,
            py::arg("mc_steps") = CoreInputDefaults::mc_steps,
            py::arg("opt_steps") = CoreInputDefaults::opt_steps,
            py::arg("refine_steps") = CoreInputDefaults::refine_steps,
            py::arg("num_pose") = CoreInputDefaults::num_pose,
            py::arg("rmsd_limit") = CoreInputDefaults::rmsd_limit,
            py::arg("energy_range") = CoreInputDefaults::energy_range,
            py::arg("seed") = CoreInputDefaults::seed,
            py::arg("constraint_docking") = CoreInputDefaults::constraint_docking,
            py::arg("use_tor_lib") = CoreInputDefaults::use_tor_lib,
            py::arg("gpu_device_id") = CoreInputDefaults::gpu_device_id,
            py::arg("name_json") = CoreInputDefaults::name_json,
            py::arg("max_gpu_mem") = CoreInputDefaults::max_gpu_memory
        )

        .def("set_receptor", &DockingPipeline::set_receptor,
            R"pbdoc(
Set the receptor molecule.

Args:
    receptor_info (list): A list containing receptor atom information.
        Each element represents an atom with its properties (type, coordinates, etc.).
            )pbdoc")
            
        .def("add_ligands", &DockingPipeline::add_ligands,
            R"pbdoc(
Add ligand molecules to the docking pipeline.

Args:
    ligands_info (dict): A dictionary containing ligand information.
        Keys are ligand identifiers, values contain atom and bond data.
            )pbdoc")

        .def("run", &DockingPipeline::run,
            R"pbdoc(
Run the docking simulation.

Executes the molecular docking calculation on GPU. Results will be written as JSON files
to the output directory specified during initialization, with the name specified by name_json.
            )pbdoc");
}