#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "format/json.h"
#include "screening/core.h"

namespace py = pybind11;

namespace {

py::object get_item(const py::dict& mapping, const char* key) {
    return mapping[py::str(key)];
}

py::dict get_dict(const py::dict& mapping, const char* key, const std::string& location) {
    py::object value = get_item(mapping, key);
    if (!py::isinstance<py::dict>(value)) {
        throw py::type_error(location + "." + key + " must be a dict");
    }
    return py::cast<py::dict>(value);
}

std::string field_location(const std::string& location, const char* key) {
    return location + "." + key;
}

std::string get_string(const py::dict& mapping, const char* key, const std::string& location) {
    py::object value = get_item(mapping, key);
    if (!py::isinstance<py::str>(value)) {
        throw py::type_error(field_location(location, key) + " must be a string");
    }
    return py::cast<std::string>(value);
}

int get_integer(const py::dict& mapping, const char* key, const std::string& location) {
    py::object value = get_item(mapping, key);
    if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value)) {
        throw py::type_error(field_location(location, key) + " must be an integer");
    }
    return py::cast<int>(value);
}

bool get_boolean(const py::dict& mapping, const char* key, const std::string& location) {
    py::object value = get_item(mapping, key);
    if (!py::isinstance<py::bool_>(value)) {
        throw py::type_error(field_location(location, key) + " must be a boolean");
    }
    return py::cast<bool>(value);
}

Real get_number(py::handle value, const std::string& location) {
    const bool is_integer = py::isinstance<py::int_>(value) && !py::isinstance<py::bool_>(value);
    if (!is_integer && !py::isinstance<py::float_>(value)) {
        throw py::type_error(location + " must be a number");
    }

    const Real result = py::cast<Real>(value);
    if (!std::isfinite(result)) {
        throw py::value_error(location + " must be finite");
    }
    return result;
}

Real get_number(const py::dict& mapping, const char* key, const std::string& location) {
    return get_number(get_item(mapping, key), field_location(location, key));
}

std::array<Real, 3> get_triplet(
    const py::dict& mapping,
    const char* key,
    const std::string& location
) {
    py::object value = get_item(mapping, key);
    if (!py::isinstance<py::list>(value)) {
        throw py::type_error(location + "." + key + " must be a JSON array");
    }

    py::list values = py::cast<py::list>(value);
    if (py::len(values) != 3) {
        throw py::value_error(location + "." + key + " must contain exactly 3 values");
    }

    std::array<Real, 3> result{};
    for (py::ssize_t index = 0; index < 3; ++index) {
        result[static_cast<std::size_t>(index)] = get_number(
            values[index],
            field_location(location, key) + "[" + std::to_string(index) + "]"
        );
    }
    return result;
}

struct ParsedEngineRequest {
    CoreInput input;
    bool use_tor_lib = false;
    py::dict molecules;
};

ParsedEngineRequest parse_engine_request(const py::dict& request) {
    const py::dict parameters = get_dict(request, "parameters", "request");
    const py::dict runtime = get_dict(request, "runtime", "request");
    py::dict molecules = get_dict(request, "molecules", "request");

    for (const auto& item : molecules) {
        if (!py::isinstance<py::str>(item.first)) {
            throw py::type_error("request.molecules keys must be strings");
        }
    }
    if (!molecules.contains(py::str("receptor"))) {
        throw py::key_error("request.molecules is missing required key: receptor");
    }

    const auto center = get_triplet(parameters, "center", "request.parameters");
    const auto box_size = get_triplet(parameters, "box_size", "request.parameters");

    ParsedEngineRequest parsed;
    parsed.input.box.x_lo = center[0] - box_size[0] / 2;
    parsed.input.box.x_hi = center[0] + box_size[0] / 2;
    parsed.input.box.y_lo = center[1] - box_size[1] / 2;
    parsed.input.box.y_hi = center[1] + box_size[1] / 2;
    parsed.input.box.z_lo = center[2] - box_size[2] / 2;
    parsed.input.box.z_hi = center[2] + box_size[2] / 2;

    parsed.input.task = get_string(parameters, "task", "request.parameters");
    parsed.input.search_mode = get_string(parameters, "search_mode", "request.parameters");
    parsed.input.exhaustiveness = get_integer(parameters, "exhaustiveness", "request.parameters");
    parsed.input.randomize = get_boolean(parameters, "randomize", "request.parameters");
    parsed.input.mc_steps = get_integer(parameters, "mc_steps", "request.parameters");
    parsed.input.opt_steps = get_integer(parameters, "opt_steps", "request.parameters");
    parsed.input.refine_steps = get_integer(parameters, "refine_steps", "request.parameters");
    parsed.input.num_pose = get_integer(parameters, "num_pose", "request.parameters");
    parsed.input.rmsd_limit = get_number(parameters, "rmsd_limit", "request.parameters");
    parsed.input.energy_range = get_number(parameters, "energy_range", "request.parameters");
    parsed.input.seed = get_integer(parameters, "seed", "request.parameters");
    parsed.input.bias = get_string(parameters, "bias", "request.parameters");
    parsed.input.bias_k = get_number(parameters, "bias_k", "request.parameters");
    parsed.use_tor_lib = get_boolean(parameters, "use_tor_lib", "request.parameters");
    parsed.input.energy_decomp = get_boolean(parameters, "energy_decomp", "request.parameters");
    parsed.input.constraint_docking = get_boolean(parameters, "constraint_docking", "request.parameters");

    parsed.input.output_dir = get_string(runtime, "output_dir", "request.runtime");
    parsed.input.name_json = get_string(runtime, "output_prefix", "request.runtime");
    parsed.input.gpu_device_id = get_integer(runtime, "gpu_device_id", "request.runtime");
    parsed.input.max_gpu_memory = get_integer(runtime, "max_gpu_memory", "request.runtime");
    parsed.molecules = std::move(molecules);
    return parsed;
}

void run_engine_request(const py::dict& request) {
    ParsedEngineRequest parsed = parse_engine_request(request);

    py::module_ json = py::module_::import("json");
    const std::string json_string = py::cast<std::string>(
        json.attr("dumps")(parsed.molecules, py::arg("allow_nan") = false)
    );

    py::gil_scoped_release release;
    parsed.input.fix_mol = UDFixMol();
    parsed.input.flex_mol_list.clear();
    parsed.input.fns_flex.clear();
    read_ud_from_json_string(
        json_string,
        parsed.input.box,
        parsed.input.fix_mol,
        parsed.input.flex_mol_list,
        parsed.input.fns_flex,
        parsed.use_tor_lib
    );

    if (core_pipeline(parsed.input) != 0) {
        throw std::runtime_error("Core pipeline failed");
    }
}

}  // namespace

PYBIND11_MODULE(pipeline, module) {
    module.doc() = "Private Python binding for the Uni-Dock2 molecular docking engine";
    module.def(
        "run",
        &run_engine_request,
        py::arg("request"),
        R"pbdoc(
Run one docking request.

The request must be a JSON-compatible dictionary produced by
``unidock2._engine.build_engine_request``. This native module is private; use
``UnidockProtocolRunner`` for the supported public workflow.
        )pbdoc"
    );
}
