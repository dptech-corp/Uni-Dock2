import os


tor_pkl_path = os.path.join(
    os.path.dirname(__file__), "torsion_library", "data", "torsion_library.pkl"
)

if not os.path.exists(tor_pkl_path):
    from unidock2.torsion_library.torsion_library_builder import (
        TorsionLibraryBuilder,
    )

    tor_lib_data_dir = os.path.dirname(tor_pkl_path)
    torsion_library_builder = TorsionLibraryBuilder(tor_lib_data_dir)
    torsion_library_builder.build_torsion_library()

    assert os.path.exists(tor_pkl_path), "Torsion library not installed correctly"
