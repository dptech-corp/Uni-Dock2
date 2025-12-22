//
// Created by Congcong Liu on 25-6-10.
//

#ifndef MAIN_H
#define MAIN_H

const char* STR_CONFIG_TEMPLATE = R"(
# For each item with type tag (!!), the value is REQUIRED


Advanced:
  seed: !!int 1234567       # explicit random seed
  exhaustiveness: !!int 512 # MC candidates count (roughly proportional to time).
                            # For `screen` task, it's only considered for `free` mode
  randomize: !!bool true    # whether to randomize input pose before performing the global search
  mc_steps: !!int 40        # Monte Carlo random walk steps
  opt_steps: !!int -1       # optimization steps after each MC walk step; -1 to use heuristic strategy
  refine_steps: !!int 5     # refinement steps after clustering
  rmsd_limit: !!float 1.0   # minimum RMSD between output poses
  num_pose: !!int 10        # number of the finally generated poses to output
  energy_range: !!float 10  # maximum energy difference between output poses and the best pose
  bias: !!str pos           # default is `no`, `pos` for positio bias (Gaussian), `align` to align two molecules
  bias_k: !!float 0.1       # scaling coefficient, default is 0.1

Hardware:
  ncpu: !!int 10            # [Not loaded] the number of CPUs to use (the default is to use all detected CPUs)
  gpu_device_id: !!int 0    # GPU device id (default 0)
  max_gpu_memory: !!int 0   # maximum gpu memory (MB) to use (default=0, use all available GPU memory)


Settings:
  task: !!str screen        # screen | score | mc
                            # screen: The most common mode, perform randomize(if true) + MC(mc_steps) +
                            #         optimization(opt_steps) + cluster(if true) + refinement(refine_steps)
                            # score: Only provide scores for input ligands, no searching or optimization
                            # mc: only perform pure mc, namely opt_steps=0; no refinement, neither

  search_mode: !!str free   # [Only for task "screen"] fast | balance | detail | free,
                            # use recommended settings of exhaustiveness and search steps

  constraint_docking: !!bool false # if True, cancel the translation & orientation DOFs
  center_x: !!float -22.3   # X coordinate of the center (Angstrom)
  center_y: !!float 1
  center_z: !!float 27.3
  size_x: !!float 30        # size in the X dimension (Angstrom)
  size_y: !!float 30.0
  size_z: !!float 30


Outputs:
  dir: !!str ./res2         # output directory, default is `./res`


Inputs:
  json: !!str ./5S8I.json   # Input json file containing receptor & ligands info.

)";


#endif //MAIN_H
