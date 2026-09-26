#!/usr/bin/env python3
"""
Entry point for the DP adsorption/TS/workflow pipeline.

Delegates to the WorkflowContext + handler modules under utils/.
"""
import argparse

from utils import config as cfgmod
from handlers import DPWorkflow


if __name__ == "__main__":
    parser = cfgmod.build_arg_parser()
    args = parser.parse_args()
    cfg = cfgmod.config_from_args(args)
    slab_entries = cfgmod.resolve_slab_paths_for_workflow(cfg)

    for entry in slab_entries:
        group_index = entry["group_index"]
        slab_path_for_run = entry["slab_path"]
        output_suffix = f"vg{group_index}" if group_index is not None else ""

        if output_suffix:
            print(f"\n===== Start workflow for vacancy group {group_index} =====")
        else:
            print("\n===== Start workflow =====")

        workflow = DPWorkflow(
            path=cfg.path,
            prepared_data_file=cfg.prepared,
            slab_path=slab_path_for_run,
            top_x=cfg.top_x,
            enable_rotation_enum_ads=cfg.enum_ads,
            enable_rotation_enum_ts=cfg.enum_ts,
            enable_ts_seed=cfg.ts_seed,
            surface_normal=cfg.surface_normal,
            normal_axis=cfg.normal_axis,
            enable_imag_mode_check=cfg.imag_mode_check,
            imag_mode_displacement=cfg.imag_mode_displacement,
            imag_mode_relax_steps=cfg.imag_mode_relax_steps,
            bottom_freeze_threshold=cfg.bottom_freeze_threshold,
            run_irc_final_state=cfg.run_irc_final_state,
            irc_fmax=cfg.irc_fmax,
            irc_steps=cfg.irc_steps,
            irc_dx=cfg.irc_dx,
            irc_eta=cfg.irc_eta,
            irc_ninner_iter=cfg.irc_ninner_iter,
            enable_irc_thermo_corrections=cfg.irc_thermo_corrections,
            irc_temperature=cfg.irc_temperature,
            enable_thermo_corrections=cfg.thermo_corrections,
            gas_species_whitelist=cfg.gas_species_whitelist,
            gas_pressure_pa=cfg.gas_pressure_pa,
            output_suffix=output_suffix,
        )
        workflow.generate_initial_adsorbate_guesses()
        workflow.generate_rxn_ts_guesses_ccqn()
        if cfg.run_irc_final_state:
            workflow.get_final_state_IRC()
        workflow.get_final_state_energy()
        workflow.get_ads_energy()

        if output_suffix:
            print(f"===== Finished workflow for vacancy group {group_index} =====\n")
        else:
            print("===== Finished workflow =====\n")
