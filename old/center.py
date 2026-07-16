import argparse
from utils import config as cfgmod
from workflow.context import WorkflowContext
from workflow.handlers.adsorption import AdsorptionHandler
from workflow.handlers.ts import TSHandler
from workflow.handlers.irc import IRCHandler
from workflow.handlers.energy import EnergyHandler

class DPWorkflow:
    def __init__(self, **kwargs):
        self.ctx = WorkflowContext(**kwargs)
        self.ads_handler = AdsorptionHandler(self.ctx)
        self.ts_handler = TSHandler(self.ctx)
        self.irc_handler = IRCHandler(self.ctx) if self.ctx.run_irc_final_state else None
        self.energy_handler = EnergyHandler(self.ctx)

    def run(self):
        self.ads_handler.run()
        self.ts_handler.run()
        if self.irc_handler:
            self.irc_handler.run()
        self.energy_handler.run_final_state_energy()
        self.energy_handler.run_adsorption_energy()


if __name__ == "__main__":
    parser = cfgmod.build_arg_parser()
    args = parser.parse_args()
    cfg = cfgmod.config_from_args(args)
    slab_entries = cfgmod.resolve_slab_paths_for_workflow(cfg)

    for entry in slab_entries:
        group_index = entry["group_index"]
        slab_path_for_run = entry["slab_path"]
        output_suffix = f"vg{group_index}" if group_index is not None else ""

        workflow = DPWorkflow(
            path=cfg.path,
            prepared_data_file=cfg.prepared,
            slab_path=slab_path_for_run,
            top_x=cfg.top_x,
            enable_rotation_enum_ads=cfg.enum_ads,
            enable_rotation_enum_ts=cfg.enum_ts,
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
        workflow.run()