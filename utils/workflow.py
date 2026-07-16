"""
DPWorkflow — orchestrator that delegates each stage to the appropriate handler module.
"""
from __future__ import annotations

from .context import WorkflowContext
from . import adsorption
from . import ts
from . import irc
from . import energy


class DPWorkflow:
    """Top-level workflow facade.

    Delegates to handler modules for adsorption, TS, IRC, and energy stages.
    """

    def __init__(self, **kwargs):
        self.ctx = WorkflowContext(**kwargs)
        # Keep alias for backward compat with some inline uses of self.xxx
        self.run_irc_final_state = self.ctx.run_irc_final_state

    # Proxy context attributes so code that accesses self.xxx still works
    def __getattr__(self, name):
        if hasattr(self.ctx, name):
            return getattr(self.ctx, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # ---- Stage methods --------------------------------------------------
    def generate_initial_adsorbate_guesses(self):
        adsorption.generate_initial_adsorbate_guesses(self.ctx)

    def generate_rxn_ts_guesses_ccqn(self):
        ts.generate_rxn_ts_guesses_ccqn(self.ctx)

    def get_final_state_IRC(self):
        return irc.get_final_state_IRC(self.ctx)

    def get_final_state_energy(self):
        return energy.get_final_state_energy(self.ctx)

    def get_ads_energy(self):
        return energy.get_ads_energy(self.ctx)


__all__ = ["DPWorkflow"]
