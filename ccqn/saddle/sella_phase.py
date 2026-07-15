import numpy as np


class SellaStepFailure(RuntimeError):
    def __init__(self, message, suggested_radius=None):
        super().__init__(message)
        self.suggested_radius = suggested_radius


class SellaPhase:
    """Adapter that exposes a PRFO-like phase interface backed by Sella."""

    def __init__(
        self,
        atoms,
        trust_radius_saddle_initial,
        logfile=None,
        trajectory=None,
        master=None,
        sella_kwargs=None,
    ):
        try:
            from sella import Sella
        except ImportError as exc:
            raise ImportError(
                "Sella is not installed. Install it with `pip install sella` to use saddle_optimizer='sella'."
            ) from exc

        kwargs = dict(sella_kwargs or {})
        if 'order' not in kwargs:
            kwargs['order'] = 1
        if 'delta0' not in kwargs:
            kwargs['delta0'] = trust_radius_saddle_initial
        self._sella_cls = Sella
        self._atoms = atoms
        self._logfile = logfile
        self._trajectory = trajectory
        self._master = master
        self._base_kwargs = dict(kwargs)
        self._sella = self._build_optimizer(Sella, atoms, logfile, trajectory, master, kwargs)

    def _build_optimizer(self, sella_cls, atoms, logfile, trajectory, master, kwargs):
        attempts = [
            dict(kwargs, logfile=logfile, trajectory=trajectory, master=master),
            dict(kwargs, logfile=logfile, trajectory=trajectory),
            dict(kwargs, logfile=logfile),
            dict(kwargs),
        ]
        last_error = None
        for init_kwargs in attempts:
            cleaned = {k: v for k, v in init_kwargs.items() if v is not None}
            try:
                return sella_cls(atoms, **cleaned)
            except TypeError as exc:
                last_error = exc
                continue
        raise TypeError(f"Unable to initialize Sella optimizer with provided arguments: {last_error}")

    def _rebuild_with_delta0(self, delta0):
        kwargs = dict(self._base_kwargs)
        kwargs['delta0'] = float(delta0)
        self._sella = self._build_optimizer(
            self._sella_cls,
            self._atoms,
            self._logfile,
            self._trajectory,
            self._master,
            kwargs,
        )

    def _step_once(self):
        try:
            self._sella.step()
        except TypeError:
            self._sella.step(None)

    def run(self, ctx):
        pos_before = ctx.atoms.get_positions().copy()

        try:
            self._step_once()
        except Exception as exc:
            base = max(float(ctx.trust_radius_saddle), 1e-3)
            retry_radii = [max(base * 0.5, 1e-3), max(base * 0.25, 1e-3), max(base * 0.1, 1e-3)]
            last_exc = exc
            for retry_delta in retry_radii:
                try:
                    if ctx.logfile:
                        ctx.logfile.write(
                            f"Sella step failed ({last_exc}); retry with smaller delta0={retry_delta:.4e}.\n"
                        )
                    self._rebuild_with_delta0(retry_delta)
                    self._step_once()
                    break
                except Exception as retry_exc:
                    last_exc = retry_exc
            else:
                raise SellaStepFailure(
                    f"Sella step failed after retries: {last_exc}",
                    suggested_radius=retry_radii[-1],
                )

        pos_after = ctx.atoms.get_positions().copy()
        s_k = (pos_after - pos_before).reshape(-1)
        new_radius = ctx.trust_radius_saddle
        for attr in ('delta', 'delta0'):
            if hasattr(self._sella, attr):
                try:
                    candidate = float(getattr(self._sella, attr))
                    if np.isfinite(candidate) and candidate > 0.0:
                        new_radius = candidate
                        break
                except Exception:
                    pass

        return s_k, new_radius
