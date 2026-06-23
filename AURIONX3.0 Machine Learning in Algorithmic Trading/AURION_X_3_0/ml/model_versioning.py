"""Versionado de modelos con rollback automático si empeora OOS."""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analytics.logger import get_logger

from .oos_validator import OOSValidator

logger = get_logger(__name__)


@dataclass
class ModelVersionRecord:
    version_id: str
    path: Path
    metrics: dict[str, float]
    created_at: str
    reason: str
    symbol: str | None = None


class ModelVersionManager:
    """Gestiona ``best_model.pkl``, versiones numeradas y rollback por símbolo."""

    def __init__(self, models_dir: Path, *, symbol: str | None = None, oos_cfg: dict[str, Any] | None = None) -> None:
        self._symbol = symbol.upper() if symbol else None
        self._dir = Path(models_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._versions_dir = self._dir / "versions"
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._versions_dir / "manifest.json"
        self._best_path = self._dir / "best_model.pkl"
        self._rollback_path = self._dir / "rollback_model.pkl"
        self._oos = OOSValidator(oos_cfg)

    def _load_manifest(self) -> list[dict[str, Any]]:
        if not self._manifest_path.is_file():
            return []
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return list(data) if isinstance(data, list) else []
        except Exception:
            return []

    def _save_manifest(self, rows: list[dict[str, Any]]) -> None:
        self._manifest_path.write_text(json.dumps(rows[-40:], indent=2), encoding="utf-8")

    def _load_best_metrics(self) -> dict[str, float]:
        p = self._versions_dir / "best_metrics.json"
        if not p.is_file():
            return {}
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                return {k: float(v) for k, v in prev.items() if isinstance(v, (int, float))}
        except Exception:
            pass
        return {}

    @staticmethod
    def _rank_metrics(metrics: dict[str, float]) -> float:
        return OOSValidator.composite_rank(metrics)

    @staticmethod
    def _payload_n_features(path: Path) -> int | None:
        """Devuelve nº de features esperado por el modelo de ``path`` (None si no se puede)."""
        if not path.is_file():
            return None
        try:
            payload = ModelVersionManager.load_payload(path)
        except Exception:
            return None

        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        for src in (payload, meta):
            for key in ("n_features_in", "n_features", "feature_count", "expected_n_features"):
                v = src.get(key)
                if isinstance(v, int) and v > 0:
                    return int(v)
            feats = src.get("feature_names") or src.get("features")
            if isinstance(feats, (list, tuple)) and feats:
                return len(feats)

        est = payload.get("estimator")
        n_in = getattr(est, "n_features_in_", None)
        if isinstance(n_in, int) and n_in > 0:
            return int(n_in)
        return None

    def _detect_schema_upgrade(self, candidate_path: Path, deploy_path: Path) -> tuple[bool, int, int]:
        """¿El candidato amplía el nº de features respecto al deploy actual?

        Devuelve (upgrade, n_deploy, n_candidate). Si no se puede determinar
        alguno, se considera ``upgrade=False`` (decisión conservadora salvo
        cuando hay diferencia clara hacia arriba).
        """
        n_cand = self._payload_n_features(candidate_path) or 0
        n_deploy = self._payload_n_features(deploy_path) or 0
        upgrade = n_cand > 0 and n_deploy > 0 and n_cand > n_deploy
        return upgrade, n_deploy, n_cand

    def register_candidate(
        self,
        candidate_path: Path,
        *,
        metrics: dict[str, float],
        reason: str,
        deploy_path: Path,
        regime_label: str | None = None,
    ) -> tuple[bool, str]:
        """Promueve candidato si supera gates OOS; si no, rollback.

        Excepción: si el candidato amplía el esquema de features respecto al
        modelo desplegado (p.ej. v1=10 → v2=25), se promueve igualmente porque
        el modelo viejo está rompiéndose con el pipeline actual y queda en
        modo neutro permanente.

        Returns:
            (promoted, version_id)
        """
        best_prev = self._load_best_metrics()
        promote, gate_reason = self._oos.should_promote(metrics, best_prev or None)
        rank_new = self._rank_metrics(metrics)

        schema_upgrade = False
        if not promote:
            schema_upgrade, n_deploy, n_cand = self._detect_schema_upgrade(candidate_path, deploy_path)
            if schema_upgrade:
                wr = float(metrics.get("winrate", 0.0) or 0.0)
                expc = float(metrics.get("expectancy", 0.0) or 0.0)
                pf = float(metrics.get("profit_factor", 0.0) or 0.0)
                sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
                toxic = (
                    wr < 0.18
                    or expc < -0.80
                    or pf < 0.30
                    or sharpe < -3.0
                )
                if toxic:
                    logger.error(
                        "ModelVersionManager[%s]: SCHEMA UPGRADE detectado pero métricas TÓXICAS "
                        "(wr=%.3f expc=%.3f pf=%.3f sharpe=%.3f) → rechazo promoción y mantengo rollback",
                        self._symbol or "GLOBAL",
                        wr,
                        expc,
                        pf,
                        sharpe,
                    )
                    schema_upgrade = False
                    gate_reason = f"{gate_reason}+schema_upgrade_blocked_toxic(wr={wr:.2f},pf={pf:.2f})"
                else:
                    logger.warning(
                        "ModelVersionManager[%s]: forzando promoción por SCHEMA UPGRADE "
                        "deploy_n_features=%d → candidate_n_features=%d (gate previo=%s, wr=%.3f pf=%.3f)",
                        self._symbol or "GLOBAL",
                        n_deploy,
                        n_cand,
                        gate_reason,
                        wr,
                        pf,
                    )
                    promote = True
                    gate_reason = f"schema_upgrade({n_deploy}->{n_cand}) over {gate_reason}"

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sym_tag = f"{self._symbol}_" if self._symbol else ""
        version_id = f"{sym_tag}model_v_{ts}"
        version_path = self._versions_dir / f"{version_id}.pkl"
        shutil.copy2(candidate_path, version_path)

        manifest = self._load_manifest()
        entry = {
            "version_id": version_id,
            "symbol": self._symbol,
            "path": str(version_path),
            "metrics": metrics,
            "rank": rank_new,
            "reason": reason,
            "gate": gate_reason,
            "schema_upgrade": bool(schema_upgrade),
            "regime_label": regime_label,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest.append(entry)
        self._save_manifest(manifest)

        if not promote:
            if self._rollback_path.is_file() and deploy_path.is_file():
                shutil.copy2(self._rollback_path, deploy_path)
                logger.warning(
                    "ModelVersionManager[%s]: no promovido (%s) rank=%.4f → rollback %s",
                    self._symbol or "GLOBAL",
                    gate_reason,
                    rank_new,
                    deploy_path,
                )
            elif self._best_path.is_file():
                shutil.copy2(self._best_path, deploy_path)
            return False, version_id

        if deploy_path.is_file():
            shutil.copy2(deploy_path, self._rollback_path)
        shutil.copy2(candidate_path, deploy_path)
        shutil.copy2(candidate_path, self._best_path)
        (self._versions_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info(
            "ModelVersionManager[%s]: promovido %s rank=%.4f → %s (%s)",
            self._symbol or "GLOBAL",
            version_id,
            rank_new,
            deploy_path,
            reason,
        )
        return True, version_id

    def force_rollback(self, deploy_path: Path) -> bool:
        src = self._rollback_path if self._rollback_path.is_file() else self._best_path
        if not src.is_file():
            return False
        shutil.copy2(src, deploy_path)
        logger.warning("ModelVersionManager[%s]: rollback manual → %s", self._symbol or "GLOBAL", deploy_path)
        return True

    @staticmethod
    def load_payload(path: Path) -> dict[str, Any]:
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"PKL inválido: {path}")
        return payload
