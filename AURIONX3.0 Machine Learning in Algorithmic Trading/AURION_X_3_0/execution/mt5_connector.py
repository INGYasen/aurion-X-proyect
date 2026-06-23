"""Conector al terminal MetaTrader 5 (inicialización, símbolos, precios)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from analytics.logger import get_logger

if TYPE_CHECKING:
    from market.tick_handler import Tick

logger = get_logger(__name__)

# MetaTrader 5: "No prices" — suele resolverse refrescando tick tras apertura o micro-pausa.
_RETCODE_NO_PRICES = 10021
_ORDER_SEND_PRICE_RETRIES = 3


class MT5Connector:
    """Encapsula la API de MetaTrader 5 para login, suscripción y datos de mercado."""

    def __init__(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
    ) -> None:
        """Guarda credenciales y servidor (sin conectar aún).

        Args:
            login: Número de cuenta MT5.
            password: Contraseña de trading.
            server: Nombre del servidor del broker.
        """
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._initialized = False
        self._mt5: Any | None = None
        self._last_init_attempt_ts = 0.0
        self._reconnect_interval_s = 30.0
        # Offset del server MT5 respecto a UTC real (en segundos).
        # MT5 reporta tick.time como segundos desde epoch en HORA DEL SERVIDOR del broker
        # (ej. GMT+3 para servers europeos). Si lo tratamos como UTC se desfasa
        # vs datetime.now(timezone.utc) usado al cerrar trades.
        self._broker_utc_offset_s: int = 0
        logger.debug("MT5Connector instanciado (login definido: %s)", login is not None)

    @property
    def initialized(self) -> bool:
        """True si la API MT5 está lista para consultas y órdenes."""
        return bool(self._initialized)

    def initialize(self) -> bool:
        """Inicializa el entorno MT5 y valida disponibilidad.

        Returns:
            ``True`` si la inicialización fue exitosa (stub).
        """
        self._last_init_attempt_ts = time.time()
        try:
            import MetaTrader5 as mt5  # type: ignore[import-untyped]
        except Exception as exc:
            logger.error("No se pudo importar MetaTrader5: %s", exc)
            return False

        init_options: list[dict[str, Any]] = []
        if self._path:
            init_options.append({"path": self._path, "timeout": 10_000})
        init_options.append({"timeout": 10_000})

        for attempt in range(1, 4):
            logger.info("MT5 initialize intento %s/3", attempt)
            inited = False
            for opt in init_options:
                if mt5.initialize(**opt):
                    inited = True
                    break
            if not inited:
                logger.warning("mt5.initialize intento %s/3 falló: %s", attempt, mt5.last_error())
                time.sleep(2.0 * attempt)
                continue

            # Espera breve para que IPC quede disponible al arrancar terminal
            time.sleep(2.0)
            # 1) Si terminal ya tiene sesión activa, preferimos reutilizarla.
            acct_current = mt5.account_info()
            if acct_current is not None:
                if self._login is None or int(acct_current.login) == int(self._login):
                    logger.info("MT5 sesión activa detectada (login=%s), se reutiliza", acct_current.login)
                    self._mt5 = mt5
                    self._initialized = True
                    self._broker_utc_offset_s = self._compute_broker_utc_offset()
                    logger.info("MT5Connector.initialize: conexión MT5 activa")
                    return True
                logger.warning(
                    "Sesión activa en MT5 con login distinto (actual=%s esperado=%s); se intentará login explícito",
                    acct_current.login,
                    self._login,
                )

            # 2) Si no hay sesión compatible, intentamos login explícito.
            if self._login is not None and self._password and self._server:
                ok = mt5.login(login=self._login, password=self._password, server=self._server)
                if not ok:
                    # Reintento suave: a veces login falla IPC pero la sesión queda activa.
                    time.sleep(1.0)
                    acct_fallback = mt5.account_info()
                    if acct_fallback is not None and int(acct_fallback.login) == int(self._login):
                        logger.warning(
                            "mt5.login falló (%s), pero se detectó sesión activa correcta; se continúa",
                            mt5.last_error(),
                        )
                    else:
                        logger.warning("mt5.login intento %s/3 falló: %s", attempt, mt5.last_error())
                        mt5.shutdown()
                        time.sleep(2.0 * attempt)
                        continue
            else:
                acct = mt5.account_info()
                if acct is None:
                    logger.warning("mt5.account_info vacío en intento %s/3", attempt)
                    mt5.shutdown()
                    time.sleep(2.0 * attempt)
                    continue
            self._mt5 = mt5
            self._initialized = True
            self._broker_utc_offset_s = self._compute_broker_utc_offset()
            logger.info("MT5Connector.initialize: conexión MT5 activa")
            return True

        logger.error("No se pudo establecer sesión MT5 tras reintentos")
        return False

    def account_equity(self) -> float | None:
        """Equity de la cuenta MT5 activa, o None si no hay sesión."""
        if not self._initialized or self._mt5 is None:
            return None
        info = self._mt5.account_info()
        if info is None:
            return None
        try:
            return float(getattr(info, "equity", 0.0) or 0.0)
        except Exception:
            return None

    def shutdown(self) -> None:
        """Libera el entorno MT5."""
        if self._mt5 is not None:
            self._mt5.shutdown()
        self._initialized = False
        logger.info("MT5Connector.shutdown")

    def subscribe_symbol(self, symbol: str) -> None:
        """Suscribe el terminal a un símbolo para ticks y book."""
        if self._mt5 is None:
            return
        ok = self._mt5.symbol_select(symbol, True)
        if not ok:
            logger.warning("symbol_select falló para %s: %s", symbol, self._mt5.last_error())

    def _compute_broker_utc_offset(self) -> int:
        """Calcula offset del server MT5 respecto a UTC real, en segundos.

        Compara ``tick.time`` de varios simbolos con ``time.time()``
        (epoch UTC real del SO). Toma el MENOR diff absoluto (el simbolo con
        tick mas reciente) para evitar contar 'ticks viejos' de fin de semana.
        Redondea a multiplo de 1800s (30 min). Acota a +-14h (rango realista
        de timezones).

        Returns:
            Offset en segundos (positivo si el server va adelantado a UTC).
        """
        if self._mt5 is None:
            return 0
        try:
            symbols = self._mt5.symbols_get()
            if not symbols:
                return 0
            now_utc = time.time()
            best_diff: float | None = None
            best_age: float = float("inf")
            for sym in symbols[:30]:
                tick = self._mt5.symbol_info_tick(sym.name)
                if tick is None or tick.time <= 0:
                    continue
                age = abs(float(tick.time) - now_utc)
                if age < best_age:
                    best_age = age
                    best_diff = float(tick.time) - now_utc
            if best_diff is None:
                logger.warning("No se pudo leer ningun tick para calcular broker offset")
                return 0
            offset_s = int(round(best_diff / 1800.0) * 1800)
            # Acotar a rango realista de timezones (+-14h).
            MAX_OFFSET_S = 14 * 3600
            if abs(offset_s) > MAX_OFFSET_S:
                logger.warning(
                    "Broker offset fuera de rango (%+.1fh) - mercado cerrado / tick viejo. Asumiendo 0.",
                    offset_s / 3600.0,
                )
                return 0
            logger.info(
                "Broker UTC offset detectado: %+d segundos (%+.1f horas)",
                offset_s,
                offset_s / 3600.0,
            )
            return offset_s
        except Exception:
            logger.exception("No se pudo calcular broker UTC offset; asumiendo 0")
            return 0

    def last_tick(self, symbol: str) -> dict[str, Any]:
        """Obtiene el último tick conocido para ``symbol``.

        Returns:
            Diccionario compatible con ``TickHandler.ingest``.
        """
        if not self._initialized or self._mt5 is None:
            return {}
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return {}
        # AURION X 4.3 (7-jun-2026): Fix bug duraciones negativas en trades.
        # tick.time NO es UTC, es server time del broker (ej. GMT+3 = +10800s).
        # Restamos el offset detectado para alinear con datetime.now(timezone.utc).
        # Si el offset esta en 0 (no se pudo calcular en init por mercado cerrado),
        # intentamos recalcular cuando recibimos un tick fresco (< 120s real).
        if self._broker_utc_offset_s == 0:
            now_utc = time.time()
            tick_age = abs(float(tick.time) - now_utc)
            # tick "fresco" = a menos de 14h (rango de TZ); si es viejo, no recalcular.
            if 0 < tick_age < (14 * 3600):
                new_offset = self._compute_broker_utc_offset()
                if new_offset != 0:
                    self._broker_utc_offset_s = new_offset
        ts = datetime.fromtimestamp(tick.time - self._broker_utc_offset_s, tz=timezone.utc)
        return {
            "symbol": symbol,
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "time_utc": ts,
            "volume": float(getattr(tick, "volume_real", 0.0) or getattr(tick, "volume", 0.0)),
        }

    def get_tick(self, symbol: str) -> Tick | None:
        """Último tick normalizado para el símbolo.

        Cuando ``last_tick`` devuelva un dict válido, aquí se puede construir un
        ``Tick`` (p. ej. vía ``TickHandler.ingest`` desde el bucle de mercado).
        """
        if (not self._initialized) and (time.time() - self._last_init_attempt_ts >= self._reconnect_interval_s):
            self.initialize()
        raw = self.last_tick(symbol)
        if not raw:
            return None
        from market.tick_handler import Tick

        try:
            return Tick(
                symbol=symbol,
                bid=Decimal(str(raw["bid"])),
                ask=Decimal(str(raw["ask"])),
                time_utc=cast(datetime, raw["time_utc"]),
                volume=Decimal(str(raw["volume"])) if raw.get("volume") is not None else None,
            )
        except Exception:
            return None

    def place_market_order(
        self,
        *,
        symbol: str,
        volume: float,
        side: str,
        sl: float | None = None,
        tp: float | None = None,
        deviation: int = 20,
        magic: int = 300000,
        comment: str = "AURION_X_3_0",
    ) -> str | None:
        """Envía orden mercado a MT5 y devuelve ticket."""
        if not self._initialized or self._mt5 is None:
            return None
        side_norm = side.lower().strip()
        if side_norm not in {"buy", "sell"}:
            return None
        self.subscribe_symbol(symbol)
        sym = self._mt5.symbol_info(symbol)
        if sym is None:
            logger.error("symbol_info vacío para %s", symbol)
            return None
        if not sym.visible:
            self._mt5.symbol_select(symbol, True)
            sym = self._mt5.symbol_info(symbol)
        if sym is None:
            return None

        vmin = float(sym.volume_min)
        vmax = float(sym.volume_max)
        vstep = float(sym.volume_step)
        vol = float(volume)
        if vstep > 0:
            vol = max(vmin, min(vmax, vol))
            steps = int((vol - vmin) / vstep + 1e-9)
            vol = max(vmin, vmin + steps * vstep)
        else:
            vol = max(vmin, min(vmax, vol))
        if vol < vmin or vol <= 0:
            logger.error("volumen inválido tras normalizar symbol=%s vol=%s min=%s", symbol, volume, vmin)
            return None

        order_type = self._mt5.ORDER_TYPE_BUY if side_norm == "buy" else self._mt5.ORDER_TYPE_SELL
        digits = int(getattr(sym, "digits", 5) or 5)

        fm = int(getattr(sym, "filling_mode", 0) or 0)
        mt5 = self._mt5
        fill_candidates: list[int] = []
        if fm & int(getattr(mt5, "SYMBOL_FILLING_IOC", 0) or 0):
            fill_candidates.append(mt5.ORDER_FILLING_IOC)
        if fm & int(getattr(mt5, "SYMBOL_FILLING_FOK", 0) or 0):
            fill_candidates.append(mt5.ORDER_FILLING_FOK)
        if fm & int(getattr(mt5, "SYMBOL_FILLING_RETURN", 0) or 0):
            fill_candidates.append(mt5.ORDER_FILLING_RETURN)
        if not fill_candidates:
            fill_candidates = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]

        last_result: Any = None
        for price_attempt in range(_ORDER_SEND_PRICE_RETRIES):
            tick = self._mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(
                    "symbol_info_tick vacío symbol=%s intento=%s/%s err=%s",
                    symbol,
                    price_attempt + 1,
                    _ORDER_SEND_PRICE_RETRIES,
                    mt5.last_error(),
                )
                time.sleep(0.2 + 0.15 * price_attempt)
                continue
            bid = float(tick.bid)
            ask = float(tick.ask)
            if bid <= 0.0 or ask <= 0.0:
                logger.warning(
                    "precios inválidos symbol=%s bid=%s ask=%s intento=%s/%s",
                    symbol,
                    bid,
                    ask,
                    price_attempt + 1,
                    _ORDER_SEND_PRICE_RETRIES,
                )
                time.sleep(0.25 + 0.15 * price_attempt)
                continue
            price = ask if side_norm == "buy" else bid
            price = round(float(price), digits)
            no_prices = False
            for filling in fill_candidates:
                req: dict[str, Any] = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": float(vol),
                    "type": order_type,
                    "price": float(price),
                    "deviation": deviation,
                    "magic": magic,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling,
                }
                if sl is not None:
                    req["sl"] = round(float(sl), digits)
                if tp is not None:
                    req["tp"] = round(float(tp), digits)
                result = mt5.order_send(req)
                last_result = result
                if result is None:
                    logger.warning("order_send None filling=%s err=%s", filling, mt5.last_error())
                    continue
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    pos_id = int(getattr(result, "position", 0) or 0)
                    if pos_id > 0:
                        return str(pos_id)
                    deal_id = int(getattr(result, "deal", 0) or 0)
                    if deal_id > 0:
                        return str(deal_id)
                    ord_id = int(getattr(result, "order", 0) or 0)
                    if ord_id > 0:
                        return str(ord_id)
                    logger.warning("order_send DONE sin position/deal/order en resultado")
                    return None
                ret_comment = getattr(result, "comment", "")
                if int(result.retcode) == _RETCODE_NO_PRICES:
                    no_prices = True
                    logger.warning(
                        "order_send retcode=10021 (No prices) symbol=%s intento_precio=%s/%s filling=%s",
                        symbol,
                        price_attempt + 1,
                        _ORDER_SEND_PRICE_RETRIES,
                        filling,
                    )
                    break
                logger.warning(
                    "order_send retcode=%s filling=%s comment=%s",
                    result.retcode,
                    filling,
                    ret_comment,
                )
            if no_prices:
                time.sleep(0.3 + 0.2 * price_attempt)
                continue
            break
        if last_result is None:
            logger.error("order_send falló sin resultado: %s", mt5.last_error())
        return None

    def close_position(self, ticket: str) -> bool:
        """Cierra posición por ticket."""
        if not self._initialized or self._mt5 is None:
            return False
        try:
            pos_id = int(float(ticket))
        except ValueError:
            return False
        positions = self._mt5.positions_get(ticket=pos_id)
        if not positions:
            return False
        pos = positions[0]
        tick = self._mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return False
        is_buy = pos.type == self._mt5.POSITION_TYPE_BUY
        req = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": self._mt5.ORDER_TYPE_SELL if is_buy else self._mt5.ORDER_TYPE_BUY,
            "price": tick.bid if is_buy else tick.ask,
            "deviation": 20,
            "magic": 300000,
            "comment": "AURION_CLOSE",
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        result = self._mt5.order_send(req)
        return bool(result and result.retcode == self._mt5.TRADE_RETCODE_DONE)

    def close_position_partial(self, ticket: str, volume: float) -> bool:
        """Cierra un volumen parcial de la posición (restante respeta volume_min del símbolo)."""
        if not self._initialized or self._mt5 is None:
            return False
        try:
            pos_id = int(float(ticket))
        except ValueError:
            return False
        positions = self._mt5.positions_get(ticket=pos_id)
        if not positions:
            return False
        pos = positions[0]
        sym = self._mt5.symbol_info(pos.symbol)
        if sym is None:
            return False
        vmin = float(sym.volume_min)
        vstep = float(sym.volume_step)
        vmax = float(sym.volume_max)
        pos_vol = float(pos.volume)
        want = float(volume)
        if want <= 0 or pos_vol <= 0:
            return False
        if want >= pos_vol - 1e-9:
            return self.close_position(ticket)
        rem = pos_vol - want
        if 0 < rem < vmin:
            want = pos_vol
            return self.close_position(ticket)
        if vstep > 0:
            want = max(vmin, min(vmax, want))
            steps = int((want - vmin) / vstep + 1e-9)
            want = max(vmin, vmin + steps * vstep)
        want = min(want, pos_vol)
        if want < vmin:
            return False
        tick = self._mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return False
        is_buy = pos.type == self._mt5.POSITION_TYPE_BUY
        mt5 = self._mt5
        fm = int(getattr(sym, "filling_mode", 0) or 0)
        fill_candidates: list[int] = []
        if fm & int(getattr(mt5, "SYMBOL_FILLING_IOC", 0) or 0):
            fill_candidates.append(mt5.ORDER_FILLING_IOC)
        if fm & int(getattr(mt5, "SYMBOL_FILLING_FOK", 0) or 0):
            fill_candidates.append(mt5.ORDER_FILLING_FOK)
        if fm & int(getattr(mt5, "SYMBOL_FILLING_RETURN", 0) or 0):
            fill_candidates.append(mt5.ORDER_FILLING_RETURN)
        if not fill_candidates:
            fill_candidates = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
        for filling in fill_candidates:
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": float(want),
                "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
                "price": tick.bid if is_buy else tick.ask,
                "deviation": 20,
                "magic": 300000,
                "comment": "AURION_PARTIAL",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            result = mt5.order_send(req)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                return True
        return False

    def modify_position(self, ticket: str, *, sl: float | None = None, tp: float | None = None) -> bool:
        """Modifica SL/TP de posición abierta."""
        if not self._initialized or self._mt5 is None:
            return False
        try:
            pos_id = int(float(ticket))
        except ValueError:
            return False
        positions = self._mt5.positions_get(ticket=pos_id)
        if not positions:
            return False
        pos = positions[0]
        req: dict[str, Any] = {
            "action": self._mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": pos.symbol,
        }
        req["sl"] = float(sl) if sl is not None else float(pos.sl)
        req["tp"] = float(tp) if tp is not None else float(pos.tp)
        result = self._mt5.order_send(req)
        return bool(result and result.retcode == self._mt5.TRADE_RETCODE_DONE)

    def close_all(self) -> None:
        """Cierra todas las posiciones abiertas."""
        if not self._initialized or self._mt5 is None:
            return
        positions = self._mt5.positions_get() or []
        for p in positions:
            self.close_position(str(p.ticket))

    def list_positions_for_reconcile(
        self,
        symbols: set[str],
        *,
        magic: int | None = 300000,
        filter_by_magic: bool = True,
    ) -> list[dict[str, Any]]:
        """Posiciones abiertas en MT5 filtradas por símbolos del bot.

        Args:
            symbols: Conjunto de símbolos en mayúsculas (p. ej. {"XAUUSD"}).
            magic: Magic number de las órdenes AURION; ignorado si ``filter_by_magic`` es falso.
            filter_by_magic: Si es falso, incluye todas las posiciones en esos símbolos (útil si el bróker no informa magic).
        """
        if not self._initialized or self._mt5 is None:
            return []
        mt5 = self._mt5
        want = {str(s).strip().upper() for s in symbols if str(s).strip()}
        out: list[dict[str, Any]] = []
        for p in mt5.positions_get() or []:
            sym = str(getattr(p, "symbol", "") or "").strip().upper()
            if sym not in want:
                continue
            mg = int(getattr(p, "magic", 0) or 0)
            if filter_by_magic and magic is not None and mg != int(magic):
                continue
            typ = int(getattr(p, "type", 0))
            is_buy = typ == mt5.POSITION_TYPE_BUY
            out.append(
                {
                    "ticket": str(int(getattr(p, "ticket", 0))),
                    "symbol": str(getattr(p, "symbol", sym)),
                    "type_buy": bool(is_buy),
                    "price_open": float(getattr(p, "price_open", 0.0) or 0.0),
                    "volume": float(getattr(p, "volume", 0.0) or 0.0),
                    "sl": float(getattr(p, "sl", 0.0) or 0.0),
                    "tp": float(getattr(p, "tp", 0.0) or 0.0),
                    "magic": mg,
                }
            )
        return out

    def count_open_positions_for_symbol(
        self,
        symbol: str,
        *,
        magic: int | None = 300000,
        filter_by_magic: bool = True,
    ) -> int:
        """Cuenta posiciones abiertas en MT5 para un símbolo (mismo criterio que reconcile)."""
        sym = str(symbol).strip().upper()
        if not sym:
            return 0
        return len(self.list_positions_for_reconcile({sym}, magic=magic, filter_by_magic=filter_by_magic))

    def is_position_open(self, ticket: str) -> bool:
        """Comprueba si el ticket sigue abierto en MT5."""
        if not self._initialized or self._mt5 is None:
            return False
        try:
            pos_id = int(float(ticket))
        except ValueError:
            return False
        positions = self._mt5.positions_get(ticket=pos_id)
        return bool(positions)

    def get_closed_summary(self, ticket: str) -> dict[str, Any] | None:
        """Obtiene resumen de cierre desde historial de deals de MT5.

        Retorna dict con ``exit_price``, ``profit`` y ``close_time`` en UTC cuando exista.
        """
        if not self._initialized or self._mt5 is None:
            return None
        try:
            pos_id = int(float(ticket))
        except ValueError:
            return None
        utc_to = datetime.now(timezone.utc)
        # 7 días de lookback, suficiente para ciclos intradía recientes.
        utc_from = utc_to - timedelta(days=7)
        deals = self._mt5.history_deals_get(utc_from, utc_to, position=pos_id)
        if not deals:
            return None
        last = max(deals, key=lambda d: getattr(d, "time", 0))
        ts = datetime.fromtimestamp(getattr(last, "time", 0), tz=timezone.utc)
        return {
            "exit_price": float(getattr(last, "price", 0.0) or 0.0),
            "profit": float(getattr(last, "profit", 0.0) or 0.0),
            "close_time": ts,
        }
