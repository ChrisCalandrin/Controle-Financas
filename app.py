from __future__ import annotations

import sqlite3
import os
import base64
import hashlib
import hmac
from contextlib import contextmanager
from datetime import date
import calendar
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# ============================================================
# Auth (multi-usuário)
# ============================================================
AUTH_PBKDF2_ITERS = 200_000
LEGACY_USER_ID = 0  # dados antigos (pré-login) ficam com user_id=0 até serem atribuídos ao admin


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))


def hash_password(password: str) -> Tuple[str, str]:
    password = str(password or "")
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, AUTH_PBKDF2_ITERS)
    return _b64e(salt), _b64e(dk)


def verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    try:
        salt = _b64d(salt_b64)
        expected = _b64d(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", str(password or "").encode("utf-8"), salt, AUTH_PBKDF2_ITERS)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def require_user_id() -> int:
    uid = st.session_state.get("user_id", None)
    if uid is None:
        raise RuntimeError("Usuário não autenticado.")
    return int(uid)


def is_admin() -> bool:
    return bool(st.session_state.get("is_admin", False))


def logout() -> None:
    for k in ["user_id", "username", "display_name", "is_admin"]:
        if k in st.session_state:
            del st.session_state[k]
    # força recarregar config do período quando logar novamente
    st.session_state._period_cfg_loaded = False
    st.rerun()

# ============================================================
# Config
# ============================================================
APP_TITLE = "Finanças Pessoais"
DB_PATH = "financas.db"

TIPOS = ["Despesa", "Receita"]

# ✅ inclui "Amortizado" para cartão (adiantamento)
STATUS = ["Pago", "Pendente", "Amortizado"]

PAGAMENTOS = ["Crédito", "Débito", "Débito Automático", "PIX", "Dinheiro", "Boleto", "Transferência"]
CATEGORIAS_DEFAULT = [
    "Alimentação",
    "Moradia",
    "Transporte",
    "Lazer",
    "Saúde",
    "Educação",
    "Assinaturas",
    "Compras",
    "Impostos",
    "Salário",
    "Renda Extra",
    "Outros",
]

PT_MONTHS = {
    1: "Janeiro",
    2: "Fevereiro",
    3: "Março",
    4: "Abril",
    5: "Maio",
    6: "Junho",
    7: "Julho",
    8: "Agosto",
    9: "Setembro",
    10: "Outubro",
    11: "Novembro",
    12: "Dezembro",
}

# ============================================================
# Investimentos
# ============================================================
# Tipos de movimento aceitos (usado para normalização e UI)
INV_MOV_TYPES = ["Aporte", "Retirada"]

# Produtos default para sugestão (usuário pode digitar/outros virão do histórico)
INV_PRODUCTS_DEFAULT = [
    "Renda Fixa (CDB/LCI/LCA)",
    "Tesouro Direto",
    "Fundo de Investimento",
    "Ações/ETF",
    "Cripto",
    "Outro",
]


# ============================================================
# Utils
# ============================================================


def brl(value: float) -> str:
    """Formatação simples para R$ sem depender de locale do SO."""
    try:
        v = float(value)
    except Exception:
        v = 0.0
    s = f"{v:,.2f}"  # 1,234.56
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def safe_float(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return 0.0
    s = s.replace(".", "").replace(",", ".") if "," in s else s
    try:
        return float(s)
    except Exception:
        return 0.0


def month_key(dt: pd.Timestamp) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def month_label(yyyy_mm: str) -> str:
    y, m = yyyy_mm.split("-")
    m = int(m)
    return f"{PT_MONTHS.get(m, m)} {y}"


def start_end_of_month(yyyy_mm: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    y, m = yyyy_mm.split("-")
    y, m = int(y), int(m)
    first = pd.Timestamp(date(y, m, 1))
    last_day = calendar.monthrange(y, m)[1]
    last = pd.Timestamp(date(y, m, last_day))
    return first, last


def clamp_day_to_month(y: int, m: int, d: int) -> int:
    last_day = calendar.monthrange(y, m)[1]
    return min(max(1, d), last_day)


def add_months(d: date, months: int) -> date:
    """Soma meses a uma date (ajusta dia para o último dia do mês, se necessário)."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = clamp_day_to_month(y, m, d.day)
    return date(y, m, day)


def month_range(start_yyyy_mm: str, end_yyyy_mm: str) -> List[str]:
    sy, sm = map(int, start_yyyy_mm.split("-"))
    ey, em = map(int, end_yyyy_mm.split("-"))
    start = date(sy, sm, 1)
    end = date(ey, em, 1)
    out = []
    cur = start
    while cur <= end:
        out.append(f"{cur.year:04d}-{cur.month:02d}")
        cur = add_months(cur, 1)
    return out


def split_installments(total: float, n: int) -> List[float]:
    """Divide total em n parcelas, ajustando a última para fechar centavos."""
    total = float(total)
    n = max(1, int(n))
    base = round(total / n, 2)
    vals = [base] * n
    diff = round(total - sum(vals), 2)
    vals[-1] = round(vals[-1] + diff, 2)
    return vals


def annual_to_monthly_rate(annual_rate_decimal: float) -> float:
    """Converte taxa anual efetiva (decimal) para mensal efetiva."""
    if annual_rate_decimal <= 0:
        return 0.0
    return (1.0 + float(annual_rate_decimal)) ** (1.0 / 12.0) - 1.0


def annual_to_daily_rate(annual_rate_decimal: float) -> float:
    """Converte taxa anual efetiva (decimal) para diária efetiva (~365)."""
    if annual_rate_decimal <= 0:
        return 0.0
    return (1.0 + float(annual_rate_decimal)) ** (1.0 / 365.0) - 1.0


def fv_with_contrib(principal: float, pmt: float, r: float, n_months: int) -> float:
    """Future Value com aportes mensais (pmt no fim do mês)."""
    principal = float(principal)
    pmt = float(pmt)
    r = float(r)
    n = int(max(0, n_months))
    if n == 0:
        return principal
    if r <= 0:
        return principal + pmt * n
    return principal * ((1 + r) ** n) + pmt * (((1 + r) ** n - 1) / r)


# ============================================================
# Período (Calendário vs Orçamento por virada)
# ============================================================


def _normalize_mode(mode: str) -> str:
    mode = str(mode or "calendar").strip().lower()
    return mode if mode in {"calendar", "budget"} else "calendar"


def period_bounds(yyyy_mm: str, mode: str, cutoff_day: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Retorna (start, end) inclusivo do período selecionado.

    - calendar: 1º ao último dia do mês
    - budget: do dia cutoff do mês anterior até (cutoff-1) do mês atual
      Ex cutoff=28, período de 2026-02 => 28/01/2026 a 27/02/2026
    """
    mode = _normalize_mode(mode)
    cutoff_day = int(max(1, min(31, int(cutoff_day))))

    if mode != "budget" or cutoff_day <= 1:
        return start_end_of_month(yyyy_mm)

    y, m = map(int, yyyy_mm.split("-"))
    prev = add_months(date(y, m, 1), -1)

    start_day = clamp_day_to_month(prev.year, prev.month, cutoff_day)
    start = pd.Timestamp(date(prev.year, prev.month, start_day))

    end_day = clamp_day_to_month(y, m, cutoff_day - 1)
    end = pd.Timestamp(date(y, m, end_day))

    return start, end


def budget_month_key_from_datetime(dts: pd.Series, cutoff_day: int) -> pd.Series:
    """
    Retorna 'YYYY-MM' aplicando regra:
    - se dia >= cutoff_day => conta para o próximo mês
    - caso contrário => mês atual
    """
    cutoff_day = max(1, min(31, int(cutoff_day)))
    x = pd.to_datetime(dts, errors="coerce")

    p = x.dt.to_period("M")
    day = x.dt.day.fillna(0).astype(int)
    shift = (day >= cutoff_day).astype(int)

    out = (p + shift).astype(str)
    out = out.where(~x.isna(), "")
    out = out.replace("NaT", "")
    return out


def filter_df_by_period(df: pd.DataFrame, date_col: str, yyyy_mm: str, mode: str, cutoff_day: int) -> pd.DataFrame:
    if df.empty:
        return df
    start, end = period_bounds(yyyy_mm, mode, cutoff_day)
    return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()


def annual_bounds(year: int, mode: str, cutoff_day: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Intervalo anual respeitando o modo:
    - calendar: 01/01 a 31/12
    - budget: do início do período de YYYY-01 até o fim do período de YYYY-12
      (pode incluir uma fatia do Dezembro do ano anterior, pela regra do cutoff)
    """
    mode = _normalize_mode(mode)
    year = int(year)
    if mode == "calendar":
        return pd.Timestamp(date(year, 1, 1)), pd.Timestamp(date(year, 12, 31))

    start, _ = period_bounds(f"{year:04d}-01", mode, cutoff_day)
    _, end = period_bounds(f"{year:04d}-12", mode, cutoff_day)
    return start, end


def filter_df_by_year(df: pd.DataFrame, date_col: str, year: int, mode: str, cutoff_day: int) -> pd.DataFrame:
    if df.empty:
        return df
    start, end = annual_bounds(int(year), mode, cutoff_day)
    return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()


# ============================================================
# Database
# ============================================================


@contextmanager
def db_conn(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _table_cols(conn: sqlite3.Connection, table: str) -> set:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {r["name"] for r in cur.fetchall()}


def init_db() -> None:
    with db_conn() as conn:
        cur = conn.cursor()

        # ---------- users ----------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                display_name TEXT,
                pw_salt TEXT NOT NULL,
                pw_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")

        # ---------- base (multi-usuário) ----------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL DEFAULT 0,
                data DATE,
                descricao TEXT,
                categoria TEXT,
                tipo TEXT,
                valor REAL,
                pagamento TEXT,
                status TEXT DEFAULT 'Pago',
                origem TEXT,
                observacao TEXT,
                cc_compra_id INTEGER,
                cc_parcela_num INTEGER,
                cc_parcela_total INTEGER,
                ref_month TEXT,
                fixo_id INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS orcamentos (
                user_id INTEGER NOT NULL DEFAULT 0,
                categoria TEXT NOT NULL,
                valor_teto REAL,
                PRIMARY KEY (user_id, categoria)
            );
            """
        )

        
        # ---------- lançamentos fixos (Débito Automático) ----------
        cur.execute(
            """
            
CREATE TABLE IF NOT EXISTS lancamentos_fixos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL DEFAULT 0,
    descricao TEXT NOT NULL,
    categoria TEXT NOT NULL,
    valor REAL NOT NULL,
    pagamento TEXT NOT NULL DEFAULT 'Débito Automático',
    status TEXT NOT NULL DEFAULT 'Pago',
    dia INTEGER NOT NULL,
    inicio_ym TEXT NOT NULL,   -- 'YYYY-MM'
    intervalo_m INTEGER NOT NULL DEFAULT 1,  -- 1=mensal, 3=trimestral, 12=anual
    fim_ym TEXT,               -- 'YYYY-MM' (opcional)
    fim DATE,                  -- opcional (data exata)
    paused_from_ym TEXT,        -- 'YYYY-MM' (opcional)
    paused_until_ym TEXT,       -- 'YYYY-MM' (opcional)
    ativo INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    cancelado_em TEXT
);
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fixos_user ON lancamentos_fixos(user_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fixos_user_ativo ON lancamentos_fixos(user_id, ativo);")

        # ---------- cartão (parcelado/recorrente + amortização) ----------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cc_compras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL DEFAULT 0,
                data_compra DATE,
                descricao TEXT,
                categoria TEXT,
                tipo_compra TEXT,          -- avista | parcelado | recorrente
                total REAL,                -- total (avista/parcelado) | valor mensal (recorrente)
                parcelas INTEGER,           -- 1 (avista) | N (parcelado) | NULL (recorrente)
                ativo INTEGER DEFAULT 1,    -- 1=ativo, 0=inativo
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                fim DATE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cc_amortizacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL DEFAULT 0,
                cc_compra_id INTEGER,
                data DATE,
                parcelas_amortizadas INTEGER DEFAULT 0,
                desconto REAL DEFAULT 0,
                valor_pago REAL,
                pagamento TEXT DEFAULT 'PIX',
                observacao TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # ---------- investimentos ----------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS investimentos_aportes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL DEFAULT 0,
                data DATE,
                produto TEXT,
                tipo TEXT,                 -- 'Aporte' | 'Retirada'
                valor REAL,
                observacao TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS investimentos_config (
                user_id INTEGER PRIMARY KEY,
                aporte_planejado REAL NOT NULL DEFAULT 0,
                cdi_anual REAL NOT NULL DEFAULT 0,
                pct_cdi REAL NOT NULL DEFAULT 100,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            "INSERT OR IGNORE INTO investimentos_config (user_id, aporte_planejado, cdi_anual, pct_cdi) VALUES (0, 0.0, 0.0, 100.0);"
        )

        # ---------- config do app (modo de período) ----------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS app_config (
                user_id INTEGER PRIMARY KEY,
                budget_mode TEXT,     -- 'calendar' | 'budget'
                cutoff_day INTEGER,   -- dia de virada do orçamento (ex: 28)
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute("INSERT OR IGNORE INTO app_config (user_id, budget_mode, cutoff_day) VALUES (0, 'calendar', 28);")

        # ============================================================
        # Migrações (compatibilidade com bancos antigos)
        # ============================================================

        # --- transacoes: user_id e colunas novas ---
        cols = _table_cols(conn, "transacoes")
        if "user_id" not in cols:
            cur.execute("ALTER TABLE transacoes ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0;")

        to_add = [
            ("origem", "TEXT"),
            ("observacao", "TEXT"),
            ("cc_compra_id", "INTEGER"),
            ("cc_parcela_num", "INTEGER"),
            ("cc_parcela_total", "INTEGER"),
            ("ref_month", "TEXT"),
            ("fixo_id", "INTEGER"),
            ("created_at", "TEXT"),
        ]
        cols = _table_cols(conn, "transacoes")
        for col, typ in to_add:
            if col not in cols:
                # created_at em SQLite via ALTER não aplica DEFAULT em linhas antigas, mas não tem problema.
                cur.execute(f"ALTER TABLE transacoes ADD COLUMN {col} {typ};")

        # --- cc_compras / cc_amortizacoes: user_id e coluna fim ---
        if _table_exists(conn, "cc_compras"):
            cc_cols = _table_cols(conn, "cc_compras")
            if "user_id" not in cc_cols:
                cur.execute("ALTER TABLE cc_compras ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0;")
            if "fim" not in cc_cols:
                cur.execute("ALTER TABLE cc_compras ADD COLUMN fim DATE;")
        if _table_exists(conn, "cc_amortizacoes"):
            am_cols = _table_cols(conn, "cc_amortizacoes")
            if "user_id" not in am_cols:
                cur.execute("ALTER TABLE cc_amortizacoes ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0;")

        # --- investimentos_aportes: user_id + tipo ---
        if _table_exists(conn, "investimentos_aportes"):
            inv_cols = _table_cols(conn, "investimentos_aportes")
            if "user_id" not in inv_cols:
                cur.execute("ALTER TABLE investimentos_aportes ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0;")
            if "tipo" not in inv_cols:
                cur.execute("ALTER TABLE investimentos_aportes ADD COLUMN tipo TEXT;")
                cur.execute("UPDATE investimentos_aportes SET tipo = CASE WHEN COALESCE(valor,0) < 0 THEN 'Retirada' ELSE 'Aporte' END WHERE tipo IS NULL OR TRIM(tipo) = '';")

        # --- orcamentos: reconstrução para PK (user_id, categoria) ---
        if _table_exists(conn, "orcamentos"):
            orc_cols = _table_cols(conn, "orcamentos")
            # banco antigo: só tem categoria/valor_teto
            if "user_id" not in orc_cols:
                cur.execute("ALTER TABLE orcamentos RENAME TO orcamentos_old;")
                cur.execute(
                    """
                    CREATE TABLE orcamentos (
                        user_id INTEGER NOT NULL DEFAULT 0,
                        categoria TEXT NOT NULL,
                        valor_teto REAL,
                        PRIMARY KEY (user_id, categoria)
                    );
                    """
                )
                cur.execute(
                    """
                    INSERT INTO orcamentos (user_id, categoria, valor_teto)
                    SELECT 0, categoria, valor_teto
                    FROM orcamentos_old
                    """
                )
                cur.execute("DROP TABLE orcamentos_old;")

        # --- investimentos_config / app_config: reconstrução para user_id PK (banco antigo tinha id=1) ---
        if _table_exists(conn, "investimentos_config"):
            icols = _table_cols(conn, "investimentos_config")
            if "id" in icols:
                cur.execute("ALTER TABLE investimentos_config RENAME TO investimentos_config_old;")
                cur.execute(
                    """
                    CREATE TABLE investimentos_config (
                        user_id INTEGER PRIMARY KEY,
                        aporte_planejado REAL NOT NULL DEFAULT 0,
                        cdi_anual REAL NOT NULL DEFAULT 0,
                        pct_cdi REAL NOT NULL DEFAULT 100,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                row = conn.execute(
                    "SELECT aporte_planejado, cdi_anual, pct_cdi FROM investimentos_config_old WHERE id = 1"
                ).fetchone()
                ap = float((row["aporte_planejado"] if row else 0.0) or 0.0)
                cdi = float((row["cdi_anual"] if row else 0.0) or 0.0)
                pct = float((row["pct_cdi"] if row else 100.0) or 100.0)
                cur.execute(
                    "INSERT OR IGNORE INTO investimentos_config (user_id, aporte_planejado, cdi_anual, pct_cdi) VALUES (0, ?, ?, ?)",
                    (ap, cdi, pct),
                )
                cur.execute("DROP TABLE investimentos_config_old;")

        if _table_exists(conn, "app_config"):
            acols = _table_cols(conn, "app_config")
            if "id" in acols:
                cur.execute("ALTER TABLE app_config RENAME TO app_config_old;")
                cur.execute(
                    """
                    CREATE TABLE app_config (
                        user_id INTEGER PRIMARY KEY,
                        budget_mode TEXT,
                        cutoff_day INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                row = conn.execute("SELECT budget_mode, cutoff_day FROM app_config_old WHERE id = 1").fetchone()
                mode = (row["budget_mode"] if row else "calendar") or "calendar"
                cutoff = int((row["cutoff_day"] if row else 28) or 28)
                cutoff = max(1, min(31, cutoff))
                cur.execute(
                    "INSERT OR IGNORE INTO app_config (user_id, budget_mode, cutoff_day) VALUES (0, ?, ?)",
                    (str(mode).strip().lower(), cutoff),
                )
                cur.execute("DROP TABLE app_config_old;")

        # --- tabelas extras (se existirem) ---
        if _table_exists(conn, "lancamentos_fixos"):
            lcols = _table_cols(conn, "lancamentos_fixos")
            if "user_id" not in lcols:
                cur.execute("ALTER TABLE lancamentos_fixos ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0;")

        # colunas novas para fixos (início_ym e cancelamento)
        lcols = _table_cols(conn, "lancamentos_fixos")
        if "inicio_ym" not in lcols:
            cur.execute("ALTER TABLE lancamentos_fixos ADD COLUMN inicio_ym TEXT;")
            # best-effort: se existir coluna antiga 'inicio', tenta derivar 'YYYY-MM'
            try:
                cur.execute("UPDATE lancamentos_fixos SET inicio_ym = substr(inicio, 1, 7) WHERE (inicio_ym IS NULL OR inicio_ym = '') AND inicio IS NOT NULL;")
            except Exception:
                pass

        lcols = _table_cols(conn, "lancamentos_fixos")
        if "cancelado_em" not in lcols:
            cur.execute("ALTER TABLE lancamentos_fixos ADD COLUMN cancelado_em TEXT;")

        # colunas novas para fixos (frequência, pausa, fim)
        lcols = _table_cols(conn, "lancamentos_fixos")
        if "fim_ym" not in lcols:
            cur.execute("ALTER TABLE lancamentos_fixos ADD COLUMN fim_ym TEXT;")

        lcols = _table_cols(conn, "lancamentos_fixos")
        if "intervalo_m" not in lcols:
            cur.execute("ALTER TABLE lancamentos_fixos ADD COLUMN intervalo_m INTEGER NOT NULL DEFAULT 1;")
        lcols = _table_cols(conn, "lancamentos_fixos")
        if "fim" not in lcols:
            cur.execute("ALTER TABLE lancamentos_fixos ADD COLUMN fim DATE;")
        lcols = _table_cols(conn, "lancamentos_fixos")
        if "paused_from_ym" not in lcols:
            cur.execute("ALTER TABLE lancamentos_fixos ADD COLUMN paused_from_ym TEXT;")
        lcols = _table_cols(conn, "lancamentos_fixos")
        if "paused_until_ym" not in lcols:
            cur.execute("ALTER TABLE lancamentos_fixos ADD COLUMN paused_until_ym TEXT;")

        if _table_exists(conn, "app_settings"):
            scols = _table_cols(conn, "app_settings")
            if "user_id" not in scols:
                # app_settings antigo: key PK. Reconstrói como (user_id, key)
                cur.execute("ALTER TABLE app_settings RENAME TO app_settings_old;")
                cur.execute(
                    """
                    CREATE TABLE app_settings (
                        user_id INTEGER NOT NULL DEFAULT 0,
                        key TEXT NOT NULL,
                        value TEXT,
                        PRIMARY KEY (user_id, key)
                    );
                    """
                )
                cur.execute(
                    """
                    INSERT INTO app_settings (user_id, key, value)
                    SELECT 0, key, value FROM app_settings_old
                    """
                )
                cur.execute("DROP TABLE app_settings_old;")

        # índices (com user_id)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transacoes_user_data ON transacoes(user_id, data);")
        # Evita duplicar geração automática de fixos no mesmo mês
        try:
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uniq_transacoes_user_fixo_ref ON transacoes(user_id, fixo_id, ref_month);")
        except Exception:
            # Se houver dados duplicados antigos, não bloqueia a inicialização.
            pass

        cur.execute("CREATE INDEX IF NOT EXISTS idx_transacoes_user_cc ON transacoes(user_id, cc_compra_id, cc_parcela_num, ref_month);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cc_compras_user_data ON cc_compras(user_id, data_compra);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cc_amort_user ON cc_amortizacoes(user_id, cc_compra_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_invest_user_data ON investimentos_aportes(user_id, data);")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,)).fetchone()
    return row is not None


def users_count() -> int:
    with db_conn() as conn:
        if not _table_exists(conn, "users"):
            return 0
        row = conn.execute("SELECT COUNT(1) AS n FROM users").fetchone()
        return int(row["n"] if row else 0)


def get_user_by_username(username: str) -> Optional[Dict[str, object]]:
    username = str(username or "").strip()
    if not username:
        return None
    with db_conn() as conn:
        row = conn.execute(
            """
            SELECT id, username, display_name, pw_salt, pw_hash, is_admin
            FROM users
            WHERE lower(username) = lower(?)
            """,
            (username,),
        ).fetchone()
    return dict(row) if row else None


def ensure_user_defaults(user_id: int) -> None:
    user_id = int(user_id)
    with db_conn() as conn:
        # app_config
        row0 = conn.execute("SELECT budget_mode, cutoff_day FROM app_config WHERE user_id = ?", (LEGACY_USER_ID,)).fetchone()
        mode0 = (row0["budget_mode"] if row0 else "calendar") or "calendar"
        cutoff0 = int((row0["cutoff_day"] if row0 else 28) or 28)
        cutoff0 = max(1, min(31, cutoff0))
        conn.execute(
            """
            INSERT OR IGNORE INTO app_config (user_id, budget_mode, cutoff_day)
            VALUES (?, ?, ?)
            """,
            (user_id, str(mode0).strip().lower(), cutoff0),
        )

        # investimentos_config
        rowi0 = conn.execute(
            "SELECT aporte_planejado, cdi_anual, pct_cdi FROM investimentos_config WHERE user_id = ?",
            (LEGACY_USER_ID,),
        ).fetchone()
        ap0 = float((rowi0["aporte_planejado"] if rowi0 else 0.0) or 0.0)
        cdi0 = float((rowi0["cdi_anual"] if rowi0 else 0.0) or 0.0)
        pct0 = float((rowi0["pct_cdi"] if rowi0 else 100.0) or 100.0)
        conn.execute(
            """
            INSERT OR IGNORE INTO investimentos_config (user_id, aporte_planejado, cdi_anual, pct_cdi)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, ap0, cdi0, pct0),
        )


def create_user(username: str, display_name: str, password: str, admin: bool = False) -> int:
    username = str(username or "").strip()
    if not username:
        raise ValueError("Username inválido.")

    salt_b64, hash_b64 = hash_password(password)

    with db_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO users (username, display_name, pw_salt, pw_hash, is_admin)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username, str(display_name or "").strip(), salt_b64, hash_b64, 1 if admin else 0),
        )
        user_id = int(cur.lastrowid)

    ensure_user_defaults(user_id)
    return user_id


def assign_legacy_data_to_user(user_id: int) -> None:
    user_id = int(user_id)
    tables = [
        "transacoes",
        "cc_compras",
        "cc_amortizacoes",
        "investimentos_aportes",
        "orcamentos",
        "investimentos_config",
        "app_config",
        # tabelas extras (se existirem)
        "lancamentos_fixos",
        "app_settings",
    ]
    with db_conn() as conn:
        for t in tables:
            if not _table_exists(conn, t):
                continue
            cols = _table_cols(conn, t)
            if "user_id" not in cols:
                continue
            conn.execute(f"UPDATE {t} SET user_id = ? WHERE user_id = ?", (user_id, LEGACY_USER_ID))


def _set_logged_user(user_row: Dict[str, object]) -> None:
    st.session_state.user_id = int(user_row["id"])
    st.session_state.username = str(user_row.get("username") or "")
    st.session_state.display_name = str(user_row.get("display_name") or user_row.get("username") or "")
    st.session_state.is_admin = bool(int(user_row.get("is_admin") or 0))
    # força recarregar config do período (multiusuário)
    st.session_state._period_cfg_loaded = False


def auth_gate() -> bool:
    """
    Retorna True quando autenticado.
    Quando não autenticado, renderiza a tela de login/criação de admin e retorna False.
    """
    if st.session_state.get("user_id", None) is not None:
        return True

    st.markdown(f"## 🔐 {APP_TITLE} — Login")

    n = users_count()
    if n == 0:
        st.info("Primeiro acesso: crie seu usuário ADMIN. Todos os dados existentes serão atribuídos a ele.")
        with st.form("create_admin"):
            u = st.text_input("Usuário (username)", value="admin")
            d = st.text_input("Nome exibido", value="Administrador")
            p1 = st.text_input("Senha", type="password")
            p2 = st.text_input("Confirmar senha", type="password")
            submitted = st.form_submit_button("Criar ADMIN")
        if submitted:
            if not u.strip():
                st.error("Preencha o usuário.")
                return False
            if not p1:
                st.error("Preencha a senha.")
                return False
            if p1 != p2:
                st.error("As senhas não conferem.")
                return False
            try:
                uid = create_user(u, d, p1, admin=True)
                assign_legacy_data_to_user(uid)
                row = get_user_by_username(u)
                if row:
                    _set_logged_user(row)
                    st.success("ADMIN criado e dados migrados com sucesso.")
                    st.rerun()
            except sqlite3.IntegrityError:
                st.error("Este usuário já existe.")
            except Exception as e:
                st.error(f"Falha ao criar ADMIN: {e}")
        return False

    with st.form("login_form"):
        u = st.text_input("Usuário")
        p = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar")
    if submitted:
        row = get_user_by_username(u)
        if not row:
            st.error("Usuário não encontrado.")
            return False
        if not verify_password(p, str(row["pw_salt"]), str(row["pw_hash"])):
            st.error("Senha incorreta.")
            return False
        _set_logged_user(row)
        st.rerun()

    st.caption("Dica: somente o ADMIN cria novos usuários (Cadastro A).")
    return False


def fetch_app_config() -> Dict[str, object]:
    uid = require_user_id()
    with db_conn() as conn:
        row = conn.execute("SELECT budget_mode, cutoff_day FROM app_config WHERE user_id = ?", (uid,)).fetchone()

        if not row:
            # cria defaults para este usuário copiando do legado (0)
            row0 = conn.execute("SELECT budget_mode, cutoff_day FROM app_config WHERE user_id = ?", (LEGACY_USER_ID,)).fetchone()
            mode0 = (row0["budget_mode"] if row0 else "calendar") or "calendar"
            cutoff0 = int((row0["cutoff_day"] if row0 else 28) or 28)
            cutoff0 = max(1, min(31, cutoff0))
            conn.execute(
                "INSERT OR IGNORE INTO app_config (user_id, budget_mode, cutoff_day) VALUES (?, ?, ?)",
                (uid, str(mode0).strip().lower(), cutoff0),
            )
            row = conn.execute("SELECT budget_mode, cutoff_day FROM app_config WHERE user_id = ?", (uid,)).fetchone()

    if not row:
        return {"budget_mode": "calendar", "cutoff_day": 28}

    mode = str(row["budget_mode"] or "calendar").strip().lower()
    if mode not in {"calendar", "budget"}:
        mode = "calendar"

    cutoff = int(row["cutoff_day"] or 28)
    cutoff = max(1, min(31, cutoff))
    return {"budget_mode": mode, "cutoff_day": cutoff}


def upsert_app_config(budget_mode: str, cutoff_day: int) -> None:
    uid = require_user_id()

    budget_mode = str(budget_mode or "calendar").strip().lower()
    if budget_mode not in {"calendar", "budget"}:
        budget_mode = "calendar"

    cutoff_day = int(cutoff_day)
    cutoff_day = max(1, min(31, cutoff_day))

    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO app_config (user_id, budget_mode, cutoff_day)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                budget_mode = excluded.budget_mode,
                cutoff_day = excluded.cutoff_day
            """,
            (uid, budget_mode, cutoff_day),
        )


def fetch_transacoes() -> pd.DataFrame:
    uid = require_user_id()
    with db_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                id, data, descricao, categoria, tipo, valor, pagamento, status,
                origem, cc_compra_id, cc_parcela_num, cc_parcela_total, ref_month
            FROM transacoes
            WHERE user_id = ?
            """,
            conn,
            params=(uid,),
        )

    if df.empty:
        return df

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["descricao"] = df["descricao"].astype(str).fillna("")
    df["categoria"] = df["categoria"].astype(str).fillna("Outros")
    df["tipo"] = df["tipo"].astype(str).fillna("Despesa")
    df["pagamento"] = df["pagamento"].astype(str).fillna("PIX")
    df["status"] = df["status"].astype(str).fillna("Pago")
    df["origem"] = df.get("origem", "").astype(str).fillna("")
    df["ref_month"] = df.get("ref_month", "").astype(str).fillna("")

    # normaliza valores numéricos
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0).astype(float)
    df["cc_compra_id"] = pd.to_numeric(df.get("cc_compra_id"), errors="coerce").fillna(pd.NA)
    df["cc_parcela_num"] = pd.to_numeric(df.get("cc_parcela_num"), errors="coerce").fillna(pd.NA)
    df["cc_parcela_total"] = pd.to_numeric(df.get("cc_parcela_total"), errors="coerce").fillna(pd.NA)
    return df


def fetch_orcamentos() -> pd.DataFrame:
    uid = require_user_id()
    with db_conn() as conn:
        df = pd.read_sql_query(
            "SELECT categoria, valor_teto FROM orcamentos WHERE user_id = ?",
            conn,
            params=(uid,),
        )

    if df.empty:
        return df

    df["categoria"] = df["categoria"].astype(str)
    df["valor_teto"] = pd.to_numeric(df["valor_teto"], errors="coerce").fillna(0.0).astype(float)
    return df


def insert_transacao(
    data: date,
    descricao: str,
    categoria: str,
    tipo: str,
    valor: float,
    pagamento: str,
    status: str,
) -> None:
    uid = require_user_id()
    data_str = pd.Timestamp(data).strftime("%Y-%m-%d")
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO transacoes (user_id, data, descricao, categoria, tipo, valor, pagamento, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (uid, data_str, descricao.strip(), categoria.strip(), tipo, float(valor), pagamento, status),
        )


def insert_transacao_extra(
    data: date,
    descricao: str,
    categoria: str,
    tipo: str,
    valor: float,
    pagamento: str,
    status: str,
    origem: Optional[str] = None,
    cc_compra_id: Optional[int] = None,
    cc_parcela_num: Optional[int] = None,
    cc_parcela_total: Optional[int] = None,
    ref_month: Optional[str] = None,
) -> None:
    uid = require_user_id()
    data_str = pd.Timestamp(data).strftime("%Y-%m-%d")
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO transacoes (
                user_id,
                data, descricao, categoria, tipo, valor, pagamento, status,
                origem, cc_compra_id, cc_parcela_num, cc_parcela_total, ref_month
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uid,
                data_str,
                descricao.strip(),
                categoria.strip(),
                tipo,
                float(valor),
                pagamento,
                status,
                (origem or "").strip(),
                int(cc_compra_id) if cc_compra_id is not None else None,
                int(cc_parcela_num) if cc_parcela_num is not None else None,
                int(cc_parcela_total) if cc_parcela_total is not None else None,
                (ref_month or "").strip(),
            ),
        )


def delete_transacoes(ids: Iterable[int]) -> int:
    uid = require_user_id()
    ids = [int(i) for i in ids if str(i).strip().isdigit()]
    if not ids:
        return 0
    with db_conn() as conn:
        cur = conn.cursor()
        cur.executemany("DELETE FROM transacoes WHERE id = ? AND user_id = ?", [(i, uid) for i in ids])
        return cur.rowcount


def update_transacao(row: Dict) -> None:
    uid = require_user_id()
    with db_conn() as conn:
        conn.execute(
            """
            UPDATE transacoes
            SET data = ?, descricao = ?, categoria = ?, tipo = ?, valor = ?, pagamento = ?, status = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                row["data"],
                row["descricao"],
                row["categoria"],
                row["tipo"],
                float(row["valor"]),
                row["pagamento"],
                row["status"],
                int(row["id"]),
                uid,
            ),
        )


def upsert_orcamento(categoria: str, valor_teto: float) -> None:
    uid = require_user_id()
    categoria = categoria.strip()
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO orcamentos (user_id, categoria, valor_teto)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, categoria) DO UPDATE SET valor_teto = excluded.valor_teto
            """,
            (uid, categoria, float(valor_teto)),
        )


def delete_orcamento(categoria: str) -> None:
    uid = require_user_id()
    with db_conn() as conn:
        conn.execute("DELETE FROM orcamentos WHERE categoria = ? AND user_id = ?", (categoria, uid))


def fetch_cc_compras() -> pd.DataFrame:
    uid = require_user_id()
    with db_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT id, data_compra, descricao, categoria, tipo_compra, total, parcelas, ativo, created_at, fim
            FROM cc_compras
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            conn,
            params=(uid,),
        )
    if df.empty:
        return df
    df["data_compra"] = pd.to_datetime(df["data_compra"], errors="coerce")
    df["descricao"] = df["descricao"].astype(str).fillna("")
    df["categoria"] = df["categoria"].astype(str).fillna("Outros")
    df["tipo_compra"] = df["tipo_compra"].astype(str).fillna("avista")
    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0).astype(float)
    df["parcelas"] = pd.to_numeric(df["parcelas"], errors="coerce").fillna(pd.NA)
    df["ativo"] = pd.to_numeric(df["ativo"], errors="coerce").fillna(1).astype(int)
    df["fim"] = pd.to_datetime(df.get("fim"), errors="coerce")
    return df


def insert_cc_compra(
    data_compra: date,
    descricao: str,
    categoria: str,
    tipo_compra: str,
    total: float,
    parcelas: Optional[int],
) -> None:
    uid = require_user_id()
    data_str = pd.Timestamp(data_compra).strftime("%Y-%m-%d")
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO cc_compras (user_id, data_compra, descricao, categoria, tipo_compra, total, parcelas, ativo)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                uid,
                data_str,
                descricao.strip(),
                categoria.strip(),
                tipo_compra.strip(),
                float(total),
                int(parcelas) if parcelas is not None else None,
            ),
        )


def set_cc_compra_ativo(cc_compra_id: int, ativo: int) -> None:
    uid = require_user_id()
    with db_conn() as conn:
        conn.execute(
            "UPDATE cc_compras SET ativo = ?, fim = CASE WHEN ? = 1 THEN NULL ELSE fim END WHERE id = ? AND user_id = ?",
            (int(ativo), int(ativo), int(cc_compra_id), uid),
        )


def fetch_cc_amortizacoes(cc_compra_id: Optional[int] = None) -> pd.DataFrame:
    uid = require_user_id()
    with db_conn() as conn:
        if cc_compra_id is None:
            df = pd.read_sql_query(
                """
                SELECT id, cc_compra_id, data, parcelas_amortizadas, desconto, valor_pago, pagamento, observacao, created_at
                FROM cc_amortizacoes
                WHERE user_id = ?
                ORDER BY id DESC
                """,
                conn,
                params=(uid,),
            )
        else:
            df = pd.read_sql_query(
                """
                SELECT id, cc_compra_id, data, parcelas_amortizadas, desconto, valor_pago, pagamento, observacao, created_at
                FROM cc_amortizacoes
                WHERE user_id = ? AND cc_compra_id = ?
                ORDER BY id DESC
                """,
                conn,
                params=(uid, int(cc_compra_id)),
            )
    if df.empty:
        return df
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["parcelas_amortizadas"] = pd.to_numeric(df["parcelas_amortizadas"], errors="coerce").fillna(0).astype(int)
    df["desconto"] = pd.to_numeric(df["desconto"], errors="coerce").fillna(0.0).astype(float)
    df["valor_pago"] = pd.to_numeric(df["valor_pago"], errors="coerce").fillna(0.0).astype(float)
    df["pagamento"] = df["pagamento"].astype(str).fillna("PIX")
    df["observacao"] = df["observacao"].astype(str).fillna("")
    return df


def insert_cc_amortizacao(
    cc_compra_id: int,
    data_: date,
    parcelas_amortizadas: int,
    desconto: float,
    valor_pago: float,
    pagamento: str,
    observacao: str = "",
) -> None:
    uid = require_user_id()
    data_str = pd.Timestamp(data_).strftime("%Y-%m-%d")
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO cc_amortizacoes (user_id, cc_compra_id, data, parcelas_amortizadas, desconto, valor_pago, pagamento, observacao)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uid,
                int(cc_compra_id),
                data_str,
                int(parcelas_amortizadas),
                float(desconto),
                float(valor_pago),
                pagamento,
                observacao.strip(),
            ),
        )


def _cc_transacao_exists(cc_compra_id: int, parcela_num: Optional[int], ref_month: str) -> bool:
    uid = require_user_id()
    with db_conn() as conn:
        cur = conn.execute(
            """
            SELECT 1
            FROM transacoes
            WHERE user_id = ?
              AND cc_compra_id = ?
              AND (cc_parcela_num IS ? OR cc_parcela_num = ?)
              AND ref_month = ?
              AND pagamento = 'Crédito'
            LIMIT 1
            """,
            (uid, int(cc_compra_id), parcela_num, parcela_num, ref_month),
        )
        return cur.fetchone() is not None



def ensure_cc_generated(horizon_months: int = 24) -> None:
    """Gera parcelas/recorrências para meses futuros, para aparecerem no Dashboard."""
    compras = fetch_cc_compras()
    if compras.empty:
        return

    today = date.today()
    horizon_end = add_months(date(today.year, today.month, 1), int(horizon_months))

    for _, r in compras.iterrows():
        try:
            if int(r.get("ativo", 1)) != 1:
                continue

            cc_id = int(r["id"])

            data_compra_ts = r.get("data_compra")
            if pd.isna(data_compra_ts):
                continue
            data_compra = pd.to_datetime(data_compra_ts).date()

            desc = str(r.get("descricao") or "").strip() or "Compra no cartão"
            cat = str(r.get("categoria") or "").strip() or "Outros"
            tipo_compra = str(r.get("tipo_compra") or "").strip().lower()
            total = float(r.get("total") or 0.0)

            fim_date = None
            fim_ts = r.get("fim", None)
            if pd.notna(fim_ts):
                try:
                    fim_date = pd.to_datetime(fim_ts).date()
                except Exception:
                    fim_date = None

            if tipo_compra == "recorrente":
                d = date(data_compra.year, data_compra.month, 1)
                k = 0
                while d <= horizon_end:
                    ref = f"{d.year:04d}-{d.month:02d}"
                    dia = clamp_day_to_month(d.year, d.month, data_compra.day)
                    dt_lanc = date(d.year, d.month, dia)

                    if fim_date is not None and dt_lanc > fim_date:
                        break

                    if not _cc_transacao_exists(cc_id, None, ref):
                        insert_transacao_extra(
                            data=dt_lanc,
                            descricao=f"{desc} (Recorrente)",
                            categoria=cat,
                            tipo="Despesa",
                            valor=total,
                            pagamento="Crédito",
                            status="Pendente",
                            origem="Cartão",
                            cc_compra_id=cc_id,
                            cc_parcela_num=None,
                            cc_parcela_total=None,
                            ref_month=ref,
                        )
                    d = add_months(d, 1)
                    k += 1
                    if k > 600:
                        break

            elif tipo_compra in {"avista", "parcelado"}:
                parcelas_raw = r.get("parcelas")
                parcelas = int(parcelas_raw) if pd.notna(parcelas_raw) else 1
                parcelas = max(1, parcelas)
                vals = split_installments(total, parcelas)

                for i in range(parcelas):
                    parc_num = i + 1
                    dt = add_months(data_compra, i)
                    ref = f"{dt.year:04d}-{dt.month:02d}"

                    if fim_date is not None and dt > fim_date:
                        break

                    if not _cc_transacao_exists(cc_id, parc_num, ref):
                        sufixo = f" ({parc_num}/{parcelas})" if parcelas > 1 else ""
                        insert_transacao_extra(
                            data=dt,
                            descricao=f"{desc}{sufixo}",
                            categoria=cat,
                            tipo="Despesa",
                            valor=float(vals[i]),
                            pagamento="Crédito",
                            status="Pendente",
                            origem="Cartão",
                            cc_compra_id=cc_id,
                            cc_parcela_num=parc_num,
                            cc_parcela_total=parcelas,
                            ref_month=ref,
                        )
            else:
                # tipo desconhecido
                continue
        except Exception:
            # não quebra o app por uma compra inválida
            continue


# ============================================================
# Fixos (Débito Automático)
# ============================================================

def parse_ym(yyyy_mm: str) -> Tuple[int, int]:
    y, m = yyyy_mm.split("-")
    return int(y), int(m)


def next_month_key(yyyy_mm: str) -> str:
    y, m = parse_ym(yyyy_mm)
    return month_key(pd.Timestamp(add_months(date(y, m, 1), 1)))


def last_day_date_of_ym(yyyy_mm: str) -> date:
    y, m = parse_ym(yyyy_mm)
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, last_day)


def fetch_fixos(include_inactive: bool = True) -> pd.DataFrame:
    uid = require_user_id()
    where = "" if include_inactive else "AND ativo = 1"
    with db_conn() as conn:
        df = pd.read_sql_query(
            f"""
            SELECT id, descricao, categoria, valor, pagamento, status, dia, inicio_ym, intervalo_m, fim_ym, fim, paused_from_ym, paused_until_ym, ativo, created_at, cancelado_em
            FROM lancamentos_fixos
            WHERE user_id = ?
            {where}
            ORDER BY ativo DESC, id DESC
            """,
            conn,
            params=(uid,),
        )
    return df


def _fixo_transacao_exists(fixo_id: int, ref_month: str) -> bool:
    uid = require_user_id()
    with db_conn() as conn:
        cur = conn.execute(
            """
            SELECT 1
            FROM transacoes
            WHERE user_id = ?
              AND fixo_id = ?
              AND ref_month = ?
            LIMIT 1
            """,
            (uid, int(fixo_id), str(ref_month)),
        )
        return cur.fetchone() is not None



def ensure_fixos_generated(horizon_months: int = 24) -> None:
    """Gera (idempotente) as despesas fixas em Débito Automático para meses futuros."""
    try:
        fixos = fetch_fixos(include_inactive=False)
    except Exception:
        return
    if fixos.empty:
        return

    today = date.today()
    horizon_end = add_months(date(today.year, today.month, 1), int(horizon_months))

    for _, r in fixos.iterrows():
        try:
            if int(r.get("ativo", 1)) != 1:
                continue

            fixo_id = int(r["id"])
            desc = str(r.get("descricao") or "").strip()
            cat = str(r.get("categoria") or "").strip()
            valor = float(r.get("valor") or 0.0)
            pagamento = str(r.get("pagamento") or "Débito Automático").strip() or "Débito Automático"
            status = str(r.get("status") or "Pago").strip() or "Pago"
            dia = int(r.get("dia") or 1)

            inicio_ym = str(r.get("inicio_ym") or "").strip()
            if not inicio_ym:
                continue

            intervalo_m = int(r.get("intervalo_m") or 1)
            intervalo_m = max(1, min(24, intervalo_m))

            fim_date = None
            fim_ts = r.get("fim", None)
            if pd.notna(fim_ts):
                try:
                    fim_date = pd.to_datetime(fim_ts).date()
                except Exception:
                    fim_date = None

            fim_ym = str(r.get("fim_ym") or "").strip() or None
            if fim_date is None and fim_ym:
                try:
                    fim_date = last_day_date_of_ym(fim_ym)
                except Exception:
                    fim_date = None

            paused_from = str(r.get("paused_from_ym") or "").strip() or None
            paused_until = str(r.get("paused_until_ym") or "").strip() or None

            # começa no mês de início e segue até o horizonte (ou fim)
            y0, m0 = parse_ym(inicio_ym)
            d = date(y0, m0, 1)

            k = 0
            while d <= horizon_end:
                ref = f"{d.year:04d}-{d.month:02d}"

                # pausa (pula meses dentro do intervalo pausado)
                if paused_from and paused_until and (paused_from <= ref <= paused_until):
                    d = add_months(d, intervalo_m)
                    k += 1
                    if k > 900:
                        break
                    continue

                dia_clamped = clamp_day_to_month(d.year, d.month, dia)
                dt_lanc = date(d.year, d.month, dia_clamped)

                if fim_date is not None and dt_lanc > fim_date:
                    break

                if not _fixo_transacao_exists(fixo_id, ref):
                    with db_conn() as conn:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO transacoes (
                                user_id,
                                data, descricao, categoria, tipo, valor, pagamento, status,
                                origem, ref_month, fixo_id
                            )
                            VALUES (?, ?, ?, ?, 'Despesa', ?, ?, ?, 'Débito Automático', ?, ?)
                            """,
                            (
                                require_user_id(),
                                pd.Timestamp(dt_lanc).strftime("%Y-%m-%d"),
                                desc,
                                cat,
                                float(valor),
                                pagamento,
                                status,
                                ref,
                                int(fixo_id),
                            ),
                        )

                d = add_months(d, intervalo_m)
                k += 1
                if k > 900:
                    break
        except Exception:
            continue




def create_fixo_debito_automatico(
    data_: date,
    descricao: str,
    categoria: str,
    valor: float,
    intervalo_m: int = 1,
    fim: Optional[date] = None,
) -> int:
    """Cria regra fixa (Débito Automático) e registra o lançamento do mês."""
    uid = require_user_id()
    ref = month_key(pd.Timestamp(data_))
    dia = int(data_.day)
    intervalo_m = max(1, min(24, int(intervalo_m or 1)))

    fim_ym = month_key(pd.Timestamp(fim)) if fim else None
    fim_iso = pd.Timestamp(fim).strftime("%Y-%m-%d") if fim else None

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO lancamentos_fixos (
                user_id, descricao, categoria, valor,
                pagamento, status, dia, inicio_ym,
                intervalo_m, fim_ym, fim,
                ativo
            )
            VALUES (?, ?, ?, ?, 'Débito Automático', 'Pago', ?, ?, ?, ?, ?, 1)
            """,
            (uid, descricao.strip(), categoria.strip(), float(valor), dia, ref, intervalo_m, fim_ym, fim_iso),
        )
        fixo_id = int(cur.lastrowid)

        # lançamento do mês de criação já entra na tabela transacoes com fixo_id
        conn.execute(
            """
            INSERT OR IGNORE INTO transacoes (
                user_id,
                data, descricao, categoria, tipo, valor, pagamento, status,
                origem, ref_month, fixo_id
            )
            VALUES (?, ?, ?, ?, 'Despesa', ?, 'Débito Automático', 'Pago', 'Débito Automático', ?, ?)
            """,
            (uid, pd.Timestamp(data_).strftime("%Y-%m-%d"), descricao.strip(), categoria.strip(), float(valor), ref, fixo_id),
        )

    # Gera meses futuros para aparecer no Dashboard
    ensure_fixos_generated(horizon_months=24)
    return fixo_id




def set_fixo_ativo(fixo_id: int, ativo: int, stop_after_ym: Optional[str] = None) -> int:
    """Ativa/desativa um fixo. Se desativar, remove lançamentos futuros a partir do próximo mês."""
    uid = require_user_id()
    removed = 0

    with db_conn() as conn:
        cur = conn.cursor()
        if int(ativo) == 0:
            stop_after_ym = (stop_after_ym or month_key(pd.Timestamp(date.today())))
            next_ym = next_month_key(stop_after_ym)
            fim_dt = last_day_date_of_ym(stop_after_ym)

            # marca como inativo e define fim (mês atual do stop_after)
            cur.execute(
                """
                UPDATE lancamentos_fixos
                SET ativo = 0,
                    fim_ym = ?,
                    fim = ?,
                    cancelado_em = CURRENT_TIMESTAMP,
                    paused_from_ym = NULL,
                    paused_until_ym = NULL
                WHERE id = ? AND user_id = ?
                """,
                (stop_after_ym, pd.Timestamp(fim_dt).strftime("%Y-%m-%d"), int(fixo_id), uid),
            )

            # remove lançamentos futuros (próximo mês em diante)
            cur.execute(
                """
                DELETE FROM transacoes
                WHERE user_id = ?
                  AND fixo_id = ?
                  AND ref_month >= ?
                """,
                (uid, int(fixo_id), str(next_ym)),
            )
            removed = int(cur.rowcount)
        else:
            # reativar: limpa fim e gera novamente
            cur.execute(
                """
                UPDATE lancamentos_fixos
                SET ativo = 1,
                    fim_ym = NULL,
                    fim = NULL,
                    paused_from_ym = NULL,
                    paused_until_ym = NULL
                WHERE id = ? AND user_id = ?
                """,
                (int(fixo_id), uid),
            )

    if int(ativo) == 1:
        ensure_fixos_generated(horizon_months=24)

    return removed



def pause_fixo(fixo_id: int, pause_months: int, current_ym: str) -> Tuple[str, str, int]:
    """Pausa um fixo por X meses (a partir do próximo mês do período atual) e remove lançamentos já gerados nesse intervalo."""
    uid = require_user_id()
    pause_months = max(1, int(pause_months))
    start_ym = next_month_key(current_ym)

    y, m = parse_ym(start_ym)
    start_date = date(y, m, 1)
    until_date = add_months(start_date, pause_months - 1)
    until_ym = month_key(pd.Timestamp(until_date))

    removed = 0
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE lancamentos_fixos
            SET paused_from_ym = ?, paused_until_ym = ?
            WHERE id = ? AND user_id = ?
            """,
            (start_ym, until_ym, int(fixo_id), uid),
        )
        cur.execute(
            """
            DELETE FROM transacoes
            WHERE user_id = ?
              AND fixo_id = ?
              AND ref_month >= ?
              AND ref_month <= ?
            """,
            (uid, int(fixo_id), start_ym, until_ym),
        )
        removed = int(cur.rowcount)

    return start_ym, until_ym, removed


def resume_fixo(fixo_id: int) -> None:
    """Remove pausa e regenera meses futuros."""
    uid = require_user_id()
    with db_conn() as conn:
        conn.execute(
            """
            UPDATE lancamentos_fixos
            SET paused_from_ym = NULL, paused_until_ym = NULL
            WHERE id = ? AND user_id = ?
            """,
            (int(fixo_id), uid),
        )
    ensure_fixos_generated(horizon_months=24)


def convert_transacao_to_fixo(
    transacao_id: int,
    intervalo_m: int = 1,
    fim: Optional[date] = None,
) -> int:
    """Transforma uma despesa existente em conta fixa por Débito Automático."""
    uid = require_user_id()
    intervalo_m = max(1, min(24, int(intervalo_m or 1)))

    with db_conn() as conn:
        row = conn.execute(
            """
            SELECT id, data, descricao, categoria, tipo, valor, pagamento, origem, status
            FROM transacoes
            WHERE id = ? AND user_id = ?
            """,
            (int(transacao_id), uid),
        ).fetchone()

        if not row:
            raise ValueError("Lançamento não encontrado.")
        if str(row["tipo"]).strip().lower() != "despesa":
            raise ValueError("Apenas despesas podem virar Débito Automático.")
        if str(row.get("pagamento") or "").strip().lower() == "crédito":
            raise ValueError("Compras no crédito devem ser geridas no Cartão de Crédito.")
        if str(row.get("origem") or "").strip().lower() == "cartão":
            raise ValueError("Compras do cartão devem ser geridas no Cartão de Crédito.")

        dt = pd.to_datetime(row["data"]).date() if row["data"] else date.today()
        descricao = str(row["descricao"] or "").strip() or "Despesa"
        categoria = str(row["categoria"] or "").strip() or "Outros"
        valor = float(row["valor"] or 0.0)

    fixo_id = create_fixo_debito_automatico(
        data_=dt,
        descricao=descricao,
        categoria=categoria,
        valor=valor,
        intervalo_m=intervalo_m,
        fim=fim,
    )

    ref = month_key(pd.Timestamp(dt))
    with db_conn() as conn:
        conn.execute(
            """
            UPDATE transacoes
            SET pagamento = 'Débito Automático',
                status = 'Pago',
                origem = 'Débito Automático',
                ref_month = ?,
                fixo_id = ?
            WHERE id = ? AND user_id = ?
            """,
            (ref, int(fixo_id), int(transacao_id), uid),
        )

    ensure_fixos_generated(horizon_months=24)
    return int(fixo_id)

def cancel_cc_recorrencia(cc_compra_id: int, stop_after_ym: str) -> int:
    """Cancela recorrência do cartão para não aparecer nos próximos meses (remove pendentes futuros)."""
    uid = require_user_id()
    next_ym = next_month_key(stop_after_ym)
    removed = 0
    with db_conn() as conn:
        cur = conn.cursor()
        # desativa compra e marca fim (último dia do mês selecionado)
        fim_dt = last_day_date_of_ym(stop_after_ym)
        cur.execute(
            "UPDATE cc_compras SET ativo = 0, fim = ? WHERE id = ? AND user_id = ?",
            (pd.Timestamp(fim_dt).strftime("%Y-%m-%d"), int(cc_compra_id), uid),
        )
        # remove lançamentos pendentes futuros
        cur.execute(
            """
            DELETE FROM transacoes
            WHERE user_id = ?
              AND cc_compra_id = ?
              AND origem = 'Cartão'
              AND pagamento = 'Crédito'
              AND status = 'Pendente'
              AND ref_month >= ?
            """,
            (uid, int(cc_compra_id), str(next_ym)),
        )
        removed = int(cur.rowcount)
    return removed

def amortizar_compra(
    cc_compra_id: int,
    data_pagamento: date,
    parcelas_amortizar: int,
    desconto: float,
    pagamento: str,
    observacao: str = "",
) -> Tuple[int, float]:
    """
    1) Marca N parcelas futuras como "Amortizado"
    2) Cria uma transação 'Despesa' agora (pagamento adiantado) NÃO no 'Crédito'
    3) Registra em cc_amortizacoes
    """
    uid = require_user_id()

    parcelas_amortizar = max(1, int(parcelas_amortizar))
    desconto = max(0.0, float(desconto))

    with db_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT id, data, descricao, valor, cc_parcela_num, cc_parcela_total
            FROM transacoes
            WHERE user_id = ?
              AND cc_compra_id = ?
              AND pagamento = 'Crédito'
              AND tipo = 'Despesa'
              AND status = 'Pendente'
            ORDER BY date(data) ASC
            """,
            conn,
            params=(uid, int(cc_compra_id)),
        )

    if df.empty:
        return 0, 0.0

    df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0).astype(float)
    take = df.head(parcelas_amortizar).copy()
    amort_ids = take["id"].astype(int).tolist()

    valor_bruto = float(take["valor"].sum())
    desconto = min(desconto, valor_bruto)
    valor_pago = round(valor_bruto - desconto, 2)

    with db_conn() as conn:
        conn.executemany(
            "UPDATE transacoes SET status = 'Amortizado' WHERE id = ? AND user_id = ?",
            [(int(i), uid) for i in amort_ids],
        )

    desc_base = str(take.iloc[0]["descricao"]) if not take.empty else "Compra no cartão"
    insert_transacao_extra(
        data=data_pagamento,
        descricao=f"Amortização do cartão: {desc_base} (adiantou {len(amort_ids)} parcela(s))",
        categoria="Cartão de Crédito",
        tipo="Despesa",
        valor=valor_pago,
        pagamento=pagamento,  # PIX/Transferência/Débito...
        status="Pago",
        origem="Amortização",
        cc_compra_id=int(cc_compra_id),
        cc_parcela_num=None,
        cc_parcela_total=None,
        ref_month=month_key(pd.Timestamp(data_pagamento)),
    )

    insert_cc_amortizacao(
        cc_compra_id=int(cc_compra_id),
        data_=data_pagamento,
        parcelas_amortizadas=len(amort_ids),
        desconto=desconto,
        valor_pago=valor_pago,
        pagamento=pagamento,
        observacao=observacao or "",
    )

    return len(amort_ids), valor_pago


def _inv_signed_value(tipo: str, valor_abs: float) -> float:
    v = abs(float(valor_abs))
    return v if str(tipo).strip() == "Aporte" else -v


def _sum_aportes(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df.loc[df["valor"] > 0, "valor"].sum())


def _sum_retiradas(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(abs(df.loc[df["valor"] < 0, "valor"].sum()))


def _sum_liquido(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["valor"].sum())


def fetch_invest_config() -> Dict[str, float]:
    uid = require_user_id()
    with db_conn() as conn:
        row = conn.execute(
            "SELECT aporte_planejado, cdi_anual, pct_cdi FROM investimentos_config WHERE user_id = ?",
            (uid,),
        ).fetchone()

        if not row:
            # cria defaults para este usuário copiando do legado (0)
            row0 = conn.execute(
                "SELECT aporte_planejado, cdi_anual, pct_cdi FROM investimentos_config WHERE user_id = ?",
                (LEGACY_USER_ID,),
            ).fetchone()
            ap0 = float((row0["aporte_planejado"] if row0 else 0.0) or 0.0)
            cdi0 = float((row0["cdi_anual"] if row0 else 0.0) or 0.0)
            pct0 = float((row0["pct_cdi"] if row0 else 100.0) or 100.0)
            conn.execute(
                "INSERT OR IGNORE INTO investimentos_config (user_id, aporte_planejado, cdi_anual, pct_cdi) VALUES (?, ?, ?, ?)",
                (uid, ap0, cdi0, pct0),
            )
            row = conn.execute(
                "SELECT aporte_planejado, cdi_anual, pct_cdi FROM investimentos_config WHERE user_id = ?",
                (uid,),
            ).fetchone()

    if not row:
        return {"aporte_planejado": 0.0, "cdi_anual": 0.0, "pct_cdi": 100.0}

    return {
        "aporte_planejado": float(row["aporte_planejado"] or 0.0),
        "cdi_anual": float(row["cdi_anual"] or 0.0),
        "pct_cdi": float(row["pct_cdi"] or 100.0),
    }


def upsert_invest_config(aporte_planejado: float, cdi_anual: float, pct_cdi: float) -> None:
    uid = require_user_id()
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO investimentos_config (user_id, aporte_planejado, cdi_anual, pct_cdi)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                aporte_planejado = excluded.aporte_planejado,
                cdi_anual = excluded.cdi_anual,
                pct_cdi = excluded.pct_cdi
            """,
            (uid, float(aporte_planejado), float(cdi_anual), float(pct_cdi)),
        )


def fetch_invest_aportes() -> pd.DataFrame:
    uid = require_user_id()
    with db_conn() as conn:
        cols = _table_cols(conn, "investimentos_aportes")
        if "tipo" in cols:
            df = pd.read_sql_query(
                """
                SELECT id, data, produto, tipo, valor, observacao, created_at
                FROM investimentos_aportes
                WHERE user_id = ?
                ORDER BY date(data) DESC, id DESC
                """,
                conn,
                params=(uid,),
            )
        else:
            df = pd.read_sql_query(
                """
                SELECT id, data, produto, valor, observacao, created_at
                FROM investimentos_aportes
                WHERE user_id = ?
                ORDER BY date(data) DESC, id DESC
                """,
                conn,
                params=(uid,),
            )
            if not df.empty:
                df["tipo"] = df["valor"].apply(lambda v: "Retirada" if safe_float(v) < 0 else "Aporte")

    if df.empty:
        return df

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["produto"] = df["produto"].astype(str).fillna("")
    df["observacao"] = df["observacao"].astype(str).fillna("")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0).astype(float)
    df["tipo"] = df["tipo"].astype(str).fillna("").replace("", "Aporte")
    # normaliza tipo conforme lista
    df["tipo"] = df["tipo"].apply(lambda t: t if t in INV_MOV_TYPES else ("Retirada" if t.lower().startswith("ret") else "Aporte"))
    return df


def insert_invest_movimento(d: date, produto: str, tipo: str, valor_abs: float, observacao: str = "") -> None:
    uid = require_user_id()
    data_str = pd.Timestamp(d).strftime("%Y-%m-%d")
    tipo = tipo if tipo in INV_MOV_TYPES else "Aporte"
    v_signed = _inv_signed_value(tipo, float(valor_abs))

    with db_conn() as conn:
        cols = _table_cols(conn, "investimentos_aportes")
        if "tipo" in cols:
            conn.execute(
                """
                INSERT INTO investimentos_aportes (user_id, data, produto, tipo, valor, observacao)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (uid, data_str, str(produto or "").strip(), tipo, float(v_signed), str(observacao or "").strip()),
            )
        else:
            conn.execute(
                """
                INSERT INTO investimentos_aportes (user_id, data, produto, valor, observacao)
                VALUES (?, ?, ?, ?, ?)
                """,
                (uid, data_str, str(produto or "").strip(), float(v_signed), str(observacao or "").strip()),
            )


def insert_invest_aporte(d: date, produto: str, valor: float, observacao: str = "") -> None:
    """Compat: mantém chamadas antigas como Aporte."""
    insert_invest_movimento(d=d, produto=produto, tipo="Aporte", valor_abs=float(valor), observacao=observacao)


def update_invest_movimento(row: Dict) -> None:
    uid = require_user_id()
    tipo = row.get("tipo", "Aporte")
    tipo = tipo if tipo in INV_MOV_TYPES else "Aporte"
    valor_abs = safe_float(row.get("valor", 0.0))
    v_signed = _inv_signed_value(tipo, valor_abs)

    with db_conn() as conn:
        cols = _table_cols(conn, "investimentos_aportes")
        if "tipo" in cols:
            conn.execute(
                """
                UPDATE investimentos_aportes
                SET data = ?, produto = ?, tipo = ?, valor = ?, observacao = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    row["data"],
                    str(row.get("produto") or "").strip(),
                    tipo,
                    float(v_signed),
                    str(row.get("observacao") or "").strip(),
                    int(row["id"]),
                    uid,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE investimentos_aportes
                SET data = ?, produto = ?, valor = ?, observacao = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    row["data"],
                    str(row.get("produto") or "").strip(),
                    float(v_signed),
                    str(row.get("observacao") or "").strip(),
                    int(row["id"]),
                    uid,
                ),
            )


def update_invest_aporte(row: Dict) -> None:
    """Compat: aceita tanto o payload antigo quanto o novo."""
    if "tipo" not in row:
        row = dict(row)
        row["tipo"] = "Aporte"
    update_invest_movimento(row)


def delete_invest_aportes(ids: Iterable[int]) -> int:
    uid = require_user_id()
    ids = [int(i) for i in ids if str(i).strip().isdigit()]
    if not ids:
        return 0
    with db_conn() as conn:
        cur = conn.cursor()
        cur.executemany("DELETE FROM investimentos_aportes WHERE id = ? AND user_id = ?", [(i, uid) for i in ids])
        return cur.rowcount


def compute_invest_annual_rate_decimal(cdi_anual_pct: float, pct_cdi: float) -> float:
    """Ex: CDI=11.5 (% a.a.), pct=100 => 0.115. pct=120 => 0.138."""
    cdi = max(0.0, float(cdi_anual_pct)) / 100.0
    pct = max(0.0, float(pct_cdi)) / 100.0
    return cdi * pct


def estimate_portfolio_value(movs: pd.DataFrame, annual_rate_decimal: float, as_of: pd.Timestamp) -> float:
    """
    Valor estimado em uma data 'as_of' (fim do período selecionado),
    aplicando juros diários aproximados por movimento.

    ✅ Retiradas: são armazenadas como valor negativo (efeito reduz o patrimônio a partir daquela data).
    """
    if movs.empty:
        return 0.0

    r_d = annual_to_daily_rate(annual_rate_decimal)
    as_of = pd.Timestamp(as_of).normalize()

    total = 0.0
    for _, r in movs.iterrows():
        v = float(r["valor"])
        dt = pd.Timestamp(r["data"]).normalize()
        days = max(0, (as_of - dt).days)
        if r_d <= 0:
            total += v
        else:
            total += v * ((1 + r_d) ** days)
    return float(total)


def avg_monthly_netflow(movs: pd.DataFrame, as_of: pd.Timestamp) -> float:
    """Média mensal do fluxo líquido (aportes - retiradas) desde o primeiro movimento até 'as_of'."""
    if movs.empty:
        return 0.0
    as_of = pd.Timestamp(as_of)
    df = movs[movs["data"] <= as_of].copy()
    if df.empty:
        return 0.0
    first = pd.Timestamp(df["data"].min()).to_period("M").to_timestamp()
    end_m = as_of.to_period("M").to_timestamp()
    months = (end_m.year - first.year) * 12 + (end_m.month - first.month) + 1
    months = max(1, months)
    return float(df["valor"].sum()) / months


def build_invest_timeline(
    movs_all: pd.DataFrame,
    annual_rate: float,
    start_yyyy_mm: str,
    end_yyyy_mm: str,
    planned_monthly: float,
    mode: str,
    cutoff_day: int,
) -> pd.DataFrame:
    """Série mensal com acumulado real/planejado (com rendimento) + acumulado de aportes/retiradas."""
    months = month_range(start_yyyy_mm, end_yyyy_mm)
    r_m = annual_to_monthly_rate(annual_rate) if annual_rate > 0 else 0.0
    planned_monthly = float(planned_monthly)

    rows = []
    for idx, mm in enumerate(months):
        _, end_ts = period_bounds(mm, mode, cutoff_day)
        sub = movs_all[movs_all["data"] <= end_ts].copy()

        real_val = estimate_portfolio_value(sub, annual_rate, end_ts) if not sub.empty else 0.0
        aporte_acum = _sum_aportes(sub)
        retirada_acum = _sum_retiradas(sub)
        net_acum = _sum_liquido(sub)

        n = idx + 1  # 1º mês => n=1
        if planned_monthly <= 0:
            plan_val = 0.0
        elif annual_rate <= 0:
            plan_val = planned_monthly * n
        else:
            plan_val = fv_with_contrib(0.0, planned_monthly, r_m, n)

        rows.append(
            {
                "mes": mm,
                "data": end_ts,
                "patrimonio_real": float(real_val),
                "patrimonio_planejado": float(plan_val),
                "aporte_acumulado": float(aporte_acum),
                "retirada_acumulada": float(retirada_acum),
                "liquido_acumulado": float(net_acum),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Mock data
# ============================================================


def generate_mock_data() -> None:
    """Insere transações fictícias se o banco estiver vazio."""
    df = fetch_transacoes()
    if not df.empty:
        return

    today = date.today()
    mock = [
        {
            "data": date(today.year, today.month, 1),
            "descricao": "Salário",
            "categoria": "Salário",
            "tipo": "Receita",
            "valor": 8100.0,
            "pagamento": "PIX",
            "status": "Pago",
        },
        {
            "data": date(today.year, today.month, 3),
            "descricao": "Supermercado",
            "categoria": "Alimentação",
            "tipo": "Despesa",
            "valor": 350.0,
            "pagamento": "Crédito",
            "status": "Pago",
        },
        {
            "data": date(today.year, today.month, 6),
            "descricao": "Aluguel",
            "categoria": "Moradia",
            "tipo": "Despesa",
            "valor": 2000.0,
            "pagamento": "Transferência",
            "status": "Pago",
        },
        {
            "data": date(today.year, today.month, 10),
            "descricao": "Uber",
            "categoria": "Transporte",
            "tipo": "Despesa",
            "valor": 45.9,
            "pagamento": "Débito",
            "status": "Pago",
        },
        {
            "data": date(today.year, today.month, 15),
            "descricao": "Freelance",
            "categoria": "Renda Extra",
            "tipo": "Receita",
            "valor": 1200.0,
            "pagamento": "PIX",
            "status": "Pago",
        },
    ]
    for r in mock:
        insert_transacao(**r)


# ============================================================
# UI building blocks
# ============================================================

CSS = """
<style>
:root{
  --card-bg: #ffffff;
  --card-border: rgba(0,0,0,0.06);
  --shadow: 0 6px 18px rgba(0,0,0,0.06);
  --text: rgba(0,0,0,0.86);
  --muted: rgba(0,0,0,0.55);
}
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
.kpi{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px; box-shadow: var(--shadow);
      padding: 16px 16px; width: 100%; }
.kpi .title{ font-size: 0.92rem; color: var(--muted); margin-bottom: 6px; }
.kpi .value{ font-size: 1.6rem; font-weight: 700; color: var(--text); line-height: 1.2; }
.kpi .hint{ font-size: 0.85rem; color: var(--muted); margin-top: 6px; }
.panel{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px; box-shadow: var(--shadow);
        padding: 14px 16px; }
.pbar-wrap{ width: 100%; height: 14px; background: rgba(0,0,0,0.06); border-radius: 999px; overflow: hidden; }
.pbar{ height: 100%; border-radius: 999px; }
.pbar.green{ background: #22c55e; }
.pbar.yellow{ background: #f59e0b; }
.pbar.red{ background: #ef4444; }
.badge{ display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 600; }
.badge.ok{ background: rgba(34,197,94,0.12); color: #16a34a; }
.badge.warn{ background: rgba(245,158,11,0.14); color: #b45309; }
.badge.danger{ background: rgba(239,68,68,0.12); color: #dc2626; }
.small-muted{ color: var(--muted); font-size: 0.9rem; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
</style>
"""


def kpi_card(title: str, value: str, hint: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi">
          <div class="title">{title}</div>
          <div class="value">{value}</div>
          {f'<div class="hint">{hint}</div>' if hint else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def progress_bar_html(percent_0_100: float, color: str) -> str:
    p = max(0.0, min(100.0, float(percent_0_100)))
    return f"<div class='pbar-wrap'><div class='pbar {color}' style='width:{p:.2f}%'></div></div>"


def badge_html(text: str, level: str) -> str:
    level = level if level in {"ok", "warn", "danger"} else "ok"
    return f"<span class='badge {level}'>{text}</span>"


# ============================================================
# App state
# ============================================================


def ensure_state():
    if "card_limit" not in st.session_state:
        st.session_state.card_limit = 10000.0

    today = date.today()
    if "selected_month" not in st.session_state:
        st.session_state.selected_month = f"{today.year:04d}-{today.month:02d}"
    if "selected_year" not in st.session_state:
        st.session_state.selected_year = int(today.year)

    # novo: visão do dashboard
    if "view_scope" not in st.session_state:
        st.session_state.view_scope = "month"  # 'month' | 'year'

    # modo de período
    if "analysis_mode" not in st.session_state:
        st.session_state.analysis_mode = "calendar"  # 'calendar' | 'budget'
    if "budget_cutoff_day" not in st.session_state:
        st.session_state.budget_cutoff_day = 28
    if "_period_cfg_loaded" not in st.session_state:
        st.session_state._period_cfg_loaded = False


# ============================================================
# Data logic
# ============================================================


def available_months(
    df_all: pd.DataFrame,
    inv_all: Optional[pd.DataFrame] = None,
    future_months: int = 12,
    past_months: int = 18,
    mode: str = "calendar",
    cutoff_day: int = 28,
) -> List[str]:
    """Permite navegar meses futuros e passados; no modo orçamento usa virada pelo cutoff_day."""
    today = date.today()
    today_key = f"{today.year:04d}-{today.month:02d}"

    end_future = add_months(date(today.year, today.month, 1), future_months)
    end_key = f"{end_future.year:04d}-{end_future.month:02d}"

    dates = pd.Series([], dtype="datetime64[ns]")
    if not df_all.empty and "data" in df_all.columns:
        dates = pd.concat([dates, pd.to_datetime(df_all["data"], errors="coerce")], ignore_index=True)

    if inv_all is not None and not inv_all.empty and "data" in inv_all.columns:
        dates = pd.concat([dates, pd.to_datetime(inv_all["data"], errors="coerce")], ignore_index=True)

    dates = dates.dropna()

    if dates.empty:
        start_past = add_months(date(today.year, today.month, 1), -past_months)
        start_key = f"{start_past.year:04d}-{start_past.month:02d}"
        return month_range(start_key, end_key)

    mode = _normalize_mode(mode)
    if mode == "budget":
        keys = budget_month_key_from_datetime(dates, cutoff_day)
        keys = keys[keys != ""]
        min_k = keys.min()
        max_k = keys.max()
        max_key = max(max_k, end_key, today_key)
        return month_range(min_k, max_key)

    min_p = dates.min().to_period("M").strftime("%Y-%m")
    max_p = dates.max().to_period("M").strftime("%Y-%m")
    max_key = max(max_p, end_key, today_key)
    return month_range(min_p, max_key)


def available_years_from_months(months: List[str]) -> List[int]:
    ys = sorted({int(m.split("-")[0]) for m in months})
    if not ys:
        ys = [date.today().year]
    return ys


def clamp_month_in_available(yyyy_mm: str, months: List[str]) -> str:
    if not months:
        return yyyy_mm
    if yyyy_mm in months:
        return yyyy_mm
    # tenta manter mesmo ano se existir
    y = yyyy_mm.split("-")[0]
    same_year = [m for m in months if m.startswith(f"{y}-")]
    return (same_year[-1] if same_year else months[-1])


def compute_kpis(df_scope: pd.DataFrame, invest_aportes_scope: float, invest_net_scope: float) -> Dict[str, float]:
    """
    KPIs do período.

    Regras:
    - "Investimentos (aportes)" continua sendo exibido separado (somente entradas para investimento).
    - "Saldo" passa a representar o SALDO NO BANCO no período:
        saldo_banco = receitas - despesas - invest_net_scope

      Onde invest_net_scope é o valor líquido assinado dos movimentos em investimentos:
        + aportes -> positivo (saída de caixa)
        + retiradas -> negativo (entrada de caixa)

    Isso faz com que o KPI "Saldo" reflita o caixa no banco (resultado operacional menos o líquido investido).
    """
    receitas = 0.0
    despesas = 0.0

    if not df_scope.empty:
        receitas = float(df_scope.loc[df_scope["tipo"] == "Receita", "valor"].sum())
        despesas = float(df_scope.loc[df_scope["tipo"] == "Despesa", "valor"].sum())

    saldo_operacional = receitas - despesas
    saldo_banco = saldo_operacional - float(invest_net_scope)

    economia_pct = (saldo_banco / receitas * 100.0) if receitas > 0 else 0.0

    return {
        "receitas": float(receitas),
        "despesas": float(despesas),
        "saldo": float(saldo_banco),                  # ✅ Saldo do banco
        "saldo_operacional": float(saldo_operacional),
        "economia_pct": float(economia_pct),
        "invest": float(invest_aportes_scope),        # ✅ só aportes
        "invest_net": float(invest_net_scope),        # líquido (debug/insight)
    }


def _set_month_from_year_month(year: int, month: int):
    st.session_state.selected_year = int(year)
    st.session_state.selected_month = f"{int(year):04d}-{int(month):02d}"


def top_bar(months: List[str], mode: str, cutoff_day: int) -> Tuple[str, str, int]:
    def _shift_month(delta: int):
        mlist = st.session_state.get("_months_list", [])
        cur = st.session_state.get("selected_month")
        if not mlist or cur not in mlist:
            return

        i = mlist.index(cur) + int(delta)
        i = max(0, min(len(mlist) - 1, i))
        new_mm = mlist[i]

        st.session_state.selected_month = new_mm
        y2, m2 = map(int, new_mm.split("-"))
        st.session_state.selected_year = y2

        # ✅ MUITO IMPORTANTE: sincroniza os selectboxes
        st.session_state["_year_month_y"] = y2
        st.session_state["_year_month_m"] = m2

    col1, col2 = st.columns([2.1, 1.9])
    with col1:
        st.markdown(f"## {APP_TITLE}")

        if _normalize_mode(mode) == "budget":
            st.markdown(
                f"<div class='small-muted'>Modo: <b>Orçamento</b> (virada dia <b>{int(cutoff_day)}</b>)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='small-muted'>Modo: <b>Calendário</b> (mês normal)</div>", unsafe_allow_html=True)

    years = available_years_from_months(months)

    st.session_state.selected_month = clamp_month_in_available(st.session_state.selected_month, months)
    cur_y, cur_m = map(int, st.session_state.selected_month.split("-"))
    if st.session_state.selected_year not in years:
        st.session_state.selected_year = cur_y

    # ✅ init dos widgets (pra poder sincronizar via setas)
    if "_year_month_y" not in st.session_state:
        st.session_state["_year_month_y"] = cur_y
    if "_year_month_m" not in st.session_state:
        st.session_state["_year_month_m"] = cur_m

    with col2:
        cA, cB = st.columns([1, 1])
        with cA:
            scope_ui = st.radio(
                "Visão",
                ["Mês", "Ano"],
                horizontal=True,
                index=0 if st.session_state.view_scope == "month" else 1,
                key="_scope_radio",
            )
        with cB:
            st.write("")

        st.session_state.view_scope = "month" if scope_ui == "Mês" else "year"

        if st.session_state.view_scope == "year":
            y_sel = st.selectbox("Ano", years, index=years.index(st.session_state.selected_year), key="_year_sel")
            st.session_state.selected_year = int(y_sel)

        else:
            y_col, m_col, left_col, right_col = st.columns([1.2, 1.2, 0.45, 0.45])

            with y_col:
                y_sel = st.selectbox(
                    "Ano",
                    years,
                    index=years.index(int(st.session_state["_year_month_y"])) if int(st.session_state["_year_month_y"]) in years else years.index(cur_y),
                    key="_year_month_y",
                )

            months_in_year = [m for m in months if m.startswith(f"{int(y_sel):04d}-")]
            if not months_in_year:
                months_in_year = [f"{int(y_sel):04d}-01"]

            m_opts = [int(m.split("-")[1]) for m in months_in_year]
            if int(st.session_state["_year_month_m"]) not in m_opts:
                st.session_state["_year_month_m"] = m_opts[-1]

            with m_col:
                m_sel = st.selectbox(
                    "Mês",
                    m_opts,
                    index=m_opts.index(int(st.session_state["_year_month_m"])),
                    format_func=lambda mm: PT_MONTHS.get(int(mm), str(mm)),
                    key="_year_month_m",
                )

            # ✅ atualiza mês selecionado baseado nos selectboxes (quando usuário muda manualmente)
            st.session_state.selected_month = f"{int(y_sel):04d}-{int(m_sel):02d}"
            st.session_state.selected_year = int(y_sel)
            st.session_state.selected_month = clamp_month_in_available(st.session_state.selected_month, months)

            cur = st.session_state.selected_month
            idx = months.index(cur) if cur in months else (len(months) - 1)
            prev_ok = idx > 0
            next_ok = idx < (len(months) - 1)

            with left_col:
                st.button(
                    "◀",
                    use_container_width=True,
                    disabled=not prev_ok,
                    key="_prev_month",
                    on_click=_shift_month,
                    args=(-1,),
                )
            with right_col:
                st.button(
                    "▶",
                    use_container_width=True,
                    disabled=not next_ok,
                    key="_next_month",
                    on_click=_shift_month,
                    args=(+1,),
                )

    st.divider()
    return st.session_state.view_scope, st.session_state.selected_month, int(st.session_state.selected_year)

# ============================================================
# Dashboard (mensal + anual)
# ============================================================


def _period_label(scope: str, yyyy_mm: str, year: int) -> str:
    return month_label(yyyy_mm) if scope == "month" else f"Ano {int(year)}"


def _scope_filters(
    scope: str,
    df_all: pd.DataFrame,
    inv_all: pd.DataFrame,
    yyyy_mm: str,
    year: int,
    mode: str,
    cutoff_day: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """
    Retorna:
      - df_scope
      - inv_scope
      - inv_aportes (somente aportes, sempre >= 0)  -> usado no KPI/Gráfico "Investimentos (aportes)"
      - inv_net (líquido assinado: aportes positivos, retiradas negativas) -> usado no Saldo (Banco)

    Nota:
      - A tabela de investimentos pode armazenar valores já assinados (retirada negativa).
      - Quando a tabela não tem coluna "tipo" (versões antigas), tudo é tratado como aporte.
    """
    if scope == "month":
        df_scope = filter_df_by_period(df_all, "data", yyyy_mm, mode, cutoff_day)
        inv_scope = filter_df_by_period(inv_all, "data", yyyy_mm, mode, cutoff_day) if not inv_all.empty else inv_all
    else:
        df_scope = filter_df_by_year(df_all, "data", year, mode, cutoff_day)
        inv_scope = filter_df_by_year(inv_all, "data", year, mode, cutoff_day) if not inv_all.empty else inv_all

    inv_aportes = _sum_aportes(inv_scope) if not inv_scope.empty else 0.0
    inv_net = _sum_liquido(inv_scope) if not inv_scope.empty else 0.0
    return df_scope, inv_scope, float(inv_aportes), float(inv_net)


def _pie_receita_despesa_invest(receitas: float, despesas: float, invest: float, title: str):
    dfp = pd.DataFrame(
        {
            "Grupo": ["Receitas", "Despesas", "Investimentos (aportes)"],
            "Valor": [max(0.0, float(receitas)), max(0.0, float(despesas)), max(0.0, float(invest))],
        }
    )
    # evita gráfico “vazio”
    if dfp["Valor"].sum() <= 0:
        st.info("Sem valores para exibir no gráfico.")
        return
    fig = px.pie(dfp, names="Grupo", values="Valor", hole=0.45)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380, title=title)
    st.plotly_chart(fig, use_container_width=True)


def screen_dashboard(df_all: pd.DataFrame, scope: str, yyyy_mm: str, year: int, mode: str, cutoff_day: int) -> None:
    inv_all = fetch_invest_aportes()
    df_scope, inv_scope, inv_aportes_scope, inv_net_scope = _scope_filters(scope, df_all, inv_all, yyyy_mm, year, mode, cutoff_day)

    if df_scope.empty and abs(inv_net_scope) <= 0:
        st.info("Sem dados para este período")
        st.markdown(
            "<div class='panel'>Dica: vá em <b>Lançamentos</b>, <b>Cartão de Crédito</b> ou <b>Investimentos</b> e adicione seus dados.</div>",
            unsafe_allow_html=True,
        )
        return

    kpis = compute_kpis(df_scope, inv_aportes_scope, inv_net_scope)
    label_scope = _period_label(scope, yyyy_mm, year)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    saldo_color = "#16a34a" if kpis["saldo"] >= 0 else "#dc2626"
    saldo_value = f"<span style='color:{saldo_color}'>{brl(kpis['saldo'])}</span>"

    with c1:
        st.markdown(
            "<div class='kpi'><div class='title'>Saldo (Banco)</div><div class='value'>" + saldo_value + "</div>"
            f"<div class='hint'>{label_scope}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        kpi_card("Total de Receitas", brl(kpis["receitas"]), hint=label_scope)
    with c3:
        kpi_card("Total de Despesas", brl(kpis["despesas"]), hint=label_scope)
    with c4:
        kpi_card("Investimentos (aportes)", brl(kpis["invest"]), hint=label_scope)
    with c5:
        hint = "(Saldo no banco / Receitas)" if kpis["receitas"] > 0 else "Sem receitas"
        kpi_card("Economia", f"{kpis['economia_pct']:.1f}%", hint=hint)

    st.write("")

    # Linha principal (categoria + distribuição)
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Gastos por Categoria")
        despesas = df_scope[df_scope["tipo"] == "Despesa"].copy() if not df_scope.empty else pd.DataFrame()
        if despesas.empty:
            st.info("Sem despesas neste período")
        else:
            g = despesas.groupby("categoria", as_index=False)["valor"].sum().sort_values("valor", ascending=False)
            fig = px.pie(g, names="categoria", values="valor", hole=0.55)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Distribuição (Receitas x Despesas x Investimentos)")
        _pie_receita_despesa_invest(kpis["receitas"], kpis["despesas"], kpis["invest"], title="")

    st.write("")

    # Visão anual: gráficos mensais + insights
    if scope == "year":
        st.markdown("### Evolução mensal no ano")

        # Série mensal: receitas/despesas
        months_year = [f"{int(year):04d}-{m:02d}" for m in range(1, 13)]
        rows = []
        for mm in months_year:
            dfm = filter_df_by_period(df_all, "data", mm, mode, cutoff_day)
            invm = filter_df_by_period(inv_all, "data", mm, mode, cutoff_day) if not inv_all.empty else inv_all
            receitas = float(dfm.loc[dfm["tipo"] == "Receita", "valor"].sum()) if not dfm.empty else 0.0
            despesas = float(dfm.loc[dfm["tipo"] == "Despesa", "valor"].sum()) if not dfm.empty else 0.0
            invest_aportes = float(_sum_aportes(invm)) if not invm.empty else 0.0
            inv_net = float(_sum_liquido(invm)) if not invm.empty else 0.0
            saldo_banco = (receitas - despesas) - inv_net
            rows.append(
                {
                    "mes": mm,
                    "mes_label": PT_MONTHS[int(mm.split("-")[1])],
                    "receitas": receitas,
                    "despesas": despesas,
                    "investimentos": invest_aportes,
                    "saldo": saldo_banco,
                }
            )
        ts = pd.DataFrame(rows)

        cA, cB = st.columns([1.35, 1.0])

        with cA:
            long = ts.melt(id_vars=["mes", "mes_label"], value_vars=["receitas", "despesas", "saldo"], var_name="serie", value_name="valor")
            long["serie"] = long["serie"].replace({"receitas": "Receitas", "despesas": "Despesas", "saldo": "Saldo"})
            fig = px.line(long, x="mes_label", y="valor", color="serie", markers=True)
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Mês", yaxis_title="R$")
            st.plotly_chart(fig, use_container_width=True)

        with cB:
            fig2 = px.bar(ts, x="mes_label", y="investimentos")
            fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Mês", yaxis_title="Investimentos (R$)")
            st.plotly_chart(fig2, use_container_width=True)

        # Insights (consultoria financeira)
        st.markdown("### Insights do ano (rápido e prático)")

        despesas_ano = df_scope[df_scope["tipo"] == "Despesa"].copy() if not df_scope.empty else pd.DataFrame()
        receitas_ano = df_scope[df_scope["tipo"] == "Receita"].copy() if not df_scope.empty else pd.DataFrame()

        top_cat = None
        if not despesas_ano.empty:
            gc = despesas_ano.groupby("categoria", as_index=False)["valor"].sum().sort_values("valor", ascending=False)
            top_cat = gc.iloc[0].to_dict()

        month_max_spend = ts.sort_values("despesas", ascending=False).iloc[0] if not ts.empty else None
        month_max_invest = ts.sort_values("investimentos", ascending=False).iloc[0] if not ts.empty else None

        i1, i2, i3 = st.columns(3)
        with i1:
            if top_cat:
                st.markdown(
                    "<div class='panel'>"
                    "<div class='small-muted'>Maior categoria de gasto</div>"
                    f"<div style='font-size:1.2rem;font-weight:900'>{top_cat['categoria']}</div>"
                    f"<div class='small-muted'>Total: <b>{brl(top_cat['valor'])}</b></div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div class='panel'><div class='small-muted'>Maior categoria de gasto</div><div><b>Sem despesas</b></div></div>", unsafe_allow_html=True)

        with i2:
            if month_max_spend is not None:
                st.markdown(
                    "<div class='panel'>"
                    "<div class='small-muted'>Mês com maior despesa</div>"
                    f"<div style='font-size:1.2rem;font-weight:900'>{month_max_spend['mes_label']}</div>"
                    f"<div class='small-muted'>Despesas: <b>{brl(month_max_spend['despesas'])}</b></div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div class='panel'><div class='small-muted'>Mês com maior despesa</div><div><b>—</b></div></div>", unsafe_allow_html=True)

        with i3:
            if month_max_invest is not None:
                st.markdown(
                    "<div class='panel'>"
                    "<div class='small-muted'>Mês com maior investimento</div>"
                    f"<div style='font-size:1.2rem;font-weight:900'>{month_max_invest['mes_label']}</div>"
                    f"<div class='small-muted'>Aportes: <b>{brl(month_max_invest['investimentos'])}</b></div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div class='panel'><div class='small-muted'>Mês com maior investimento</div><div><b>—</b></div></div>", unsafe_allow_html=True)

        if top_cat and (kpis["despesas"] > 0):
            share = (float(top_cat["valor"]) / float(kpis["despesas"])) * 100.0
            if share >= 35:
                st.warning(
                    f"**Concentração de gastos:** {top_cat['categoria']} representa **{share:.1f}%** das despesas do ano. "
                    "Isso é ótimo para atacar com ações diretas (renegociar, reduzir frequência, trocar fornecedor)."
                )

        if kpis["receitas"] > 0:
            inv_rate = (kpis["invest"] / kpis["receitas"]) * 100.0
            st.info(
                f"**Taxa de investimento no ano:** {inv_rate:.1f}% da sua receita foi para aportes. "
                "Se a meta for acelerar patrimônio, o próximo passo é estabilizar gastos fixos e criar regra automática de aporte."
            )

    # Tabelas recentes / escopo
    st.write("")

    if scope == "month" and not df_scope.empty:
        st.markdown("### Últimos lançamentos (mês)")
        recent = df_scope.sort_values("data", ascending=False).head(10).copy()
        recent["data"] = recent["data"].dt.strftime("%d/%m/%Y")
        recent["valor"] = recent["valor"].apply(brl)
        st.dataframe(
            recent[["id", "data", "descricao", "categoria", "tipo", "pagamento", "status", "valor"]],
            use_container_width=True,
            hide_index=True,
        )

    if not inv_scope.empty:
        st.markdown(f"### Movimentos em investimentos ({'ano' if scope == 'year' else 'período'})")
        inv_show = inv_scope.sort_values("data", ascending=False).copy()
        inv_show["data"] = inv_show["data"].dt.strftime("%d/%m/%Y")
        inv_show["valor"] = inv_show["valor"].abs().apply(brl)
        st.dataframe(inv_show[["id", "data", "tipo", "produto", "valor", "observacao"]], use_container_width=True, hide_index=True)


# ============================================================
# Screens (Lançamentos / Cartão / Planejamento / Investimentos)
# (mantidos do seu código com mínimos ajustes para compat)
# ============================================================


def screen_lancamentos(df_all: pd.DataFrame, yyyy_mm: str, mode: str, cutoff_day: int) -> None:
    st.markdown("### Histórico de Lançamentos")

    with st.expander("➕ Novo lançamento", expanded=False):
        cats = sorted(set(CATEGORIAS_DEFAULT) | set(df_all["categoria"].unique().tolist() if not df_all.empty else []))
        pags = sorted(set(PAGAMENTOS) | set(df_all["pagamento"].unique().tolist() if not df_all.empty else []))

        with st.form("form_add", clear_on_submit=True):
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                d = st.date_input("Data", value=date.today())
            with c2:
                desc = st.text_input("Descrição", placeholder="Ex: Supermercado, Aluguel, Salário")
            with c3:
                tipo = st.selectbox("Tipo", TIPOS)

            c4, c5, c6 = st.columns([2, 1, 1])
            with c4:
                cat_choice = st.selectbox("Categoria", cats + ["(Outra)"])
                categoria = cat_choice
                if cat_choice == "(Outra)":
                    categoria = st.text_input("Digite a categoria", placeholder="Ex: Academia, Internet, Luz")
            with c5:
                valor = st.text_input("Valor (R$)", placeholder="Ex: 199,90")
            with c6:
                pagamento = st.selectbox("Pagamento", pags)
            c7, c8 = st.columns([1, 2])
            is_auto = (pagamento == "Débito Automático") and (tipo == "Despesa")
            with c7:
                idx_pago = STATUS.index("Pago") if "Pago" in STATUS else 0
                status = st.selectbox("Status", STATUS, index=(idx_pago if is_auto else 0), disabled=is_auto)
                if is_auto:
                    status = "Pago"
            with c8:
                if is_auto:
                    st.caption("✅ **Débito Automático** será tratado como **conta fixa mensal** e entra como **Pago**. Você pode cancelar depois em 'Débito Automático (contas fixas)'.")
                else:
                    st.caption("Dica: parcelado/recorrente use **Cartão de Crédito**. Aportes/retiradas use **Investimentos**.")
            # Opções avançadas para Débito Automático (fixo)
            intervalo_m = 1
            fim_dt_opt: Optional[date] = None
            if is_auto:
                st.write("")
                cfa, cfb, cfc = st.columns([1.2, 1.2, 1.6])
                with cfa:
                    freq_label = st.selectbox("Frequência", ["Mensal", "Trimestral", "Anual"], index=0)
                    intervalo_m = {"Mensal": 1, "Trimestral": 3, "Anual": 12}.get(freq_label, 1)
                with cfb:
                    use_fim = st.checkbox("Definir data de fim", value=False)
                with cfc:
                    if use_fim:
                        fim_dt_opt = st.date_input("Data de fim", value=date.today())
                        if fim_dt_opt and fim_dt_opt < d:
                            st.warning("A data de fim está antes da data de início. Ajuste para evitar que o fixo não gere meses futuros.")
                    else:
                        st.caption("Sem data de fim (gera indefinidamente).")

            submitted = st.form_submit_button("Salvar lançamento")

            if submitted:
                valor_f = safe_float(valor)
                if not desc.strip():
                    st.error("Descrição é obrigatória")
                elif not str(categoria).strip():
                    st.error("Categoria é obrigatória")
                elif valor_f <= 0:
                    st.error("Valor deve ser maior que zero")
                else:
                    try:
                        if (pagamento == "Débito Automático") and (tipo == "Despesa"):
                            create_fixo_debito_automatico(d, desc, str(categoria), float(valor_f), intervalo_m=intervalo_m, fim=fim_dt_opt)
                            st.success("Conta fixa em Débito Automático criada! Próximos meses já foram gerados.")
                        else:
                            insert_transacao(d, desc, str(categoria), tipo, valor_f, pagamento, status)
                            st.success("Lançamento salvo!")

                        st.rerun()
                    except Exception as e:
                        st.error(f"Falha ao salvar: {e}")

    st.write("")


    with st.expander("🔁 Débito Automático (contas fixas)", expanded=False):
        fixos = fetch_fixos(include_inactive=True)
        if fixos.empty:
            st.info(
                "Nenhuma conta fixa em Débito Automático cadastrada ainda. "
                "Crie uma em 'Novo lançamento' escolhendo Pagamento = Débito Automático."
            )
        else:
            show = fixos.copy()
            show["ativo"] = show["ativo"].apply(lambda x: "Sim" if int(x) == 1 else "Não")
            show["valor"] = show["valor"].apply(brl)
            show["frequência"] = show["intervalo_m"].apply(lambda n: {1: "Mensal", 3: "Trimestral", 12: "Anual"}.get(int(n), f"{int(n)}m"))

            st.dataframe(
                show[
                    [
                        "id",
                        "ativo",
                        "descricao",
                        "categoria",
                        "valor",
                        "dia",
                        "inicio_ym",
                        "frequência",
                        "fim_ym",
                        "paused_from_ym",
                        "paused_until_ym",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            ids = show["id"].tolist()
            c1, c2, c3 = st.columns([1.2, 1.4, 2])
            with c1:
                fixo_sel = st.selectbox("Selecionar fixo (ID)", ids, index=0, key="fixo_sel")

            row = fixos[fixos["id"] == int(fixo_sel)].iloc[0]
            ativo_now = int(row.get("ativo", 1))

            with c2:
                if ativo_now == 1:
                    if st.button("⛔ Cancelar (parar a partir do próximo mês)", use_container_width=True, key=f"fixo_cancel_{fixo_sel}"):
                        try:
                            removed = set_fixo_ativo(int(fixo_sel), 0, stop_after_ym=yyyy_mm)
                            st.success(f"Cancelado. Removidos {removed} lançamento(s) futuro(s).")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Falha ao cancelar: {e}")

                    paused_from = str(row.get("paused_from_ym") or "").strip()
                    paused_until = str(row.get("paused_until_ym") or "").strip()

                    if paused_from and paused_until:
                        st.info(f"⏸️ Pausado de **{paused_from}** até **{paused_until}**.")
                        if st.button("▶️ Retomar (voltar a gerar)", use_container_width=True, key=f"fixo_resume_{fixo_sel}"):
                            try:
                                resume_fixo(int(fixo_sel))
                                st.success("Pausa removida. Próximos meses foram gerados novamente.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Falha ao retomar: {e}")
                    else:
                        pm = st.number_input(
                            "Pausar por quantos meses?",
                            min_value=1,
                            max_value=24,
                            value=1,
                            step=1,
                            key=f"fixo_pause_m_{fixo_sel}",
                        )
                        if st.button("⏸️ Pausar", use_container_width=True, key=f"fixo_pause_btn_{fixo_sel}"):
                            try:
                                start_ym, until_ym, removed2 = pause_fixo(int(fixo_sel), int(pm), current_ym=yyyy_mm)
                                st.success(
                                    f"Pausado de {start_ym} até {until_ym}. "
                                    f"Removidos {removed2} lançamento(s) já gerado(s) nesse intervalo."
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"Falha ao pausar: {e}")
                else:
                    if st.button("✅ Reativar", use_container_width=True, key=f"fixo_reactivate_{fixo_sel}"):
                        try:
                            set_fixo_ativo(int(fixo_sel), 1)
                            st.success("Reativado. Próximos meses foram gerados novamente.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Falha ao reativar: {e}")

            with c3:
                st.caption(
                    "⚠️ **Cancelar** remove lançamentos **pendentes/futuros** a partir do **próximo mês** "
                    "(o histórico do mês atual/anteriores fica). "
                    "**Pausar** remove apenas por um intervalo e volta automaticamente depois (ou você pode retomar)."
                )

    with st.expander("🧲 Transformar despesa em Débito Automático", expanded=False):
        st.caption(
            "Pegue uma despesa já lançada e transforme em **conta fixa** (Débito Automático). "
            "A partir daí, os próximos meses serão gerados automaticamente e você poderá pausar/cancelar."
        )
        cand = df_all.copy()
        if not cand.empty:
            cand["data"] = pd.to_datetime(cand["data"], errors="coerce")

        if cand.empty:
            st.info("Não há lançamentos disponíveis.")
        else:
            cand2 = cand[
                (cand["tipo"] == "Despesa")
                & (cand["pagamento"].astype(str).str.lower() != "crédito")
                & (~cand.get("origem", "").astype(str).str.lower().eq("cartão"))
            ].copy()

            if cand2.empty:
                st.info("Nenhuma despesa elegível para transformar (exclui cartão/crédito).")
            else:
                cand2 = cand2.sort_values("data", ascending=False).head(200)
                cand2["label"] = cand2.apply(
                    lambda rr: f"#{int(rr['id'])} — {rr['data'].date() if pd.notna(rr['data']) else ''} — {str(rr['descricao'])[:40]} — {brl(rr['valor'])}",
                    axis=1,
                )
                sel = st.selectbox("Selecione a despesa", cand2["label"].tolist(), index=0, key="conv_sel")
                tid = int(cand2[cand2["label"] == sel].iloc[0]["id"])

                cta, ctb, ctc = st.columns([1.2, 1.2, 1.6])
                with cta:
                    freq_label2 = st.selectbox("Frequência (fixo)", ["Mensal", "Trimestral", "Anual"], index=0, key="conv_freq")
                    intervalo_m2 = {"Mensal": 1, "Trimestral": 3, "Anual": 12}.get(freq_label2, 1)
                with ctb:
                    use_fim2 = st.checkbox("Definir data de fim", value=False, key="conv_usefim")
                with ctc:
                    fim2 = None
                    if use_fim2:
                        fim2 = st.date_input("Data de fim", value=date.today(), key="conv_fim")
                    else:
                        st.caption("Sem data de fim.")

                if st.button("✅ Transformar em Débito Automático", use_container_width=True, key="conv_btn"):
                    try:
                        new_id = convert_transacao_to_fixo(tid, intervalo_m=int(intervalo_m2), fim=fim2)
                        st.success(f"Transformado com sucesso! Fixo criado (ID {new_id}).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Falha ao transformar: {e}")

    f1, f2, f3, f4 = st.columns([1, 1, 1, 2])


    df = df_all.copy()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    with f1:
        only_month = st.toggle("Filtrar pelo período selecionado", value=True)
    with f2:
        tipo_f = st.selectbox("Tipo", ["Todos"] + TIPOS)
    with f3:
        status_f = st.selectbox("Status", ["Todos"] + STATUS)
    with f4:
        q = st.text_input("Buscar", placeholder="Procure por descrição ou categoria...")

    if only_month:
        df = filter_df_by_period(df, "data", yyyy_mm, mode, cutoff_day)
    if tipo_f != "Todos":
        df = df[df["tipo"] == tipo_f]
    if status_f != "Todos":
        df = df[df["status"] == status_f]
    if q.strip():
        qq = q.strip().lower()
        df = df[
            df["descricao"].str.lower().str.contains(qq, na=False)
            | df["categoria"].str.lower().str.contains(qq, na=False)
        ]

    if df.empty:
        st.info("Nenhum lançamento encontrado com os filtros atuais.")
        return

    st.markdown("#### Tabela (editar e excluir)")
    df_show = df.sort_values("data", ascending=False).copy()
    df_show["data"] = df_show["data"].dt.date
    df_show["Excluir"] = False

    cols_editable = ["data", "descricao", "categoria", "tipo", "valor", "pagamento", "status", "Excluir"]

    edited = st.data_editor(
        df_show[["id"] + cols_editable],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True, format="%d"),
            "data": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
            "tipo": st.column_config.SelectboxColumn("Tipo", options=TIPOS, required=True),
            "status": st.column_config.SelectboxColumn("Status", options=STATUS, required=True),
            "pagamento": st.column_config.SelectboxColumn(
                "Pagamento",
                options=sorted(set(PAGAMENTOS + (df_all["pagamento"].unique().tolist() if not df_all.empty else []))),
                required=True,
            ),
            "valor": st.column_config.NumberColumn("Valor", format="%.2f", step=0.5),
            "Excluir": st.column_config.CheckboxColumn("Excluir"),
        },
        key="editor_transacoes",
    )

    b1, b2, b3 = st.columns([1, 1, 2])

    with b1:
        if st.button("💾 Salvar alterações", use_container_width=True):
            try:
                base = df_show.set_index("id")
                ed = edited.set_index("id")

                changed_ids = []
                for tid in ed.index:
                    if tid not in base.index:
                        continue
                    row_base = base.loc[tid]
                    row_new = ed.loc[tid]

                    diffs = []
                    for col in ["data", "descricao", "categoria", "tipo", "valor", "pagamento", "status"]:
                        a = row_base[col]
                        b = row_new[col]
                        if col == "valor":
                            diffs.append((pd.isna(a) != pd.isna(b)) or (abs(float(a) - float(b)) > 1e-6))
                        else:
                            diffs.append(str(a).strip() != str(b).strip())
                    if any(diffs):
                        changed_ids.append(int(tid))

                if not changed_ids:
                    st.info("Nenhuma alteração detectada.")
                else:
                    for tid in changed_ids:
                        r = ed.loc[tid]
                        data_str = pd.Timestamp(r["data"]).strftime("%Y-%m-%d")
                        payload = {
                            "id": int(tid),
                            "data": data_str,
                            "descricao": str(r["descricao"]).strip(),
                            "categoria": str(r["categoria"]).strip() or "Outros",
                            "tipo": str(r["tipo"]).strip() if str(r["tipo"]).strip() in TIPOS else "Despesa",
                            "valor": safe_float(r["valor"]),
                            "pagamento": str(r["pagamento"]).strip() or "PIX",
                            "status": str(r["status"]).strip() if str(r["status"]).strip() in STATUS else "Pago",
                        }
                        update_transacao(payload)

                    st.success(f"Atualizado: {len(changed_ids)} lançamento(s).")
                    st.rerun()

            except Exception as e:
                st.error(f"Falha ao salvar alterações: {e}")

    with b2:
        if st.button("🗑️ Excluir selecionados", use_container_width=True):
            try:
                to_delete = edited.loc[edited["Excluir"] == True, "id"].tolist()  # noqa: E712
                if not to_delete:
                    st.warning("Selecione pelo menos um item na coluna 'Excluir'.")
                else:
                    n = delete_transacoes(to_delete)
                    st.success(f"Excluídos: {n} lançamento(s).")
                    st.rerun()
            except Exception as e:
                st.error(f"Falha ao excluir: {e}")

    with b3:
        st.markdown(
            "<div class='small-muted'>"
            "Você pode editar diretamente na tabela e salvar. Para excluir, marque <b>Excluir</b> e clique em <b>Excluir selecionados</b>."
            "</div>",
            unsafe_allow_html=True,
        )


def screen_cartao(df_all: pd.DataFrame, yyyy_mm: str, mode: str, cutoff_day: int) -> None:
    st.markdown("### Cartão de Crédito")

    ensure_cc_generated(horizon_months=24)
    df_all = fetch_transacoes()
    dfm = filter_df_by_period(df_all, "data", yyyy_mm, mode, cutoff_day)

    credit = dfm[
        (dfm["tipo"] == "Despesa") & (dfm["pagamento"].str.lower() == "crédito") & (dfm["status"] != "Amortizado")
    ].copy()
    fatura = float(credit["valor"].sum()) if not credit.empty else 0.0

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.number_input(
            "Limite total do cartão (R$)",
            min_value=0.0,
            value=float(st.session_state.card_limit),
            step=100.0,
            key="card_limit",
        )
        st.caption("Este valor fica salvo na sua sessão (não no banco).")

    limit_total = float(st.session_state.card_limit) if float(st.session_state.card_limit) > 0 else 1.0
    pct_used = (fatura / limit_total) * 100.0

    if pct_used < 50:
        color = "green"
        level = "ok"
    elif pct_used < 80:
        color = "yellow"
        level = "warn"
    else:
        color = "red"
        level = "danger"

    with col2:
        label = f"{pct_used:.1f}% do limite usado"
        st.markdown(
            "<div class='panel'>"
            f"<div class='small-muted'>Fatura no período</div><div style='font-size:1.8rem;font-weight:800'>{brl(fatura)}</div>"
            f"<div style='margin-top:10px'>{progress_bar_html(pct_used, color)}</div>"
            f"<div style='margin-top:8px'>{badge_html(label, level)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col3:
        disponivel = max(0.0, limit_total - fatura)
        st.markdown(
            "<div class='panel'>"
            f"<div class='small-muted'>Disponível</div><div style='font-size:1.8rem;font-weight:800'>{brl(disponivel)}</div>"
            "<div class='small-muted' style='margin-top:10px'>Dica: evite passar de 80% para manter folga no mês.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### Lançar compra no cartão (à vista / parcelado / recorrente)")
    compras = fetch_cc_compras()
    cats = sorted(set(CATEGORIAS_DEFAULT) | set(df_all["categoria"].unique().tolist() if not df_all.empty else []))

    with st.expander("➕ Nova compra no cartão", expanded=False):
        with st.form("form_cc_compra", clear_on_submit=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                d_compra = st.date_input("Data da compra", value=date.today())
            with c2:
                desc = st.text_input("Descrição", placeholder="Ex: Notebook, Mercado, Assinatura...")

            c3, c4, c5 = st.columns([2, 1, 1])
            with c3:
                cat = st.selectbox("Categoria", cats + ["(Outra)"])
                if cat == "(Outra)":
                    cat = st.text_input("Digite a categoria", placeholder="Ex: Streaming, Internet, Academia")
            with c4:
                modo_ui = st.selectbox("Tipo", ["À vista", "Parcelado", "Recorrente (mensal)"])
            with c5:
                valor_txt = st.text_input("Valor (R$)", placeholder="Ex: 1999,90")

            parcelas = None
            if modo_ui == "Parcelado":
                parcelas = st.slider("Número de parcelas", min_value=2, max_value=36, value=10)

            st.caption("⚙️ Parcelado/Recorrente gera lançamentos nos próximos meses automaticamente.")

            submit = st.form_submit_button("Salvar compra no cartão")
            if submit:
                v = safe_float(valor_txt)
                if not str(desc).strip():
                    st.error("Descrição é obrigatória.")
                elif not str(cat).strip():
                    st.error("Categoria é obrigatória.")
                elif v <= 0:
                    st.error("Valor deve ser maior que zero.")
                else:
                    try:
                        if modo_ui == "À vista":
                            tipo_compra = "avista"
                            parcelas_db = 1
                        elif modo_ui == "Parcelado":
                            tipo_compra = "parcelado"
                            parcelas_db = int(parcelas or 2)
                        else:
                            tipo_compra = "recorrente"
                            parcelas_db = None

                        insert_cc_compra(d_compra, str(desc), str(cat), tipo_compra, float(v), parcelas_db)
                        ensure_cc_generated(horizon_months=24)
                        st.success("Compra cadastrada! Parcelas/recorrências foram geradas.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Falha ao salvar compra: {e}")

    st.write("")
    st.markdown("#### Compras cadastradas (com parcelas / recorrências)")

    if compras.empty:
        st.info("Nenhuma compra cadastrada ainda. Use 'Nova compra no cartão' acima.")
    else:
        df_cc = df_all[(df_all["pagamento"].str.lower() == "crédito") & (df_all["cc_compra_id"].notna())].copy()
        df_cc["cc_compra_id"] = pd.to_numeric(df_cc["cc_compra_id"], errors="coerce")

        resumo = []
        for _, r in compras.iterrows():
            cc_id = int(r["id"])
            ativo = int(r.get("ativo", 1))
            tipo_compra = str(r["tipo_compra"]).lower()
            total = float(r["total"])
            parcelas = int(r["parcelas"]) if pd.notna(r["parcelas"]) else None

            sub = df_cc[df_cc["cc_compra_id"] == cc_id].copy()
            pend = sub[sub["status"] == "Pendente"]
            pagos = sub[sub["status"] == "Pago"]
            amort = sub[sub["status"] == "Amortizado"]

            if tipo_compra == "recorrente":
                info = f"Recorrente | pendentes no horizonte: {len(pend)}"
            else:
                info = f"{parcelas}x | {len(pagos)} pago | {len(amort)} amort. | {len(pend)} pend."

            resumo.append(
                {
                    "ID": cc_id,
                    "Ativo": "Sim" if ativo == 1 else "Não",
                    "Data": r["data_compra"].date() if pd.notna(r["data_compra"]) else "",
                    "Descrição": r["descricao"],
                    "Categoria": r["categoria"],
                    "Tipo": "Recorrente" if tipo_compra == "recorrente" else ("Parcelado" if parcelas and parcelas > 1 else "À vista"),
                    "Valor": brl(total),
                    "Status": info,
                }
            )

        df_resumo = pd.DataFrame(resumo)
        st.dataframe(df_resumo, use_container_width=True, hide_index=True)

        c_act1, c_act2, c_act3 = st.columns([1.2, 1.2, 2])
        with c_act1:
            ids = df_resumo["ID"].tolist()
            id_sel = st.selectbox("Selecionar compra (ID)", ids, index=0)
        with c_act2:
            row_sel = compras[compras["id"] == int(id_sel)].iloc[0]
            ativo_now = int(row_sel.get("ativo", 1))
            tipo_sel = str(row_sel.get("tipo_compra", "")).lower()

            label_btn = "⛔ Desativar" if ativo_now == 1 else "✅ Reativar"
            if st.button(label_btn, use_container_width=True):
                try:
                    if ativo_now == 1:
                        # Desativa a compra (para evitar novas gerações)
                        set_cc_compra_ativo(int(id_sel), 0)

                        # Para recorrente: remove lançamentos futuros pendentes para não aparecer nos próximos meses
                        if tipo_sel == "recorrente":
                            removed = cancel_cc_recorrencia(int(id_sel), stop_after_ym=yyyy_mm)
                            st.success(f"Recorrência cancelada. Removidos {removed} lançamento(s) futuro(s).")
                        else:
                            st.success("Compra desativada.")
                    else:
                        set_cc_compra_ativo(int(id_sel), 1)
                        ensure_cc_generated(horizon_months=24)
                        st.success("Reativado. Próximos meses foram gerados novamente.")

                    st.rerun()
                except Exception as e:
                    st.error(f"Falha: {e}")
        with c_act3:
            st.caption("Desativar não apaga histórico. Para **Recorrente**, remove os lançamentos **futuros pendentes** para não aparecer nos próximos meses. Para parcelado/à vista, apenas para a geração futura (se houver).")

    st.write("")
    st.markdown("#### Amortizar (adiantar) parcelas do cartão")

    compras_ativas = fetch_cc_compras()
    if compras_ativas.empty:
        st.info("Cadastre compras no cartão para habilitar amortização.")
    else:
        parc = compras_ativas[(compras_ativas["tipo_compra"].str.lower() == "parcelado") & (compras_ativas["ativo"] == 1)].copy()
        if parc.empty:
            st.info("Nenhuma compra parcelada ativa encontrada.")
        else:
            df_cc = df_all[(df_all["pagamento"].str.lower() == "crédito") & (df_all["cc_compra_id"].notna())].copy()
            df_cc["cc_compra_id"] = pd.to_numeric(df_cc["cc_compra_id"], errors="coerce")

            options = []
            meta = {}
            for _, r in parc.iterrows():
                cc_id = int(r["id"])
                sub = df_cc[(df_cc["cc_compra_id"] == cc_id) & (df_cc["status"] == "Pendente")].sort_values("data")
                if sub.empty:
                    continue
                per_parc = float(sub.iloc[0]["valor"])
                remaining = int(len(sub))
                label = f"#{cc_id} • {r['descricao']} • restantes: {remaining} • {brl(per_parc)}/parc"
                options.append(label)
                meta[label] = {"cc_id": cc_id, "remaining": remaining, "per_parc": per_parc, "desc": r["descricao"]}

            if not options:
                st.info("Não há parcelas pendentes para amortizar.")
            else:
                with st.expander("💳 Amortizar agora", expanded=False):
                    with st.form("form_amort", clear_on_submit=True):
                        opt = st.selectbox("Compra parcelada", options)
                        cc_id = meta[opt]["cc_id"]
                        remaining = meta[opt]["remaining"]
                        per_parc = meta[opt]["per_parc"]

                        c1, c2, c3 = st.columns([1, 1, 1])
                        with c1:
                            dt_pay = st.date_input("Data do pagamento", value=date.today())
                        with c2:
                            qtd = st.number_input("Qtd. parcelas a amortizar", min_value=1, max_value=remaining, value=min(2, remaining), step=1)
                        with c3:
                            desconto = st.number_input("Desconto (R$)", min_value=0.0, value=0.0, step=10.0)

                        c4, c5 = st.columns([1, 2])
                        with c4:
                            pag = st.selectbox("Como pagou", ["PIX", "Transferência", "Débito", "Dinheiro", "Boleto"])
                        with c5:
                            obs = st.text_input("Observação (opcional)", placeholder="Ex: desconto por antecipação; negociação; etc.")

                        bruto = round(float(qtd) * float(per_parc), 2)
                        desc_eff = min(float(desconto), bruto)
                        pagar = round(bruto - desc_eff, 2)
                        st.markdown(
                            f"<div class='panel'>"
                            f"<div class='small-muted'>Resumo</div>"
                            f"<div style='font-size:1.4rem;font-weight:900'>Pagar agora: {brl(pagar)}</div>"
                            f"<div class='small-muted'>Bruto: {brl(bruto)} | Desconto: {brl(desc_eff)} | Parcelas: {int(qtd)}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        ok = st.form_submit_button("Confirmar amortização")
                        if ok:
                            try:
                                amort_qtd, valor_pago = amortizar_compra(
                                    cc_compra_id=int(cc_id),
                                    data_pagamento=dt_pay,
                                    parcelas_amortizar=int(qtd),
                                    desconto=float(desc_eff),
                                    pagamento=str(pag),
                                    observacao=str(obs),
                                )
                                st.success(f"Amortizado: {amort_qtd} parcela(s). Pago agora: {brl(valor_pago)}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Falha ao amortizar: {e}")

    st.write("")
    st.markdown("#### Lançamentos no crédito (período)")
    if credit.empty:
        st.info("Nenhuma despesa no crédito neste período (ou foram amortizadas).")
    else:
        credit_show = credit.sort_values("data", ascending=False).copy()
        credit_show["data"] = credit_show["data"].dt.strftime("%d/%m/%Y")
        credit_show["valor"] = credit_show["valor"].apply(brl)
        st.dataframe(
            credit_show[["id", "data", "descricao", "categoria", "status", "valor"]],
            use_container_width=True,
            hide_index=True,
        )

    st.write("")
    st.markdown("#### Histórico de amortizações")
    am = fetch_cc_amortizacoes()
    if am.empty:
        st.info("Nenhuma amortização registrada ainda.")
    else:
        am_show = am.copy()
        am_show["data"] = am_show["data"].dt.strftime("%d/%m/%Y")
        am_show["desconto"] = am_show["desconto"].apply(brl)
        am_show["valor_pago"] = am_show["valor_pago"].apply(brl)
        st.dataframe(
            am_show[["id", "cc_compra_id", "data", "parcelas_amortizadas", "desconto", "valor_pago", "pagamento", "observacao"]],
            use_container_width=True,
            hide_index=True,
        )


def screen_planejamento(df_all: pd.DataFrame, yyyy_mm: str, mode: str, cutoff_day: int) -> None:
    st.markdown("### Planejamento")
    st.markdown("Defina tetos de gasto por categoria e acompanhe seu progresso no período.")

    dfm = filter_df_by_period(df_all, "data", yyyy_mm, mode, cutoff_day)
    orc = fetch_orcamentos()

    st.markdown("#### Tetos por categoria")
    cats_hist = sorted(set(CATEGORIAS_DEFAULT) | set(df_all["categoria"].unique().tolist() if not df_all.empty else []))

    with st.expander("✍️ Definir / ajustar teto", expanded=False):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            cat = st.selectbox("Categoria", cats_hist + ["(Outra)"], key="orc_cat_sel")
            if cat == "(Outra)":
                cat = st.text_input("Digite a categoria", key="orc_cat_txt")
        with c2:
            teto = st.number_input("Teto (R$)", min_value=0.0, value=500.0, step=50.0, key="orc_teto")
        with c3:
            if st.button("Salvar teto", use_container_width=True, key="orc_save"):
                if not str(cat).strip():
                    st.error("Informe a categoria")
                else:
                    try:
                        upsert_orcamento(str(cat), float(teto))
                        st.success("Teto salvo!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Falha ao salvar teto: {e}")

    if orc.empty:
        st.info("Você ainda não definiu nenhum teto. Use a seção acima para começar.")
        return

    orc_show = orc.sort_values("categoria").copy()
    orc_show["Excluir"] = False

    edited = st.data_editor(
        orc_show,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "categoria": st.column_config.TextColumn("Categoria", disabled=True),
            "valor_teto": st.column_config.NumberColumn("Teto (R$)", format="%.2f", step=50.0),
            "Excluir": st.column_config.CheckboxColumn("Excluir"),
        },
        key="editor_orcamentos",
    )

    b1, b2 = st.columns([1, 3])

    with b1:
        if st.button("💾 Salvar alterações", key="save_orc", use_container_width=True):
            try:
                base = orc_show.set_index("categoria")
                ed = edited.set_index("categoria")

                changed = []
                for cat in ed.index:
                    if cat not in base.index:
                        continue
                    a = float(base.loc[cat, "valor_teto"])
                    b = float(ed.loc[cat, "valor_teto"])
                    if abs(a - b) > 1e-6:
                        changed.append((cat, b))

                for cat, new_val in changed:
                    upsert_orcamento(cat, new_val)

                to_del = edited.loc[edited["Excluir"] == True, "categoria"].tolist()  # noqa: E712
                for cat in to_del:
                    delete_orcamento(str(cat))

                if changed or to_del:
                    st.success("Orçamentos atualizados!")
                    st.rerun()
                else:
                    st.info("Nenhuma alteração detectada.")

            except Exception as e:
                st.error(f"Falha ao atualizar: {e}")

    with b2:
        st.markdown("<div class='small-muted'>Acompanhe abaixo o progresso por categoria no período selecionado.</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Progresso no período")

    despesas = dfm[dfm["tipo"] == "Despesa"].copy() if not dfm.empty else pd.DataFrame(columns=df_all.columns)
    gastos_cat = despesas.groupby("categoria", as_index=False)["valor"].sum() if not despesas.empty else pd.DataFrame({"categoria": [], "valor": []})

    merged = orc.merge(gastos_cat, on="categoria", how="left").fillna({"valor": 0.0})
    merged["pct"] = merged.apply(
        lambda r: (float(r["valor"]) / float(r["valor_teto"]) * 100.0) if float(r["valor_teto"]) > 0 else 0.0,
        axis=1,
    )
    merged = merged.sort_values("pct", ascending=False)

    for _, r in merged.iterrows():
        cat = str(r["categoria"])
        teto = float(r["valor_teto"])
        gasto = float(r["valor"])
        pct = float(r["pct"])

        if teto <= 0:
            color = "yellow"
            level = "warn"
            status_txt = "Defina um teto"
        elif pct < 70:
            color = "green"
            level = "ok"
            status_txt = "Dentro do orçamento"
        elif pct < 100:
            color = "yellow"
            level = "warn"
            status_txt = "Atenção"
        else:
            color = "red"
            level = "danger"
            status_txt = "Estourou o teto"

        st.markdown(
            "<div class='panel' style='margin-bottom:10px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;gap:10px'>"
            f"  <div style='font-size:1.05rem;font-weight:800'>{cat}</div>"
            f"  <div>{badge_html(status_txt, level)}</div>"
            f"</div>"
            f"<div class='small-muted' style='margin-top:6px'>Gasto: <b>{brl(gasto)}</b> &nbsp;|&nbsp; Teto: <b>{brl(teto)}</b> &nbsp;|&nbsp; {pct:.1f}%</div>"
            f"<div style='margin-top:10px'>{progress_bar_html(pct, color)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    if not merged.empty:
        top = merged.iloc[0]
        if float(top["valor_teto"]) > 0 and float(top["pct"]) >= 80:
            st.warning(
                f"Categoria mais pressionada: **{top['categoria']}** ({top['pct']:.1f}% do teto). "
                "Considere revisar o teto ou reduzir gastos nesta categoria."
            )


def screen_investimentos(yyyy_mm: str, mode: str, cutoff_day: int) -> None:
    st.markdown("### Investimentos (Renda Fixa)")
    st.markdown("Registre **aportes e retiradas**, acompanhe o **acumulado** e veja projeções com base no CDI.")

    cfg = fetch_invest_config()
    movs_all = fetch_invest_aportes()

    _, as_of = period_bounds(yyyy_mm, mode, cutoff_day)
    movs_upto = movs_all[movs_all["data"] <= as_of].copy() if not movs_all.empty else movs_all

    movs_mes = filter_df_by_period(movs_all, "data", yyyy_mm, mode, cutoff_day) if not movs_all.empty else movs_all
    aporte_mes = _sum_aportes(movs_mes)
    retirada_mes = _sum_retiradas(movs_mes)
    liquido_mes = _sum_liquido(movs_mes)

    aporte_total = _sum_aportes(movs_upto) if not movs_upto.empty else 0.0
    retirada_total = _sum_retiradas(movs_upto) if not movs_upto.empty else 0.0
    liquido_total = _sum_liquido(movs_upto) if not movs_upto.empty else 0.0

    with st.expander("⚙️ Configurações (Meta + CDI)", expanded=False):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])
        with c1:
            aporte_planejado = st.number_input("Meta de aporte mensal (R$)", min_value=0.0, value=float(cfg["aporte_planejado"]), step=50.0)
        with c2:
            cdi_anual = st.number_input("CDI anual (%)", min_value=0.0, value=float(cfg["cdi_anual"]), step=0.1, help="Ex: 11.5")
        with c3:
            pct_cdi = st.number_input("Rendimento (% do CDI)", min_value=0.0, value=float(cfg["pct_cdi"]), step=1.0, help="Ex: 100, 120...")
        with c4:
            if st.button("💾 Salvar configurações", use_container_width=True):
                try:
                    upsert_invest_config(float(aporte_planejado), float(cdi_anual), float(pct_cdi))
                    st.success("Configurações salvas!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Falha ao salvar: {e}")

        st.caption("Observação: estimativa simplificada (não considera IR/IOF).")

    annual_rate = compute_invest_annual_rate_decimal(cfg["cdi_anual"], cfg["pct_cdi"])
    patrimonio_est = estimate_portfolio_value(movs_upto, annual_rate, as_of) if not movs_upto.empty else 0.0

    planned = float(cfg["aporte_planejado"])
    pct_meta = (aporte_mes / planned * 100.0) if planned > 0 else 0.0
    if planned <= 0:
        color = "yellow"
        level = "warn"
        meta_txt = "Defina uma meta"
    elif pct_meta < 70:
        color = "yellow"
        level = "warn"
        meta_txt = "Abaixo da meta"
    elif pct_meta < 100:
        color = "green"
        level = "ok"
        meta_txt = "Quase lá"
    else:
        color = "green"
        level = "ok"
        meta_txt = "Meta batida!"

    if not movs_upto.empty:
        start_mm = movs_upto["data"].min().to_period("M").strftime("%Y-%m")
    else:
        start_mm = yyyy_mm

    timeline_tmp = build_invest_timeline(
        movs_all=movs_upto if not movs_upto.empty else pd.DataFrame(columns=["data", "valor"]),
        annual_rate=annual_rate,
        start_yyyy_mm=start_mm,
        end_yyyy_mm=yyyy_mm,
        planned_monthly=planned,
        mode=mode,
        cutoff_day=cutoff_day,
    )
    planned_acc = float(timeline_tmp.iloc[-1]["patrimonio_planejado"]) if not timeline_tmp.empty else 0.0
    delta_acc = float(patrimonio_est - planned_acc)
    if planned_acc <= 0:
        delta_pct = 0.0
        adv_txt = "Sem planejado"
        adv_level = "warn"
    else:
        delta_pct = (patrimonio_est / planned_acc - 1.0) * 100.0
        if delta_acc >= 0:
            adv_txt = "Adiantado"
            adv_level = "ok"
        else:
            adv_txt = "Atrasado"
            adv_level = "danger"

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        kpi_card(
            "Patrimônio estimado (até o período)",
            brl(patrimonio_est),
            hint=f"Até: {month_label(yyyy_mm)} | Base: {cfg['pct_cdi']:.0f}% do CDI | CDI {cfg['cdi_anual']:.2f}% a.a.",
        )
    with k2:
        kpi_card("Total aportado (acumulado)", brl(aporte_total))
    with k3:
        kpi_card("Total retirado (acumulado)", brl(retirada_total))
    with k4:
        kpi_card("Aportes no período", brl(aporte_mes), hint=f"Período: {month_label(yyyy_mm)}")
    with k5:
        kpi_card("Retiradas no período", brl(retirada_mes), hint=f"Líquido: {brl(liquido_mes)}")
    with k6:
        st.markdown(
            "<div class='kpi'>"
            "<div class='title'>Meta mensal (Aporte x Planejado)</div>"
            f"<div class='value'>{brl(aporte_mes)} / {brl(planned)}</div>"
            f"<div class='hint'>{badge_html(meta_txt, level)} &nbsp; {pct_meta:.1f}%</div>"
            f"<div style='margin-top:10px'>{progress_bar_html(pct_meta, color)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    with st.expander("➕ Novo movimento (Aporte / Retirada)", expanded=False):
        products = sorted(set(INV_PRODUCTS_DEFAULT + (movs_all["produto"].unique().tolist() if not movs_all.empty else [])))
        with st.form("form_inv_mov", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns([1, 1.2, 1.8, 1.2])
            with c1:
                d = st.date_input("Data", value=date.today())
            with c2:
                tipo = st.selectbox("Tipo", INV_MOV_TYPES, index=0)
            with c3:
                prod = st.selectbox("Produto", products + ["(Outro)"])
                if prod == "(Outro)":
                    prod = st.text_input("Digite o produto", placeholder="Ex: CDB 120% CDI")
            with c4:
                val_txt = st.text_input("Valor (R$)", placeholder="Ex: 500,00")

            obs = st.text_input("Observação (opcional)", placeholder="Ex: corretora, vencimento, banco...")

            reg_receita = False
            receb_pag = "Transferência"
            if tipo == "Retirada":
                st.caption("Retirada reduz o patrimônio estimado a partir desta data.")
                reg_receita = st.checkbox("Também registrar esta retirada como RECEITA no Dashboard", value=False)
                if reg_receita:
                    receb_pag = st.selectbox("Recebido via", ["PIX", "Transferência", "Débito", "Dinheiro", "Boleto"])

            ok = st.form_submit_button("Salvar movimento")
            if ok:
                v = safe_float(val_txt)
                if not str(prod).strip():
                    st.error("Produto é obrigatório.")
                elif v <= 0:
                    st.error("Valor deve ser maior que zero.")
                else:
                    try:
                        insert_invest_movimento(d, str(prod), str(tipo), float(v), str(obs))

                        if tipo == "Retirada" and reg_receita:
                            insert_transacao(
                                data=d,
                                descricao=f"Resgate investimento ({str(prod).strip()})",
                                categoria="Resgate de Investimento",
                                tipo="Receita",
                                valor=float(v),
                                pagamento=str(receb_pag),
                                status="Pago",
                            )

                        st.success("Movimento registrado!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Falha ao salvar: {e}")

    st.write("")
    st.markdown("#### Acumulado mês a mês (Real x Planejado)")
    timeline = timeline_tmp.copy()

    if timeline.empty:
        st.info("Registre pelo menos 1 movimento para ver o acumulado mês a mês.")
    else:
        cA, cB = st.columns([1.2, 2.8])
        with cA:
            st.markdown(
                "<div class='panel'>"
                f"<div class='small-muted'>Acumulado (até {month_label(yyyy_mm)})</div>"
                f"<div style='font-size:1.55rem;font-weight:900'>{brl(patrimonio_est)}</div>"
                f"<div class='small-muted' style='margin-top:10px'>Planejado: <b>{brl(planned_acc)}</b></div>"
                f"<div class='small-muted'>Diferença: <b>{brl(delta_acc)}</b> &nbsp; {badge_html(adv_txt + f' ({delta_pct:+.1f}%)', adv_level)}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        with cB:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=timeline["data"], y=timeline["patrimonio_real"], mode="lines", name="Acumulado (Real)"))
            fig_t.add_trace(go.Scatter(x=timeline["data"], y=timeline["patrimonio_planejado"], mode="lines", name="Acumulado (Planejado)"))
            fig_t.add_trace(go.Scatter(x=[pd.Timestamp(as_of)], y=[patrimonio_est], mode="markers", name="Período selecionado"))
            fig_t.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Tempo",
                yaxis_title="Patrimônio estimado (R$)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_t, use_container_width=True)

    st.write("")
    st.markdown("#### Movimentos (editar e excluir)")
    if movs_all.empty:
        st.info("Nenhum movimento registrado ainda.")
    else:
        df_show = movs_all.sort_values("data", ascending=False).copy()
        df_show["data"] = df_show["data"].dt.date
        df_show["valor_abs"] = df_show["valor"].abs()
        df_show["Excluir"] = False

        edited = st.data_editor(
            df_show[["id", "data", "tipo", "produto", "valor_abs", "observacao", "Excluir"]],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "id": st.column_config.NumberColumn("ID", disabled=True, format="%d"),
                "data": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
                "tipo": st.column_config.SelectboxColumn("Tipo", options=INV_MOV_TYPES, required=True),
                "produto": st.column_config.TextColumn("Produto"),
                "valor_abs": st.column_config.NumberColumn("Valor (R$)", format="%.2f", step=50.0),
                "observacao": st.column_config.TextColumn("Observação"),
                "Excluir": st.column_config.CheckboxColumn("Excluir"),
            },
            key="editor_invest",
        )

        b1, b2, b3 = st.columns([1, 1, 2])

        with b1:
            if st.button("💾 Salvar alterações", use_container_width=True, key="inv_save"):
                try:
                    base = df_show.set_index("id")
                    ed = edited.set_index("id")

                    changed = []
                    for rid in ed.index:
                        if rid not in base.index:
                            continue
                        rb = base.loc[rid]
                        rn = ed.loc[rid]
                        diffs = []
                        for col in ["data", "tipo", "produto", "valor_abs", "observacao"]:
                            a = rb[col]
                            b = rn[col]
                            if col == "valor_abs":
                                diffs.append((pd.isna(a) != pd.isna(b)) or (abs(float(a) - float(b)) > 1e-6))
                            else:
                                diffs.append(str(a).strip() != str(b).strip())
                        if any(diffs):
                            changed.append(int(rid))

                    for rid in changed:
                        r = ed.loc[rid]
                        payload = {
                            "id": int(rid),
                            "data": pd.Timestamp(r["data"]).strftime("%Y-%m-%d"),
                            "tipo": str(r["tipo"]).strip() if str(r["tipo"]).strip() in INV_MOV_TYPES else "Aporte",
                            "produto": str(r["produto"]).strip() or "Outro (Renda Fixa)",
                            "valor": safe_float(r["valor_abs"]),
                            "observacao": str(r.get("observacao", "") or "").strip(),
                        }
                        update_invest_movimento(payload)

                    if changed:
                        st.success(f"Atualizado: {len(changed)} movimento(s).")
                        st.rerun()
                    else:
                        st.info("Nenhuma alteração detectada.")

                except Exception as e:
                    st.error(f"Falha ao salvar: {e}")

        with b2:
            if st.button("🗑️ Excluir selecionados", use_container_width=True, key="inv_del"):
                try:
                    to_delete = edited.loc[edited["Excluir"] == True, "id"].tolist()  # noqa: E712
                    if not to_delete:
                        st.warning("Selecione pelo menos um item na coluna 'Excluir'.")
                    else:
                        n = delete_invest_aportes(to_delete)
                        st.success(f"Excluídos: {n} movimento(s).")
                        st.rerun()
                except Exception as e:
                    st.error(f"Falha ao excluir: {e}")

        with b3:
            st.markdown(
                "<div class='small-muted'>"
                "Dica: investimentos não entram como despesa e não reduzem o saldo do período no Dashboard. "
                "Retiradas reduzem o patrimônio estimado (e opcionalmente podem virar Receita no Dashboard)."
                "</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("#### Visão de longo prazo (Projeção)")
    if annual_rate <= 0:
        st.info("Defina CDI e % do CDI nas configurações para habilitar a projeção com rendimento.")
        r_m = 0.0
    else:
        r_m = annual_to_monthly_rate(annual_rate)

    principal = float(patrimonio_est)
    real_pmt_default = max(0.0, avg_monthly_netflow(movs_upto, as_of)) if not movs_upto.empty else 0.0

    c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.8])
    with c1:
        years = st.slider("Horizonte (anos)", min_value=1, max_value=30, value=10)
    with c2:
        real_pmt = st.number_input("Aporte mensal (Real)", min_value=0.0, value=float(real_pmt_default), step=50.0)
    with c3:
        planned_pmt = st.number_input("Aporte mensal (Planejado)", min_value=0.0, value=float(cfg["aporte_planejado"]), step=50.0)
    with c4:
        st.markdown(
            "<div class='panel'>"
            f"<div class='small-muted'>Base de projeção</div>"
            f"<div style='font-size:1.2rem;font-weight:900'>{brl(principal)}</div>"
            f"<div class='small-muted'>Referência: fim de {month_label(yyyy_mm)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    n_months = int(years * 12)
    xs = list(range(0, n_months + 1))
    dates = [pd.Timestamp(as_of) + pd.DateOffset(months=i) for i in xs]

    y_real = [fv_with_contrib(principal, float(real_pmt), r_m, i) for i in xs]
    y_plan = [fv_with_contrib(principal, float(planned_pmt), r_m, i) for i in xs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_real, mode="lines", name="Projeção (Real)"))
    fig.add_trace(go.Scatter(x=dates, y=y_plan, mode="lines", name="Projeção (Planejado)"))
    fig.add_hline(
        y=principal,
        line_dash="dash",
        annotation_text=f"Acumulado atual: {brl(principal)}",
        annotation_position="top left",
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Tempo", yaxis_title="Patrimônio estimado (R$)")
    st.plotly_chart(fig, use_container_width=True)

    final_real = float(y_real[-1])
    final_plan = float(y_plan[-1])
    delta = final_plan - final_real


    months = int(years) * 12
    aportes_real = principal + (real_pmt * months)
    aportes_plan = principal + (planned_pmt * months)
    juros_real = final_real - aportes_real
    juros_plan = final_plan - aportes_plan
    st.markdown(
        "<div class='panel'>"
        f"<div style='font-size:1.15rem;font-weight:800'>Em {years} ano(s):</div>"
        f"<div class='small-muted' style='margin-top:6px'>Real: <b>{brl(final_real)}</b> &nbsp;|&nbsp; Planejado: <b>{brl(final_plan)}</b> &nbsp;|&nbsp; Diferença: <b>{brl(delta)}</b></div>"
        f"<div class='small-muted' style='margin-top:6px'>Aportado (Real/Plan.): <b>{brl(aportes_real)}</b> &nbsp;|&nbsp; <b>{brl(aportes_plan)}</b></div>"
        f"<div class='small-muted' style='margin-top:6px'>Juros (Real/Plan.): <b>{brl(juros_real)}</b> &nbsp;|&nbsp; <b>{brl(juros_plan)}</b></div>"
        f"<div class='small-muted' style='margin-top:6px'>Taxa usada: {(annual_rate*100):.2f}% a.a. (CDI {cfg['cdi_anual']:.2f}% * {cfg['pct_cdi']:.0f}%)</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# Main
# ============================================================


def screen_admin() -> None:
    if not is_admin():
        st.error("Acesso restrito ao ADMIN.")
        return

    st.markdown("## 👥 Admin — Usuários")

    with st.expander("➕ Criar novo usuário", expanded=True):
        with st.form("admin_create_user"):
            username = st.text_input("Username (login)", placeholder="ex: maria")
            display_name = st.text_input("Nome exibido", placeholder="ex: Maria")
            password = st.text_input("Senha", type="password")
            password2 = st.text_input("Confirmar senha", type="password")
            make_admin = st.checkbox("Tornar este usuário ADMIN", value=False)
            submitted = st.form_submit_button("Criar usuário")

        if submitted:
            if not username.strip():
                st.error("Preencha o username.")
                return
            if not password:
                st.error("Preencha a senha.")
                return
            if password != password2:
                st.error("As senhas não conferem.")
                return
            try:
                create_user(username=username, display_name=display_name, password=password, admin=bool(make_admin))
                st.success("Usuário criado com sucesso.")
            except sqlite3.IntegrityError:
                st.error("Este username já existe.")
            except Exception as e:
                st.error(f"Falha ao criar usuário: {e}")

    st.divider()
    st.markdown("### Usuários cadastrados")
    with db_conn() as conn:
        df = pd.read_sql_query(
            "SELECT id, username, display_name, is_admin, created_at FROM users ORDER BY id ASC",
            conn,
        )
    if df.empty:
        st.info("Nenhum usuário cadastrado.")
        return
    df["is_admin"] = df["is_admin"].apply(lambda x: "Sim" if int(x or 0) == 1 else "Não")
    st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="💰")
    st.markdown(CSS, unsafe_allow_html=True)

    init_db()
    ensure_state()

    # ---------- Login ----------
    if not auth_gate():
        return

    # garante config/seed do usuário
    ensure_user_defaults(require_user_id())

    # carrega config persistida 1x por sessão (por usuário)
    if not st.session_state._period_cfg_loaded:
        app_cfg = fetch_app_config()
        st.session_state.analysis_mode = app_cfg["budget_mode"]
        st.session_state.budget_cutoff_day = int(app_cfg["cutoff_day"])
        st.session_state._period_cfg_loaded = True

    mode = st.session_state.analysis_mode
    cutoff_day = int(st.session_state.budget_cutoff_day)

    # Gera/atualiza parcelas futuras do cartão para ESTE usuário
    ensure_cc_generated(horizon_months=24)
    # Gera/atualiza lançamentos fixos (Débito Automático) para ESTE usuário
    ensure_fixos_generated(horizon_months=24)

    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown("### 💼 Finanças")
        st.caption(f"Logado como: **{st.session_state.get('display_name','')}**")
        cols = st.columns([1, 1])
        with cols[1]:
            if st.button("Sair", use_container_width=True):
                logout()

        with st.expander("🧾 Período / mês de orçamento", expanded=False):
            opt = st.radio(
                "Modo de análise",
                ["Calendário", "Orçamento (salário fim do mês paga o próximo)"],
                index=0 if _normalize_mode(mode) == "calendar" else 1,
            )
            mode_new = "calendar" if opt.startswith("Calendário") else "budget"

            cutoff_new = cutoff_day
            if mode_new == "budget":
                cutoff_new = st.number_input("Dia de virada (>= vira próximo mês)", min_value=1, max_value=31, value=int(cutoff_day), step=1)
                st.caption("Recomendado: **28** (aproxima o último dia útil na prática).")

            if st.button("💾 Salvar período", use_container_width=True):
                upsert_app_config(mode_new, int(cutoff_new))
                st.session_state.analysis_mode = mode_new
                st.session_state.budget_cutoff_day = int(cutoff_new)
                st.success("Período salvo!")
                st.rerun()

        pages = ["Dashboard", "Lançamentos", "Cartão de Crédito", "Planejamento", "Investimentos"]
        icons = ["speedometer2", "list-check", "credit-card", "bullseye", "graph-up-arrow"]
        if is_admin():
            pages.append("Admin")
            icons.append("people")

        selected = option_menu(
            None,
            pages,
            icons=icons,
            menu_icon="wallet2",
            default_index=0,
            styles={
                "container": {"padding": "0!important"},
                "icon": {"font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "padding": "10px 12px", "border-radius": "10px"},
                "nav-link-selected": {"background-color": "rgba(59,130,246,0.14)", "font-weight": "700"},
            },
        )
        st.markdown("---")
        st.caption("Dados locais em SQLite (financas.db)")

    # ---------- Dados ----------
    df_all = fetch_transacoes()
    inv_all = fetch_invest_aportes()

    months = available_months(df_all, inv_all=inv_all, future_months=12, past_months=18, mode=mode, cutoff_day=cutoff_day)
    st.session_state["_months_list"] = months
    scope, yyyy_mm, year = top_bar(months, mode, cutoff_day)

    # ---------- Rotas ----------
    if selected == "Dashboard":
        screen_dashboard(df_all, scope=scope, yyyy_mm=yyyy_mm, year=year, mode=mode, cutoff_day=cutoff_day)
    elif selected == "Lançamentos":
        screen_lancamentos(df_all, yyyy_mm, mode, cutoff_day)
    elif selected == "Cartão de Crédito":
        screen_cartao(df_all, yyyy_mm, mode, cutoff_day)
    elif selected == "Planejamento":
        screen_planejamento(df_all, yyyy_mm, mode, cutoff_day)
    elif selected == "Investimentos":
        screen_investimentos(yyyy_mm, mode, cutoff_day)
    elif selected == "Admin":
        screen_admin()


if __name__ == "__main__":
    main()

