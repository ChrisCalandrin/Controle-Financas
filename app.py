from __future__ import annotations

import sqlite3
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
# Config
# ============================================================
APP_TITLE = "Finanças Pessoais"
DB_PATH = "financas.db"

TIPOS = ["Despesa", "Receita"]

# ✅ inclui "Amortizado" para cartão (adiantamento)
STATUS = ["Pago", "Pendente", "Amortizado"]

PAGAMENTOS = ["Crédito", "Débito", "PIX", "Dinheiro", "Boleto", "Transferência"]
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

        # ---------- base ----------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data DATE,
                descricao TEXT,
                categoria TEXT,
                tipo TEXT,
                valor REAL,
                pagamento TEXT,
                status TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS orcamentos (
                categoria TEXT PRIMARY KEY,
                valor_teto REAL
            );
            """
        )

        # ---------- cartão (parcelado/recorrente + amortização) ----------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cc_compras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_compra DATE,
                descricao TEXT,
                categoria TEXT,
                tipo_compra TEXT,          -- avista | parcelado | recorrente
                total REAL,                -- total (avista/parcelado) | valor mensal (recorrente)
                parcelas INTEGER,           -- 1 (avista) | N (parcelado) | NULL (recorrente)
                ativo INTEGER DEFAULT 1,    -- 1=ativo, 0=inativo
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cc_amortizacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cc_compra_id INTEGER,
                data DATE,
                parcelas_amortizadas INTEGER,
                desconto REAL,
                valor_pago REAL,
                pagamento TEXT,
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
                data DATE,
                produto TEXT,
                valor REAL,
                observacao TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS investimentos_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                aporte_planejado REAL,
                cdi_anual REAL,
                pct_cdi REAL
            );
            """
        )
        cur.execute(
            "INSERT OR IGNORE INTO investimentos_config (id, aporte_planejado, cdi_anual, pct_cdi) VALUES (1, 0.0, 0.0, 100.0);"
        )

        # ---------- config do app (modo de período) ----------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS app_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                budget_mode TEXT,     -- 'calendar' | 'budget'
                cutoff_day INTEGER    -- dia de virada do orçamento (ex: 28)
            );
            """
        )
        cur.execute("INSERT OR IGNORE INTO app_config (id, budget_mode, cutoff_day) VALUES (1, 'calendar', 28);")

        # ---------- migração transacoes ----------
        cols = _table_cols(conn, "transacoes")
        to_add = [
            ("origem", "TEXT"),
            ("cc_compra_id", "INTEGER"),
            ("cc_parcela_num", "INTEGER"),
            ("cc_parcela_total", "INTEGER"),
            ("ref_month", "TEXT"),
        ]
        for col, typ in to_add:
            if col not in cols:
                cur.execute(f"ALTER TABLE transacoes ADD COLUMN {col} {typ};")

        # ---------- migração investimentos: tipo (Aporte/Retirada) ----------
        inv_cols = _table_cols(conn, "investimentos_aportes")
        if "tipo" not in inv_cols:
            cur.execute("ALTER TABLE investimentos_aportes ADD COLUMN tipo TEXT;")
            cur.execute("UPDATE investimentos_aportes SET tipo = 'Aporte' WHERE tipo IS NULL OR TRIM(tipo) = '';")

        # índices
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transacoes_data ON transacoes(data);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transacoes_cc ON transacoes(cc_compra_id, cc_parcela_num, ref_month);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_invest_data ON investimentos_aportes(data);")


# ---------- App config CRUD ----------
def fetch_app_config() -> Dict[str, object]:
    with db_conn() as conn:
        row = conn.execute("SELECT budget_mode, cutoff_day FROM app_config WHERE id = 1").fetchone()

    if not row:
        return {"budget_mode": "calendar", "cutoff_day": 28}

    mode = str(row["budget_mode"] or "calendar").strip().lower()
    if mode not in {"calendar", "budget"}:
        mode = "calendar"

    cutoff = int(row["cutoff_day"] or 28)
    cutoff = max(1, min(31, cutoff))
    return {"budget_mode": mode, "cutoff_day": cutoff}


def upsert_app_config(budget_mode: str, cutoff_day: int) -> None:
    budget_mode = str(budget_mode or "calendar").strip().lower()
    if budget_mode not in {"calendar", "budget"}:
        budget_mode = "calendar"

    cutoff_day = int(cutoff_day)
    cutoff_day = max(1, min(31, cutoff_day))

    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO app_config (id, budget_mode, cutoff_day)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                budget_mode = excluded.budget_mode,
                cutoff_day = excluded.cutoff_day
            """,
            (budget_mode, cutoff_day),
        )


# ============================================================
# Fetch/CRUD (Transações)
# ============================================================


def fetch_transacoes() -> pd.DataFrame:
    with db_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                id, data, descricao, categoria, tipo, valor, pagamento, status,
                origem, cc_compra_id, cc_parcela_num, cc_parcela_total, ref_month
            FROM transacoes
            """,
            conn,
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
    df["cc_compra_id"] = pd.to_numeric(df.get("cc_compra_id", None), errors="coerce")
    df["cc_parcela_num"] = pd.to_numeric(df.get("cc_parcela_num", None), errors="coerce")
    df["cc_parcela_total"] = pd.to_numeric(df.get("cc_parcela_total", None), errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0).astype(float)

    df = df.dropna(subset=["data"]).copy()
    return df


def fetch_orcamentos() -> pd.DataFrame:
    with db_conn() as conn:
        df = pd.read_sql_query("SELECT categoria, valor_teto FROM orcamentos", conn)

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
    data_str = pd.Timestamp(data).strftime("%Y-%m-%d")
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO transacoes (data, descricao, categoria, tipo, valor, pagamento, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (data_str, descricao.strip(), categoria.strip(), tipo, float(valor), pagamento, status),
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
    data_str = pd.Timestamp(data).strftime("%Y-%m-%d")
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO transacoes (
                data, descricao, categoria, tipo, valor, pagamento, status,
                origem, cc_compra_id, cc_parcela_num, cc_parcela_total, ref_month
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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
    ids = [int(i) for i in ids if str(i).strip().isdigit()]
    if not ids:
        return 0
    with db_conn() as conn:
        cur = conn.cursor()
        cur.executemany("DELETE FROM transacoes WHERE id = ?", [(i,) for i in ids])
        return cur.rowcount


def update_transacao(row: Dict) -> None:
    with db_conn() as conn:
        conn.execute(
            """
            UPDATE transacoes
            SET data = ?, descricao = ?, categoria = ?, tipo = ?, valor = ?, pagamento = ?, status = ?
            WHERE id = ?
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
            ),
        )


def upsert_orcamento(categoria: str, valor_teto: float) -> None:
    categoria = categoria.strip()
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO orcamentos (categoria, valor_teto)
            VALUES (?, ?)
            ON CONFLICT(categoria) DO UPDATE SET valor_teto = excluded.valor_teto
            """,
            (categoria, float(valor_teto)),
        )


def delete_orcamento(categoria: str) -> None:
    with db_conn() as conn:
        conn.execute("DELETE FROM orcamentos WHERE categoria = ?", (categoria,))


# ============================================================
# Cartão: compras + amortização
# ============================================================


def fetch_cc_compras() -> pd.DataFrame:
    with db_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT id, data_compra, descricao, categoria, tipo_compra, total, parcelas, ativo, created_at
            FROM cc_compras
            ORDER BY id DESC
            """,
            conn,
        )
    if df.empty:
        return df
    df["data_compra"] = pd.to_datetime(df["data_compra"], errors="coerce")
    df["descricao"] = df["descricao"].astype(str).fillna("")
    df["categoria"] = df["categoria"].astype(str).fillna("Outros")
    df["tipo_compra"] = df["tipo_compra"].astype(str).fillna("avista")
    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0).astype(float)
    df["parcelas"] = pd.to_numeric(df["parcelas"], errors="coerce")
    df["ativo"] = pd.to_numeric(df["ativo"], errors="coerce").fillna(1).astype(int)
    return df


def insert_cc_compra(
    data_compra: date,
    descricao: str,
    categoria: str,
    tipo_compra: str,
    total: float,
    parcelas: Optional[int],
) -> None:
    data_str = pd.Timestamp(data_compra).strftime("%Y-%m-%d")
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO cc_compras (data_compra, descricao, categoria, tipo_compra, total, parcelas, ativo)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (
                data_str,
                descricao.strip(),
                categoria.strip(),
                tipo_compra.strip(),
                float(total),
                int(parcelas) if parcelas is not None else None,
            ),
        )


def set_cc_compra_ativo(cc_compra_id: int, ativo: int) -> None:
    with db_conn() as conn:
        conn.execute("UPDATE cc_compras SET ativo = ? WHERE id = ?", (int(ativo), int(cc_compra_id)))


def fetch_cc_amortizacoes(cc_compra_id: Optional[int] = None) -> pd.DataFrame:
    with db_conn() as conn:
        if cc_compra_id is None:
            df = pd.read_sql_query(
                """
                SELECT id, cc_compra_id, data, parcelas_amortizadas, desconto, valor_pago, pagamento, observacao, created_at
                FROM cc_amortizacoes
                ORDER BY id DESC
                """,
                conn,
            )
        else:
            df = pd.read_sql_query(
                """
                SELECT id, cc_compra_id, data, parcelas_amortizadas, desconto, valor_pago, pagamento, observacao, created_at
                FROM cc_amortizacoes
                WHERE cc_compra_id = ?
                ORDER BY id DESC
                """,
                conn,
                params=(int(cc_compra_id),),
            )
    if df.empty:
        return df
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["desconto"] = pd.to_numeric(df["desconto"], errors="coerce").fillna(0.0).astype(float)
    df["valor_pago"] = pd.to_numeric(df["valor_pago"], errors="coerce").fillna(0.0).astype(float)
    df["parcelas_amortizadas"] = pd.to_numeric(df["parcelas_amortizadas"], errors="coerce").fillna(0).astype(int)
    df["pagamento"] = df["pagamento"].astype(str).fillna("")
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
    data_str = pd.Timestamp(data_).strftime("%Y-%m-%d")
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO cc_amortizacoes (cc_compra_id, data, parcelas_amortizadas, desconto, valor_pago, pagamento, observacao)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(cc_compra_id),
                data_str,
                int(parcelas_amortizadas),
                float(desconto),
                float(valor_pago),
                pagamento.strip(),
                (observacao or "").strip(),
            ),
        )


def _cc_transacao_exists(cc_compra_id: int, cc_parcela_num: Optional[int], ref_month: str) -> bool:
    with db_conn() as conn:
        cur = conn.execute(
            """
            SELECT 1
            FROM transacoes
            WHERE cc_compra_id = ?
              AND (cc_parcela_num IS ? OR cc_parcela_num = ?)
              AND ref_month = ?
              AND pagamento = 'Crédito'
            LIMIT 1
            """,
            (int(cc_compra_id), cc_parcela_num, cc_parcela_num, ref_month),
        )
        return cur.fetchone() is not None


def ensure_cc_generated(horizon_months: int = 24) -> None:
    """Gera parcelas/recorrências para meses futuros, para aparecerem no Dashboard."""
    compras = fetch_cc_compras()
    if compras.empty:
        return

    today = date.today()
    horizon_end = add_months(date(today.year, today.month, 1), horizon_months)

    for _, r in compras.iterrows():
        if int(r.get("ativo", 1)) != 1:
            continue

        cc_id = int(r["id"])
        data_compra_ts = r["data_compra"]
        if pd.isna(data_compra_ts):
            continue
        data_compra = data_compra_ts.date()

        desc = str(r["descricao"]).strip() or "Compra no cartão"
        cat = str(r["categoria"]).strip() or "Outros"
        tipo_compra = str(r["tipo_compra"]).strip().lower()
        total = float(r["total"])

        if tipo_compra == "recorrente":
            d0 = date(data_compra.year, data_compra.month, 1)
            d = d0
            k = 0
            while d <= horizon_end:
                ref = f"{d.year:04d}-{d.month:02d}"
                dia = clamp_day_to_month(d.year, d.month, data_compra.day)
                dt_lanc = date(d.year, d.month, dia)

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
                if k > 500:
                    break

        elif tipo_compra in {"avista", "parcelado"}:
            parcelas = int(r["parcelas"]) if pd.notna(r["parcelas"]) else 1
            parcelas = max(1, parcelas)
            vals = split_installments(total, parcelas)

            for i in range(parcelas):
                parc_num = i + 1
                dt = add_months(data_compra, i)
                ref = f"{dt.year:04d}-{dt.month:02d}"

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
    parcelas_amortizar = max(1, int(parcelas_amortizar))
    desconto = max(0.0, float(desconto))

    with db_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT id, data, descricao, valor, cc_parcela_num, cc_parcela_total
            FROM transacoes
            WHERE cc_compra_id = ?
              AND pagamento = 'Crédito'
              AND tipo = 'Despesa'
              AND status = 'Pendente'
            ORDER BY date(data) ASC
            """,
            conn,
            params=(int(cc_compra_id),),
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
        conn.executemany("UPDATE transacoes SET status = 'Amortizado' WHERE id = ?", [(int(i),) for i in amort_ids])

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


# ============================================================
# Investimentos
# ============================================================

INV_PRODUCTS_DEFAULT = [
    "CDB",
    "LCI/LCA",
    "Tesouro Selic",
    "Tesouro IPCA",
    "Tesouro Prefixado",
    "Fundo DI",
    "Poupança",
    "Outro (Renda Fixa)",
]

INV_MOV_TYPES = ["Aporte", "Retirada"]


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
    with db_conn() as conn:
        row = conn.execute(
            "SELECT aporte_planejado, cdi_anual, pct_cdi FROM investimentos_config WHERE id = 1"
        ).fetchone()
    if not row:
        return {"aporte_planejado": 0.0, "cdi_anual": 0.0, "pct_cdi": 100.0}
    return {
        "aporte_planejado": float(row["aporte_planejado"] or 0.0),
        "cdi_anual": float(row["cdi_anual"] or 0.0),
        "pct_cdi": float(row["pct_cdi"] or 100.0),
    }


def upsert_invest_config(aporte_planejado: float, cdi_anual: float, pct_cdi: float) -> None:
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO investimentos_config (id, aporte_planejado, cdi_anual, pct_cdi)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                aporte_planejado = excluded.aporte_planejado,
                cdi_anual = excluded.cdi_anual,
                pct_cdi = excluded.pct_cdi
            """,
            (float(aporte_planejado), float(cdi_anual), float(pct_cdi)),
        )


def fetch_invest_aportes() -> pd.DataFrame:
    with db_conn() as conn:
        cols = _table_cols(conn, "investimentos_aportes")
        if "tipo" in cols:
            df = pd.read_sql_query(
                "SELECT id, data, produto, tipo, valor, observacao FROM investimentos_aportes ORDER BY date(data) DESC, id DESC",
                conn,
            )
        else:
            df = pd.read_sql_query(
                "SELECT id, data, produto, 'Aporte' as tipo, valor, observacao FROM investimentos_aportes ORDER BY date(data) DESC, id DESC",
                conn,
            )

    if df.empty:
        return df
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["produto"] = df["produto"].astype(str).fillna("Outro (Renda Fixa)")
    df["tipo"] = df.get("tipo", "Aporte").astype(str).fillna("Aporte")
    df["tipo"] = df["tipo"].apply(lambda x: x if x in INV_MOV_TYPES else "Aporte")
    df["observacao"] = df["observacao"].astype(str).fillna("")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0).astype(float)
    df = df.dropna(subset=["data"]).copy()
    return df


def insert_invest_movimento(d: date, produto: str, tipo: str, valor_abs: float, observacao: str = "") -> None:
    data_str = pd.Timestamp(d).strftime("%Y-%m-%d")
    tipo = tipo if tipo in INV_MOV_TYPES else "Aporte"
    v_signed = _inv_signed_value(tipo, float(valor_abs))
    with db_conn() as conn:
        cols = _table_cols(conn, "investimentos_aportes")
        if "tipo" in cols:
            conn.execute(
                """
                INSERT INTO investimentos_aportes (data, produto, tipo, valor, observacao)
                VALUES (?, ?, ?, ?, ?)
                """,
                (data_str, (produto or "").strip(), tipo, float(v_signed), (observacao or "").strip()),
            )
        else:
            conn.execute(
                """
                INSERT INTO investimentos_aportes (data, produto, valor, observacao)
                VALUES (?, ?, ?, ?)
                """,
                (data_str, (produto or "").strip(), float(v_signed), (observacao or "").strip()),
            )


def insert_invest_aporte(d: date, produto: str, valor: float, observacao: str = "") -> None:
    """Compat: mantém chamadas antigas como Aporte."""
    insert_invest_movimento(d=d, produto=produto, tipo="Aporte", valor_abs=float(valor), observacao=observacao)


def update_invest_movimento(row: Dict) -> None:
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
                WHERE id = ?
                """,
                (
                    row["data"],
                    row["produto"],
                    tipo,
                    float(v_signed),
                    row.get("observacao", "") or "",
                    int(row["id"]),
                ),
            )
        else:
            conn.execute(
                """
                UPDATE investimentos_aportes
                SET data = ?, produto = ?, valor = ?, observacao = ?
                WHERE id = ?
                """,
                (
                    row["data"],
                    row["produto"],
                    float(v_signed),
                    row.get("observacao", "") or "",
                    int(row["id"]),
                ),
            )


def update_invest_aporte(row: Dict) -> None:
    """Compat: aceita tanto o payload antigo quanto o novo."""
    if "tipo" not in row:
        row = dict(row)
        row["tipo"] = "Aporte"
    update_invest_movimento(row)


def delete_invest_aportes(ids: Iterable[int]) -> int:
    ids = [int(i) for i in ids if str(i).strip().isdigit()]
    if not ids:
        return 0
    with db_conn() as conn:
        cur = conn.cursor()
        cur.executemany("DELETE FROM investimentos_aportes WHERE id = ?", [(i,) for i in ids])
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


def compute_kpis(df_scope: pd.DataFrame, invest_aportes_scope: float) -> Dict[str, float]:
    """
    ✅ Regra importante:
    - Investimentos NÃO entram como despesa e NÃO reduzem o Saldo.
    - Investimentos aparecem em KPI separado (APORTES no escopo).
    """
    if df_scope.empty:
        return {
            "receitas": 0.0,
            "despesas": 0.0,
            "saldo": 0.0,
            "economia_pct": 0.0,
            "invest": float(invest_aportes_scope),
        }

    receitas = df_scope.loc[df_scope["tipo"] == "Receita", "valor"].sum()
    despesas = df_scope.loc[df_scope["tipo"] == "Despesa", "valor"].sum()
    saldo = receitas - despesas
    economia_pct = (saldo / receitas * 100.0) if receitas > 0 else 0.0
    return {
        "receitas": float(receitas),
        "despesas": float(despesas),
        "saldo": float(saldo),
        "economia_pct": float(economia_pct),
        "invest": float(invest_aportes_scope),
    }


# ============================================================
# Top bar (Ano + Mês separado + setas) + Visão Anual
# ============================================================


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
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    if scope == "month":
        df_scope = filter_df_by_period(df_all, "data", yyyy_mm, mode, cutoff_day)
        inv_scope = filter_df_by_period(inv_all, "data", yyyy_mm, mode, cutoff_day) if not inv_all.empty else inv_all
        inv_aportes = _sum_aportes(inv_scope) if not inv_scope.empty else 0.0
        return df_scope, inv_scope, float(inv_aportes)

    df_scope = filter_df_by_year(df_all, "data", year, mode, cutoff_day)
    inv_scope = filter_df_by_year(inv_all, "data", year, mode, cutoff_day) if not inv_all.empty else inv_all
    inv_aportes = _sum_aportes(inv_scope) if not inv_scope.empty else 0.0
    return df_scope, inv_scope, float(inv_aportes)


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
    df_scope, inv_scope, inv_aportes_scope = _scope_filters(scope, df_all, inv_all, yyyy_mm, year, mode, cutoff_day)

    if df_scope.empty and inv_aportes_scope <= 0:
        st.info("Sem dados para este período")
        st.markdown(
            "<div class='panel'>Dica: vá em <b>Lançamentos</b>, <b>Cartão de Crédito</b> ou <b>Investimentos</b> e adicione seus dados.</div>",
            unsafe_allow_html=True,
        )
        return

    kpis = compute_kpis(df_scope, inv_aportes_scope)
    label_scope = _period_label(scope, yyyy_mm, year)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    saldo_color = "#16a34a" if kpis["saldo"] >= 0 else "#dc2626"
    saldo_value = f"<span style='color:{saldo_color}'>{brl(kpis['saldo'])}</span>"

    with c1:
        st.markdown(
            "<div class='kpi'><div class='title'>Saldo</div><div class='value'>" + saldo_value + "</div>"
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
        hint = "(Saldo / Receitas)" if kpis["receitas"] > 0 else "Sem receitas"
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
            invest = float(_sum_aportes(invm)) if not invm.empty else 0.0
            rows.append(
                {
                    "mes": mm,
                    "mes_label": PT_MONTHS[int(mm.split("-")[1])],
                    "receitas": receitas,
                    "despesas": despesas,
                    "investimentos": invest,
                    "saldo": receitas - despesas,
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
            with c7:
                status = st.selectbox("Status", STATUS)
            with c8:
                st.caption("Dica: parcelado/recorrente use **Cartão de Crédito**. Aportes/retiradas use **Investimentos**.")

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
                        insert_transacao(d, desc, str(categoria), tipo, valor_f, pagamento, status)
                        st.success("Lançamento salvo!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Falha ao salvar: {e}")

    st.write("")

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
            if st.button("⛔ Desativar" if ativo_now == 1 else "✅ Reativar", use_container_width=True):
                try:
                    set_cc_compra_ativo(int(id_sel), 0 if ativo_now == 1 else 1)
                    st.success("Atualizado.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Falha: {e}")
        with c_act3:
            st.caption("Desativar não apaga histórico: só para de gerar novas recorrências (as já geradas continuam).")

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


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="💰")
    st.markdown(CSS, unsafe_allow_html=True)

    init_db()
    generate_mock_data()
    ensure_state()

    # carrega config persistida 1x por sessão
    if not st.session_state._period_cfg_loaded:
        app_cfg = fetch_app_config()
        st.session_state.analysis_mode = app_cfg["budget_mode"]
        st.session_state.budget_cutoff_day = int(app_cfg["cutoff_day"])
        st.session_state._period_cfg_loaded = True

    mode = st.session_state.analysis_mode
    cutoff_day = int(st.session_state.budget_cutoff_day)

    ensure_cc_generated(horizon_months=24)

    with st.sidebar:
        st.markdown("### 💼 Finanças")

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

        selected = option_menu(
            None,
            ["Dashboard", "Lançamentos", "Cartão de Crédito", "Planejamento", "Investimentos"],
            icons=["speedometer2", "list-check", "credit-card", "bullseye", "graph-up-arrow"],
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

    df_all = fetch_transacoes()
    inv_all = fetch_invest_aportes()

    months = available_months(df_all, inv_all=inv_all, future_months=12, past_months=18, mode=mode, cutoff_day=cutoff_day)
    st.session_state["_months_list"] = months
    scope, yyyy_mm, year = top_bar(months, mode, cutoff_day)

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


if __name__ == "__main__":
    main()
