"""Unit tests for the named-param → asyncpg positional binder."""

from echo.databases.postgres.binder import bind_named


def test_single_param():
    sql = "SELECT * FROM t WHERE id = %(id)s"
    rewritten, values = bind_named(sql, {"id": 42})
    assert rewritten == "SELECT * FROM t WHERE id = $1"
    assert values == [42]


def test_multiple_params():
    sql = "SELECT * FROM t WHERE a = %(a)s AND b = %(b)s"
    rewritten, values = bind_named(sql, {"a": 1, "b": 2})
    assert rewritten == "SELECT * FROM t WHERE a = $1 AND b = $2"
    assert values == [1, 2]


def test_repeated_param_uses_same_slot():
    sql = "SELECT %(x)s, %(y)s, %(x)s"
    rewritten, values = bind_named(sql, {"x": "X", "y": "Y"})
    assert rewritten == "SELECT $1, $2, $1"
    assert values == ["X", "Y"]


def test_no_params():
    sql = "SELECT 1"
    rewritten, values = bind_named(sql, {})
    assert rewritten == "SELECT 1"
    assert values == []


def test_missing_param_resolves_to_none():
    sql = "SELECT %(present)s, %(missing)s"
    rewritten, values = bind_named(sql, {"present": "yes"})
    assert rewritten == "SELECT $1, $2"
    assert values == ["yes", None]


def test_param_order_follows_first_appearance():
    sql = "SELECT %(b)s, %(a)s, %(b)s, %(c)s"
    rewritten, values = bind_named(sql, {"a": 1, "b": 2, "c": 3})
    assert rewritten == "SELECT $1, $2, $1, $3"
    assert values == [2, 1, 3]


def test_workspace_id_repeated_in_real_query_shape():
    # Mirrors search_doctors.sql: workspace_id appears once, query_text twice.
    sql = """
        WITH q AS (SELECT immutable_unaccent(%(query_text)s) AS qt,
                          plainto_tsquery('simple', immutable_unaccent(%(query_text)s)) AS tsq)
        SELECT * FROM datasets_doctor d
        WHERE d.workspace_id = %(workspace_id)s
    """
    rewritten, values = bind_named(
        sql, {"query_text": "cardio", "workspace_id": "ws_abc"}
    )
    assert "$1" in rewritten and "$2" in rewritten
    # query_text appears twice, must reuse $1; workspace_id is $2
    assert rewritten.count("$1") == 2
    assert rewritten.count("$2") == 1
    assert values == ["cardio", "ws_abc"]
