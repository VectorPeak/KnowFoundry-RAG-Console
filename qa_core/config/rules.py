"""Shared rule configuration loaded from config/rules.toml."""

from __future__ import annotations

import re
import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path

from qa_core.config.settings import PROJECT_ROOT


DEFAULT_RULE_CONFIG_PATH = PROJECT_ROOT / "config" / "rules.toml"


@dataclass(frozen=True)
class FaqFastPathRules:
    """Rules that decide whether a short query is worth probing in FAQ first."""

    max_chars: int
    hints: tuple[str, ...]

    def hint_matches(self, query: str) -> bool:
        """Return whether query contains any configured FAQ fast-path hint."""

        if not self.hints:
            return False
        pattern = re.compile("|".join(re.escape(item) for item in self.hints), re.IGNORECASE)
        return bool(pattern.search(query or ""))


@dataclass(frozen=True)
class QueryVariantReplacementRule:
    """One configured deterministic query-variant replacement rule."""

    when_any: tuple[str, ...]
    when_all: tuple[str, ...]
    replacements: tuple[tuple[str, str], ...]
    ignore_case: bool = False

    def matches(self, query: str) -> bool:
        """Return whether this replacement rule should run for query."""

        source = query.lower() if self.ignore_case else query
        any_terms = tuple(item.lower() for item in self.when_any) if self.ignore_case else self.when_any
        all_terms = tuple(item.lower() for item in self.when_all) if self.ignore_case else self.when_all
        if any_terms and not any(term in source for term in any_terms):
            return False
        if all_terms and not all(term in source for term in all_terms):
            return False
        return bool(any_terms or all_terms)


@dataclass(frozen=True)
class QueryVariantRules:
    """Configured deterministic query-variant rules."""

    short_structured_max_chars: int
    short_structured_markers: tuple[str, ...]
    replacements: tuple[QueryVariantReplacementRule, ...]

    def is_short_structured_question(self, query: str) -> bool:
        """Return whether query is short and specific enough to skip expansion."""

        compact = query.strip()
        if not compact or len(compact) > self.short_structured_max_chars:
            return False
        return any(marker in compact for marker in self.short_structured_markers)


@dataclass(frozen=True)
class RuleConfig:
    """Runtime rules shared by pipeline modules."""

    faq_fast_path: FaqFastPathRules
    query_variants: QueryVariantRules


def get_rule_config(path: str | Path | None = None) -> RuleConfig:
    """Load routing rules from TOML.

    The file is intentionally read on demand so local rule edits take effect on
    the next request or test run, matching the scenario.toml workflow.
    """

    config_path = Path(path) if path else DEFAULT_RULE_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    faq_payload = dict(payload.get("faq_fast_path") or {})
    max_chars = int(faq_payload.get("max_chars") or 0)
    hints = tuple(str(item).strip() for item in faq_payload.get("hints", ()) if str(item).strip())
    if max_chars <= 0:
        raise ValueError(f"faq_fast_path.max_chars 必须大于 0：{config_path}")
    if not hints:
        raise ValueError(f"faq_fast_path.hints 不能为空：{config_path}")
    query_variants = _load_query_variant_rules(payload, config_path)
    return RuleConfig(
        faq_fast_path=FaqFastPathRules(max_chars=max_chars, hints=hints),
        query_variants=query_variants,
    )


def _load_query_variant_rules(payload: dict, config_path: Path) -> QueryVariantRules:
    """Parse query variant rules from TOML payload."""

    variant_payload = dict(payload.get("query_variants") or {})
    max_chars = int(variant_payload.get("short_structured_max_chars") or 0)
    markers = _clean_tuple(variant_payload.get("short_structured_markers", ()))
    if max_chars <= 0:
        raise ValueError(f"query_variants.short_structured_max_chars 必须大于 0：{config_path}")
    if not markers:
        raise ValueError(f"query_variants.short_structured_markers 不能为空：{config_path}")

    replacements = tuple(
        _parse_replacement_rule(item, config_path)
        for item in variant_payload.get("replacements", ())
    )
    if not replacements:
        raise ValueError(f"query_variants.replacements 不能为空：{config_path}")

    return QueryVariantRules(
        short_structured_max_chars=max_chars,
        short_structured_markers=markers,
        replacements=replacements,
    )


def _parse_replacement_rule(payload: dict, config_path: Path) -> QueryVariantReplacementRule:
    """Parse one query variant replacement rule."""

    rule_payload = dict(payload or {})
    when_any = _clean_tuple(rule_payload.get("when_any", ()))
    when_all = _clean_tuple(rule_payload.get("when_all", ()))
    replacements = tuple(
        (str(pair[0]).strip(), str(pair[1]).strip())
        for pair in rule_payload.get("replace", ())
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and str(pair[0]).strip() and str(pair[1]).strip()
    )
    if not when_any and not when_all:
        raise ValueError(f"query_variants.replacements 中每条规则必须配置 when_any 或 when_all：{config_path}")
    if not replacements:
        raise ValueError(f"query_variants.replacements 中每条规则必须配置 replace：{config_path}")
    return QueryVariantReplacementRule(
        when_any=when_any,
        when_all=when_all,
        replacements=replacements,
        ignore_case=bool(rule_payload.get("ignore_case", False)),
    )


def _clean_tuple(items: object) -> tuple[str, ...]:
    """Return non-empty strings as a tuple."""

    return tuple(str(item).strip() for item in items or () if str(item).strip())
