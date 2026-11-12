"""
Modulo de Guardrail de Seguranca para deteccao deterministica de Prompt Injection.

Responsabilidades:
1. Inspecionar o relato textual do cliente antes de qualquer chamada ao modelo ou LLM.
2. Detectar padroes de Prompt Injection, Jailbreak, tentativa de sobrescrita de instrucoes
   de sistema e falsificacao de alcada administrativa (Clausula POL-SEC-01).
3. Retornar diagnostico auditavel em tempo sub-milissegundo (latencia O(1)).
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class GuardrailInspectionResult:
    """
    Resultado imutavel da inspecao de seguranca sobre o texto de entrada.
    """
    is_blocked: bool
    matched_rule_name: Optional[str]
    normalized_text: str


class SecurityGuardrail:
    """
    Motor heuristico baseado em expressoes regulares compiladas para bloqueio de ataques
    adversariais em portugues e ingles no fluxo de disputas financeiras.
    """

    def __init__(self) -> None:
        # Lista de tuplas (nome_da_regra, padrao_regex) compiladas na inicializacao
        raw_rules: List[Tuple[str, str]] = [
            (
                "instruction_override",
                r"\b(ignore|ignorar|desconsidere|esqueca|forget|disregard)\b.{0,40}\b(instrucoes|instructions|regras|rules|anteriores|previous|acima|politica)\b",
            ),
            (
                "system_prompt_injection",
                r"(\bsystem\s*prompt\b|\bprompt\s*de\s*sistema\b|\byou\s*are\s*now\b|\bvoce\s*agora\s*e\b|\bmodo\s*desenvolvedor\b|\bdeveloper\s*mode\b)",
            ),
            (
                "forced_output_manipulation",
                r"\b(retorne|return|output|responda|imprima)\b.{0,35}\b(approved_full_refund|approved|aprovado|reembolso\s*integral|json)\b",
            ),
            (
                "fake_authority_escalation",
                r"\b(autorizado|aprovado|liberado)\s+pel[oa]\s+(diretoria|diretor|ceo|gerencia|sistema\s*central|auditoria)\b",
            ),
            (
                "delimiter_escape_attack",
                r"(```|<\s*system\s*>|<\s*/\s*system\s*>|\[\s*system\s*\]|###\s*instruction)",
            ),
        ]
        self._compiled_rules: List[Tuple[str, re.Pattern]] = [
            (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in raw_rules
        ]

    @staticmethod
    def _strip_accents(text: str) -> str:
        """
        Remove acentuacao grafica para impedir que o atacante burle o filtro usando
        variacoes de acentos (ex: 'instruções' vs 'instrucoes').
        """
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    def inspect(self, raw_claim_text: str) -> GuardrailInspectionResult:
        """
        Executa a varredura de seguranca sobre o texto da disputa.
        Retorna is_blocked=True caso qualquer assinatura de ataque seja identificada.
        """
        clean_text = " ".join(raw_claim_text.strip().split())
        unaccented_text = self._strip_accents(clean_text).lower()

        for rule_name, compiled_regex in self._compiled_rules:
            if compiled_regex.search(unaccented_text):
                return GuardrailInspectionResult(
                    is_blocked=True,
                    matched_rule_name=rule_name,
                    normalized_text=clean_text,
                )

        return GuardrailInspectionResult(
            is_blocked=False,
            matched_rule_name=None,
            normalized_text=clean_text,
        )
