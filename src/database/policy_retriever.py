"""
Modulo de Recuperacao de Conhecimento Normativo (Policy RAG Retriever) para o Agente Tier 2.

Responsabilidades:
1. Realizar o parsing estruturado do documento data/policies/refund_policy_v1.md por clausula (POL-*).
2. Indexar as clausulas normativas em espaco vetorial TF-IDF com N-Grams (1, 2) e normalizacao de acentos.
3. Recuperar as Top-K clausulas mais relevantes para a disputa combinando a semantica do relato
   do cliente com sinais contextuais do pedido (atraso, valor financeiro e reincidencia).
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "data" / "policies" / "refund_policy_v1.md"


@dataclass(frozen=True)
class PolicyChunk:
    """
    Representa uma clausula individual extraida do documento normativo de politicas.
    """
    clause_id: str
    title: str
    content: str
    similarity_score: float = 0.0


class RefundPolicyRetriever:
    """
    Motor RAG leve e deterministico para busca de clausulas da Politica de Reembolso.
    """

    CLAUSE_HEADER_REGEX = re.compile(
        r"###\s+Clausula\s+(POL-[A-Z]+-\d+):\s*(.+)",
        re.IGNORECASE
    )

    def __init__(self, policy_path: Optional[Path] = None) -> None:
        self.policy_path = policy_path or DEFAULT_POLICY_PATH
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Arquivo de politica nao encontrado em: {self.policy_path}")

        self.chunks: List[PolicyChunk] = self._parse_markdown_clauses()
        if not self.chunks:
            raise ValueError("Nenhuma clausula no formato '### Clausula POL-XX-00' foi encontrada.")

        # Vetorizador TF-IDF com unigramas e bigramas sobre o texto normalizado das clausulas
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        corpus = [
            self._normalize_text(f"{c.clause_id} {c.title} {c.content}")
            for c in self.chunks
        ]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Remove acentos e converte para minusculas para maximizar o recall da busca vetorial.
        """
        norm = unicodedata.normalize("NFD", text)
        clean = "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")
        return clean.lower()

    def _parse_markdown_clauses(self) -> List[PolicyChunk]:
        """
        Divide o arquivo Markdown em blocos correspondentes a cada clausula POL-*.
        """
        raw_text = self.policy_path.read_text(encoding="utf-8")
        lines = raw_text.splitlines()

        parsed_chunks: List[PolicyChunk] = []
        current_id: Optional[str] = None
        current_title: Optional[str] = None
        current_body: List[str] = []

        for line in lines:
            match = self.CLAUSE_HEADER_REGEX.match(line.strip())
            if match:
                if current_id and current_title:
                    parsed_chunks.append(
                        PolicyChunk(
                            clause_id=current_id.upper(),
                            title=current_title.strip(),
                            content="\n".join(current_body).strip(),
                        )
                    )
                current_id = match.group(1)
                current_title = match.group(2)
                current_body = []
            elif current_id is not None:
                # Ignora divisores horizontais e cabecalhos de secao macro (##)
                if not line.strip().startswith("## ") and line.strip() != "---":
                    current_body.append(line)

        if current_id and current_title:
            parsed_chunks.append(
                PolicyChunk(
                    clause_id=current_id.upper(),
                    title=current_title.strip(),
                    content="\n".join(current_body).strip(),
                )
            )

        return parsed_chunks

    def retrieve_relevant_clauses(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> List[PolicyChunk]:
        """
        Retorna as Top-K clausulas mais relevantes ordenadas por similaridade de cosseno.
        """
        norm_query = self._normalize_text(query_text)
        query_vec = self.vectorizer.transform([norm_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[PolicyChunk] = []
        for idx in top_indices:
            base_chunk = self.chunks[int(idx)]
            results.append(
                PolicyChunk(
                    clause_id=base_chunk.clause_id,
                    title=base_chunk.title,
                    content=base_chunk.content,
                    similarity_score=round(float(scores[int(idx)]), 4),
                )
            )
        return results
