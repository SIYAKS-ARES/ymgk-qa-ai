"""
R-06: Soru Uretim Pipeline
generate_question(config, examples) fonksiyonu ve tam pipeline
"""

import json
import uuid
from typing import Optional
from datetime import datetime

from ..models.question import GeneratedQuestion, RetrievedQuestion
from ..models.combination import Combination
from .retriever import QuestionRetriever
from .output_formatter import RAGOutputFormatter
from .prompt_builder import PromptBuilder
from .llm_client import BaseLLMClient, parse_llm_response


class QuestionGenerationError(Exception):
    """Soru uretim hatasi"""
    pass


class ValidationError(Exception):
    """Validasyon hatasi"""
    pass


class InsufficientExamplesError(Exception):
    """Yetersiz ornek hatasi"""
    pass


class QuestionGenerator:
    """
    Tam soru uretim pipeline'i
    
    Akis:
    1. Benzer sorulari retrieve et
    2. Ornekleri formatla
    3. Prompt olustur
    4. LLM cagir
    5. Parse ve validate
    """

    def __init__(
        self,
        retriever: QuestionRetriever,
        formatter: RAGOutputFormatter,
        prompt_builder: PromptBuilder,
        llm_client: BaseLLMClient,
    ):
        """
        Args:
            retriever: Soru retrieval servisi
            formatter: RAG cikti formatlayici
            prompt_builder: Prompt olusturucu
            llm_client: LLM API client
        """
        self.retriever = retriever
        self.formatter = formatter
        self.prompt_builder = prompt_builder
        self.llm = llm_client

    async def generate_question(
        self,
        combination: dict,
        style_instruction: Optional[str] = None,
        max_attempts: int = 3,
    ) -> GeneratedQuestion:
        """
        Tek soru uret
        
        Args:
            combination: {alt_konu, zorluk, gorsel_tipi, lgs_skor}
            style_instruction: Stil talimati (cesitlilik icin)
            max_attempts: Maksimum deneme sayisi
            
        Returns:
            GeneratedQuestion instance
            
        Raises:
            QuestionGenerationError: Uretim basarisiz
            InsufficientExamplesError: Yeterli ornek yok
        """
        # Step 1: Retrieval
        examples = self.retriever.retrieve_examples(combination, top_k=5)

        if len(examples) < 3:
            raise InsufficientExamplesError(
                f"Yetersiz ornek: {len(examples)} bulundu, minimum 3 gerekli"
            )

        # Step 2: Format
        formatted_examples = self.formatter.format_examples(examples, combination)

        # Step 3: Prompt
        system_prompt = self.prompt_builder.system_prompt
        user_prompt = self.prompt_builder.build_user_prompt(
            combination=combination,
            formatted_examples=formatted_examples,
            additional_instructions=style_instruction,
        )

        # Step 4 & 5: Generate with retry
        last_error = None
        for attempt in range(max_attempts):
            try:
                # Temperature: her denemede biraz artir
                temperature = 0.7 + (attempt * 0.1)

                # LLM cagri
                response = await self.llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                )

                # Parse JSON
                question_data = parse_llm_response(response)

                # Validate
                self._validate_question(question_data, combination)

                # Return
                return GeneratedQuestion(
                    id=str(uuid.uuid4()),
                    alt_konu=combination.get("alt_konu", ""),
                    zorluk=combination.get("zorluk", 3),
                    gorsel_tipi=combination.get("gorsel_tipi", "yok"),
                    hikaye=question_data.get("hikaye", ""),
                    soru=question_data.get("soru", ""),
                    gorsel_aciklama=question_data.get("gorsel_aciklama"),
                    secenekler=question_data.get("secenekler", {}),
                    dogru_cevap=question_data.get("dogru_cevap", "A"),
                    cozum=question_data.get("cozum", []),
                    kontroller=question_data.get("kontroller"),
                    metadata={
                        "attempt": attempt + 1,
                        "retrieval_count": len(examples),
                        "style_instruction": style_instruction,
                        "temperature": temperature,
                        "combination_lgs_skor": combination.get("lgs_skor"),
                    },
                    created_at=datetime.now(),
                )

            except json.JSONDecodeError as e:
                last_error = f"JSON parse hatasi: {e}"
            except ValidationError as e:
                last_error = f"Validasyon hatasi: {e}"
            except Exception as e:
                last_error = f"Beklenmeyen hata: {e}"

        raise QuestionGenerationError(
            f"Max {max_attempts} deneme sonrasi basarisiz. Son hata: {last_error}"
        )

    def _validate_question(self, data: dict, combination: dict):
        """
        Uretilen soruyu dogrula
        
        Args:
            data: LLM ciktisi
            combination: Hedef kombinasyon
            
        Raises:
            ValidationError: Validasyon basarisiz
        """
        errors = []

        # Zorunlu alanlar
        required_fields = ["hikaye", "soru", "secenekler", "dogru_cevap", "cozum"]
        for field in required_fields:
            if field not in data or not data[field]:
                errors.append(f"Eksik alan: {field}")

        # Secenek kontrolu
        if "secenekler" in data:
            secenekler = data["secenekler"]
            if len(secenekler) != 4:
                errors.append(f"4 secenek olmali, {len(secenekler)} var")

            expected_keys = {"A", "B", "C", "D"}
            if set(secenekler.keys()) != expected_keys:
                errors.append("Secenekler A, B, C, D olmali")

        # Dogru cevap kontrolu
        if "dogru_cevap" in data:
            dc = data["dogru_cevap"]
            if dc not in ["A", "B", "C", "D"]:
                errors.append(f"Gecersiz dogru_cevap: {dc}")

            if "secenekler" in data and dc not in data["secenekler"]:
                errors.append("dogru_cevap seceneklerde yok")

        # Cozum kontrolu
        if "cozum" in data:
            cozum = data["cozum"]
            if not isinstance(cozum, list) or len(cozum) < 1:
                errors.append("cozum en az 1 adim icermeli")

        # Gorsel kontrolu
        gorsel_tipi = combination.get("gorsel_tipi", "yok")
        if gorsel_tipi != "yok":
            if not data.get("gorsel_aciklama"):
                # Uyari, hata degil
                pass

        if errors:
            raise ValidationError(", ".join(errors))

    async def generate_batch(
        self,
        combinations: list[dict],
        ensure_diversity: bool = True,
        max_per_combination: int = 1,
    ) -> list[GeneratedQuestion]:
        """
        Toplu soru uretimi
        
        Args:
            combinations: Kombinasyon listesi
            ensure_diversity: Cesitlilik garantisi
            max_per_combination: Kombinasyon basina maksimum soru
            
        Returns:
            GeneratedQuestion listesi
        """
        results = []
        used_styles = []

        for combo in combinations:
            for _ in range(max_per_combination):
                try:
                    # Cesitlilik icin rastgele stil
                    style = None
                    if ensure_diversity:
                        style = self.prompt_builder.get_random_style(
                            exclude_recent=3, recent_styles=used_styles
                        )
                        used_styles.append(style)

                    question = await self.generate_question(
                        combination=combo,
                        style_instruction=style,
                    )
                    results.append(question)

                except (QuestionGenerationError, InsufficientExamplesError) as e:
                    # Hatali kombinasyonu atla, devam et
                    print(f"Kombinasyon atlandi: {combo}. Hata: {e}")
                    continue

        return results


async def generate_single_question(
    retriever: QuestionRetriever,
    llm_client: BaseLLMClient,
    combination: dict,
    style_instruction: Optional[str] = None,
) -> GeneratedQuestion:
    """
    Basit tek soru uretimi (factory fonksiyonu)
    
    Args:
        retriever: Retriever instance
        llm_client: LLM client instance
        combination: Hedef kombinasyon
        style_instruction: Stil talimati
        
    Returns:
        GeneratedQuestion
    """
    generator = QuestionGenerator(
        retriever=retriever,
        formatter=RAGOutputFormatter(format_type="markdown"),
        prompt_builder=PromptBuilder(),
        llm_client=llm_client,
    )

    return await generator.generate_question(
        combination=combination,
        style_instruction=style_instruction,
    )

