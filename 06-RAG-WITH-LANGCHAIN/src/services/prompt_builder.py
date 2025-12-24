"""
R-05: Prompt Tasarimi
LGS tarzi soru uretimi icin system ve user prompt sablonlari
"""

from typing import Optional
import random


class PromptBuilder:
    """
    LGS soru uretimi icin prompt olusturucu
    
    Ozellikler:
    - LGS formatina uygun system prompt
    - Dinamik user prompt
    - Stil varyasyonlari (cesitlilik icin)
    - JSON cikti formati
    """

    def __init__(self):
        self.system_prompt = self._build_system_prompt()
        self._style_variations = self._get_style_variations()

    def _build_system_prompt(self) -> str:
        """LGS soru yazari sistem promptu"""
        return """Sen, LGS (Liseye Gecis Sinavi) matematik sorulari yazan uzman bir egitimcisin.

## Rol ve Gorev
- 8. sinif ogrencilerine yonelik matematik sorulari yaziyorsun
- Sorular MEB mufredat ve LGS formatlarina tam uygun olmali
- Her soru tek dogru cevapli, 4 secenekli (A, B, C, D) olmali

## Konu Alani: Carpanlar ve Katlar
Asagidaki alt konularda soru yazabilirsin:
1. **carpanlar**: Bir dogal sayinin carpanlari, asal carpanlar, carpan sayisi
2. **ebob_ekok**: En buyuk ortak bolen, en kucuk ortak kat, problemler
3. **aralarinda_asal**: Aralarinda asal sayilar, ozellikler

## LGS Soru Ozellikleri
1. **Hikaye/Baglam**: Gercek hayat senaryolari (alisveris, paylasim, zaman, geometri)
2. **Cok Adimli**: 2-4 adim cozum gerektiren yapilar
3. **Yaniltici Secenekler**: Dikkat gerektiren, mantikli yanlis secenekler
4. **Gorsel Destek**: Gerektiginde tablo, sema, sekil aciklamasi

## Zorluk Seviyeleri
| Seviye | Aciklama | Adim Sayisi |
|--------|----------|-------------|
| 1 | Temel kavram, dogrudan uygulama | 1 |
| 2 | Iki islemli, basit problem | 2 |
| 3 | Orta karmasiklik, mantik gerektiren | 2-3 |
| 4 | Ileri duzey, cok adimli | 3-4 |
| 5 | Olimpiyat tarzi, en zor | 4+ |

## Gorsel Tipi Kurallari
- **yok**: Gorsel aciklama verme
- **tablo**: Satirlar, sutunlar ve basliklar iceren tablo tanimla
- **sematik**: Ok, kutu, iliskiler iceren sema tanimla
- **geometrik_sekil**: Uzunluk, alan, cevre iceren sekil tanimla
- **resimli**: Gercek hayat objelerini icerecek sekilde tanimla

## Cikti Kurallari
1. JSON formatinda yanit ver
2. Turkce dil kurallarina uy
3. Matematiksel notasyon dogru kullan (^, sqrt, x, vb.)
4. Secenekler arasi mantikli aralik birak
5. Cozum adimlarini acik yaz
6. SADECE istenen JSON ciktisini ver, baska aciklama ekleme"""

    def _get_style_variations(self) -> list[str]:
        """Cesitlilik icin stil varyasyonlari"""
        return [
            "Hikayede bir ogrenci senaryosu kullan.",
            "Hikayede bir is yeri/fabrika senaryosu kullan.",
            "Hikayede bir spor/oyun senaryosu kullan.",
            "Hikayede bir aile/ev senaryosu kullan.",
            "Hikayede bir okul/sinif senaryosu kullan.",
            "Sayilari 100'den buyuk sec.",
            "Sayilari 50'den kucuk sec.",
            "Negatif bir durum iceren senaryo kullan (eksik, kayip, vb.).",
            "Karsilastirma iceren bir senaryo kullan.",
            "Zaman iceren bir senaryo kullan (gun, saat, dakika).",
            "Alisveris senaryosu kullan.",
            "Paylasim/bolme senaryosu kullan.",
            "Yarisma veya yarismaci senaryosu kullan.",
            "Toplama veya birlestirme senaryosu kullan.",
            "Seyahat veya yolculuk senaryosu kullan.",
        ]

    def build_user_prompt(
        self,
        combination: dict,
        formatted_examples: str,
        additional_instructions: Optional[str] = None,
    ) -> str:
        """
        Kullanici promptu olustur
        
        Args:
            combination: Hedef kombinasyon
            formatted_examples: Formatlanmis ornek sorular
            additional_instructions: Ek talimatlar (stil, vb.)
            
        Returns:
            User prompt string
        """
        prompt_parts = []

        # Gorev tanimi
        prompt_parts.append(
            "Asagidaki kombinasyona uygun YENI ve OZGUN bir LGS matematik sorusu uret."
        )
        prompt_parts.append("")

        # Ornekler
        prompt_parts.append(formatted_examples)
        prompt_parts.append("")

        # Ozel talimatlar
        if additional_instructions:
            prompt_parts.append("## Ek Talimatlar")
            prompt_parts.append(additional_instructions)
            prompt_parts.append("")

        # Cikti formati
        prompt_parts.append("## Beklenen JSON Ciktisi")
        prompt_parts.append(self._get_output_format())
        prompt_parts.append("")

        # Onemli uyarilar
        prompt_parts.append("## Onemli Uyarilar")
        prompt_parts.append("- Orneklerden FARKLI, tamamen yeni bir soru uret")
        prompt_parts.append("- Sayilari, hikayeyi ve baglami degistir")
        prompt_parts.append(f"- Zorluk seviyesi {combination.get('zorluk', 3)}/5 olmali")

        gorsel_tipi = combination.get("gorsel_tipi", "yok")
        if gorsel_tipi != "yok":
            prompt_parts.append(
                f"- {gorsel_tipi} tipinde gorsel icin DETAYLI aciklama ver"
            )
        else:
            prompt_parts.append("- gorsel_aciklama alanini null yap")

        prompt_parts.append("- Sadece JSON ciktisi ver, baska aciklama ekleme")

        return "\n".join(prompt_parts)

    def _get_output_format(self) -> str:
        """JSON cikti formati sablonu"""
        return """```json
{
  "hikaye": "Sorunun gercek hayat baglami/hikayesi...",
  "soru": "Asil soru metni (Buna gore... ile bitebilir)",
  "gorsel_aciklama": "Gorsel tipi 'yok' degilse detayli gorsel tanimi, degilse null",
  "secenekler": {
    "A": "Secenek A degeri",
    "B": "Secenek B degeri",
    "C": "Secenek C degeri",
    "D": "Secenek D degeri"
  },
  "dogru_cevap": "A/B/C/D",
  "cozum": [
    "Adim 1: Ilk islem aciklamasi",
    "Adim 2: Ikinci islem aciklamasi",
    "Adim 3: Sonuc"
  ],
  "kontroller": {
    "tek_dogru_mu": true,
    "format_ok": true,
    "secenek_sayisi": 4
  }
}
```"""

    def get_random_style(self, exclude_recent: int = 3, recent_styles: Optional[list[str]] = None) -> str:
        """
        Rastgele stil secimi
        
        Args:
            exclude_recent: Son N stili haric tut
            recent_styles: Son kullanilan stiller
            
        Returns:
            Stil talimati
        """
        recent_styles = recent_styles or []
        available = [s for s in self._style_variations if s not in recent_styles[-exclude_recent:]]

        if not available:
            available = self._style_variations

        return random.choice(available)

    def get_all_styles(self) -> list[str]:
        """Tum stil varyasyonlarini don"""
        return self._style_variations.copy()


# Hazir prompt sablonlari
PROMPT_TEMPLATES = {
    "default": PromptBuilder(),
}


def get_prompt_builder(template: str = "default") -> PromptBuilder:
    """
    Prompt builder instance al
    
    Args:
        template: Sablon adi
        
    Returns:
        PromptBuilder instance
    """
    if template not in PROMPT_TEMPLATES:
        PROMPT_TEMPLATES[template] = PromptBuilder()
    return PROMPT_TEMPLATES[template]

