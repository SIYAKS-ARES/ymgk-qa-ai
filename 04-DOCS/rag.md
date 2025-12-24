Model Katmanı Çıktıları ve RAG Soru Üretim Mantığı – Teknik Sözleşme (v1.0)

Amaç ve Kapsam

Bu doküman, LGS (Çarpanlar ve Katlar) soru üretim projesinde Model (offline) katmanının üreteceği çıktıları ve bu çıktılarla çalışan RAG + LLM (online) katmanının işleyiş mantığını “sözleşme (contract)” seviyesinde tanımlar.
Hedef: RAG tarafının, model detaylarını bilmeden yalnızca model çıktılarıyla sistem kurabilmesi.

Tanımlar

• Kombinasyon: (alt_konu, zorluk, gorsel_tipi) üçlüsü
• LGS Uygunluk Skoru (lgs_skor): Kombinasyonun “LGS soru yapısına benzerlik derecesi”ni ifade eden 0–1 arası değer
• Offline: Model eğitimi + skor üretimi + çıktı dosyalarının üretilmesi
• Online: configs.json kullanılarak RAG ile soru üretimi (LLM çağrıları)

Girdi Veri Seti (Model Tarafı)

Model katmanı şu tabloyu (CSV) temel alır: dataset_model_final.csv.
Örnek sütunlar: Alt_Konu, Zorluk, Gorsel_Tipi, Kaynak_Tipi, egitim_agirligi, is_LGS, ocr_*.
Not: Model eğitiminde egitim_agirligi örnek ağırlığı (cikmis>ornek>baslangic) olarak kullanılır; hedef etiket is_LGS yalnızca “LGS profili”ni öğrenmek içindir.

Model Katmanının Rolü (Netleştirme)

Model katmanı doğrudan soru üretmez ve “bu soru LGS’de kesin çıkar” şeklinde bir tahmin iddiası taşımaz.
Modelin görevi:

1) Veri setinden “LGS sorularının yapısal profilini” öğrenmek
2) Tüm olası kombinasyonlar için 0–1 arası LGS Uygunluk Skoru üretmek
3) Bu skorları RAG’in kullanacağı şekilde dışa aktarmak

Kesin Çıktılar (RAG’in Bağımlı Olduğu Dosyalar)

Model katmanı RAG tarafına aşağıdaki kesin çıktıları verir. RAG, modeli çalıştırmadan yalnızca bu çıktılarla çalışabilir.

A) configs.json (Zorunlu)
B) combination_scores.csv (Opsiyonel ama önerilir – debug/analiz)
C) model.pkl (Opsiyonel – yeniden skor üretmek/güncelleme için)
D) metadata.json (Opsiyonel – sürümleme/izlenebilirlik için)

A) configs.json – Şema (Zorunlu)

RAG tarafının ana sözleşme dosyasıdır. İçerik:
• model sürümü
• skor eşiği (threshold)
• kombinasyon listesi (her satır: alt_konu, zorluk, gorsel_tipi, lgs_skor)
• önerilen seçim politikası parametreleri (opsiyonel)

Önerilen şema ve örnek aşağıdadır.

configs.json – Örnek

{
  "schema_version": "1.0",
  "model_version": "lgs-struct-score-1.0",
  "created_at": "2025-12-23",
  "threshold": 0.75,
  "selection_policy": {
    "mode": "weighted_sampling",
    "temperature": 1.0,
    "top_k": 20
  },
  "combinations": [
    {
      "alt_konu": "carpanlar",
      "zorluk": 4,
      "gorsel_tipi": "tablo",
      "lgs_skor": 0.84
    },
    {
      "alt_konu": "ebob_ekok",
      "zorluk": 3,
      "gorsel_tipi": "sematik",
      "lgs_skor": 0.81
    }
  ]
}

B) combination_scores.csv (Opsiyonel, Önerilir)

Tüm kombinasyonların (yalnızca threshold üstü değil) skorlarını içerir. RAG bunu şart koşmaz; ancak analiz, raporlama ve hata ayıklama için faydalıdır.

Sütunlar (öneri): alt_konu, zorluk, gorsel_tipi, lgs_skor, rank, threshold_pass (0/1)

D) metadata.json (Opsiyonel)

Sürümleme ve izlenebilirlik için:
• eğitim veri seti hash / dosya adı
• feature listesi
• eğitim parametreleri
• eğitim metrikleri (varsa)
• random_seed

RAG Katmanı Mantığı (Online)

RAG tarafı şu adımlarla çalışır:

1) configs.json yüklenir
2) Kombinasyon seçilir (Top-K veya olasılıksal seçim)
3) Seçilen kombinasyona uygun örnek sorular veri deposundan çekilir (retrieval)
4) LLM promptu; kombinasyon + örnekler + çıktı formatı ile hazırlanır
5) LLM’den yeni soru üretilir
6) (Opsiyonel) Kalite kontrol / format doğrulama / tekrar üretim döngüsü

Kombinasyon Seçimi – Önerilen Politika

Deterministik tekrarları azaltmak için öneri:
• Önce threshold ile filtrele
• Kalanlar arasından weighted sampling: ağırlık = (lgs_skor^alpha), örn. alpha=2
• İsteğe bağlı sıcaklık (temperature) ile dağılımı yumuşat

Böylece hem yüksek skorlu yapılar öncelikli olur hem de çeşitlilik korunur.

Retrieval – Beklenen Girdi/Çıktı

Girdi: seçilen kombinasyon (alt_konu, zorluk, gorsel_tipi)
Çıktı: 3–8 adet benzer örnek soru (tercihen çıkmış + örnek)

Strateji (öneri):
• aynı alt_konu
• yakın zorluk (±1)
• aynı/benzer gorsel_tipi
• gerekiyorsa metin benzerliği (embedding) ile sıralama

LLM Prompt Sözleşmesi – Önerilen Alanlar

LLM’e verilen paket:
• hedef kombinasyon (alt_konu, zorluk, gorsel_tipi)
• 3–8 örnek soru (kısa içerik / şablon)
• çıktı formatı (JSON)
• kısıtlar (LGS tarzı, tek doğru seçenek, çözüm adımları vb.)

LLM Çıktı Formatı – Önerilen JSON

{
  "alt_konu": "carpanlar",
  "zorluk": 4,
  "gorsel_tipi": "tablo",
  "hikaye": "...",
  "soru": "...",
  "gorsel_aciklama": "...",
  "secenekler": {"A":"...", "B":"...", "C":"...", "D":"..."},
  "dogru_cevap": "B",
  "cozum": ["Adim 1 ...", "Adim 2 ..."],
  "kontroller": {
    "tek_dogru_mu": true,
    "format_ok": true
  }
}

Kalite Kontrol Önerileri

Hızlı kontroller:
• 4 seçenek var mı?
• doğru cevap seçeneklerde mi?
• alt_konu ile uyumlu mu? (basit kural kontrolü)
• zorluk: adım sayısı / işlem yoğunluğu ile tutarlı mı?

Başarısızsa: aynı kombinasyonla yeniden üretim veya bir alt sıradaki kombinasyona geçiş.

Repo Konum Önerisi

ymgk-qa-ai/
└── lgs-model/
    ├── data/processed/dataset_model_final.csv
    ├── outputs/
    │   ├── configs.json
    │   ├── combination_scores.csv
    │   ├── model.pkl
    │   └── metadata.json
    └── notebooks/ (colab ipynb)

Teslimat Notu

RAG tarafı için minimum teslimat: configs.json.
Önerilen teslimat paketi: configs.json + combination_scores.csv + metadata.json.
model.pkl yalnızca yeniden üretim/güncelleme için gereklidir.
