# Yazılım Mühendisliği Güncel Konular QA AI

**Yazılım Mühendisliği Güncel Konular QA AI**, LGS (Liseye Geçiş Sınavı) için matematik sorularını yapay zekâ ile **tahmin eden**, **üreten** ve **çözümlerini artırılmış gerçeklik (AR)** ile öğrencilere oyunlaştırılmış bir şekilde öğreten mobil bir eğitim platformudur.

---

## İçindekiler

- [Yazılım Mühendisliği Güncel Konular QA AI](#Yazılım-Mühendisliği-Güncel-Konular-QA-AI)
  - [İçindekiler](#i̇çindekiler)
  - [Proje Hakkında](#proje-hakkında)
  - [Özellikler](#özellikler)
  - [Mimari ve Teknolojiler](#mimari-ve-teknolojiler)
  - [Nasıl Kurulur](#nasıl-kurulur)
  - [Kullanım](#kullanım)
    - [Model Eğitimi](#model-eğitimi)
    - [AR / Unity Tarafı](#ar--unity-tarafı)
  - [Model Eğitimi Süreci](#model-eğitimi-süreci)
  - [AR / Oyunlaştırma](#ar--oyunlaştırma)
  - [Katkı Sağlama](#katkı-sağlama)
  - [Takım](#takım)
  - [Lisans](#lisans)
  - [İletişim](#i̇letişim)

---

## Proje Hakkında

Bu proje, LGS Matematik sınavı için gelecek yıl çıkabilecek soru kalıplarını **yapay zekâ kullanarak tahmin etmeyi ve yeni sorular üretmeyi** amaçlar. Üretilen sorular, **artırılmış gerçeklik (Unity ile)** bir ortamda öğrencinin etkileşimli bir şekilde çözmesine ve kavramları görsel olarak anlamasına olanak tanır.

Model, özellikle cebirsel ifadeler (örneğin çarpanlara ayırma, özdeşlikler) konusuna odaklanmıştır. Soru üretimi için **GAN benzeri mimariler** ve Manim kütüphanesi kullanılarak dinamik, parametreyle oynanabilir sorular üretilmesi planlanmaktadır.

---

## Özellikler

- Geleceğe yönelik LGS soru tahmini  
- Yapay zekâ ile **soru üretimi**  
- **Parametrik sorular** (değişkenler ile dinamik yapı)  
- Eğitim için AR tabanlı görselleştirme  
- Mobil platform desteği  
- Modüler altyapı — ileride Türkçe ve diğer konular eklenebilir  

---

## Mimari ve Teknolojiler

- **Model Eğitimi**  
  - Kaggle veri setleri (geometri problem otomatik formülasyonu)  
  - DeepMind AlphaGeometry araştırmaları  
  - GeoGen proje altyapısı  
  - GAN benzeri üretici modeller  
  - Manim ile soru şekli oluşturma  

- **Mobil Uygulama & AR**  
  - Unity  
  - AR kütüphaneleri (mobil destek)  
  - Oyunlaştırılmış kullanıcı deneyimi  

- **Proje Yönetimi**  
  - GitHub (kod & dokümantasyon)  
  - Trello (görev takibi)  
  - SWOT & SMART analizleri  

---

## Nasıl Kurulur

Aşağıda projeyi başlatmak için temel adımlar yer almaktadır:

1. Depoyu klonlayın:  
   ```bash
   git clone https://github.com/SIYAKS-ARES/ymgk-qa-ai.git
   cd ymgk-qa-ai
   ```

2. Python sanal ortam oluşturun ve etkinleştirin:

   ```bash
   python3 -m venv venv  
   source venv/bin/activate  # macOS / Linux  
   venv\Scripts\activate     # Windows  
   ```

3. Bağımlılıkları yükleyin:

   ```bash
   pip install -r requirements.txt
   ```

4. Manim ile soru görselleştirme için kurulum:

   * Manim’i belgelere göre kurun ve yapılandırın.
   * Gerekirse ek parametreler ve modüller ekleyin.

---

## Kullanım

### Model Eğitimi

* Verileri hazırlayın: `data/` dizinine geçmiş LGS sorularını yerleştirin.
* Modeli çalıştırmak için eğitim betiğini çalıştırın:

  ```bash
  python train_model.py --config configs/model_config.yaml
  ```
* Eğitilmiş modeli değerlendirin ve örnek sorular üretin:

  ```bash
  python generate_questions.py --model_path path/to/model --output_dir generated/
  ```

### AR / Unity Tarafı

* Unity projesini açın ve **AR sahnesini** yükleyin.
* Eğitilmiş modelden gelen soruları JSON veya uygun formatta alın.
* Unity içinden soruları görselleştirin ve kullanıcı etkileşimini test edin.

---

## Model Eğitimi Süreci

1. **Veri Toplama**: Geçmiş LGS matematik sorularını toplama ve etiketleme.
2. **Geometrik Kalıp Analizi**: Soruların geometrik senaryolarını sınıflandırma (örneğin kağıt katlama, alan hesabı vb.).
3. **Model Geliştirme**: GAN benzeri bir model ile yeni soru üretimi.
4. **Test & Doğrulama**: Üretilen soruların mantıksal ve pedagojik tutarlılığını inceleme.
5. **Yükleme / Dağıtım**: Eğitilmiş modeli Unity’ye entegre ederek AR deneyimi sunma.

---

## AR / Oyunlaştırma

* Öğrenci, artırılmış gerçeklik ortamında sorularla etkileşime girer.
* Soruların geometrik yapılarını 3B olarak görselleştirir.
* Adım adım çözüm süreci, animasyonlarla gösterilir (örneğin Manim ile oluşturulmuş şekiller).
* Geri bildirim mekanizmalarıyla öğrenci ilerlemesi ölçülür.

---

## Katkı Sağlama

Projeye katkıda bulunmak istersen:

1. Bu repoyu fork’la.
2. Yeni bir feature ya da düzeltme için branch oluştur (`git checkout -b feature-xyz`).
3. Değişikliklerini commit et (`git commit -m "Açıklayıcı mesaj"`).
4. Fork’undan main branch’ine pull request gönder.

Ayrıca proje yönetiminde Trello ve SWOT / SMART analizleri gibi alanlarda katkıda bulunmak isteyenler için belgeler mevcuttur.

---

## Takım

* Barış Kaya - Scrum Master, VR/AR geliştirme
* Mehmet Said Hüseyinoğlu — Yapay zekâ modelleme, soru üretimi
* Mehmet Fatih Akbaş - Frontend
* Utku Alyüz - Backend
* Özgün Deniz Sevilmiş — Backend

---

## Lisans

Bu proje açık kaynaklıdır. Lisans bilgisi burada: **(Lisans türünü buraya ekleyin, örneğin MIT, Apache 2.0)**

---

## İletişim

Herhangi bir sorunuz ya da katkı öneriniz olursa, lütfen **issue** açın ya da doğrudan ekip üyeleriyle iletişime geçin.

---
