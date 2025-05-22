Online Retail CRM Analytics
Bu proje, UCI Online Retail veri setini kullanarak kapsamlı müşteri davranışı analizi gerçekleştirir. RFM analizi, churn prediction ve BG/NBD modeli ile Customer Lifetime Value (CLV) tahmini yapar.
🚀 Özellikler

Veri Yükleme ve Temizleme: Otomatik veri indirme ve ön işleme
Keşifsel Veri Analizi (EDA): Zamansal desenler, ülke ve ürün analizleri
RFM Analizi: Müşteri segmentasyonu (Recency, Frequency, Monetary)
Churn Analizi: Müşteri kaybı tahmini ve risk faktörleri
BG/NBD Modeli: Gelecekteki satın alma davranışı tahmini
CLV Hesaplama: Müşteri yaşam boyu değeri tahmini
Görselleştirmeler: Interaktif grafikler ve raporlar

## 📊 Temel Bulgular

### 1. Zamansal Analizler
<p align="center">
  <img src="Figure_1.png" width="45%">
  <img src="Figure_2.png" width="45%">
</p>

- **Günlük işlemlerde** belirgin mevsimsel dalgalanmalar gözlemlendi
- **Haftanın günleri** bazında işlem dağılımı:
  - En yoğun gün: Perşembe
  - En düşük işlem: Pazar

### 2. Coğrafi ve Ürün Analizleri
<p align="center">
  <img src="Figure_4.png" width="45%">
  <img src="Figure_5.png" width="45%">
</p>

- **Ülke bazında** işlemler:
  - %85'ten fazlası UK kaynaklı
  - İkinci sırada Almanya (%4.5)
  
- **En çok satan ürünler**:
  - Paper Craft ürünleri lider
  - Ev dekorasyon ürünleri öne çıkıyor

## 🛠 Teknik Yapı

### Klasör Yapısı
online-retail-crm/
├── data/ # Ham ve işlenmiş veriler
├── notebooks/ # Analiz not defterleri
├── src/ # Kaynak kodlar
│ ├── data_loader.py # Veri yükleme
│ ├── eda_analyzer.py # Keşifsel analiz
│ ├── rfm_analyzer.py # RFM segmentasyonu
│ ├── churn_analyzer.py # Churn tahmini
│ └── clv_calculator.py # CLV modelleri
├── main.py # Ana uygulama
├── requirements.txt # Bağımlılıklar
└── README.md # 

