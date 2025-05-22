"""
Online Retail CRM Analytics - Ana Uygulama
Bu dosya tüm analiz süreçlerini koordine eder ve ana akışı yönetir.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os

# Kendi modüllerimizi import et
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.eda_analyzer import EDAAnalyzer
from src.rfm_analyzer import RFMAnalyzer
from src.clv_calculator import CLVCalculator, BuyTillYouDieModels
from src.churn_analyzer import CustomerChurnAnalyzer

# Grafik görüntüleme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
warnings.filterwarnings('ignore')

# Çalışma dizinini ayarla
def print_header(title):
    """Başlık yazdırma fonksiyonu"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_summary_metrics(data, churn_data=None):
    """Özet metriklerini yazdırır"""
    print_header("ÖZET METRİKLER")
    
    # Temel metrikler
    total_customers = data['CustomerID'].nunique()
    total_revenue = data['TotalPrice'].sum()
    total_orders = data['InvoiceNo'].nunique()
    avg_order_value = total_revenue / total_orders
    
    print(f"📊 Toplam Müşteri Sayısı: {total_customers:,}")
    print(f"💰 Toplam Gelir: £{total_revenue:,.2f}")
    print(f"🛒 Toplam Sipariş Sayısı: {total_orders:,}")
    print(f"💳 Ortalama Sipariş Değeri: £{avg_order_value:.2f}")
    
    # Churn metrikleri
    if churn_data is not None:
        active_customers = (~churn_data['IsChurned']).sum()
        churn_rate = churn_data['IsChurned'].mean()
        print(f"✅ Aktif Müşteri Sayısı: {active_customers:,} ({1-churn_rate:.2%})")
        print(f"⚠️  Churn Oranı: {churn_rate:.2%}")


def segment_clv_comparison(clv_predictions, rfm_segments):
    """Segment ve CLV karşılaştırması yapar"""
    if clv_predictions is not None and rfm_segments is not None:
        print_header("SEGMENT - CLV KARŞILAŞTIRMASI")
        
        # CLV ve segment bilgilerini birleştir
        segment_clv = pd.merge(
            clv_predictions, 
            rfm_segments[['Segment']], 
            left_index=True, 
            right_index=True
        )
        
        # Segmente göre ortalama CLV
        segment_clv_avg = segment_clv.groupby('Segment')['clv'].mean().sort_values(ascending=False)
        
        print("Segmentlere Göre Ortalama CLV:")
        for segment, avg_clv in segment_clv_avg.items():
            print(f"  {segment}: £{avg_clv:.2f}")
        
        # Görselleştirme
        plt.figure(figsize=(12, 6))
        sns.barplot(x=segment_clv_avg.index, y=segment_clv_avg.values)
        plt.title('Segmentlere Göre Ortalama Müşteri Yaşam Boyu Değeri (CLV)', fontsize=14)
        plt.xlabel('Segment')
        plt.ylabel('Ortalama CLV (£)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def main():
    """Ana uygulama fonksiyonu"""
    
    print_header("ONLINE RETAIL CRM ANALİTİKS")
    print("🚀 Müşteri Davranışı Analizi ve BG/NBD Modeli ile CLV Tahmini")
    print("📊 Veri Seti: UCI Online Retail Dataset")
    
    try:
        # 1. VERİ YÜKLEME
        print_header("1. VERİ YÜKLEME")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.zip"
        raw_data = DataLoader.download_and_extract(url)

        if raw_data is None:
            print("❌ Veri seti yüklenemedi. Program sonlandırılıyor.")
            return

        print(f"✅ Veri seti başarıyla yüklendi: {raw_data.shape[0]} satır, {raw_data.shape[1]} sütun")
        
        # 2. VERİ TEMİZLEME
        print_header("2. VERİ TEMİZLEME")
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_data(raw_data)
        
        print(f"✅ Veri temizleme tamamlandı")
        print(f"📊 Temizlenmiş veri boyutu: {clean_data.shape[0]} satır, {clean_data.shape[1]} sütun")
        
        # 3. KEŞİFSEL VERİ ANALİZİ (EDA)
        print_header("3. KEŞİFSEL VERİ ANALİZİ")
        eda = EDAAnalyzer(clean_data)
        
        print("📈 Temel veri özeti:")
        eda.show_basic_info()
        
        print("\n📅 Zamansal desen analizi...")
        eda.analyze_temporal_patterns()
        
        print("\n🌍 Ülke bazlı analiz...")
        eda.analyze_top_countries()
        
        print("\n🛍️ Ürün bazlı analiz...")
        eda.analyze_top_products()
        
        # 4. RFM ANALİZİ
        print_header("4. RFM ANALİZİ VE MÜŞTERİ SEGMENTASYONU")
        rfm_analyzer = RFMAnalyzer(clean_data)
        rfm_data = rfm_analyzer.calculate_rfm()
        rfm_segments = rfm_analyzer.segment_customers()
        rfm_analyzer.visualize_segments()
        
        print("✅ RFM analizi ve müşteri segmentasyonu tamamlandı")
        
        # 5. CHURN ANALİZİ
        print_header("5. MÜŞTERİ KAYBI (CHURN) ANALİZİ")
        churn_analyzer = CustomerChurnAnalyzer(clean_data)
        churn_data = churn_analyzer.define_churn(inactivity_threshold=90)
        
        print("📊 Churn prediction özellikleri hazırlanıyor...")
        churn_features = churn_analyzer.churn_prediction_features()
        
        print("🤖 Churn prediction modeli eğitiliyor...")
        churn_model, X_test, y_test, feature_importances = churn_analyzer.train_churn_model(churn_features)
        
        if churn_model is not None:
            print("✅ Churn prediction modeli başarıyla eğitildi")
        else:
            print("⚠️ Churn prediction modeli eğitilemedi (scikit-learn gerekli)")
        
        # 6. BG/NBD MODELİ VE CLV HESAPLAMA
        print_header("6. BG/NBD MODELİ VE CLV HESAPLAMA")
        
        # Basit CLV hesaplayıcısı
        print("📊 Basit CLV hesaplaması...")
        clv_calculator = CLVCalculator(clean_data)
        bgf_model, summary, predicted_purchases = clv_calculator.fit_bgnbd_model()
        
        # Gelişmiş BTYD modelleri
        print("🧮 Gelişmiş BG/NBD ve Gamma-Gamma modelleri...")
        btyd_models = BuyTillYouDieModels(clean_data)
        btyd_summary = btyd_models.prepare_transaction_data()
        
        bgf = btyd_models.fit_bgnbd_model()
        ggf = btyd_models.fit_gamma_gamma_model()
        
        clv_predictions = None
        if bgf is not None and ggf is not None:
            clv_predictions = btyd_models.predict_customer_ltv(time_horizon=12)
            print("✅ CLV tahmini tamamlandı")
        else:
            print("⚠️ CLV modelleri eğitilemedi (lifetimes paketi gerekli)")
        
        # 7. SONUÇLARIN KARŞILAŞTIRILMASI
        print_header("7. SONUÇLARIN ANALİZİ")
        
        # Özet metrikler
        print_summary_metrics(clean_data, churn_data)
        
        # Segment-CLV karşılaştırması
        segment_clv_comparison(clv_predictions, rfm_segments)
        
        # 8. ÖNERİLER VE AKSİYON PLANLARİ
        print_header("8. İŞ ÖNERİLERİ VE AKSİYON PLANLARI")
        
        print("💡 Ana Bulgular ve Öneriler:")
        print("\n🎯 MÜŞTERİ SEGMENTASYONU ÖNERİLERİ:")
        print("   • Champions: Premium hizmet ve özel kampanyalar sunun")
        print("   • Loyal Customers: Sadakat programları ile elde tutun")  
        print("   • At Risk: Geri kazanma kampanyaları düzenleyin")
        print("   • Lost: Win-back kampanyaları veya müşteri araştırması yapın")
        
        print("\n📈 CHURN PREVENTION STRATEJİLERİ:")
        print("   • Erken uyarı sistemi kurarak risk altındaki müşterileri belirleyin")
        print("   • Kişiselleştirilmiş teklif ve iletişim stratejileri geliştirin")
        print("   • Müşteri deneyimini iyileştirecek aksiyonlar alın")
        
        print("\n💰 CLV OPTIMIZATION:")
        print("   • Yüksek CLV'li müşterilere odaklanın")
        print("   • Cross-sell ve up-sell fırsatlarını değerlendirin")
        print("   • Müşteri kazanım maliyetlerini CLV ile karşılaştırın")
        
        print("\n📊 VERİ DRIVEN PAZARLAMA:")
        print("   • RFM skorlarına göre pazarlama bütçesi dağıtın")
        print("   • Seasonal patternleri göz önünde bulundurarak kampanya planlayın")
        print("   • A/B testleri ile stratejileri sürekli optimize edin")
        
        # 9. NEXT STEPS
        print_header("9. GELİŞTİRME ÖNERİLERİ")
        print("🔮 Gelecek Geliştirmeler:")
        print("   • Real-time dashboard oluşturma")
        print("   • Otomatik alert sistemi kurma")
        print("   • Machine Learning model ensemble")
        print("   • Cohort analysis ekleme")
        print("   • Product recommendation engine")
        
        print_header("ANALİZ TAMAMLANDI! 🎉")
        print("📈 Tüm analizler başarıyla gerçekleştirildi.")
        print("💼 Sonuçları iş stratejilerinizde kullanabilirsiniz.")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {str(e)}")
        print("🔧 Lütfen gerekli paketlerin yüklendiğinden emin olun:")
        print("   pip install pandas numpy matplotlib seaborn scikit-learn lifetimes")


if __name__ == "__main__":
    print("🔄 Program başlatılıyor...")
    main()
    print("\n👋 Program tamamlandı. İyi günler!")