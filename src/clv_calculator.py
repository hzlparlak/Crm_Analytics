"""
Customer Lifetime Value (Müşteri Yaşam Boyu Değeri) hesaplama işlemlerini gerçekleştirir.
BG/NBD (Beta-Geometric/Negative Binomial Distribution) ve Gamma-Gamma modelleri kullanılır.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
warnings.filterwarnings('ignore')


class CLVCalculator:
    """Customer Lifetime Value (Müşteri Yaşam Boyu Değeri) hesaplama işlemlerini gerçekleştirir."""
    
    def __init__(self, data):
        """
        Args:
            data (pd.DataFrame): CLV hesaplaması yapılacak veri seti
        """
        self.data = data
        
    def prepare_for_bgnbd(self):
        """
        BG/NBD modeli için verileri hazırlar.
        
        Returns:
            pd.DataFrame: Müşteri başına RFM değerlerini içeren veri seti
            datetime: Veri setindeki son tarih
        """
        # Veri setindeki son tarih
        last_date = self.data['InvoiceDate'].max()
        
        # Müşteri bazında özet metrikleri hesapla
        summary = self.data.groupby('CustomerID').agg({
            'InvoiceDate': [
                lambda x: (last_date - x.min()).days,  # Müşteri yaşı (T)
                lambda x: (last_date - x.max()).days   # Son alışverişten bu yana geçen gün (recency)
            ],
            'InvoiceNo': lambda x: len(x.unique()),    # Fatura sayısı (frequency)
        })
        
        # Sütun isimlerini düzenle
        summary.columns = ['T', 'recency', 'frequency']
        
        # frequency-1 hesapla (BG/NBD modeli için)
        summary['frequency'] = summary['frequency'] - 1
        
        # İlk alışverişi yok sayarak tekrar satın alma var mı?
        summary = summary[summary['frequency'] > 0]
        
        return summary, last_date
    
    def fit_bgnbd_model(self):
        """
        BG/NBD modelini kurar ve müşteri yaşam boyu değerini tahmin eder.
        
        Returns:
            tuple: (model, summary DataFrame, predicted_purchases DataFrame)
        """
        try:
            from lifetimes import BetaGeoFitter
            from lifetimes.plotting import plot_frequency_recency_matrix
            from lifetimes.plotting import plot_probability_alive_matrix
            
            # Veriyi hazırla
            summary, last_date = self.prepare_for_bgnbd()
            
            # BG/NBD modelini oluştur ve fit et
            bgf = BetaGeoFitter(penalizer_coef=0.0)
            bgf.fit(summary['frequency'], summary['recency'], summary['T'])
            
            print("BG/NBD Model Parametreleri:")
            print(bgf.summary)
            
            # Frekans/Recency Matrisi
            plt.figure(figsize=(12, 8))
            plot_frequency_recency_matrix(bgf, T=summary['T'].max())
            plt.title('Frekans/Recency Matrisi: Beklenen Alışveriş Sayısı')
            plt.show()
            
            # Müşteri Hayatta Kalma Olasılık Matrisi
            plt.figure(figsize=(12, 8))
            plot_probability_alive_matrix(bgf)
            plt.title('Müşteri Hayatta Kalma Olasılık Matrisi')
            plt.show()
            
            # Gelecek 30/60/90 günlük satın alma tahminleri
            t_values = [30, 60, 90]
            predicted_purchases = pd.DataFrame()
            
            for t in t_values:
                predicted_purchases[f'predicted_purchases_{t}d'] = bgf.predict(t, 
                                                                            summary['frequency'],
                                                                            summary['recency'], 
                                                                            summary['T'])
            
            # Tahmin edilen satın alımları görselleştir
            plt.figure(figsize=(12, 6))
            ax = predicted_purchases[f'predicted_purchases_{t_values[0]}d'].hist(bins=50)
            plt.title(f'Gelecek {t_values[0]} Gün İçin Tahmin Edilen Satın Alma Dağılımı')
            plt.xlabel('Tahmin Edilen Satın Alma Sayısı')
            plt.ylabel('Müşteri Sayısı')
            plt.show()
            
            # En yüksek tahmin edilen satın alımlara sahip müşteriler
            predicted_purchases_with_id = predicted_purchases.copy()
            predicted_purchases_with_id.index = summary.index
            
            top_predicted = predicted_purchases_with_id[f'predicted_purchases_{t_values[0]}d'].sort_values(ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_predicted.values, y=top_predicted.index.astype(str))
            plt.title(f'En Yüksek {t_values[0]} Günlük Tahmin Edilen Satın Alma - Top 10 Müşteri')
            plt.xlabel(f'Tahmin Edilen {t_values[0]} Günlük Satın Alma')
            plt.ylabel('Müşteri ID')
            plt.tight_layout()
            plt.show()
            
            return bgf, summary, predicted_purchases_with_id
            
        except ImportError:
            print("'lifetimes' paketi bulunamadı. Lütfen pip install lifetimes komutu ile yükleyin.")
            return None, None, None


class BuyTillYouDieModels:
    """BG/NBD ve Gamma-Gamma modellerini kullanarak CLV tahmini yapar."""
    
    def __init__(self, data):
        """
        Args:
            data (pd.DataFrame): CLV hesaplaması yapılacak veri seti
        """
        self.data = data
        self.bgf_model = None
        self.ggf_model = None
        self.summary = None
    
    def prepare_transaction_data(self):
        """
        İşlem verilerini BG/NBD ve Gamma-Gamma modelleri için hazırlar.
        
        Returns:
            pd.DataFrame: Müşteri bazında hazırlanmış RFM değerleri
        """
        # Veri setindeki son tarih
        last_date = self.data['InvoiceDate'].max()
        
        # Müşteri bazında frecency ve monetary değerleri hesapla
        rfm = self.data.groupby('CustomerID').agg({
            'InvoiceDate': [
                lambda x: (last_date - x.min()).days,  # T (customer age)
                lambda x: (last_date - x.max()).days,  # recency
            ],
            'InvoiceNo': lambda x: len(x.unique()),  # frequency
            'TotalPrice': lambda x: x.sum()  # monetary value
        })
        
        # Sütun isimlerini düzenle
        rfm.columns = ['T', 'recency', 'frequency', 'monetary_value']
        
        # Monetary değeri 0'dan büyük olan ve en az bir kez alışveriş yapan müşterileri seç
        rfm = rfm[(rfm['monetary_value'] > 0) & (rfm['frequency'] > 0)]
        
        # frequency değerini 1 azalt (ilk satın almayı saymamak için)
        rfm['frequency'] = rfm['frequency'] - 1
        
        self.summary = rfm
        return rfm
    
    def fit_bgnbd_model(self):
        """
        BG/NBD modelini eğitir.
        
        Returns:
            lifetimes.BetaGeoFitter: Eğitilmiş BG/NBD modeli
        """
        try:
            from lifetimes import BetaGeoFitter
            
            # Veriyi hazırla
            if self.summary is None:
                self.prepare_transaction_data()
            
            # BG/NBD modelini oluştur ve eğit
            print("BG/NBD modeli eğitiliyor...")
            bgf = BetaGeoFitter(penalizer_coef=0.01)
            bgf.fit(self.summary['frequency'], self.summary['recency'], self.summary['T'])
            
            print("\nBG/NBD Model Parametreleri:")
            print(bgf.summary)
            
            self.bgf_model = bgf
            return bgf
            
        except ImportError:
            print("'lifetimes' paketi bulunamadı. Lütfen pip install lifetimes komutu ile yükleyin.")
            return None
    
    def fit_gamma_gamma_model(self):
        """
        Gamma-Gamma modelini eğitir (ortalama sipariş değeri tahmini için).
        
        Returns:
            lifetimes.GammaGammaFitter: Eğitilmiş Gamma-Gamma modeli
        """
        try:
            from lifetimes import GammaGammaFitter
            
            # Veriyi hazırla
            if self.summary is None:
                self.prepare_transaction_data()
            
            # Sadece birden fazla alışveriş yapan müşterileri seç
            ggf_summary = self.summary[self.summary['frequency'] > 0].copy()
            
            # Gamma-Gamma modelini oluştur ve eğit
            print("Gamma-Gamma modeli eğitiliyor...")
            ggf = GammaGammaFitter(penalizer_coef=0.01)
            ggf.fit(ggf_summary['frequency'], ggf_summary['monetary_value'])
            
            print("\nGamma-Gamma Model Parametreleri:")
            print(ggf.summary)
            
            self.ggf_model = ggf
            return ggf
            
        except ImportError:
            print("'lifetimes' paketi bulunamadı. Lütfen pip install lifetimes komutu ile yükleyin.")
            return None
    
    def predict_customer_ltv(self, time_horizon=12, discount_rate=0.01):
        """
        Müşteri yaşam boyu değeri (CLV) tahmini yapar.
        
        Args:
            time_horizon (int): Tahmin dönemi (ay cinsinden)
            discount_rate (float): İndirim oranı
            
        Returns:
            pd.DataFrame: CLV tahmini ve diğer müşteri metrikleri
        """
        # BG/NBD ve Gamma-Gamma modellerini eğit
        if self.bgf_model is None:
            self.fit_bgnbd_model()
        
        if self.ggf_model is None:
            self.fit_gamma_gamma_model()
        
        if self.bgf_model is None or self.ggf_model is None:
            print("Modeller eğitilemedi, CLV hesaplanamıyor.")
            return None
        
        # Gelecek 1 yıl (veya time_horizon ay) için beklenen satın alma sayısı
        self.summary['predicted_purchases'] = self.bgf_model.predict(
            time_horizon * 30,  # Gün cinsinden
            self.summary['frequency'],
            self.summary['recency'],
            self.summary['T']
        )
        
        # Beklenen ortalama sipariş değeri
        self.summary['expected_avg_value'] = self.ggf_model.conditional_expected_average_profit(
            self.summary['frequency'],
            self.summary['monetary_value']
        )
        
        # Müşteri yaşam boyu değeri hesaplama
        self.summary['clv'] = self.ggf_model.customer_lifetime_value(
            self.bgf_model,
            self.summary['frequency'],
            self.summary['recency'],
            self.summary['T'],
            self.summary['monetary_value'],
            time=time_horizon,
            discount_rate=discount_rate
        )
        
        # CLV Dağılımı
        plt.figure(figsize=(10, 6))
        sns.histplot(self.summary['clv'].clip(0, self.summary['clv'].quantile(0.99)), bins=50)
        plt.title(f'{time_horizon} Aylık Tahmini Müşteri Yaşam Boyu Değeri (CLV) Dağılımı')
        plt.xlabel('Tahmini CLV')
        plt.ylabel('Müşteri Sayısı')
        plt.tight_layout()
        plt.show()
        
        # En yüksek CLV'ye sahip müşteriler
        top_clv = self.summary.sort_values('clv', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='clv', y=top_clv.index.astype(str), data=top_clv)
        plt.title('En Yüksek Tahmini CLV - Top 10 Müşteri')
        plt.xlabel('Tahmini CLV')
        plt.ylabel('Müşteri ID')
        plt.tight_layout()
        plt.show()
        
        # Correlation between metrics
        correlation = self.summary[['frequency', 'recency', 'T', 'monetary_value', 
                                  'predicted_purchases', 'expected_avg_value', 'clv']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Müşteri Metrikleri Arasındaki Korelasyon')
        plt.tight_layout()
        plt.show()
        
        return self.summary