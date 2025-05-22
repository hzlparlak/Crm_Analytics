"""
RFM (Recency, Frequency, Monetary) analizi işlemlerini gerçekleştirir.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import warnings

# Grafik görüntüleme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')


class RFMAnalyzer:
    """RFM (Recency, Frequency, Monetary) analizi işlemlerini gerçekleştirir."""
    
    def __init__(self, data):
        """
        Args:
            data (pd.DataFrame): RFM analizi yapılacak veri seti
        """
        self.data = data
        self.rfm = None
        self.rfm_segments = None
    
    def calculate_rfm(self, reference_date=None):
        """
        RFM metriklerini hesaplar.
        
        Args:
            reference_date: RFM hesabı için referans tarihi. None ise veri setindeki en son tarih kullanılır.
        
        Returns:
            pd.DataFrame: RFM metrikleri hesaplanmış veri seti
        """
        # Referans tarihi belirleme (belirtilmemişse veri setindeki son tarihten 1 gün sonrası)
        if reference_date is None:
            reference_date = self.data['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        # Her müşteri için RFM değerlerini hesapla
        rfm_data = self.data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalPrice': 'sum'  # Monetary
        }).reset_index()
        
        # Sütun isimlerini güncelle
        rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        self.rfm = rfm_data
        print("RFM metrikleri hesaplandı.")
        return rfm_data
    
    def segment_customers(self, r_bins=5, f_bins=5, m_bins=5):
        """
        Müşterileri RFM değerlerine göre segmentlere ayırır.
        
        Args:
            r_bins (int): Recency için bölme sayısı
            f_bins (int): Frequency için bölme sayısı
            m_bins (int): Monetary için bölme sayısı
        
        Returns:
            pd.DataFrame: Segment bilgisi eklenmiş RFM veri seti
        """
        if self.rfm is None:
            self.calculate_rfm()
        
        # RFM skorları hesapla (r_score için düşük değerler iyi, f ve m için yüksek değerler iyi)
        self.rfm['R_Score'] = pd.qcut(self.rfm['Recency'], q=r_bins, labels=range(r_bins, 0, -1))
        self.rfm['F_Score'] = pd.qcut(self.rfm['Frequency'], q=f_bins, labels=range(1, f_bins+1))
        self.rfm['M_Score'] = pd.qcut(self.rfm['Monetary'], q=m_bins, labels=range(1, m_bins+1))
        
        # RFM skorlarını birleştirerek RFM segmentini oluştur
        self.rfm['RFM_Score'] = self.rfm['R_Score'].astype(str) + self.rfm['F_Score'].astype(str) + self.rfm['M_Score'].astype(str)
        
        # RFM Segment tanımlamaları
        self.rfm['Segment'] = 'Low-Value'
        self.rfm.loc[self.rfm['RFM_Score'].str[0].astype(int) >= 4, 'Segment'] = 'Champions'
        self.rfm.loc[(self.rfm['RFM_Score'].str[0].astype(int) >= 2) & 
                     (self.rfm['RFM_Score'].str[0].astype(int) < 4) & 
                     (self.rfm['RFM_Score'].str[1].astype(int) >= 3), 'Segment'] = 'Loyal Customers'
        self.rfm.loc[(self.rfm['RFM_Score'].str[0].astype(int) >= 3) & 
                     (self.rfm['RFM_Score'].str[1].astype(int) < 3), 'Segment'] = 'Potential Loyalists'
        self.rfm.loc[(self.rfm['RFM_Score'].str[0].astype(int) < 2) & 
                     (self.rfm['RFM_Score'].str[1].astype(int) >= 4), 'Segment'] = 'At Risk'
        self.rfm.loc[(self.rfm['RFM_Score'].str[0].astype(int) < 2) & 
                     (self.rfm['RFM_Score'].str[1].astype(int) < 2), 'Segment'] = 'Lost'
        
        self.rfm_segments = self.rfm
        
        print("Müşteri segmentasyonu tamamlandı.")
        return self.rfm_segments
    
    def visualize_segments(self):
        """Segment dağılımını görselleştirir."""
        if self.rfm_segments is None:
            self.segment_customers()
        
        plt.figure(figsize=(10, 6))
        segment_counts = self.rfm_segments['Segment'].value_counts()
        sns.barplot(x=segment_counts.index, y=segment_counts.values)
        plt.title('Müşteri Segment Dağılımı', fontsize=14)
        plt.xlabel('Segment')
        plt.ylabel('Müşteri Sayısı')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Segmentlere göre ortalama RFM değerleri
        segment_means = self.rfm_segments.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        })
        
        print("Segmentlere Göre Ortalama RFM Değerleri:")
        print(segment_means)
        
        # Radar chart ile segment karakteristiklerini gösterme
        segment_means_norm = segment_means.copy()
        for col in segment_means_norm.columns:
            if col == 'Recency':  # Recency için düşük değerler daha iyi
                segment_means_norm[col] = 1 - (segment_means_norm[col] - segment_means_norm[col].min()) / (segment_means_norm[col].max() - segment_means_norm[col].min())
            else:
                segment_means_norm[col] = (segment_means_norm[col] - segment_means_norm[col].min()) / (segment_means_norm[col].max() - segment_means_norm[col].min())
        
        # Radar chart
        categories = list(segment_means_norm.columns)
        N = len(categories)
        
        # Her segment için bir radar chart çizimi
        for segment in segment_means_norm.index:
            values = segment_means_norm.loc[segment].values.tolist()
            values += values[:1]  # Çemberi kapatmak için
            
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Çemberi kapatmak için
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.25)
            
            plt.xticks(angles[:-1], categories, size=12)
            plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], size=10, color="grey")
            plt.title(f'Segment: {segment}', size=15)
            plt.tight_layout()
            plt.show()