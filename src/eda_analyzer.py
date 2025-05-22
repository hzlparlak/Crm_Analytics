"""
Keşifsel veri analizi işlemlerini gerçekleştirir.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Grafik görüntüleme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
warnings.filterwarnings('ignore')


class EDAAnalyzer:
    """Keşifsel veri analizi işlemlerini gerçekleştirir."""
    
    def __init__(self, data):
        """
        Args:
            data (pd.DataFrame): Analiz edilecek veri seti
        """
        self.data = data
    
    def show_basic_info(self):
        """Veri seti hakkında temel bilgileri gösterir."""
        print("Veri Seti Özeti:")
        print(f"Satır sayısı: {self.data.shape[0]}")
        print(f"Sütun sayısı: {self.data.shape[1]}")
        print("\nSütun tipleri:")
        print(self.data.dtypes)
        print("\nİlk 5 satır:")
        print(self.data.head())
        
        print("\nTemel istatistikler:")
        print(self.data.describe())
    
    def analyze_temporal_patterns(self):
        """Zamansal desenleri analiz eder."""
        # Günlük işlem sayısı 
        daily_transactions = self.data.groupby(self.data['InvoiceDate'].dt.date).size()
        
        plt.figure(figsize=(12, 6))
        daily_transactions.plot()
        plt.title('Günlük İşlem Sayısı', fontsize=14)
        plt.xlabel('Tarih')
        plt.ylabel('İşlem Sayısı')
        plt.tight_layout()
        plt.show()
        
        # Haftanın günlerine göre analiz
        self.data['DayOfWeek'] = self.data['InvoiceDate'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        plt.figure(figsize=(10, 6))
        day_counts = self.data['DayOfWeek'].value_counts().reindex(day_order)
        sns.barplot(x=day_counts.index, y=day_counts.values)
        plt.title('Haftanın Günlerine Göre İşlem Sayısı', fontsize=14)
        plt.xlabel('Gün')
        plt.ylabel('İşlem Sayısı')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Saatlere göre analiz
        self.data['Hour'] = self.data['InvoiceDate'].dt.hour
        
        plt.figure(figsize=(12, 6))
        hour_counts = self.data['Hour'].value_counts().sort_index()
        sns.barplot(x=hour_counts.index, y=hour_counts.values)
        plt.title('Saatlere Göre İşlem Sayısı', fontsize=14)
        plt.xlabel('Saat')
        plt.ylabel('İşlem Sayısı')
        plt.tight_layout()
        plt.show()
    
    def analyze_top_countries(self):
        """En çok işlem yapılan ülkeleri analiz eder."""
        country_counts = self.data['Country'].value_counts().head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=country_counts.values, y=country_counts.index)
        plt.title('En Çok İşlem Yapılan 10 Ülke', fontsize=14)
        plt.xlabel('İşlem Sayısı')
        plt.ylabel('Ülke')
        plt.tight_layout()
        plt.show()
    
    def analyze_top_products(self):
        """En çok satılan ürünleri analiz eder."""
        product_quantity = self.data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=product_quantity.values, y=product_quantity.index)
        plt.title('En Çok Satılan 10 Ürün (Miktar)', fontsize=14)
        plt.xlabel('Satış Miktarı')
        plt.ylabel('Ürün')
        plt.tight_layout()
        plt.show()