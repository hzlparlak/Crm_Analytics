"""
Veri ön işleme işlemlerini gerçekleştirir.
"""

import pandas as pd


class DataPreprocessor:
    """Veri ön işleme işlemlerini gerçekleştirir."""
    
    @staticmethod
    def clean_data(data):
        """Veri setini temizler ve ön işleme adımlarını uygular."""
        print("Veri seti temizleniyor...")
        
        # Veri seti kopyasını oluştur
        df = data.copy()
        
        # Eksik değerleri kontrol et
        missing_values = df.isnull().sum()
        print(f"Eksik değerler:\n{missing_values}")
        
        # CustomerID olmayan kayıtları çıkar
        df = df[~df['CustomerID'].isna()]
        
        # Negatif veya sıfır miktarlı işlemleri çıkar
        df = df[df['Quantity'] > 0]
        
        # Negatif fiyatlı işlemleri çıkar
        df = df[df['UnitPrice'] > 0]
        
        # İptal işlemlerini çıkar (InvoiceNo C ile başlayanlar)
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        
        # CustomerID'yi integer'a çevir
        df['CustomerID'] = df['CustomerID'].astype(int)
        
        # InvoiceDate'i datetime formatına çevir
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Toplam tutar hesapla
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        print(f"Veri seti temizlendi. Yeni boyut: {df.shape}")
        return df