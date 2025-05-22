"""
Müşteri kaybı (churn) analizi işlemlerini gerçekleştirir.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')


class CustomerChurnAnalyzer:
    """Müşteri kaybı (churn) analizi işlemlerini gerçekleştirir."""
    
    def __init__(self, data):
        """
        Args:
            data (pd.DataFrame): Churn analizi yapılacak veri seti
        """
        self.data = data
    
    def define_churn(self, inactivity_threshold=90):
        """
        Churn tanımını belirler ve müşterileri churn olarak işaretler.
        
        Args:
            inactivity_threshold (int): Churn olarak tanımlanacak inaktif gün sayısı eşiği
            
        Returns:
            pd.DataFrame: Churn bilgisi eklenmiş müşteri veri seti
        """
        # Son tarih ve müşteri bazlı son işlem tarihi
        last_date = self.data['InvoiceDate'].max()
        
        # Müşteri bazında son işlem tarihini hesapla
        customer_last_purchase = self.data.groupby('CustomerID')['InvoiceDate'].max().reset_index()
        customer_last_purchase['DaysSinceLastPurchase'] = (last_date - customer_last_purchase['InvoiceDate']).dt.days
        
        # Churn tanımı: Son X günden fazla süredir işlem yapmayan müşteriler
        customer_last_purchase['IsChurned'] = customer_last_purchase['DaysSinceLastPurchase'] > inactivity_threshold
        
        # Churn oranını hesapla
        churn_rate = customer_last_purchase['IsChurned'].mean()
        print(f"Churn Oranı ({inactivity_threshold} gün inaktiflik eşiği): {churn_rate:.2%}")
        
        # Churn ve aktif müşteri sayısı
        churn_count = customer_last_purchase['IsChurned'].sum()
        active_count = len(customer_last_purchase) - churn_count
        
        # Pasta grafiği ile görselleştirme
        plt.figure(figsize=(8, 8))
        plt.pie([active_count, churn_count], 
                labels=['Aktif', 'Churn'], 
                autopct='%1.1f%%',
                colors=['green', 'red'],
                explode=[0, 0.1],
                startangle=90)
        plt.title(f'Müşteri Churn Oranı ({inactivity_threshold} Gün İnaktiflik)', fontsize=14)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        # Son alışverişten bu yana geçen gün dağılımı
        plt.figure(figsize=(12, 6))
        sns.histplot(data=customer_last_purchase, x='DaysSinceLastPurchase', bins=30)
        plt.axvline(x=inactivity_threshold, color='r', linestyle='--', 
                   label=f'{inactivity_threshold} Gün Eşiği')
        plt.title('Son Alışverişten Bu Yana Geçen Gün Dağılımı', fontsize=14)
        plt.xlabel('Son Alışverişten Bu Yana Geçen Gün')
        plt.ylabel('Müşteri Sayısı')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return customer_last_purchase
    
    def churn_prediction_features(self, inactivity_threshold=90):
        """
        Churn tahmini için özellikler oluşturur.
        
        Args:
            inactivity_threshold (int): Churn olarak tanımlanacak inaktif gün sayısı eşiği
            
        Returns:
            pd.DataFrame: Churn tahmini için özellikler içeren veri seti
        """
        # Son tarih
        last_date = self.data['InvoiceDate'].max()
        
        # Müşteri bazında özetleme
        customer_features = self.data.groupby('CustomerID').agg({
            'InvoiceDate': [
                lambda x: (x.max() - x.min()).days,  # Müşteri ilişki süresi
                lambda x: (last_date - x.max()).days,  # Son alışverişten bu yana geçen gün
                'count'  # Toplam işlem sayısı
            ],
            'InvoiceNo': 'nunique',  # Farklı fatura sayısı
            'Quantity': ['sum', 'mean', 'std'],  # Miktar istatistikleri
            'TotalPrice': ['sum', 'mean', 'std']  # Tutar istatistikleri
        })
        
        # Sütun isimlerini düzenle
        customer_features.columns = [
            'CustomerLifetime', 'DaysSinceLastPurchase', 'TotalTransactions',
            'UniqueInvoices', 'TotalQuantity', 'AvgQuantity', 'StdQuantity',
            'TotalSpend', 'AvgSpend', 'StdSpend'
        ]
        
        # Ortalama sipariş değeri
        customer_features['AvgOrderValue'] = customer_features['TotalSpend'] / customer_features['UniqueInvoices']
        
        # Sipariş frekansı (ayda kaç sipariş)
        customer_features['PurchaseFrequency'] = customer_features['UniqueInvoices'] / (customer_features['CustomerLifetime'] / 30)
        
        # Churn durumu ekle
        customer_features['IsChurned'] = customer_features['DaysSinceLastPurchase'] > inactivity_threshold
        
        # Eksik değerleri doldur
        customer_features = customer_features.fillna(0)
        
        return customer_features
    
    def train_churn_model(self, features=None, inactivity_threshold=90):
        """
        Churn tahmin modeli eğitir.
        
        Args:
            features (pd.DataFrame): Model için özellikler. None ise oluşturulur.
            inactivity_threshold (int): Churn olarak tanımlanacak inaktif gün sayısı eşiği
            
        Returns:
            tuple: (model, X_test, y_test, feature_importances)
        """
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
            
            # Özellikleri hazırla
            if features is None:
                features = self.churn_prediction_features(inactivity_threshold)
            
            # Aşırı dengesiz veri kontrolü
            churn_ratio = features['IsChurned'].mean()
            print(f"Churn Oranı: {churn_ratio:.2%}")
            
            # Özellikler ve hedef
            X = features.drop(['IsChurned'], axis=1)
            y = features['IsChurned']
            
            # Eğitim ve test setlerine ayır
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Model eğitimi
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=8,
                min_samples_split=10,
                random_state=42,
                class_weight='balanced'  # Dengesiz veri için ağırlık uygula
            )
            
            model.fit(X_train, y_train)
            
            # Tahmin
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Model değerlendirme
            print("\nModel Değerlendirme:")
            print(classification_report(y_test, y_pred))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            
            # ROC Eğrisi
            auc_score = roc_auc_score(y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Churn Prediction')
            plt.legend()
            plt.show()
            
            # Özellik önemliliği
            feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
            plt.title('En Önemli 10 Özellik - Churn Tahmini')
            plt.tight_layout()
            plt.show()
            
            return model, X_test, y_test, feature_importances
            
        except ImportError:
            print("scikit-learn paketi bulunamadı. Lütfen pip install scikit-learn komutu ile yükleyin.")
            return None, None, None, None