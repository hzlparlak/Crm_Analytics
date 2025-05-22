import pandas as pd
import urllib.request
import zipfile
import io

class DataLoader:
    """Veri setini yükleme ve hazırlama işlemlerini yönetir."""
    
    @staticmethod
    def download_and_extract(url):
        print("Veri seti indiriliyor...")
        
        try:
            response = urllib.request.urlopen(url)
            zip_data = io.BytesIO(response.read())
            
            with zipfile.ZipFile(zip_data) as zip_ref:
                file_list = zip_ref.namelist()
                excel_files = [f for f in file_list if f.endswith('.xlsx')]
                
                if excel_files:
                    excel_file = excel_files[0]
                    with zip_ref.open(excel_file) as f:
                        data = pd.read_excel(io.BytesIO(f.read()))
                    print(f"Veri seti başarıyla yüklendi: {excel_file}")
                    return data
                else:
                    print("ZIP dosyasında excel dosyası bulunamadı.")
                    return None
        except Exception as e:
            print(f"Veri seti indirme hatası: {e}")
            try:
                direct_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
                print("Alternatif yol deneniyor...")
                data = pd.read_excel(direct_url)
                print("Veri seti başarıyla yüklendi")
                return data
            except Exception as e2:
                print(f"Alternatif yol hatası: {e2}")
                return None
