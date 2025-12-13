# 🍎 Derin Öğrenme ile Otomatik Meyve Tazelik Tespiti (Fruit Freshness Detection)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📖 Proje Hakkında (Abstract)
Bu proje, **Bilimsel Araştırma Yöntemleri** dersi kapsamında geliştirilmiştir. Gıda israfını önlemek ve gıda güvenliğini sağlamak amacıyla, derin öğrenme (Deep Learning) teknikleri kullanılarak meyvelerin taze veya çürük olduğunu tespit eden otomatik bir sistem tasarlanmıştır.

Proje, insan gözünün gözden kaçırabileceği mikroskobik bozulmaları ve doku değişimlerini analiz etmek için **CNN (Convolutional Neural Networks)** mimarisini kullanır. Ayrıca son kullanıcılar için **Streamlit** tabanlı interaktif bir arayüz geliştirilmiştir.


## 🎯 Özellikler
* **Çoklu Sınıflandırma:** Elma, Muz ve Portakal için hem "Taze" hem "Çürük" tespiti.
* **Yüksek Doğruluk:** CNN algoritması ile %90+ başarı oranı.
* **Gıda Güvenliği Uyarıları:** Tespit edilen çürük türüne göre (Örn: Patulin toksini, Fermentasyon riski) kullanıcıya özel bilimsel uyarılar verir.
* **Kullanıcı Dostu Arayüz:** Sürükle-bırak yöntemiyle çalışan web arayüzü.

## 📂 Veri Seti (Dataset)
Bu çalışmada literatürde kabul görmüş **"Fruits Fresh and Rotten for Classification"** veri seti kullanılmıştır.
* **Kaynak:** Kaggle (Sriram R.)
* **Sınıflar:** * `freshapples`, `freshbanana`, `freshoranges`
    * `rottenapples`, `rottenbanana`, `rottenoranges`
* **Veri Ön İşleme:** Görüntüler 150x150 piksel boyutuna getirilmiş, normalize edilmiş (0-1) ve Data Augmentation (Veri Artırma) teknikleri uygulanmıştır.

## 🛠️ Kurulum (Installation)

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1.  **Repoyu Klonlayın:**
    ```bash
    git clone [https://github.com/KULLANICI_ADIN/fruit-freshness-detection.git](https://github.com/KULLANICI_ADIN/fruit-freshness-detection.git)
    cd fruit-freshness-detection
    ```

2.  **Sanal Ortam Oluşturun (Önerilen):**
    ```bash
    conda create -n meyve_projesi python=3.9
    conda activate meyve_projesi
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install tensorflow streamlit matplotlib numpy pillow
    ```

## 🚀 Kullanım (Usage)

### 1. Modeli Eğitmek (Opsiyonel)
Eğer modeli sıfırdan eğitmek isterseniz Jupyter Notebook dosyalarını kullanabilirsiniz:
* `notebooks/01_veri_inceleme.ipynb`: Veri setini analiz eder.
* `notebooks/02_model_egitimi.ipynb`: CNN modelini eğitir ve `.h5` olarak kaydeder.

### 2. Arayüzü Başlatmak
Eğitilmiş model ile arayüzü çalıştırmak için terminale şu komutu yazın:
```bash
streamlit run app.py