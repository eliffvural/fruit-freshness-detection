import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Meyve Tazelik Testi",
    page_icon="🍎",
    layout="centered"
)

# --- BAŞLIK VE AÇIKLAMA ---
st.title("🍎 Meyve Tazelik ve Çürük Tespit Sistemi")
st.markdown("""
Bu sistem, Derin Öğrenme (CNN) teknolojisi kullanarak yüklediğiniz meyve fotoğrafının 
**Taze** mi yoksa **Çürük** mü olduğunu analiz eder.
*Desteklenen Meyveler: Elma, Muz, Portakal*
""")

# --- MODELİ YÜKLEME (CACHE İLE HIZLANDIRMA) ---
# Modeli her seferinde tekrar yüklememek için belleğe alıyoruz
@st.cache_resource
def load_my_model():
    # Modelin yolunu buraya yazıyoruz. Klasör yapına göre:
    model = tf.keras.models.load_model('models/fruit_cnn_model.h5')
    return model

# Modeli yüklemeyi dene, hata varsa kullanıcıya söyle
try:
    model = load_my_model()
    st.success("Yapay Zeka Modeli Başarıyla Yüklendi ve Hazır! ✅")
except:
    st.error("Model dosyası bulunamadı! Lütfen 'models/fruit_cnn_model.h5' yolunu kontrol edin.")

# --- SINIF İSİMLERİ (ALFABETİK SIRA ÖNEMLİ) ---
class_names = ['Taze Elma', 'Taze Muz', 'Taze Portakal', 
               'Çürük Elma', 'Çürük Muz', 'Çürük Portakal']

# --- RESİM YÜKLEME ALANI ---
uploaded_file = st.file_uploader("Lütfen bir meyve fotoğrafı yükleyin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Resmi Göster
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Fotoğraf', use_column_width=True)
    
    # 2. Resmi Modele Hazırla (Pre-processing)
    st.write("Analiz ediliyor...")
    
    # Resmi modelin istediği boyuta (150x150) getir
    img_resized = image.resize((150, 150))
    
    # NumPy dizisine çevir
    img_array = np.array(img_resized)
    
    # Eğer resim PNG ise ve 4 kanallıysa (RGBA), RGB'ye çevir
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # Boyut genişlet (1, 150, 150, 3) yap
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize et (0-1 arası) - Eğitimde yaptığımızın aynısı!
    img_array = img_array / 255.0
    
    # 3. Tahmin Yap
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) # Olasılıkları hesapla
    
    predicted_class_index = np.argmax(predictions)
    predicted_label = class_names[predicted_class_index]
    confidence = np.max(predictions) * 100
    
    # 4. Sonucu Ekrana Bas
    st.write("---")
    
    # Eğer sonuç "Taze" ise Yeşil, "Çürük" ise Kırmızı gösterelim
    if "Taze" in predicted_label:
        st.success(f"Sonuç: **{predicted_label}**")
        st.balloons() # Ekrana balonlar atar :)
    else:
        st.error(f"Sonuç: **{predicted_label}**")
    
    st.info(f"Yapay Zeka Güven Oranı: **%{confidence:.2f}**")