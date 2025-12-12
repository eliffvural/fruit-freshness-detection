import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Meyve Tazelik Testi",
    page_icon="🍎",
    layout="wide" # Yan yana düzen için sayfayı genişletiyoruz
)

# --- BAŞLIK VE AÇIKLAMA ---
st.title("🍎 Meyve Tazelik ve Çürük Tespit Sistemi")
st.markdown("""
Bu sistem, Derin Öğrenme (CNN) teknolojisi kullanarak yüklediğiniz meyve fotoğrafının 
**Taze** mi yoksa **Çürük** mü olduğunu analiz eder.
""")
st.write("---") # Ayırıcı çizgi

# --- MODELİ YÜKLEME ---
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('models/fruit_cnn_model.h5')
    return model

try:
    model = load_my_model()
except:
    st.error("Model dosyası bulunamadı! Lütfen 'models/fruit_cnn_model.h5' yolunu kontrol edin.")

# --- SINIF İSİMLERİ ---
class_names = ['Taze Elma', 'Taze Muz', 'Taze Portakal', 
               'Çürük Elma', 'Çürük Muz', 'Çürük Portakal']

# --- RESİM YÜKLEME ALANI ---
# Yükleyiciyi merkeze veya sol tarafa koyabiliriz, burada tam genişlikte kalması daha iyi.
uploaded_file = st.file_uploader("Lütfen bir meyve fotoğrafı yükleyin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- ÖN İŞLEME (PRE-PROCESSING) ---
    # Resmi aç ve modele hazırla
    image = Image.open(uploaded_file)
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized)
    
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Tahmin Yap
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_label = class_names[predicted_class_index]
    confidence = np.max(predictions) * 100

    # --- LAYOUT DÜZENİ (BURASI DEĞİŞTİ) ---
    # Ekranı iki sütuna bölüyoruz: Sol (Resim) - Sağ (Sonuç)
    col1, col2 = st.columns([1, 1], gap="medium") # [1,1] eşit genişlik demek

    with col1:
        st.info("📷 Yüklenen Fotoğraf")
        st.image(image, use_container_width=True)

    with col2:
        st.info("📊 Analiz Sonucu")
        
        # --- TAZE İSE ---
        if "Taze" in predicted_label:
            st.success(f"Sonuç: **{predicted_label}** ✅")
            st.markdown(f"**Güven Oranı:** %{confidence:.2f}")
            
            st.balloons() 
            
            st.markdown("""
            ---
            **🥗 Beslenme İpucu:** Taze meyveler vitamin deposudur. Yıkamadan yemeyiniz!
            """)

       # --- ÇÜRÜK İSE (ÖZELLEŞTİRİLMİŞ UYARI SİSTEMİ) ---
        else:
            st.error(f"DİKKAT! Tespit Edilen: **{predicted_label}** ⚠️")
            st.markdown(f"**Güven Oranı:** %{confidence:.2f}")
            
            st.warning("⛔ **GIDA GÜVENLİĞİ ANALİZİ**")
            
            # --- ELMA İÇİN ÖZEL UYARI ---
            if "Elma" in predicted_label:
                st.markdown("""
                **Tespit Edilen Risk: Patulin Toksini** 🍎
                
                Çürük elmalarda sıkça görülen *Penicillium expansum* küfü, **Patulin** adı verilen bir toksin üretir.
                * **Risk:** Bu toksin ısıya dayanıklıdır (pişirmekle geçmez) ve meyvenin sağlam görünen kısımlarına da yayılabilir.
                * **Öneri:** Çürük kısım küçükse çok geniş kesip atın. Ancak çürük meyvenin %30'unu kaplıyorsa **tamamını atın.**
                """)

            # --- MUZ İÇİN ÖZEL UYARI ---
            elif "Muz" in predicted_label:
                st.markdown("""
                **Tespit Edilen Risk: Doku Bozulması ve Fermentasyon** 🍌
                
                Muzun kabuğundaki siyah noktalar (şekerleşme) normaldir ancak iç kısımdaki cıvıklaşma ve siyah küf tehlikelidir.
                * **Risk:** *Nigrospora* mantarı (Muzun merkezinde siyahlık) alerjik reaksiyonlara sebep olabilir.
                * **Öneri:** Eğer muzun içi tamamen kahverengileşmiş ve alkol kokusu geliyorsa fermentasyon başlamıştır, tüketmeyiniz. Sadece dışı kararmışsa kek yapımında kullanılabilir.
                """)

            # --- PORTAKAL İÇİN ÖZEL UYARI ---
            elif "Portakal" in predicted_label:
                st.markdown("""
                **Tespit Edilen Risk: Mavi/Yeşil Küf Sporları** 🍊
                
                Narenciyelerde görülen yeşil küf (*Penicillium digitatum*), çok hızlı yayılan uçucu sporlara sahiptir.
                * **Risk:** Portakal sulu bir meyve olduğu için küf kökleri meyvenin tamamına çok hızlı yayılır. Dışarıdan sadece bir nokta gibi görünse de içi bozulmuş olabilir.
                * **Öneri:** **Kesinlikle tüketmeyiniz.** Yanındaki diğer meyvelere de spor bulaştırmış olabileceği için onları da yıkayınız.
                """)