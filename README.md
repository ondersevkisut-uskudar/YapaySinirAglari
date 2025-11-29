# 🧬 Meme Kanseri Teşhisi: MLP Sınıflandırma ve XAI (SHAP) Analizi

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Scikit--Learn-MLP-orange)
![Optimization](https://img.shields.io/badge/Optuna-Hyperparameter-green)
![XAI](https://img.shields.io/badge/SHAP-Explainable%20AI-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Bu proje, **Üsküdar Üniversitesi Yapay Sinir Ağları Dersi Ara Sınavı** kapsamında geliştirilmiştir. **Breast Cancer Wisconsin** veri seti kullanılarak Yapay Sinir Ağı (MLP) tabanlı bir sınıflandırma modeli oluşturulmuş, model performansı **Optuna** ile optimize edilmiş ve karar mekanizması **SHAP** (Explainable AI) kütüphanesi ile şeffaf hale getirilmiştir.

## 📋 İçindekiler
- [Proje Özeti](#proje-özeti)
- [Dosya Yapısı](#dosya-yapısı)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Proje Adımları](#proje-adımları)
- [Sonuçlar](#sonuçlar)

## 🔍 Proje Özeti
Bu çalışmanın amacı, meme kanseri hücrelerinin özelliklerine (yarıçap, doku, alan vb.) dayanarak tümörün **İyi Huylu (Benign)** veya **Kötü Huylu (Malignant)** olduğunu yüksek doğrulukla tahmin etmektir. 

Proje sadece tahminde bulunmakla kalmayıp, **"Model neden bu kararı verdi?"** sorusunu yanıtlayarak tıbbi teşhis süreçlerinde güvenilirliği artırmayı hedeflemektedir.

## 📂 Dosya Yapısı
Repo içerisindeki temel dosyalar şunlardır:

* **`254329023_onder_sevki_sut.ipynb`**: Projenin tüm kodlarını, analizlerini ve grafiklerini içeren ana Jupyter Notebook dosyası.
* **`254329023_onder_sevki_sut.html`**: Notebook dosyasının tarayıcıda görüntülenebilir rapor formatı (Kod çalıştırmadan incelemek için).
* **`README.md`**: Proje dokümantasyonu.

## 🛠 Kullanılan Teknolojiler
Proje **Python** dili ile geliştirilmiş olup aşağıdaki kütüphaneler kullanılmıştır:

* **Veri İşleme:** `pandas`, `numpy`
* **Görselleştirme:** `matplotlib`, `seaborn`
* **Makine Öğrenmesi:** `scikit-learn` (MLPClassifier, StandardScaler, Metrics)
* **Optimizasyon:** `optuna` (Otomatik Hiperparametre Ayarı)
* **Açıklanabilir Yapay Zeka (XAI):** `shap`

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  **Repoyu klonlayın:**
    ```bash
    git clone [https://github.com/ondersevkisut-uskudar/YapaySinirAglari.git](https://github.com/ondersevkisut-uskudar/YapaySinirAglari.git)
    cd YapaySinirAglari
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn optuna shap
    ```

3.  **Notebook'u çalıştırın:**
    ```bash
    jupyter notebook 254329023_onder_sevki_sut.ipynb
    ```
    *Alternatif olarak `.html` dosyasını tarayıcınızda açarak kodları ve çıktıları doğrudan inceleyebilirsiniz.*

## 📊 Proje Adımları

### 1. Veri Analizi ve Ön İşleme
* Veri setinde eksik değer kontrolü yapıldı (Eksik veri bulunmadı).
* **Boxplot** analizi ile aykırı değerler tespit edildi.
* **Korelasyon Matrisi** ile özellikler arası ilişkiler incelendi (Çoklu doğrusallık tespit edildi).
* Tüm veriler **StandardScaler** ile ölçeklendirildi.
* Veri seti; **%70 Eğitim**, **%10 Doğrulama** ve **%20 Test** olarak ayrıldı.

### 2. MLP Modelleme
Farklı mimarilere sahip 5 adet MLP modeli (Basit, Orta, Geniş, Derin, Düşük LR) doğrulama seti üzerinde karşılaştırıldı.
* **En Başarılı Model:** Model 3 (Geniş - 64x64 nöron, `tanh` aktivasyonu).

### 3. Hiperparametre Optimizasyonu (Optuna)
Model performansını maksimize etmek için **Optuna** kütüphanesi ile **150 deneme (trial)** gerçekleştirildi.
* Katman sayısı, nöron sayısı, öğrenme oranı (Learning Rate), aktivasyon fonksiyonu vb. optimize edildi.
* **Sonuç:** Manuel modellerden daha yüksek bir **F1 Skoru (%98.6)** elde edildi.

### 4. Açıklanabilirlik (SHAP)
Modellerin karar mekanizması incelendi:
* **Manuel Model:** Genelleme yaparak ortalama (`mean`) değerlere odaklandı.
* **Optuna Modeli:** Tıbbi teşhis mantığına uygun olarak **uç/kötü (`worst`)** değerlere (örn: `worst concave points`) odaklanmayı öğrendi.

## 🏆 Sonuçlar

| Model | Accuracy | Recall (Duyarlılık) | F1-Score | ROC-AUC |
|-------|----------|---------------------|----------|---------|
| **Manuel En İyi (Model 3)** | %96.49 | %98.61 | %97.26 | 0.9967 |
| **Optuna Optimize Model** | **~%97.5** | **~%99.0** | **%98.63** | **0.9980** |

* Model, kanserli vakaları tespit etmede çok yüksek başarı göstermiştir.
* SHAP analizi, modelin **"İçbükeylik" (Concavity)** ve **"Alan" (Area)** özelliklerini en kritik belirteçler olarak kullandığını kanıtlamıştır.

---
**Yazar:** Önder Şevki Süt  
**Ders:** Yapay Sinir Ağları - Ara Sınav Ödevi
