# 🎬 Geliştirilmiş Video Analiz Sistemi

## 📋 Proje Özeti

Bu sistem, ses içeren video kayıtlarını tamamen lokal olarak metne dönüştürür, anlamlı bölümlere ayırır ve LLM destekli derinlemesine analiz yapar. İnternet bağlantısı gerektirmez ve tüm işlemler gizlilik odaklı olarak gerçekleştirilir.

### 🎯 Temel Özellikler
- ✅ **Video → Ses → Metin** dönüşümü (FFmpeg + Whisper)
- ✅ **Akıllı Segmentasyon** (doğal kırılma noktalarında bölme)
- ✅ **LLM Destekli Analiz** (Ollama ile lokal LLM)
- ✅ **Fallback Sistemi** (LLM yoksa geleneksel NLP)
- ✅ **Çoklu Format Çıktı** (TXT, JSON)
- ✅ **Kapsamlı İçgörüler** (duygusal ton, anahtar kavramlar, yapısal analiz)

## 🛠️ Kurulum

### Ön Gereksinimler

#### 1. Python Kütüphaneleri
```bash
pip install openai-whisper requests pandas scikit-learn numpy
```

#### 2. FFmpeg Kurulumu
**Windows:**
```bash
# Chocolatey ile
choco install ffmpeg

# Manuel: https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

#### 3. Ollama Kurulumu (Opsiyonel - LLM için)
```bash
# Ollama indirin: https://ollama.ai
curl -fsSL https://ollama.ai/install.sh | sh

# Model indirin
ollama pull llama3.2
# veya
ollama pull mistral
```

### Kurulum Doğrulama
```bash
python -c "import whisper; print('Whisper: OK')"
ffmpeg -version
ollama --version  # Opsiyonel
```

## 🚀 Kullanım

### Temel Kullanım
```bash
python main.py
```

### Adım Adım İşleyiş

1. **Girdi Tipi Seçimi:**
   ```
   📁 Girdi tipi seçin:
   1. Video dosyası (.mp4, .avi, .mov, .mkv vb.) 🎥
   2. Hazır transkript dosyası (.txt) 📄
   ```

2. **Video Analizi için:**
   - Video dosya yolu girin
   - Whisper model boyutu seçin (`tiny`, `base`, `small`, `medium`, `large`)
   - Dil belirtîn (`en`, `auto`)
   - LLM desteği açıp kapatın

3. **Sistem otomatik olarak:**
   - Videodan ses çıkarır
   - Ses tanıma yapar
   - Akıllı segmentasyon uygular
   - LLM ile analiz yapar
   - Sonuçları kaydeder

## 📁 Çıktı Dosyaları

```
output/
├── 📁 llm_insights/
│   ├── segment_analysis.txt      # Detaylı segment analizleri
│   └── overall_summary.txt       # Genel video özeti
├── 📁 enhanced_analysis/
│   ├── key_insights.txt          # Derinlemesine içgörüler
│   └── executive_summary.txt     # Yönetici özeti
├── 📁 structured_data/
│   ├── analysis_data.json        # Yapılandırılmış veriler
└── [video_name]_transcript.txt   # Ham transkript
```

## 🔧 Konfigürasyon

### Whisper Model Boyutları
| Model | Boyut | Hız | Doğruluk | Önerilen |
|-------|--------|-----|----------|----------|
| `tiny` | 39 MB | En hızlı | Düşük | Test için |
| `base` | 74 MB | Hızlı | Orta | **Varsayılan** |
| `small` | 244 MB | Orta | İyi | Kaliteli sonuç |
| `medium` | 769 MB | Yavaş | Yüksek | Profesyonel |
| `large` | 1550 MB | En yavaş | En yüksek | En iyi kalite |

### LLM Modelleri (Ollama)
```bash
# Hafif ve hızlı
ollama pull llama3.2:1b

# Dengeli (önerilen)
ollama pull llama3.2

# Büyük ve güçlü
ollama pull mistral:7b
ollama pull gemma:7b
```

## 📊 Örnek Kullanım Senaryoları

### Senaryo 1: Eğitim Videosu Analizi
```bash
# 20 dakikalık ders videosu
Input: education_video.mp4
Model: base
Language: tr
Output: 8 segment, pozitif ton, eğitim odaklı içgörüler
```

### Senaryo 2: Podcast Analizi
```bash
# 45 dakikalık Türkçe podcast
Input: podcast_episode.mp3
Model: small
Language: tr
Output: 15 segment, konu geçişleri, anahtar mesajlar
```

### Senaryo 3: Toplantı Kaydı
```bash
# 30 dakikalık iş toplantısı
Input: meeting_record.mp4
Model: medium
Language: en
Output: Karar noktaları, aksiyon öğeleri, katılımcı analizleri
```

## ⚙️ Gelişmiş Özellikler

### 1. Hibrit Analiz Sistemi
- **LLM Aktif:** Ollama ile semantik analiz
- **LLM Pasif:** TF-IDF + geleneksel NLP
- **Otomatik Geçiş:** Bağlantı yoksa fallback

### 2. Akıllı Segmentasyon
```python
# Doğal kırılma noktaları
transition_words = ['next', 'however', 'but', 'also', 'furthermore']
adaptive_length = 200-300  # kelime
semantic_breaks = True
```

### 3. Çok Katmanlı Analiz
- **Duygusal Ton:** Pozitif/Negatif/Nötr
- **Konu Kategorileri:** Health, Tech, Business, Education
- **Yapısal Metrikler:** Denge, akış, yoğunluk
- **Anahtar Terimler:** TF-IDF + N-gram analizi

## 🔍 Sorun Giderme

### Yaygın Hatalar ve Çözümleri

#### FFmpeg Bulunamadı
```bash
❌ Hata: FFmpeg bulunamadı!
✅ Çözüm: 
# Windows: https://ffmpeg.org/download.html
# macOS: brew install ffmpeg  
# Linux: sudo apt install ffmpeg
```

#### Whisper Yükleme Hatası
```bash
❌ Hata: Whisper kurulu değil!
✅ Çözüm: pip install openai-whisper
```

#### Ollama Bağlantısı
```bash
❌ Hata: Ollama bağlantısı yok
✅ Çözüm: 
ollama serve  # Ollama'yı başlatın
# veya LLM desteğini kapatın
```

#### Bellek Yetersizliği
```bash
❌ Hata: Büyük video dosyası bellek hatası
✅ Çözüm: 
# Daha küçük Whisper modeli kullanın
model_size = "tiny"  # veya "base"
```

### Log Dosyaları
Sistem otomatik olarak hata logları oluşturur:
```
temp_audio/error.log
output/analysis_errors.log
```

## 📈 Performans Optimizasyonu

### Hız İyileştirmeleri
```python
# Whisper model cache
export WHISPER_CACHE_DIR="./models"

# FFmpeg optimizasyonu
ffmpeg_threads = 4  # CPU çekirdek sayısı

# Segment büyüklüğü
max_segment_length = 250  # Daha küçük = daha hızlı
```

### Bellek Optimizasyonu
```python
# Büyük videolar için
chunk_processing = True
temp_file_cleanup = True
memory_efficient_mode = True
```

## 🤝 Katkıda Bulunma

Bu bir stajyer projesidir. Geliştirme önerileri:

1. **Yeni Dil Desteği:** Whisper model fine-tuning
2. **Görsel Analiz:** Video frame analizi ekleme
3. **Real-time Processing:** Canlı stream analizi
4. **Web Interface:** Flask/Streamlit GUI
5. **API Endpoint:** REST API hizmeti

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir. Ticari kullanım için izin gereklidir.

## 📞 İletişim

**Kişi:** [Sueda Zehra Ergül]  
**E-posta:** [ergulsuedazehra@gmail.com]  

## 🙏 Teşekkürler

- **OpenAI Whisper:** Ses tanıma teknolojisi
- **Ollama:** Lokal LLM desteği  
- **FFmpeg:** Video işleme altyapısı
- **Python Ecosystem:** Scikit-learn, Pandas, NumPy

---
