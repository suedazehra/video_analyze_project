#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Video Analiz Sistemi """

import re
import os
import json
import requests
import subprocess
import whisper
from collections import Counter, defaultdict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pathlib import Path
import time

class VideoProcessor:
    """Video'dan ses çıkarma ve transkript oluşturma"""
    
    def __init__(self):
        self.temp_dir = "temp_audio"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def extract_audio_from_video(self, video_path, audio_path=None):
        """Video'dan ses çıkar"""
        if audio_path is None:
            video_name = Path(video_path).stem
            audio_path = f"{self.temp_dir}/{video_name}.wav"
        
        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # video stream'i dahil etme
                '-c:a', 'pcm_s16le',  # wav formatı,  # wav formatı
                '-ar', '16000',  # 16kHz sampling rate
                '-ac', '1',  # mono kanal
                '-y',  # üzerine yaz
                audio_path
            ]
            
            print(f"🎵 Video'dan ses çıkarılıyor: {video_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Ses dosyası oluşturuldu: {audio_path}")
            return audio_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg hatası: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return None
        except FileNotFoundError:
            print("❌ FFmpeg bulunamadı! Lütfen FFmpeg'i yükleyin:")
            return None
    
    def check_dependencies(self):
        """Gerekli bağımlılıkları kontrol et"""
        dependencies = {
            'ffmpeg': self.check_ffmpeg(),
            'whisper': self.check_whisper()
        }
        return dependencies
    
    def check_ffmpeg(self):
        """FFmpeg kurulu mu kontrol et"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            return True
        except:
            return False
    
    def check_whisper(self):
        """Whisper kurulu mu kontrol et"""
        try:
            import whisper
            return True
        except ImportError:
            return False

class WhisperTranscriber:
    """Whisper ile ses tanıma"""
    
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Whisper modelini yükle"""
        try:
            print(f"🤖 Whisper {self.model_size} modeli yükleniyor...")
            self.model = whisper.load_model(self.model_size)
            print("✅ Whisper modeli hazır")
            return True
        except ImportError:
            print("❌ Whisper kurulu değil!")
            return False
        except Exception as e:
            print(f"❌ Whisper yükleme hatası: {e}")
            return False
    
    def transcribe_audio(self, audio_path, language="en"):
        """Ses dosyasını metne çevir"""
        if not self.model:
            print("❌ Whisper modeli yüklü değil!")
            return None
        
        try:
            print(f"🎤 Ses tanıma başlıyor: {audio_path}")
            
            # Whisper transcription options
            options = {
                "language": language,  # None = auto-detect, 
                "task": "transcribe",  # "transcribe" or "translate" 
                "verbose": False
            }
            
            result = self.model.transcribe(audio_path, **options)
            
            transcript_text = result["text"].strip()
            detected_language = result.get("language", "unknown")
            
            print(f"✅ Transkripsiyon tamamlandı")
            print(f"📊 Tespit edilen dil: {detected_language}")
            print(f"📝 Metin uzunluğu: {len(transcript_text)} karakter")
            
            return {
                "text": transcript_text,
                "language": detected_language,
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            print(f"❌ Transkripsiyon hatası: {e}")
            return None

class LocalLLMAnalyzer:
    """Local LLM (Ollama) ile analiz yapan sınıf"""
    
    def __init__(self, model_name="llama3.2", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
    def check_ollama_connection(self):
        """Ollama bağlantısını kontrol et"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt, max_tokens=1000):
        """LLM'den yanıt al"""
        if not self.check_ollama_connection():
            print("⚠️ Ollama bağlantısı yok, basit analiz kullanılıyor...")
            return None
            
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(self.api_url, json=data, timeout=30)
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                print(f"LLM API hatası: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"LLM bağlantı hatası: {e}")
            return None

class EnhancedTranscriptAnalyzer:
    def __init__(self, file_path, use_llm=True):
        """Geliştirilmiş transkript analiz sistemi"""
        self.file_path = file_path
        self.use_llm = use_llm
        self.text = self.load_text()
        
        # LLM analyzer'ı başlat
        if self.use_llm:
            self.llm = LocalLLMAnalyzer()
            if not self.llm.check_ollama_connection():
                print("🔄 Ollama bulunamadı, sadece geleneksel analiz yapılacak")
                self.use_llm = False
        
        # Basit tokenization
        self.sentences = self.safe_tokenize_sentences(self.text)
        self.words = self.safe_tokenize_words(self.text.lower())
        
        # Genişletilmiş stop words listesi
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their',
            'so', 'very', 'just', 'now', 'also', 'only', 'here', 'there', 'where', 'when',
            'what', 'how', 'why', 'who', 'which', 'than', 'more', 'most', 'some', 'any'
        }
        
    def load_text(self):
        """Text dosyasını yükle"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Dosya bulunamadı: {self.file_path}")
            return ""
        except UnicodeDecodeError:
            # Farklı encoding'leri dene
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(self.file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except:
                    continue
            print(f"Dosya encoding hatası: {self.file_path}")
            return ""
    
    def safe_tokenize_sentences(self, text):
        """Geliştirilmiş cümle tokenization"""
        # Nokta, ünlem, soru işareti ile ayır
        sentences = re.split(r'[.!?]+', text)
        # Boş ve çok kısa cümleleri filtrele
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences
    
    def safe_tokenize_words(self, text):
        """Geliştirilmiş kelime tokenization"""
        words = re.findall(r'\b\w+\b', text.lower())
        # Çok kısa kelimeleri filtrele
        words = [w for w in words if len(w) > 2]
        return words
    
    def llm_analyze_segment(self, segment_text, segment_id):
        """LLM ile segment analizi"""
        if not self.use_llm:
            return self.traditional_segment_analysis(segment_text, segment_id)
        
        prompt = f"""Bu video transkriptinin bir bölümünü analiz et:

SEGMENT {segment_id}:
"{segment_text}"

Lütfen şunları sağla:
1. ANA KONU: Bu bölümün ana konusu nedir? (tek cümle)
2. ANAHTAR MESAJLAR: En önemli 2-3 mesaj (madde halinde)
3. DUYGUSAL TON: Pozitif/Negatif/Nötr ve neden
4. ÖNEMLİ KAVRAMLAR: Bahsedilen önemli terimler (5 tane max)
5. SEGMENT TİPİ: Bu bölüm nedir? (örnek: tanım, açıklama, hikaye, istatistik, tavsiye)

Yanıtını JSON formatında ver:
{{
    "ana_konu": "...",
    "anahtar_mesajlar": ["...", "..."],
    "duygusal_ton": {{"ton": "pozitif/negatif/notr", "aciklama": "..."}},
    "onemli_kavramlar": ["...", "...", "..."],
    "segment_tipi": "..."
}}"""

        try:
            response = self.llm.generate_response(prompt, max_tokens=500)
            if response:
                # JSON parse etmeye çalış
                try:
                    # Yanıttan JSON kısmını çıkar
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        analysis = json.loads(json_str)
                        return analysis
                except:
                    # JSON parse edilemezse, metin analizi yap
                    return self.parse_llm_text_response(response, segment_id)
            else:
                return self.traditional_segment_analysis(segment_text, segment_id)
                
        except Exception as e:
            print(f"LLM analiz hatası (Segment {segment_id}): {e}")
            return self.traditional_segment_analysis(segment_text, segment_id)
    
    def parse_llm_text_response(self, response, segment_id):
        """LLM'nin metin yanıtını parse et"""
        try:
            lines = response.split('\n')
            analysis = {
                "ana_konu": "Belirlenemedi",
                "anahtar_mesajlar": [],
                "duygusal_ton": {"ton": "nötr", "aciklama": "LLM yanıtı parse edilemedi"},
                "onemli_kavramlar": [],
                "segment_tipi": "genel"
            }
            
            for line in lines:
                line = line.strip()
                if "ana konu" in line.lower() or "main topic" in line.lower():
                    analysis["ana_konu"] = line.split(':', 1)[-1].strip()
                elif "mesaj" in line.lower() or "message" in line.lower():
                    analysis["anahtar_mesajlar"].append(line.split(':', 1)[-1].strip())
                elif "ton" in line.lower() or "sentiment" in line.lower():
                    if "pozitif" in line.lower() or "positive" in line.lower():
                        analysis["duygusal_ton"]["ton"] = "pozitif"
                    elif "negatif" in line.lower() or "negative" in line.lower():
                        analysis["duygusal_ton"]["ton"] = "negatif"
            
            return analysis
            
        except:
            return self.traditional_segment_analysis("", segment_id)
    
    def traditional_segment_analysis(self, segment_text, segment_id):
        """Geleneksel segment analizi (LLM olmadan)"""
        words = self.safe_tokenize_words(segment_text)
        
        # Ana konu tahmin etme
        topic_keywords = {
            'health': ['health', 'wellness', 'medical', 'body', 'mind', 'physical', 'mental'],
            'social': ['social', 'people', 'relationship', 'community', 'friend', 'family'],
            'technology': ['technology', 'digital', 'computer', 'internet', 'data'],
            'education': ['education', 'learning', 'student', 'school', 'knowledge'],
            'business': ['business', 'work', 'career', 'job', 'company']
        }
        
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for word in words if word in keywords)
            topic_scores[topic] = score
        
        ana_konu = max(topic_scores, key=topic_scores.get) if max(topic_scores.values()) > 0 else "genel konu"
        
        # Basit sentiment analizi
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'happy', 'joy', 'positive', 'better', 'best', 'success'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'worried', 'problem', 'difficult', 'hard'}
        
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        if pos_count > neg_count:
            ton = "pozitif"
        elif neg_count > pos_count:
            ton = "negatif"
        else:
            ton = "nötr"
        
        # En sık geçen kelimeleri bul
        word_freq = Counter([w for w in words if w not in self.stop_words])
        onemli_kavramlar = [word for word, count in word_freq.most_common(5)]
        
        return {
            "ana_konu": f"{ana_konu} konusu",
            "anahtar_mesajlar": [f"Bu bölümde {ana_konu} hakkında bilgi veriliyor"],
            "duygusal_ton": {"ton": ton, "aciklama": f"{pos_count} pozitif, {neg_count} negatif kelime"},
            "onemli_kavramlar": onemli_kavramlar,
            "segment_tipi": "bilgi paylaşımı"
        }
    
    def llm_generate_overall_summary(self, segments_analysis):
        """LLM ile genel özet oluştur"""
        if not self.use_llm:
            return self.traditional_overall_summary(segments_analysis)
        
        # Segment özetlerini birleştir
        segment_summaries = []
        for i, analysis in enumerate(segments_analysis, 1):
            summary = f"Segment {i}: {analysis['ana_konu']} - {', '.join(analysis['anahtar_mesajlar'][:2])}"
            segment_summaries.append(summary)
        
        combined_segments = '\n'.join(segment_summaries)
        
        prompt = f"""Bu video transkriptinin segment analizlerine dayanarak genel bir özet oluştur:

SEGMENT ANALİZLERİ:
{combined_segments}

Lütfen şunları içeren bir özet hazırla:
1. VİDEONUN ANA KONUSU: Video genel olarak neyi anlatıyor?
2. ANA MESAJLAR: Videonun vermek istediği en önemli 3-4 mesaj
3. İÇERİK AKIŞI: Video nasıl bir yapıda ilerliyor?
4. HEDEF KİTLE: Bu video kime hitap ediyor?
5. ANA ÇIKARIMLAR: İzleyicinin öğreneceği ana noktalar

Yanıtını düzenli paragraflar halinde, açık ve anlaşılır şekilde yaz. 
Toplam 300-400 kelime olsun."""

        try:
            response = self.llm.generate_response(prompt, max_tokens=800)
            if response:
                return response
            else:
                return self.traditional_overall_summary(segments_analysis)
        except Exception as e:
            print(f"LLM özet oluşturma hatası: {e}")
            return self.traditional_overall_summary(segments_analysis)
    
    def traditional_overall_summary(self, segments_analysis):
        """Geleneksel özet oluşturma"""
        # En sık geçen konuları bul
        topics = [analysis['ana_konu'] for analysis in segments_analysis]
        topic_freq = Counter(topics)
        main_topic = topic_freq.most_common(1)[0][0] if topic_freq else "çeşitli konular"
        
        # Duygusal ton dağılımı
        tones = [analysis['duygusal_ton']['ton'] for analysis in segments_analysis]
        tone_freq = Counter(tones)
        dominant_tone = tone_freq.most_common(1)[0][0]
        
        # Tüm önemli kavramları topla
        all_concepts = []
        for analysis in segments_analysis:
            all_concepts.extend(analysis['onemli_kavramlar'])
        
        concept_freq = Counter(all_concepts)
        top_concepts = [concept for concept, count in concept_freq.most_common(8)]
        
        summary = f"""Bu video öncelikli olarak {main_topic} üzerine odaklanmaktadir. 
        
Video toplamda {len(segments_analysis)} anlamlı bölümden oluşuyor ve genel olarak {dominant_tone} bir ton taşıyor.

Video boyunca en sık bahsedilen kavramlar: {', '.join(top_concepts[:5])}.

İçerik, izleyicilere {main_topic} konusunda bilgi vermeyi ve farkındalık oluşturmayı amaçlıyor. 
Video sistematik bir şekilde konuyu ele alarak, farklı açılardan yaklaşım sunuyor.

Bu içerik, {main_topic} ile ilgilenen herkese hitap eden, bilgilendirici bir yapıya sahip."""
        
        return summary
    
    def extract_key_insights(self):
        """Ana içgörüleri çıkar"""
        insights = {
            'video_length_analysis': self.analyze_video_length(),
            'content_density': self.analyze_content_density(),
            'topic_progression': self.analyze_topic_progression(),
            'key_terminology': self.extract_key_terminology(),
            'structural_analysis': self.analyze_content_structure()
        }
        return insights
    
    def analyze_video_length(self):
        """Video uzunluğu analizi"""
        word_count = len(self.words)
        sentence_count = len(self.sentences)
        
        # Ortalama konuşma hızı 150-160 kelime/dakika
        estimated_duration = word_count / 150  # dakika
        
        if estimated_duration < 5:
            length_category = "kısa"
        elif estimated_duration < 15:
            length_category = "orta"
        else:
            length_category = "uzun"
            
        return {
            'estimated_duration_minutes': round(estimated_duration, 1),
            'word_count': word_count,
            'sentence_count': sentence_count,
            'category': length_category,
            'density_score': round(sentence_count / max(estimated_duration, 1), 1)
        }
    
    def analyze_content_density(self):
        """İçerik yoğunluğu analizi"""
        # Benzersiz kelime oranı
        unique_ratio = len(set(self.words)) / len(self.words) if self.words else 0
        
        # Ortalama cümle uzunluğu
        avg_sentence_length = len(self.words) / len(self.sentences) if self.sentences else 0
        
        # Karmaşıklık skoru
        complexity_score = unique_ratio * avg_sentence_length
        
        if complexity_score < 5:
            complexity_level = "basit"
        elif complexity_score < 10:
            complexity_level = "orta"
        else:
            complexity_level = "karmaşık"
            
        return {
            'unique_word_ratio': round(unique_ratio, 3),
            'avg_sentence_length': round(avg_sentence_length, 1),
            'complexity_score': round(complexity_score, 2),
            'complexity_level': complexity_level
        }
    
    def analyze_topic_progression(self):
        """Konu ilerleyişi analizi"""
        segments = self.segment_content_enhanced()
        progression = []
        
        for i, segment in enumerate(segments):
            # Her segmentin temel konusunu belirle
            words = self.safe_tokenize_words(segment['text'])
            
            # Konu belirleyici kelimeler
            topic_indicators = {
                'introduction': ['introduce', 'begin', 'start', 'first', 'welcome'],
                'explanation': ['explain', 'because', 'reason', 'how', 'what', 'define'],
                'example': ['example', 'instance', 'like', 'such as', 'consider'],
                'conclusion': ['conclude', 'finally', 'end', 'summary', 'therefore'],
                'transition': ['next', 'now', 'then', 'however', 'but', 'also']
            }
            
            segment_type = 'content'  # default
            max_score = 0
            
            for topic_type, indicators in topic_indicators.items():
                score = sum(1 for word in words if word in indicators)
                if score > max_score:
                    max_score = score
                    segment_type = topic_type
            
            progression.append({
                'segment': i + 1,
                'type': segment_type,
                'word_count': len(words)
            })
        
        return progression
    
    def extract_key_terminology(self):
        """Anahtar terminoloji çıkarımı"""
        # TF-IDF ile önemli terimleri bul
        try:
            sentences_for_tfidf = [' '.join(self.sentences[i:i+2]) for i in range(0, len(self.sentences), 2)]
            
            vectorizer = TfidfVectorizer(
                max_features=30,
                stop_words='english',
                ngram_range=(1, 3),  # 1-3 kelimelik terimler
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences_for_tfidf)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            
            term_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            term_scores.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'technical_terms': [term for term, score in term_scores[:15]],
                'key_phrases': [term for term, score in term_scores if len(term.split()) > 1][:10]
            }
            
        except Exception as e:
            print(f"Terminoloji çıkarım hatası: {e}")
            return {'technical_terms': [], 'key_phrases': []}
    
    def analyze_content_structure(self):
        """İçerik yapısı analizi"""
        segments = self.segment_content_enhanced()
        
        structure_analysis = {
            'total_segments': len(segments),
            'segment_balance': self.calculate_segment_balance(segments),
            'content_flow': self.analyze_content_flow(segments),
            'information_distribution': self.analyze_information_distribution(segments)
        }
        
        return structure_analysis
    
    def calculate_segment_balance(self, segments):
        """Segment dengesi hesapla"""
        word_counts = [seg['word_count'] for seg in segments]
        avg_words = np.mean(word_counts)
        std_words = np.std(word_counts)
        
        balance_score = 1 - (std_words / avg_words) if avg_words > 0 else 0
        
        return {
            'average_segment_length': round(avg_words, 1),
            'length_variation': round(std_words, 1),
            'balance_score': round(balance_score, 3),
            'balance_level': 'dengeli' if balance_score > 0.7 else 'dengesiz'
        }
    
    def analyze_content_flow(self, segments):
        """İçerik akışı analizi"""
        flow_patterns = []
        
        for i in range(len(segments) - 1):
            current_length = segments[i]['word_count']
            next_length = segments[i + 1]['word_count']
            
            if next_length > current_length * 1.5:
                flow_patterns.append('expansion')
            elif next_length < current_length * 0.5:
                flow_patterns.append('contraction')
            else:
                flow_patterns.append('stable')
        
        pattern_freq = Counter(flow_patterns)
        
        return {
            'flow_patterns': dict(pattern_freq),
            'dominant_pattern': pattern_freq.most_common(1)[0][0] if pattern_freq else 'stable'
        }
    
    def analyze_information_distribution(self, segments):
        """Bilgi dağılımı analizi"""
        total_words = sum(seg['word_count'] for seg in segments)
        
        distribution = []
        for i, seg in enumerate(segments):
            percentage = (seg['word_count'] / total_words) * 100
            distribution.append({
                'segment': i + 1,
                'percentage': round(percentage, 1),
                'word_count': seg['word_count']
            })
        
        # En bilgi yoğun segmentleri bul
        sorted_segments = sorted(distribution, key=lambda x: x['percentage'], reverse=True)
        
        return {
            'distribution': distribution,
            'most_dense_segments': sorted_segments[:3],
            'information_concentration': 'concentrated' if sorted_segments[0]['percentage'] > 30 else 'distributed'
        }
    
    def segment_content_enhanced(self):
        """Geliştirilmiş içerik segmentasyonu"""
        segments = []
        current_segment = []
        word_count = 0
        
        for sentence in self.sentences:
            sentence_words = len(sentence.split())
            current_segment.append(sentence)
            word_count += sentence_words
            
            # Segment sonu koşulları (daha akıllı)
            should_break = False
            
            # Uzunluk bazlı koşullar
            if word_count > 200:  # Minimum segment uzunluğu
                # Doğal kırılma noktalarını ara
                if any(phrase in sentence.lower() for phrase in [
                    'next', 'now', 'then', 'however', 'but', 'also', 'another',
                    'furthermore', 'moreover', 'additionally', 'in conclusion'
                ]):
                    should_break = True
                elif word_count > 300:  # Maksimum segment uzunluğu
                    should_break = True
            
            # Son cümle ise kesinlikle kır
            if sentence == self.sentences[-1]:
                should_break = True
            
            if should_break and current_segment:
                segment_text = ' '.join(current_segment)
                segments.append({
                    'segment_id': len(segments) + 1,
                    'text': segment_text,
                    'sentence_count': len(current_segment),
                    'word_count': word_count,
                    'char_count': len(segment_text)
                })
                current_segment = []
                word_count = 0
        
        return segments
    
    def save_enhanced_results(self, results):
        """Geliştirilmiş sonuçları kaydet"""
        # Dizinleri oluştur
        os.makedirs("output/enhanced_analysis", exist_ok=True)
        os.makedirs("output/llm_insights", exist_ok=True)
        os.makedirs("output/structured_data", exist_ok=True)
        
        # 1. LLM İçgörüleri
        with open("output/llm_insights/segment_analysis.txt", "w", encoding="utf-8") as f:
            f.write("LLM DESTEKLİ SEGMENT ANALİZİ\n")
            f.write("=" * 50 + "\n\n")
            
            for i, analysis in enumerate(results['llm_segment_analysis'], 1):
                f.write(f"SEGMENT {i}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Ana Konu: {analysis['ana_konu']}\n")
                f.write(f"Segment Tipi: {analysis['segment_tipi']}\n")
                f.write(f"Duygusal Ton: {analysis['duygusal_ton']['ton']} ({analysis['duygusal_ton']['aciklama']})\n")
                f.write(f"\nAnahtar Mesajlar:\n")
                for mesaj in analysis['anahtar_mesajlar']:
                    f.write(f"• {mesaj}\n")
                f.write(f"\nÖnemli Kavramlar: {', '.join(analysis['onemli_kavramlar'])}\n")
                f.write("\n" + "="*50 + "\n\n")
        
        # 2. Genel Özet
        with open("output/llm_insights/overall_summary.txt", "w", encoding="utf-8") as f:
            f.write("VIDEO GENEL ÖZETİ (LLM Destekli)\n")
            f.write("=" * 50 + "\n\n")
            f.write(results['overall_summary'])
            f.write("\n\n")
        
        # 3. Anahtar İçgörüler
        with open("output/enhanced_analysis/key_insights.txt", "w", encoding="utf-8") as f:
            f.write("ANAHTAR İÇGÖRÜLER VE DERINLEMESINE ANALİZ\n")
            f.write("=" * 60 + "\n\n")
            
            insights = results['key_insights']
            
            # Video uzunluk analizi
            length_analysis = insights['video_length_analysis']
            f.write("1. VİDEO UZUNLUK ANALİZİ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Tahmini Süre: {length_analysis['estimated_duration_minutes']} dakika\n")
            f.write(f"Toplam Kelime: {length_analysis['word_count']}\n")
            f.write(f"Toplam Cümle: {length_analysis['sentence_count']}\n")
            f.write(f"Video Kategorisi: {length_analysis['category']}\n")
            f.write(f"İçerik Yoğunluk Skoru: {length_analysis['density_score']}\n\n")
            
            # İçerik yoğunluğu
            density = insights['content_density']
            f.write("2. İÇERİK YOĞUNLUK ANALİZİ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Benzersiz Kelime Oranı: {density['unique_word_ratio']}\n")
            f.write(f"Ortalama Cümle Uzunluğu: {density['avg_sentence_length']} kelime\n")
            f.write(f"Karmaşıklık Seviyesi: {density['complexity_level']}\n")
            f.write(f"Karmaşıklık Skoru: {density['complexity_score']}\n\n")
            
            # Konu ilerleyişi
            progression = insights['topic_progression']
            f.write("3. KONU İLERLEYİŞİ ANALİZİ\n")
            f.write("-" * 30 + "\n")
            for item in progression:
                f.write(f"Segment {item['segment']}: {item['type']} ({item['word_count']} kelime)\n")
            f.write("\n")
            
            # Anahtar terminoloji
            terminology = insights['key_terminology']
            f.write("4. ANAHTAR TERMİNOLOJİ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Teknik Terimler: {', '.join(terminology['technical_terms'][:10])}\n")
            f.write(f"Önemli İfadeler: {', '.join(terminology['key_phrases'][:5])}\n\n")
            
            # Yapısal analiz
            structure = insights['structural_analysis']
            f.write("5. YAPISAL ANALİZ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Toplam Segment: {structure['total_segments']}\n")
            f.write(f"Ortalama Segment Uzunluğu: {structure['segment_balance']['average_segment_length']}\n")
            f.write(f"İçerik Dengesi: {structure['segment_balance']['balance_level']}\n")
            f.write(f"Bilgi Dağılımı: {structure['information_distribution']['information_concentration']}\n")
            f.write(f"Dominant Akış Paterni: {structure['content_flow']['dominant_pattern']}\n\n")
        
        # 4. Yapılandırılmış veri (JSON)
        with open("output/structured_data/analysis_data.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 5. Özet rapor
        with open("output/enhanced_analysis/executive_summary.txt", "w", encoding="utf-8") as f:
            f.write("YÖNETİCİ ÖZETİ - VIDEO ANALİZ RAPORU\n")
            f.write("=" * 50 + "\n\n")
            
            length_info = results['key_insights']['video_length_analysis']
            f.write(f"📊 VIDEO İSTATİSTİKLERİ:\n")
            f.write(f"• Tahmini Süre: {length_info['estimated_duration_minutes']} dakika\n")
            f.write(f"• Toplam Kelime: {length_info['word_count']:,}\n")
            f.write(f"• Video Kategorisi: {length_info['category']}\n\n")
            
            f.write(f"🎯 ANA BULGULAR:\n")
            f.write(f"• Toplam {len(results['llm_segment_analysis'])} anlamlı segment tespit edildi\n")
            
            # En sık geçen konuları listele
            topics = [seg['ana_konu'] for seg in results['llm_segment_analysis']]
            topic_freq = Counter(topics)
            f.write(f"• En yaygın konu kategorileri: {', '.join([topic for topic, count in topic_freq.most_common(3)])}\n")
            
            # Duygusal ton dağılımı
            tones = [seg['duygusal_ton']['ton'] for seg in results['llm_segment_analysis']]
            tone_freq = Counter(tones)
            f.write(f"• Duygusal ton dağılımı: {dict(tone_freq)}\n\n")
            
            f.write(f"📈 İÇERİK KALİTESİ:\n")
            density_info = results['key_insights']['content_density']
            f.write(f"• Karmaşıklık Seviyesi: {density_info['complexity_level']}\n")
            f.write(f"• İçerik Yoğunluğu: {density_info['complexity_score']}/10\n\n")
            
            f.write(f"🔍 ÖNERİLER:\n")
            if density_info['complexity_level'] == 'karmaşık':
                f.write("• İçerik karmaşık seviyede, hedef kitleye uygunluğu değerlendirilmeli\n")
            if length_info['category'] == 'uzun':
                f.write("• Video uzun kategorisinde, bölümlere ayrılması düşünülebilir\n")
            
            structure_info = results['key_insights']['structural_analysis']
            if structure_info['segment_balance']['balance_level'] == 'dengesiz':
                f.write("• Segment uzunlukları dengesiz, içerik yapısı gözden geçirilmeli\n")
            
            f.write(f"\n📄 DETAYLAR:\n")
            f.write(f"• Tam analiz sonuçları: output/llm_insights/ klasöründe\n")
            f.write(f"• Yapılandırılmış veriler: output/structured_data/ klasöründe\n")
            f.write(f"• Rapor oluşturulma zamanı: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print("\n✅ SONUÇLAR KAYDEDİLDİ:")
        print("📁 output/llm_insights/segment_analysis.txt - Detaylı segment analizleri")
        print("📁 output/llm_insights/overall_summary.txt - Genel video özeti") 
        print("📁 output/enhanced_analysis/key_insights.txt - Derinlemesine içgörüler")
        print("📁 output/enhanced_analysis/executive_summary.txt - Yönetici özeti")
        print("📁 output/structured_data/analysis_data.json - Yapılandırılmış veriler")
    
    def run_complete_analysis(self):
        """Tam analizi çalıştır"""
        print("🚀 Geliştirilmiş Video Analiz Sistemi Başlatılıyor...")
        print(f"📄 Analiz edilen dosya: {self.file_path}")
        print(f"🤖 LLM Desteği: {'Aktif' if self.use_llm else 'Pasif'}")
        print("-" * 60)
        
        if not self.text:
            print("❌ Dosya okunamadı veya boş!")
            return None
        
        # 1. İçerik segmentasyonu
        print("📊 1. İçerik segmentasyonu yapılıyor...")
        segments = self.segment_content_enhanced()
        
        if not segments:
            print("❌ Segment oluşturulamadı!")
            return None
        
        print(f"✅ {len(segments)} segment oluşturuldu")
        
        # 2. LLM ile segment analizi
        print("🧠 2. Segment analizi yapılıyor...")
        llm_analyses = []
        
        for i, segment in enumerate(segments, 1):
            print(f"   📝 Segment {i}/{len(segments)} analiz ediliyor...")
            analysis = self.llm_analyze_segment(segment['text'], i)
            llm_analyses.append(analysis)
        
        print("✅ Segment analizleri tamamlandı")
        
        # 3. Genel özet oluşturma
        print("📋 3. Genel özet oluşturuluyor...")
        overall_summary = self.llm_generate_overall_summary(llm_analyses)
        print("✅ Genel özet hazırlandı")
        
        # 4. Anahtar içgörüler
        print("🔍 4. Anahtar içgörüler çıkarılıyor...")
        key_insights = self.extract_key_insights()
        print("✅ İçgörüler analiz edildi")
        
        # 5. Sonuçları birleştir
        results = {
            'analysis_metadata': {
                'file_path': self.file_path,
                'llm_enabled': self.use_llm,
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_segments': len(segments)
            },
            'segments': segments,
            'llm_segment_analysis': llm_analyses,
            'overall_summary': overall_summary,
            'key_insights': key_insights
        }
        
        # 6. Sonuçları kaydet
        print("💾 5. Sonuçlar kaydediliyor...")
        self.save_enhanced_results(results)
        
        print("\n🎉 ANALİZ TAMAMLANDI!")
        return results

def complete_video_analysis_pipeline(video_path, use_llm=True, whisper_model="base", language=None):
    """Tam video analiz pipeline'ı"""
    
    # 1. Bağımlılık kontrolü
    video_processor = VideoProcessor()
    deps = video_processor.check_dependencies()
    
    if not deps['ffmpeg']:
        print("❌ FFmpeg bulunamadı! Video işleme yapılamaz.")
        print("Kurulum:")
        print("  Windows: https://ffmpeg.org/download.html")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt install ffmpeg")
        return None
    
    if not deps['whisper']:
        print("❌ Whisper bulunamadı! Ses tanıma yapılamaz.")
        print("Yüklemek için: pip install openai-whisper")
        return None
    
    # 2. Video'dan ses çıkar
    audio_path = video_processor.extract_audio_from_video(video_path)
    if not audio_path:
        return None
    
    # 3. Ses tanıma ile transkript oluştur
    transcriber = WhisperTranscriber(model_size=whisper_model)
    transcript_result = transcriber.transcribe_audio(audio_path, language=language)
    
    if not transcript_result:
        return None
    
    # 4. Transkripti dosyaya kaydet
    transcript_text = transcript_result["text"]
    video_name = Path(video_path).stem
    transcript_path = f"output/{video_name}_transcript.txt"
    
    os.makedirs("output", exist_ok=True)
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript_text)
    
    print(f"📄 Transkript kaydedildi: {transcript_path}")
    
    # 5. Ana analiz sistemini çalıştır
    analyzer = EnhancedTranscriptAnalyzer(transcript_path, use_llm=use_llm)
    results = analyzer.run_complete_analysis()
    
    # 6. Geçici ses dosyasını temizle
    try:
        os.remove(audio_path)
        print(f"🗑️ Geçici dosya temizlendi: {audio_path}")
    except:
        pass
    
    return results

def main():
    """Ana çalıştırma fonksiyonu - Video Processing Dahil"""
    print("🎬 TAM VİDEO ANALİZ SİSTEMİ")
    print("Video'dan Metne + LLM Destekli Analiz")
    print("=" * 60)
    
    # Dosya tipini belirle
    input_type = input("""
📁 Girdi tipi seçin:
1. Video dosyası (.mp4, .avi, .mov, .mkv vb.) 🎥
2. Hazır transkript dosyası (.txt) 📄
Seçiminiz (1/2): """).strip()
    
    if input_type == "1":
        # Video dosyası analizi
        video_path = input("🎥 Video dosya yolunu girin: ").strip()
        
        if not os.path.exists(video_path):
            print(f"❌ Video dosyası bulunamadı: {video_path}")
            return
        
        # Video analiz parametreleri
        print("\n🔧 AYARLAR:")
        whisper_model = input("🤖 Whisper model boyutu (tiny/base/small/medium/large) [base]: ").strip() or "base"
        
        language_input = input("🌍 Dil belirtin (en/auto) [auto]: ").strip() or "auto"
        language = None if language_input == "auto" else language_input
        
        use_llm = input("🧠 LLM desteği kullanılsın mı? (y/n) [y]: ").strip().lower() != 'n'
        
        print(f"\n🚀 TAM VİDEO ANALİZ PIPELINE'I BAŞLATILIYOR...")
        print(f"📹 Video: {Path(video_path).name}")
        print(f"🤖 Whisper Model: {whisper_model}")
        print(f"🌍 Dil: {language or 'otomatik tespit'}")
        print(f"🧠 LLM: {'Aktif' if use_llm else 'Pasif'}")
        print("-" * 60)
        
        results = complete_video_analysis_pipeline(video_path, use_llm, whisper_model, language)
        
    elif input_type == "2":
        # Hazır transkript analizi
        transcript_path = input("📄 Transkript dosya yolunu girin: ").strip()
        
        if not os.path.exists(transcript_path):
            print(f"❌ Transkript dosyası bulunamadı: {transcript_path}")
            return
        
        use_llm = input("🧠 LLM desteği kullanılsın mı? (y/n) [y]: ").strip().lower() != 'n'
        
        print(f"\n🚀 TRANSKRİPT ANALİZ SİSTEMİ BAŞLATILIYOR...")
        print(f"📄 Dosya: {Path(transcript_path).name}")
        print(f"🧠 LLM: {'Aktif' if use_llm else 'Pasif'}")
        print("-" * 60)
        
        # Analiz sistemini çalıştır
        analyzer = EnhancedTranscriptAnalyzer(transcript_path, use_llm=use_llm)
        results = analyzer.run_complete_analysis()
    
    else:
        print("❌ Geçersiz seçim!")
        return
    
    if results:
        print(f"\n📊 ANALİZ İSTATİSTİKLERİ:")
        print(f"• Toplam kelime: {results['key_insights']['video_length_analysis']['word_count']:,}")
        print(f"• Toplam cümle: {results['key_insights']['video_length_analysis']['sentence_count']:,}")
        print(f"• Segment sayısı: {len(results['segments'])}")
        print(f"• Tahmini video süresi: {results['key_insights']['video_length_analysis']['estimated_duration_minutes']} dakika")
        print(f"• Karmaşıklık seviyesi: {results['key_insights']['content_density']['complexity_level']}")
        
        print("\n📁 ÇIKTI DOSYALARI:")
        print("• output/llm_insights/segment_analysis.txt")
        print("• output/llm_insights/overall_summary.txt") 
        print("• output/enhanced_analysis/key_insights.txt")
        print("• output/enhanced_analysis/executive_summary.txt")
        print("• output/structured_data/analysis_data.json")
        print("• output/structured_data/segments_data.csv")
        
        if input_type == "1":
            video_name = Path(video_path).stem
            print(f"• output/{video_name}_transcript.txt")
        
        print(f"\n🎉 ANALİZ BAŞARIYLA TAMAMLANDI!")
        print("📂 Tüm sonuçlar 'output/' klasöründe kaydedildi.")
    else:
        print("\n❌ Analiz başarısız oldu!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ İşlem kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()