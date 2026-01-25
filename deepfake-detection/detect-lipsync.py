import sys
import os
import argparse
import importlib

def find_syncnet_class():
    """Kütüphanenin içindeki doğru sınıfı bulur (Pipeline vs Instance)"""
    try:
        import syncnet_python
        print(f"📦 Kütüphane bulundu: {syncnet_python.__file__}")
        
        # Olası sınıf isimlerini kontrol et
        candidates = ['SyncNetPipeline', 'SyncNetInstance', 'SyncNet']
        
        for name in candidates:
            if hasattr(syncnet_python, name):
                cls = getattr(syncnet_python, name)
                if cls is not None:
                    print(f"✅ Geçerli sınıf bulundu: {name}")
                    return cls
        
        print("❌ Kütüphane yüklü ama beklenen sınıflar (SyncNetPipeline/Instance) bulunamadı.")
        print(f"   Mevcut içerik: {dir(syncnet_python)}")
        return None

    except ImportError:
        print("❌ 'syncnet-python' kütüphanesi bulunamadı.")
        return None

def check_lipsync(video_path):
    print(f"\n👄 Dudak Senkronizasyonu Analiz Ediliyor: {video_path}")
    
    # 1. Doğru sınıfı bul
    SyncNetClass = find_syncnet_class()
    if SyncNetClass is None:
        print("🚨 Lütfen kütüphaneyi tekrar kurun: pip install --force-reinstall syncnet-python")
        return

    # 2. Modeli Başlat
    try:
        print("   ⏳ Model yükleniyor (CPU)...")
        # Bazı sürümler parametre ister, bazıları istemez. Güvenli başlatma:
        try:
            pipeline = SyncNetClass(device='cpu')
        except TypeError:
            # Eğer parametre hatası verirse boş deneyelim
            pipeline = SyncNetClass()
            
    except Exception as e:
        print(f"❌ Model başlatılamadı: {e}")
        return

    # 3. Analiz Yap
    try:
        print("   🎥 Video taranıyor (Bu işlem biraz sürebilir)...")
        
        # Kütüphanenin sürümüne göre doğru metodu çağır
        results = None
        if hasattr(pipeline, 'inference'):
            results = pipeline.inference(video_path)
        elif hasattr(pipeline, 'evaluate'):
            # Eski sürüm uyumluluğu
            results = pipeline.evaluate(video_path)
        else:
            print("❌ Modelde 'inference' veya 'evaluate' metodu bulunamadı.")
            return

        if not results:
            print("⚠️ Sonuç döndürülemedi (Yüz tespit edilememiş olabilir).")
            return

        # Sonuçları Ayrıştır (Genellikle: offset, confidence, dists)
        # Bazen tuple döner, bazen dict. Kontrol edelim:
        offset, conf = 0, 0
        
        if isinstance(results, tuple) or isinstance(results, list):
            offset = results[0]
            conf = results[1]
            if isinstance(offset, list): # Eğer liste içinde liste döndüyse ilkini al
                offset = offset[0]
                conf = conf[0]
        else:
            print(f"⚠️ Bilinmeyen sonuç formatı: {type(results)}")
            print(results)
            return

        print("\n" + "="*50)
        print("📊 ANALİZ SONUCU")
        print(f"   Güven Skoru: {conf:.2f} ( > 6.0 İyidir)")
        print(f"   Ses Kayması: {offset} kare")
        print("="*50)

        if conf > 5.0:
            print("✅ GERÇEK: Dudak hareketleri sesle mükemmel uyumlu.")
        elif conf < 3.0:
            print("🚨 SAHTE (DEEPFAKE): Dudak hareketleri sesle uyuşmuyor! (Wav2Lip Şüphesi)")
        else:
            print("⚠️ BELİRSİZ: Video kalitesi düşük veya hafif kayma var.")

    except Exception as e:
        print(f"❌ Analiz sırasında hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Dosya bulunamadı: {args.input}")
    else:
        check_lipsync(args.input)
        
