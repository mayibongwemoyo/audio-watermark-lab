from audioseal import AudioSeal
import os

def download_models():
    print("Downloading AudioSeal models...")
    try:
        # Download generator model
        generator = AudioSeal.load_generator("audioseal_wm_16bits")
        print("Generator model downloaded successfully!")
        
        # Download detector model
        detector = AudioSeal.load_detector("audioseal_detector_16bits")
        print("Detector model downloaded successfully!")
        
        print("All models downloaded successfully!")
    except Exception as e:
        print(f"Error downloading models: {e}")

if __name__ == "__main__":
    download_models() 