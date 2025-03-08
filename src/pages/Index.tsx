
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import HeroSection from "@/components/hero/HeroSection";
import AudioUploader from "@/components/upload/AudioUploader";
import WatermarkControls from "@/components/watermark/WatermarkControls";
import AnalysisSection from "@/components/analysis/AnalysisSection";
import AboutSection from "@/components/about/AboutSection";

const Index = () => {
  return (
    <div className="relative min-h-screen w-full overflow-x-hidden">
      <div className="grain-overlay" />
      <Header />
      <main>
        <HeroSection />
        <AudioUploader />
        <WatermarkControls />
        <AnalysisSection />
        <AboutSection />
      </main>
      <Footer />
    </div>
  );
};

export default Index;
