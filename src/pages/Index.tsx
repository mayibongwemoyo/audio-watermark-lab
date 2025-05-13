
import { useContext } from "react";
import { Link, useNavigate } from "react-router-dom";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import HeroSection from "@/components/hero/HeroSection";
import { Button } from "@/components/ui/button";
import { AuthContext } from "@/contexts/AuthContext";
import AudioUploader from "@/components/upload/AudioUploader";
import WatermarkControls from "@/components/watermark/WatermarkControls";
import AnalysisSection from "@/components/analysis/AnalysisSection";
import AboutSection from "@/components/about/AboutSection";
import { Card, CardContent } from "@/components/ui/card";
import { Beaker, Users } from "lucide-react";

const Index = () => {
  const { isAuthenticated } = useContext(AuthContext);
  const navigate = useNavigate();

  const handleModeSelect = (mode: "research" | "application") => {
    if (mode === "research") {
      document.getElementById("research")?.scrollIntoView({ behavior: "smooth" });
    } else {
      // For application mode, check if user is authenticated
      if (isAuthenticated) {
        navigate("/app");
      } else {
        navigate("/login");
      }
    }
  };

  return (
    <div className="relative min-h-screen w-full overflow-x-hidden">
      <div className="grain-overlay" />
      <Header />
      
      {/* Mode Selection */}
      <section className="py-16 px-6 md:px-10">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">Audio Watermark Lab</h1>
          <p className="text-xl mb-12 text-black/70 dark:text-white/70">
            Choose your experience mode
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardContent className="p-6 flex flex-col items-center">
                <Beaker className="h-16 w-16 mb-4 text-primary" />
                <h2 className="text-2xl font-semibold mb-2">Research Mode</h2>
                <p className="text-muted-foreground mb-6 text-center">
                  Advanced configuration for audio watermarking experiments and analysis
                </p>
                <Button 
                  size="lg" 
                  className="w-full"
                  onClick={() => handleModeSelect("research")}
                >
                  Enter Research Mode
                </Button>
              </CardContent>
            </Card>
            
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardContent className="p-6 flex flex-col items-center">
                <Users className="h-16 w-16 mb-4 text-primary" />
                <h2 className="text-2xl font-semibold mb-2">Application Mode</h2>
                <p className="text-muted-foreground mb-6 text-center">
                  User-friendly interface for embedding and detecting watermarks with authentication
                </p>
                <Button 
                  size="lg" 
                  className="w-full"
                  onClick={() => handleModeSelect("application")}
                >
                  {isAuthenticated ? "Open Dashboard" : "Login to Access"}
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>
      
      {/* Research Mode Section */}
      <div id="research">
        <main>
          <HeroSection />
          <AudioUploader />
          <WatermarkControls />
          <AnalysisSection />
          <AboutSection />
        </main>
      </div>
      
      <Footer />
    </div>
  );
};

export default Index;
