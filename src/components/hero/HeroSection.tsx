
import { ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";

const HeroSection = () => {
  const scrollToUpload = () => {
    const element = document.getElementById('upload');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };
  
  return (
    <section className="relative min-h-screen flex flex-col justify-center items-center pt-16 px-6 md:px-10 overflow-hidden">
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-[500px] bg-gradient-to-b from-[#f5f5f5] to-transparent dark:from-[#111] opacity-80 z-0"></div>
        <div className="grain-overlay"></div>
      </div>
      
      <div className="relative z-10 max-w-4xl mx-auto text-center">
        <div className="inline-block rounded-full px-3 py-1 text-xs bg-black/5 dark:bg-white/10 backdrop-blur-md mb-6 animate-fade-in">
          <span className="text-black/70 dark:text-white/70">PCA-Based Research Project</span>
        </div>
        
        <h1 className="text-4xl md:text-6xl font-bold leading-tight tracking-tight mb-6 animate-slide-up text-balance">
          Dynamic Multi-Layer
          <br />
          Audio Watermarking
        </h1>
        
        <p className="text-lg md:text-xl text-black/70 dark:text-white/70 max-w-2xl mx-auto mb-10 animation-delay-100 animate-slide-up text-balance">
          Explore our innovative approach to embedding multiple watermarks in audio 
          using Principal Component Analysis for enhanced robustness and security.
        </p>
        
        <div className="flex flex-col md:flex-row gap-4 justify-center animation-delay-200 animate-slide-up">
          <Button
            onClick={scrollToUpload}
            className="bg-black hover:bg-black/90 text-white dark:bg-white dark:hover:bg-white/90 dark:text-black px-6 py-6 rounded-full h-12"
          >
            Try It Now
          </Button>
          <Button
            variant="outline"
            className="border-black/20 dark:border-white/20 hover:bg-black/5 dark:hover:bg-white/5 rounded-full px-6 h-12"
            onClick={() => document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' })}
          >
            Learn More
          </Button>
        </div>
      </div>
      
      <div className="absolute bottom-10 left-0 right-0 flex justify-center animation-delay-500 animate-fade-in">
        <button 
          onClick={() => document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' })}
          className="flex flex-col items-center text-black/50 dark:text-white/50 hover:text-black dark:hover:text-white transition-colors"
        >
          <span className="text-xs mb-2">Scroll Down</span>
          <ChevronDown size={20} className="animate-bounce" />
        </button>
      </div>
    </section>
  );
};

export default HeroSection;
