
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ExternalLink, ArrowUp } from "lucide-react";

const AboutSection = () => {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
  
  return (
    <section id="about" className="py-24 px-6 md:px-10 bg-black/5 dark:bg-white/5">
      <div className="max-w-4xl mx-auto">
        <div className="flex flex-col gap-3 mb-12 animate-fade-in">
          <div className="flex items-center gap-2">
            <div className="h-8 w-1 bg-black dark:bg-white rounded-full"></div>
            <h2 className="text-xl md:text-2xl font-semibold">About This Research</h2>
          </div>
          <p className="text-black/70 dark:text-white/70 ml-3 pl-3">
            Learn more about the Dynamic Multi-Layer Audio Watermarking project
          </p>
        </div>
        
        <div className="space-y-10">
          <Card className="rounded-xl overflow-hidden animate-scale-in">
            <div className="p-6 md:p-8">
              <h3 className="text-xl font-medium mb-6">Research Background</h3>
              
              <div className="space-y-6 text-balance">
                <p className="text-black/80 dark:text-white/80">
                  Audio watermarking plays a crucial role in content protection and ownership verification. 
                  Our research introduces a novel approach that utilizes Principal Component Analysis (PCA) 
                  to identify optimal frequency bands for embedding multiple watermarks in audio signals.
                </p>
                
                <p className="text-black/80 dark:text-white/80">
                  Unlike traditional methods that use fixed frequency bands or simple bit replacement, 
                  our dynamic approach adapts to the specific characteristics of each audio signal, 
                  providing enhanced robustness against common attacks while maintaining audio quality.
                </p>
                
                <div className="bg-black/5 dark:bg-white/5 p-4 rounded-lg mb-4">
                  <h4 className="font-medium mb-2">Key Innovations</h4>
                  <ul className="list-disc list-inside space-y-2 text-black/80 dark:text-white/80">
                    <li>Adaptive band selection using PCA</li>
                    <li>Multi-layer watermark embedding for redundancy</li>
                    <li>Enhanced robustness against noise, compression, and resampling</li>
                    <li>Comparative analysis with blockchain-based verification</li>
                  </ul>
                </div>
              </div>
            </div>
          </Card>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="rounded-xl animate-slide-up">
              <div className="p-6">
                <h3 className="text-lg font-medium mb-4">Methodology</h3>
                
                <ol className="list-decimal list-inside space-y-4 text-black/80 dark:text-white/80">
                  <li className="mb-2">
                    <span className="font-medium">Feature Extraction</span>
                    <p className="text-sm text-black/60 dark:text-white/60 pl-5 mt-1">
                      Transforming audio to frequency domain using STFT
                    </p>
                  </li>
                  
                  <li className="mb-2">
                    <span className="font-medium">PCA Analysis</span>
                    <p className="text-sm text-black/60 dark:text-white/60 pl-5 mt-1">
                      Identifying principal components for optimal embedding
                    </p>
                  </li>
                  
                  <li className="mb-2">
                    <span className="font-medium">Multi-Layer Embedding</span>
                    <p className="text-sm text-black/60 dark:text-white/60 pl-5 mt-1">
                      Distributing watermarks across different frequency bands
                    </p>
                  </li>
                  
                  <li className="mb-2">
                    <span className="font-medium">Robustness Testing</span>
                    <p className="text-sm text-black/60 dark:text-white/60 pl-5 mt-1">
                      Evaluating against noise, compression, and resampling attacks
                    </p>
                  </li>
                </ol>
              </div>
            </Card>
            
            <Card className="rounded-xl animate-slide-up animation-delay-100">
              <div className="p-6">
                <h3 className="text-lg font-medium mb-4">Future Work</h3>
                
                <div className="space-y-4">
                  <p className="text-black/80 dark:text-white/80">
                    Our research continues to explore several promising directions:
                  </p>
                  
                  <div className="space-y-3">
                    <div className="flex items-start gap-3">
                      <div className="w-1 h-1 rounded-full bg-black dark:bg-white mt-2.5"></div>
                      <p className="text-black/80 dark:text-white/80">
                        Integration with blockchain for tamper-evident verification
                      </p>
                    </div>
                    
                    <div className="flex items-start gap-3">
                      <div className="w-1 h-1 rounded-full bg-black dark:bg-white mt-2.5"></div>
                      <p className="text-black/80 dark:text-white/80">
                        Machine learning approaches to further optimize embedding locations
                      </p>
                    </div>
                    
                    <div className="flex items-start gap-3">
                      <div className="w-1 h-1 rounded-full bg-black dark:bg-white mt-2.5"></div>
                      <p className="text-black/80 dark:text-white/80">
                        Support for real-time watermarking in streaming applications
                      </p>
                    </div>
                    
                    <div className="flex items-start gap-3">
                      <div className="w-1 h-1 rounded-full bg-black dark:bg-white mt-2.5"></div>
                      <p className="text-black/80 dark:text-white/80">
                        Extension to other media types including speech and music
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>
          
          <div className="flex flex-col items-center justify-center text-center space-y-6 animate-fade-in">
            <h3 className="text-xl font-medium">University of Zimbabwe</h3>
            <p className="text-black/60 dark:text-white/60 max-w-xl">
              This research is conducted under the supervision of the Faculty of Engineering
              at the University of Zimbabwe, as part of ongoing work in digital signal processing
              and media security.
            </p>
            
            <Button 
              variant="outline"
              className="rounded-full"
              onClick={scrollToTop}
            >
              <ArrowUp size={16} className="mr-2" />
              Back to Top
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;
