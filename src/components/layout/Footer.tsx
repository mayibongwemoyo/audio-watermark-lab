
import { Github } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-12 px-6 md:px-10 border-t border-black/10 dark:border-white/10 mt-20">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="flex space-x-1 h-4">
              {[1, 2, 3].map((i) => (
                <div 
                  key={i} 
                  className={`w-[2px] h-full bg-black dark:bg-white rounded-full animate-waveform-${i}`}
                />
              ))}
            </div>
            <span className="text-sm font-medium">Dynamic Watermarking</span>
          </div>
          
          <div className="flex flex-col md:flex-row gap-4 md:gap-8 items-center">
            <span className="text-xs text-muted-foreground">University of Zimbabwe</span>
            <span className="text-xs text-muted-foreground">© {new Date().getFullYear()} Research Project</span>
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-xs flex items-center gap-1 text-muted-foreground hover:text-black dark:hover:text-white transition-colors"
            >
              <Github size={14} />
              <span>Source Code</span>
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
