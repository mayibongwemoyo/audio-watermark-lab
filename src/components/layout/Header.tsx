
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Moon, Sun } from 'lucide-react';

const Header = () => {
  const [isDark, setIsDark] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };
    
    const checkDarkMode = () => {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setIsDark(prefersDark);
      if (prefersDark) {
        document.documentElement.classList.add('dark');
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    checkDarkMode();
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const toggleDarkMode = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle('dark');
  };

  return (
    <header className={`fixed top-0 left-0 right-0 z-50 px-6 md:px-10 py-4 transition-all duration-300 ${scrolled ? 'bg-white/80 dark:bg-black/80 backdrop-blur-lg shadow-sm' : ''}`}>
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex space-x-1 h-6">
            {[1, 2, 3, 4, 5].map((i) => (
              <div 
                key={i} 
                className={`w-[3px] h-full bg-black dark:bg-white rounded-full animate-waveform-${i}`}
              />
            ))}
          </div>
          <h1 className="text-lg font-medium tracking-tight ml-2">
            Dynamic Watermarking
          </h1>
        </div>
        
        <div className="flex items-center gap-4">
          <nav className="hidden md:flex gap-8">
            <a href="#upload" className="text-sm hover:text-black/70 dark:hover:text-white/70 transition-colors">Upload</a>
            <a href="#watermark" className="text-sm hover:text-black/70 dark:hover:text-white/70 transition-colors">Watermark</a>
            <a href="#analysis" className="text-sm hover:text-black/70 dark:hover:text-white/70 transition-colors">Analysis</a>
            <a href="#about" className="text-sm hover:text-black/70 dark:hover:text-white/70 transition-colors">About</a>
          </nav>
          
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleDarkMode}
            className="rounded-full"
          >
            {isDark ? <Sun size={18} /> : <Moon size={18} />}
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;
