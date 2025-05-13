
import { createContext, useState, useEffect, ReactNode } from "react";
import { toast } from "sonner";

// User roles for the application
export type UserRole = "voice_actor" | "producer" | "editor" | "marketer" | "auditor";

export interface User {
  id: number;
  email: string;
  role: UserRole;
  name?: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string, role: UserRole) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
}

export const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: false,
  login: async () => {},
  register: async () => {},
  logout: () => {},
  isAuthenticated: false,
});

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider = ({ children }: AuthProviderProps) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Load user from localStorage on mount
  useEffect(() => {
    const storedUser = localStorage.getItem("awl_user");
    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setUser(parsedUser);
        setIsAuthenticated(true);
      } catch (error) {
        console.error("Failed to parse stored user:", error);
        localStorage.removeItem("awl_user");
      }
    }
    setLoading(false);
  }, []);

  // Login function - for now we'll implement a mock version
  // In the future, this would integrate with Supabase
  const login = async (email: string, password: string) => {
    setLoading(true);
    
    try {
      // Mock successful login for demonstration
      // In a real app, this would make an API call to authenticate
      if (email && password) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // Mock user data - in real app, this would come from an API
        const mockUser: User = {
          id: 1,
          email,
          role: "voice_actor",
          name: email.split('@')[0] // Just use part of the email as a name
        };
        
        // Store in localStorage for persistence
        localStorage.setItem("awl_user", JSON.stringify(mockUser));
        
        setUser(mockUser);
        setIsAuthenticated(true);
        toast.success("Successfully logged in!");
        
        return;
      }
      
      throw new Error("Invalid credentials");
    } catch (error) {
      toast.error("Login failed: " + (error instanceof Error ? error.message : "Unknown error"));
      throw error;
    } finally {
      setLoading(false);
    }
  };

  // Register function - mock implementation
  const register = async (email: string, password: string, name: string, role: UserRole) => {
    setLoading(true);
    
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock user creation - in real app, this would be an API call
      const mockUser: User = {
        id: Math.floor(Math.random() * 1000),
        email,
        name,
        role
      };
      
      // Store in localStorage for persistence
      localStorage.setItem("awl_user", JSON.stringify(mockUser));
      
      setUser(mockUser);
      setIsAuthenticated(true);
      toast.success("Account created successfully!");
    } catch (error) {
      toast.error("Registration failed: " + (error instanceof Error ? error.message : "Unknown error"));
      throw error;
    } finally {
      setLoading(false);
    }
  };

  // Logout function
  const logout = () => {
    localStorage.removeItem("awl_user");
    setUser(null);
    setIsAuthenticated(false);
    toast.info("You have been logged out");
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        register,
        logout,
        isAuthenticated
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
