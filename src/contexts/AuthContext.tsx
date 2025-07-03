import { createContext, useState, useEffect, ReactNode } from "react";
import { toast } from "sonner";
import { userRegistry } from "@/services/userRegistry";

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


  const login = async (email: string, password: string) => {
    setLoading(true);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API delay
      
      if (!email || !password) {
        throw new Error("Email and password are required.");
      }

      // --- THIS IS THE NEW, SIMPLE LOGIC ---
      // 1. Get all our known users.
      const allUsers = userRegistry.getAllUsers();
      
      // 2. Find the one user where BOTH email and password match.
      const foundUser = allUsers.find(u => 
        u.email.toLowerCase() === email.toLowerCase() && u.password === password
      );

      // 3. If we found a match, log them in.
      if (foundUser) {
        const userToLogin: User = {
            id: foundUser.id, // The ID is now guaranteed to be small and correct (1-5)
            email: foundUser.email,
            role: foundUser.role as UserRole,
            name: foundUser.name
        };
        
        localStorage.setItem("awl_user", JSON.stringify(userToLogin));
        setUser(userToLogin);
        setIsAuthenticated(true);
        toast.success(`Welcome back, ${userToLogin.name}!`);
        return;
      } else {
        // 4. If no match, the login fails.
        throw new Error("Invalid email or password.");
      }
    } catch (error) {
      toast.error(`Login failed: ${error instanceof Error ? error.message : "Unknown error"}`);
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
      
      // Register user in the registry
      userRegistry.registerUser({
        id: mockUser.id,
        name: mockUser.name || email.split('@')[0],
        role: mockUser.role,
        email: mockUser.email
      });
      
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
