export interface UserInfo {
  id: number;
  name: string;
  role: string;
  email: string;
  password?: string; // Add optional password field
}

class UserRegistry {
  private static instance: UserRegistry;
  private users: Map<number, UserInfo> = new Map();
  private storageKey = 'awl_user_registry';

  private constructor() {
    this.loadFromStorage();
    this.initializeMockUsers();
  }

  public static getInstance(): UserRegistry {
    if (!UserRegistry.instance) {
      UserRegistry.instance = new UserRegistry();
    }
    return UserRegistry.instance;
  }

  private loadFromStorage(): void {
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        const userArray = JSON.parse(stored);
        this.users = new Map(userArray.map((user: UserInfo) => [user.id, user]));
      }
    } catch (error) {
      console.error('Failed to load user registry from storage:', error);
    }
  }

  private saveToStorage(): void {
    try {
      const userArray = Array.from(this.users.values());
      localStorage.setItem(this.storageKey, JSON.stringify(userArray));
    } catch (error) {
      console.error('Failed to save user registry to storage:', error);
    }
  }

  private initializeMockUsers(): void {
    const mockUsers: UserInfo[] = [
      { id: 1, name: "Mayi", role: "voice_actor", email: "mayi@example.com", password: "password1" },
      { id: 2, name: "Alice", role: "producer", email: "alice@example.com", password: "password2" },
      { id: 3, name: "Bob", role: "editor", email: "bob@example.com", password: "password3" },
      { id: 4, name: "Charlie", role: "marketer", email: "charlie@example.com", password: "password4" },
      { id: 5, name: "Diana", role: "auditor", email: "diana@example.com", password: "password5" }
    ];

    let needsSave = false;
    mockUsers.forEach(user => {
      if (!this.users.has(user.id)) {
        this.users.set(user.id, user);
        needsSave = true;
      }
    });
    
    if (needsSave) {
      this.saveToStorage();
    }
  }

  public registerUser(user: UserInfo): void {
    this.users.set(user.id, user);
    this.saveToStorage();
  }

  public getUser(id: number): UserInfo | undefined {
    return this.users.get(id);
  }

  public getAllUsers(): UserInfo[] {
    return Array.from(this.users.values());
  }

  public getUserName(id: number): string {
    const user = this.users.get(id);
    return user ? user.name : `User ${id}`;
  }
}

export const userRegistry = UserRegistry.getInstance();