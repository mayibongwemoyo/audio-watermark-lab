
import { useContext } from "react";
import { Outlet } from "react-router-dom";
import { 
  SidebarProvider, 
  Sidebar, 
  SidebarContent, 
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem, 
  SidebarMenuButton,
  SidebarFooter,
  SidebarInset,
  SidebarTrigger
} from "@/components/ui/sidebar";
import { AuthContext } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Home, FileAudio, BarChart2, BookOpen, LogOut } from "lucide-react";
import { Link, useLocation } from "react-router-dom";

const AppLayout = () => {
  const { user, logout } = useContext(AuthContext);
  const location = useLocation();
  
  return (
    <SidebarProvider defaultOpen>
      <div className="flex min-h-screen w-full">
        <Sidebar>
          <SidebarHeader className="px-6 py-3">
            <div className="flex items-center">
              <FileAudio className="w-6 h-6 mr-2" />
              <h1 className="text-lg font-semibold">Audio Watermark Lab</h1>
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              Application Mode
            </div>
          </SidebarHeader>
          
          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupLabel>Navigation</SidebarGroupLabel>
              <SidebarMenu>
                <SidebarMenuItem>
                  <SidebarMenuButton asChild isActive={location.pathname === "/app"}>
                    <Link to="/app">
                      <Home />
                      <span>Dashboard</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
                
                <SidebarMenuItem>
                  <SidebarMenuButton asChild isActive={location.pathname === "/app/detect"}>
                    <Link to="/app/detect">
                      <BarChart2 />
                      <span>Detect & Trace</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
                
                <SidebarMenuItem>
                  <SidebarMenuButton asChild isActive={location.pathname === "/app/ledger"}>
                    <Link to="/app/ledger">
                      <BookOpen />
                      <span>Watermark Ledger</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              </SidebarMenu>
            </SidebarGroup>
            
            <SidebarGroup>
              <SidebarGroupLabel>Account</SidebarGroupLabel>
              <div className="px-4 py-2">
                <div className="font-medium">{user?.name || 'User'}</div>
                <div className="text-xs text-muted-foreground">{user?.role?.replace('_', ' ')}</div>
                <div className="text-xs text-muted-foreground mt-1">{user?.email}</div>
              </div>
            </SidebarGroup>
          </SidebarContent>
          
          <SidebarFooter>
            <div className="flex flex-col gap-2 px-4 py-2">
              <Link to="/">
                <Button variant="outline" size="sm" className="w-full">
                  Research Mode
                </Button>
              </Link>
              <Button 
                variant="ghost" 
                size="sm" 
                className="w-full flex items-center justify-center" 
                onClick={logout}
              >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </Button>
            </div>
          </SidebarFooter>
        </Sidebar>
        
        <SidebarInset>
          <div className="relative flex-1 p-6">
            <div className="absolute top-4 left-4">
              <SidebarTrigger />
            </div>
            <Outlet />
          </div>
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
};

export default AppLayout;
