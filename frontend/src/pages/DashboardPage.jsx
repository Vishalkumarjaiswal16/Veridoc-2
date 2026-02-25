import React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { logout } from "@/services/authService";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Logo } from "@/components/Logo";

export default function DashboardPage() {
    const navigate = useNavigate();

    const handleLogout = () => {
        logout();
        navigate("/login");
    };

    return (
        <div className="p-8 min-h-screen transition-colors duration-300">
            <div className="flex justify-between items-center mb-8">
                <div className="flex items-center gap-4">
                    <Logo className="h-8 w-auto text-primary" />
                    <h1 className="text-3xl font-bold">Dashboard</h1>
                </div>
                <div className="flex items-center gap-4">
                    <ThemeToggle />
                    <Button variant="outline" onClick={handleLogout}>Logout</Button>
                </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card>
                    <CardHeader>
                        <CardTitle>Welcome to Veridoc</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <p className="text-muted-foreground">You have successfully logged in.</p>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
