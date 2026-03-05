"use client";
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { isAuthenticated, isTokenExpired, getToken, clearAuth } from '@/app/lib/auth';

export default function RootPage() {
  const router = useRouter();

  useEffect(() => {
    const token = getToken();

    if (!token || !isAuthenticated()) {
      router.replace('/login');
      return;
    }

    if (isTokenExpired(token)) {
      clearAuth();
      router.replace('/login');
      return;
    }

    router.replace('/home');
  }, [router]);

  // Minimal loading state while redirecting
  return (
    <div className="flex h-screen items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-3">
        <div className="w-6 h-6 border-2 border-foreground border-t-transparent rounded-full animate-spin opacity-40" />
        <span className="text-muted-foreground text-sm font-mono">Initializing...</span>
      </div>
    </div>
  );
}
