"use client";
import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { setToken, setUser, isAuthenticated, getToken, isTokenExpired, clearAuth } from '@/app/lib/auth';

// -- Simulated API call --
// Replace the body of this function with your real fetch() to your backend.
async function loginRequest(email: string, password: string): Promise<{ token: string; user: { id: string; email: string; name: string; createdAt: string } }> {
  // TODO: replace with real API call, e.g.:
  // const res = await fetch('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) });
  // if (!res.ok) throw new Error((await res.json()).message ?? 'Login failed');
  // return res.json();

  await new Promise((r) => setTimeout(r, 1200)); // simulate latency

  if (email === 'demo@example.com' && password === 'password') {
    // A real JWT would come from your backend; this is a mock payload
    const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
    const payload = btoa(JSON.stringify({
      sub: 'usr_001',
      email,
      name: 'Demo User',
      exp: Math.floor(Date.now() / 1000) + 60 * 60 * 8, // 8h
    }));
    const sig = btoa('mock-signature');
    return {
      token: `${header}.${payload}.${sig}`,
      user: { id: 'usr_001', email, name: 'Demo User', createdAt: '2025-01-01T00:00:00Z' },
    };
  }
  throw new Error('Invalid email or password');
}

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [mounted, setMounted] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setMounted(true);
    const token = getToken();
    if (token && isAuthenticated() && !isTokenExpired(token)) {
      router.replace('/home');
      return;
    }
    if (token) clearAuth();
    setTimeout(() => inputRef.current?.focus(), 400);
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please fill in all fields.');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const { token, user } = await loginRequest(email, password);
      setToken(token);
      setUser({ ...user, avatarUrl: undefined });
      router.push('/home');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center bg-background relative overflow-hidden"
      style={{ opacity: mounted ? 1 : 0, transition: 'opacity 0.4s ease' }}
    >
      {/* Subtle grid background */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px',
        }}
      />

      {/* Glow blob */}
      <div
        aria-hidden
        className="pointer-events-none absolute"
        style={{
          width: 480,
          height: 480,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(100,255,150,0.06) 0%, transparent 70%)',
          top: '50%',
          left: '50%',
          transform: 'translate(-60%, -50%)',
        }}
      />

      <div
        className="relative w-full max-w-sm mx-4"
        style={{
          animation: 'slideUp 0.5s cubic-bezier(0.16,1,0.3,1) both',
          animationDelay: '0.1s',
        }}
      >
        {/* Terminal chrome bar */}
        <div
          className="flex items-center gap-2 px-4 py-3 rounded-t-xl"
          style={{ background: 'rgba(255,255,255,0.04)', borderBottom: '1px solid rgba(255,255,255,0.06)' }}
        >
          <span className="w-3 h-3 rounded-full bg-red-500 opacity-70" />
          <span className="w-3 h-3 rounded-full bg-yellow-500 opacity-70" />
          <span className="w-3 h-3 rounded-full bg-green-500 opacity-70" />
          <span className="ml-2 text-xs font-mono text-muted-foreground tracking-widest">auth — login</span>
        </div>

        {/* Card body */}
        <div
          className="px-8 py-8 rounded-b-xl"
          style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderTop: 'none',
            backdropFilter: 'blur(12px)',
          }}
        >
          <div className="mb-7">
            <p className="font-mono text-xs text-muted-foreground mb-1">$ auth --mode login</p>
            <h1 className="text-foreground text-2xl font-semibold tracking-tight">Welcome back</h1>
            <p className="text-muted-foreground text-sm mt-1">Sign in to your account to continue.</p>
          </div>

          <form onSubmit={handleSubmit} noValidate>
            <div className="flex flex-col gap-4">
              <label className="flex flex-col gap-1.5">
                <span className="font-mono text-xs text-muted-foreground">email</span>
                <input
                  ref={inputRef}
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  disabled={loading}
                  className="input-field"
                  style={inputStyle}
                />
              </label>

              <label className="flex flex-col gap-1.5">
                <span className="font-mono text-xs text-muted-foreground">password</span>
                <input
                  type="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  disabled={loading}
                  style={inputStyle}
                />
              </label>

              {error && (
                <div
                  className="font-mono text-xs px-3 py-2 rounded-md"
                  style={{ background: 'rgba(239,68,68,0.12)', color: '#f87171', border: '1px solid rgba(239,68,68,0.2)' }}
                >
                  ✗ {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                style={submitButtonStyle(loading)}
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-3.5 h-3.5 border border-current border-t-transparent rounded-full animate-spin" />
                    Authenticating...
                  </span>
                ) : (
                  '→ Sign in'
                )}
              </button>
            </div>
          </form>

          <p className="mt-6 text-center text-sm text-muted-foreground font-mono">
            No account?{' '}
            <Link href="/signup" className="text-foreground underline underline-offset-4 decoration-dotted hover:opacity-70 transition-opacity">
              Sign up
            </Link>
          </p>

          {/* Demo hint */}
          <p
            className="mt-4 text-center font-mono text-xs px-3 py-2 rounded-md"
            style={{ background: 'rgba(255,255,255,0.04)', color: 'rgba(255,255,255,0.3)' }}
          >
            demo: demo@example.com / password
          </p>
        </div>
      </div>

      <style>{`
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(20px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        input::placeholder { opacity: 0.3; }
        input:focus { outline: none; border-color: rgba(255,255,255,0.25) !important; }
      `}</style>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 8,
  padding: '10px 14px',
  color: 'inherit',
  fontFamily: 'var(--font-geist-mono, monospace)',
  fontSize: 13,
  transition: 'border-color 0.15s',
};

const submitButtonStyle = (loading: boolean): React.CSSProperties => ({
  width: '100%',
  marginTop: 4,
  padding: '11px 16px',
  background: loading ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.1)',
  border: '1px solid rgba(255,255,255,0.15)',
  borderRadius: 8,
  color: loading ? 'rgba(255,255,255,0.4)' : 'inherit',
  fontFamily: 'var(--font-geist-mono, monospace)',
  fontSize: 13,
  cursor: loading ? 'not-allowed' : 'pointer',
  transition: 'background 0.15s, opacity 0.15s',
  letterSpacing: '0.02em',
});
