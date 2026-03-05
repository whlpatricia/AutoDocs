"use client";
import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { setToken, setUser, isAuthenticated, getToken, isTokenExpired, clearAuth } from '@/app/lib/auth';

// -- Simulated API call --
// Replace with your real backend endpoint.
async function signupRequest(name: string, email: string, password: string): Promise<{ token: string; user: { id: string; email: string; name: string; createdAt: string } }> {
  // TODO: replace with real API call, e.g.:
  // const res = await fetch('/api/auth/signup', { method: 'POST', body: JSON.stringify({ name, email, password }) });
  // if (!res.ok) throw new Error((await res.json()).message ?? 'Signup failed');
  // return res.json();

  await new Promise((r) => setTimeout(r, 1400));

  if (email === 'taken@example.com') {
    throw new Error('An account with this email already exists.');
  }

  const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
  const payload = btoa(JSON.stringify({
    sub: 'usr_' + Math.random().toString(36).slice(2, 8),
    email,
    name,
    exp: Math.floor(Date.now() / 1000) + 60 * 60 * 8,
  }));
  const sig = btoa('mock-signature');

  return {
    token: `${header}.${payload}.${sig}`,
    user: { id: 'usr_new', email, name, createdAt: new Date().toISOString() },
  };
}

function validatePassword(pw: string): string | null {
  if (pw.length < 8) return 'Password must be at least 8 characters.';
  return null;
}

export default function SignupPage() {
  const router = useRouter();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
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
    if (!name || !email || !password || !confirm) {
      setError('Please fill in all fields.');
      return;
    }
    const pwError = validatePassword(password);
    if (pwError) { setError(pwError); return; }
    if (password !== confirm) { setError('Passwords do not match.'); return; }

    setError('');
    setLoading(true);
    try {
      const { token, user } = await signupRequest(name, email, password);
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
      {/* Grid background */}
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
          width: 500,
          height: 500,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(100,180,255,0.05) 0%, transparent 70%)',
          top: '50%',
          left: '50%',
          transform: 'translate(-40%, -50%)',
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
          <span className="ml-2 text-xs font-mono text-muted-foreground tracking-widest">auth — signup</span>
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
            <p className="font-mono text-xs text-muted-foreground mb-1">$ auth --mode signup</p>
            <h1 className="text-foreground text-2xl font-semibold tracking-tight">Create account</h1>
            <p className="text-muted-foreground text-sm mt-1">Start managing your terminal sessions.</p>
          </div>

          <form onSubmit={handleSubmit} noValidate>
            <div className="flex flex-col gap-4">
              <label className="flex flex-col gap-1.5">
                <span className="font-mono text-xs text-muted-foreground">display name</span>
                <input
                  ref={inputRef}
                  type="text"
                  autoComplete="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Your name"
                  disabled={loading}
                  style={inputStyle}
                />
              </label>

              <label className="flex flex-col gap-1.5">
                <span className="font-mono text-xs text-muted-foreground">email</span>
                <input
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  disabled={loading}
                  style={inputStyle}
                />
              </label>

              <label className="flex flex-col gap-1.5">
                <span className="font-mono text-xs text-muted-foreground">password</span>
                <input
                  type="password"
                  autoComplete="new-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Min. 8 characters"
                  disabled={loading}
                  style={inputStyle}
                />
              </label>

              <label className="flex flex-col gap-1.5">
                <span className="font-mono text-xs text-muted-foreground">confirm password</span>
                <input
                  type="password"
                  autoComplete="new-password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  placeholder="••••••••"
                  disabled={loading}
                  style={{
                    ...inputStyle,
                    borderColor: confirm && confirm !== password ? 'rgba(239,68,68,0.4)' : undefined,
                  }}
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
                    Creating account...
                  </span>
                ) : (
                  '→ Create account'
                )}
              </button>
            </div>
          </form>

          <p className="mt-6 text-center text-sm text-muted-foreground font-mono">
            Already have an account?{' '}
            <Link href="/login" className="text-foreground underline underline-offset-4 decoration-dotted hover:opacity-70 transition-opacity">
              Sign in
            </Link>
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
  transition: 'background 0.15s',
  letterSpacing: '0.02em',
});
