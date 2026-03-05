// Auth utilities — JWT stored in sessionStorage (cleared on tab close)

const TOKEN_KEY = 'auth_token';
const USER_KEY = 'auth_user';

export interface AuthUser {
  id: string;
  email: string;
  name: string;
  avatarUrl?: string;
  createdAt: string;
}

export function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return sessionStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  sessionStorage.setItem(TOKEN_KEY, token);
}

export function getUser(): AuthUser | null {
  if (typeof window === 'undefined') return null;
  const raw = sessionStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AuthUser;
  } catch {
    return null;
  }
}

export function setUser(user: AuthUser): void {
  sessionStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearAuth(): void {
  sessionStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(USER_KEY);
}

export function isAuthenticated(): boolean {
  return !!getToken();
}

// Decode JWT payload without verification (verification happens server-side)
export function decodeToken(token: string): Record<string, unknown> | null {
  try {
    const payload = token.split('.')[1];
    return JSON.parse(atob(payload));
  } catch {
    return null;
  }
}

export function isTokenExpired(token: string): boolean {
  const decoded = decodeToken(token);
  if (!decoded || typeof decoded.exp !== 'number') return true;
  return Date.now() / 1000 > decoded.exp;
}
