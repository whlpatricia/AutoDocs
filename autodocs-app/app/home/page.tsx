"use client"
import { Sidebar } from '@/app/components/Sidebar';
import { UploadTile } from '@/app/components/UploadTile';
import { SessionTile } from '@/app/components/SessionTile';
import { SessionDetailModal } from '@/app/components/SessionDetailModal';
import { getSessionUser } from '@/app/lib/auth';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

export interface SessionEvent {
  label: string;
  depth: number; // -1 = subevent, 0 = independent, >=1 = exiting
}

// Shape returned by the API
export interface ApiSession {
  id: string;
  title: string;
  durationSeconds: number;
  content: string; // raw .txt file content
  createdAt: string;
}

// Shape used internally / passed to components
export interface Session {
  id: string;
  title: string;
  duration: string; // formatted, e.g. "42m 17s"
  content: string;  // raw .txt file content
  createdAt: string;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return s > 0 ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem > 0 ? `${h}h ${rem}m` : `${h}h`;
}

function formatDate(isoString: string): string {
  return new Date(isoString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}

/** Parse raw .txt content into structured SessionEvent[]. */
export function parseSessionContent(raw: string): SessionEvent[] {
  const lines = raw.split('\n');
  const events: SessionEvent[] = [];
  let i = 0;

  while (i < lines.length) {
    const label = lines[i].trim();
    if (!label) { i++; continue; }

    // Next non-empty line should be the depth value
    let j = i + 1;
    while (j < lines.length && lines[j].trim() === '') j++;

    if (j < lines.length) {
      const depthVal = parseInt(lines[j].trim(), 10);
      if (!isNaN(depthVal)) {
        events.push({ label, depth: depthVal });
        i = j + 1;
        continue;
      }
    }

    // No depth line found — skip this line
    i++;
  }

  return events;
}

export default function App() {
  const router = useRouter();
  const [checkingAuth, setCheckingAuth] = useState(true);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [sessionsError, setSessionsError] = useState<string | null>(null);
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Auth check
  useEffect(() => {
    const run = async () => {
      const user = await getSessionUser();
      if (!user) {
        router.replace('/login');
        return;
      }
      setCheckingAuth(false);
    };
    void run();
  }, [router]);

  const fetchSessions = async () => {
    setLoadingSessions(true);
    setSessionsError(null);
    try {
      const res = await fetch('/api/terminal-sessions');
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.message ?? `Request failed with status ${res.status}`);
      }
      const data = await res.json() as { terminalSessions: ApiSession[] };
      setSessions(
        data.terminalSessions.map((s) => ({
          id: s.id,
          title: s.title,
          duration: formatDuration(s.durationSeconds),
          content: s.content,
          createdAt: formatDate(s.createdAt),
        }))
      );
    } catch (err) {
      setSessionsError(err instanceof Error ? err.message : 'Failed to load sessions.');
    } finally {
      setLoadingSessions(false);
    }
  };

  // Fetch sessions once auth is confirmed
  useEffect(() => {
    if (checkingAuth) return;

    void fetchSessions();
  }, [checkingAuth]);

  const handleSessionClick = (session: Session) => {
    setSelectedSession(session);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedSession(null);
  };

  if (checkingAuth) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-3">
          <div className="w-6 h-6 border-2 border-foreground border-t-transparent rounded-full animate-spin opacity-40" />
          <span className="text-muted-foreground text-sm font-mono">Checking session...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className="p-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-foreground mb-2">Terminal Sessions</h1>
            <p className="text-muted-foreground">
              Upload and explore hierarchical event logs from your terminal sessions
            </p>
          </div>

          {/* Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Upload Tile — always first */}
            <UploadTile onUploadSuccess={fetchSessions}/>

            {/* Loading skeleton */}
            {loadingSessions &&
              Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="bg-card rounded-xl border border-border p-6 animate-pulse">
                  <div className="w-12 h-12 bg-muted rounded-lg mb-4" />
                  <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                  <div className="h-3 bg-muted rounded w-1/2 mb-4" />
                  <div className="h-20 bg-muted rounded" />
                </div>
              ))
            }

            {/* Error state */}
            {!loadingSessions && sessionsError && (
              <div className="col-span-full text-sm text-destructive font-mono bg-destructive/10 rounded-lg p-4 border border-destructive/20">
                {sessionsError}
              </div>
            )}

            {/* Empty state */}
            {!loadingSessions && !sessionsError && sessions.length === 0 && (
              <div className="col-span-full text-sm text-muted-foreground font-mono p-4">
                No sessions yet. Upload your first terminal session to get started.
              </div>
            )}

            {/* Session Tiles */}
            {!loadingSessions && sessions.map((session) => (
              <SessionTile
                key={session.id}
                session={session}
                onClick={() => handleSessionClick(session)}
              />
            ))}
          </div>
        </div>
      </main>

      {/* Session Detail Modal */}
      <SessionDetailModal
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        session={selectedSession}
      />
    </div>
  );
}