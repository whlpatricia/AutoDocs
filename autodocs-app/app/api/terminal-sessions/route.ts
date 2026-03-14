import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/app/lib/server/db';
import { getRequestUser } from '@/app/lib/server/request-auth';

interface TerminalSessionBody {
  title?: string;
  durationSeconds?: number;
  content?: string;
}

interface TerminalSessionRow {
  id: string;
  title: string;
  durationSeconds: number;
  content: string;
  createdAt: string;
}

export async function GET(request: NextRequest) {
  try {
    const user = await getRequestUser(request);
    if (!user) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const result = await db.query<TerminalSessionRow>(
      `SELECT ts.id,
              ts.title,
              ts.duration_seconds AS "durationSeconds",
              ts.content,
              ts.created_at AS "createdAt"
       FROM terminal_sessions ts
       INNER JOIN terminal_session_access tsa
         ON tsa.terminal_session_id = ts.id
       WHERE tsa.user_id = $1 AND tsa.owner = TRUE
       ORDER BY ts.created_at DESC`,
      [user.id],
    );
    return NextResponse.json({ terminalSessions: result.rows });
  } catch (e){
    console.log(e);
    return NextResponse.json({ message: e}, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  let transactionStarted = false;

  try {
    const user = await getRequestUser(request);
    if (!user) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const body = (await request.json()) as TerminalSessionBody;
    const title = body.title?.trim();
    const durationSeconds = body.durationSeconds;
    const content = body.content?.trim();

    if (!title) {
      return NextResponse.json({ message: 'title is required.' }, { status: 400 });
    }

    if (typeof durationSeconds !== 'number' || !Number.isInteger(durationSeconds) || durationSeconds < 0) {
      return NextResponse.json(
        { message: 'durationSeconds must be a non-negative integer.' },
        { status: 400 },
      );
    }

    if (!content) {
      return NextResponse.json({ message: 'content is required.' }, { status: 400 });
    }

    await db.query('BEGIN');
    transactionStarted = true;

    const sessionInsert = await db.query<TerminalSessionRow>(
      `INSERT INTO terminal_sessions (title, duration_seconds, content)
       VALUES ($1, $2, $3)
       RETURNING id,
                 title,
                 duration_seconds AS "durationSeconds",
                 content,
                 created_at AS "createdAt"`,
      [title, durationSeconds, content],
    );

    const terminalSession = sessionInsert.rows[0];

    await db.query(
      `INSERT INTO terminal_session_access (user_id, terminal_session_id, owner)
       VALUES ($1, $2, TRUE)`,
      [user.id, terminalSession.id],
    );

    await db.query('COMMIT');
    transactionStarted = false;

    return NextResponse.json({ terminalSession }, { status: 201 });
  } catch {
    if (transactionStarted) {
      await db.query('ROLLBACK');
    }
    return NextResponse.json({ message: 'Failed to create terminal session.' }, { status: 500 });
  }
}
