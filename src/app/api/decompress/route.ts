import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { existsSync } from 'fs';
import path from 'path';
import { spawn } from 'child_process';

export const runtime = 'nodejs';
export const maxDuration = 600;

const ENGINE_PATH = path.join(process.cwd(), 'NFR-Compressor', 'nfr_v6_engine.py');
const PYTHON_BIN = process.env.NFR_PYTHON || '/home/z/.venv/bin/python';

/**
 * Decompress an NFR file. Returns immediately; client polls /api/files for status.
 */
export async function POST(req: NextRequest) {
  try {
    const { id } = await req.json();

    if (!id) {
      return NextResponse.json({ error: 'Missing id' }, { status: 400 });
    }

    const record = await db.nFRFile.findUnique({ where: { id } });
    if (!record) {
      return NextResponse.json({ error: 'Not found' }, { status: 404 });
    }
    if (!record.outputPath || !existsSync(record.outputPath)) {
      return NextResponse.json({ error: 'Compressed file missing' }, { status: 404 });
    }

    const decompressedPath = record.outputPath + '.out';

    // Spawn Python decompress in background
    const child = spawn(PYTHON_BIN, [ENGINE_PATH, 'decompress', record.outputPath, decompressedPath], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    child.on('close', async (code) => {
      try {
        if (code === 0 && existsSync(decompressedPath)) {
          await db.nFRFile.update({
            where: { id },
            data: { decompressedPath },
          });
        }
      } catch (e) {
        console.error('Decompress finalize error:', e);
      }
    });

    return NextResponse.json({ id, status: 'decompressing' });
  } catch (e: any) {
    console.error('Decompress error:', e);
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
