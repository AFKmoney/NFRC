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
 * Starts a compression job in the background.
 * The job streams progress events to a log file that the /api/status/[id]
 * SSE endpoint reads from.
 *
 * Query params:
 *   id   - NFRFile id (required)
 *   mode - force mode: auto | video | image | binary (optional, default: auto)
 */
export async function POST(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const id = searchParams.get('id');
    const forceMode = searchParams.get('mode') || 'auto';

    if (!id) {
      return NextResponse.json({ error: 'Missing id' }, { status: 400 });
    }

    const record = await db.nFRFile.findUnique({ where: { id } });
    if (!record) {
      return NextResponse.json({ error: 'Record not found' }, { status: 404 });
    }
    if (!existsSync(record.inputPath)) {
      return NextResponse.json({ error: 'Input file missing' }, { status: 404 });
    }

    // Determine output path
    const ext = record.detectedType === 'video' || record.detectedType === 'image' ? '.nf6' : '.nfg';
    const outputPath = record.inputPath + ext;

    await db.nFRFile.update({
      where: { id },
      data: { status: 'compressing', outputPath, errorMessage: null },
    });

    // Build CLI args
    const args = [ENGINE_PATH, 'compress', record.inputPath, outputPath, '--json'];
    if (forceMode === 'binary') args.push('--mode-bin');
    else if (forceMode === 'video' || forceMode === 'image') args.push('--mode-media');

    // Spawn Python subprocess in background
    const child = spawn(PYTHON_BIN, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, NFR_JSON: '1' },
    });

    // Pipe stdout (JSON events) to a log file that the SSE endpoint reads
    const logPath = record.inputPath + '.events.log';
    const { createWriteStream } = await import('fs');
    const logStream = createWriteStream(logPath, { flags: 'w' });

    child.stdout.on('data', (chunk: Buffer) => {
      logStream.write(chunk);
    });
    child.stderr.on('data', (chunk: Buffer) => {
      // stderr is informational; we ignore for now
    });

    child.on('close', async (code) => {
      logStream.end();
      try {
        if (code === 0) {
          const { statSync } = await import('fs');
          const stats = statSync(outputPath);
          await db.nFRFile.update({
            where: { id },
            data: {
              status: 'done',
              compressedSize: stats.size,
              ratio: record.originalSize / stats.size,
              outputPath,
            },
          });
          // Write final "done" marker if engine didn't emit one
          const { appendFileSync } = await import('fs');
          appendFileSync(logPath, JSON.stringify({
            phase: 'final',
            status: 'done',
            compressed_size: stats.size,
            ratio: record.originalSize / stats.size,
          }) + '\n');
        } else {
          await db.nFRFile.update({
            where: { id },
            data: { status: 'error', errorMessage: `Process exited with code ${code}` },
          });
          const { appendFileSync } = await import('fs');
          appendFileSync(logPath, JSON.stringify({
            phase: 'final',
            status: 'error',
            message: `Process exited with code ${code}`,
          }) + '\n');
        }
      } catch (e) {
        console.error('Failed to finalize record:', e);
      }
    });

    child.on('error', async (err) => {
      logStream.end();
      try {
        await db.nFRFile.update({
          where: { id },
          data: { status: 'error', errorMessage: err.message },
        });
      } catch {}
    });

    return NextResponse.json({
      id,
      status: 'compressing',
      message: 'Compression started',
      eventsLog: logPath,
    });
  } catch (e: any) {
    console.error('Compress error:', e);
    return NextResponse.json({ error: e.message || 'Internal error' }, { status: 500 });
  }
}
