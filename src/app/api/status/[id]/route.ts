import { NextRequest } from 'next/server';
import { db } from '@/lib/db';
import { createReadStream, existsSync, statSync } from 'fs';
import path from 'path';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/**
 * SSE endpoint that streams compression progress events for a given job id.
 * Reads from the events log file written by the /api/compress route.
 *
 * Usage: GET /api/status/nfr_xxx
 * Returns: text/event-stream of JSON events
 */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  const record = await db.nFRFile.findUnique({ where: { id } });
  if (!record) {
    return new Response('Not found', { status: 404 });
  }

  const logPath = record.inputPath + '.events.log';

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      let sentBytes = 0;
      let finished = false;

      const send = (obj: any) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
      };

      // Send initial state
      send({
        phase: 'init',
        id,
        status: record.status,
        originalSize: record.originalSize,
        originalName: record.originalName,
        detectedType: record.detectedType,
        predictedRatio: record.predictedRatio,
      });

      // Poll the events log file, sending new lines as they appear
      const pollInterval = setInterval(async () => {
        try {
          if (finished) return;
          if (!existsSync(logPath)) return;

          const stats = statSync(logPath);
          if (stats.size <= sentBytes) {
            // Check if process is done (DB status)
            const fresh = await db.nFRFile.findUnique({ where: { id } });
            if (fresh && (fresh.status === 'done' || fresh.status === 'error')) {
              send({
                phase: 'final',
                status: fresh.status,
                originalSize: fresh.originalSize,
                compressedSize: fresh.compressedSize,
                ratio: fresh.ratio,
                errorMessage: fresh.errorMessage,
              });
              finished = true;
              clearInterval(pollInterval);
              controller.close();
            }
            return;
          }

          // Read new bytes
          const { open } = await import('fs/promises');
          const handle = await open(logPath, 'r');
          const buf = Buffer.alloc(stats.size - sentBytes);
          await handle.read(buf, 0, buf.length, sentBytes);
          await handle.close();

          const lines = buf.toString('utf-8').split('\n').filter(Boolean);
          for (const line of lines) {
            try {
              const evt = JSON.parse(line);
              send(evt);
              if (evt.phase === 'done' || evt.phase === 'error' || evt.phase === 'final') {
                finished = true;
                clearInterval(pollInterval);
                controller.close();
                return;
              }
            } catch {
              // skip malformed lines
            }
          }
          sentBytes = stats.size;
        } catch (e) {
          console.error('SSE poll error:', e);
        }
      }, 200);

      // Cleanup on abort
      req.signal.addEventListener('abort', () => {
        clearInterval(pollInterval);
        try { controller.close(); } catch {}
      });

      // Safety timeout: 10 minutes
      setTimeout(() => {
        if (!finished) {
          clearInterval(pollInterval);
          try {
            send({ phase: 'final', status: 'timeout', message: 'Stream timed out' });
            controller.close();
          } catch {}
        }
      }, 10 * 60 * 1000);
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  });
}
