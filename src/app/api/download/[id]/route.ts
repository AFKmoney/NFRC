import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { existsSync } from 'fs';
import path from 'path';

export const runtime = 'nodejs';

/**
 * Download a file by NFRFile id and kind.
 * Query params: kind = compressed | decompressed | original
 */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const kind = req.nextUrl.searchParams.get('kind') || 'compressed';

  const record = await db.nFRFile.findUnique({ where: { id } });
  if (!record) {
    return NextResponse.json({ error: 'Not found' }, { status: 404 });
  }

  let filePath: string | null = null;
  let downloadName: string | null = null;

  if (kind === 'compressed') {
    filePath = record.outputPath;
    if (filePath) {
      const ext = path.extname(filePath);
      downloadName = record.originalName + ext;
    }
  } else if (kind === 'decompressed') {
    filePath = record.decompressedPath;
    downloadName = record.originalName;
  } else if (kind === 'original') {
    filePath = record.inputPath;
    downloadName = record.originalName;
  }

  if (!filePath || !existsSync(filePath)) {
    return NextResponse.json({ error: 'File not on disk' }, { status: 404 });
  }

  const { statSync, createReadStream } = await import('fs');
  const stats = statSync(filePath);

  // Use streaming response for large files
  const stream = createReadStream(filePath);
  // @ts-ignore - ReadStream is a ReadableStream-like
  const webStream = stream; // Node Web Stream compat in Next 16

  return new Response(webStream as any, {
    headers: {
      'Content-Type': 'application/octet-stream',
      'Content-Length': stats.size.toString(),
      'Content-Disposition': `attachment; filename="${encodeURIComponent(downloadName || 'download')}"`,
    },
  });
}
