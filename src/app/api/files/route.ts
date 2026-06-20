import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';

export const runtime = 'nodejs';

/**
 * List all NFR files (newest first).
 */
export async function GET() {
  try {
    const files = await db.nFRFile.findMany({
      orderBy: { createdAt: 'desc' },
      take: 200,
    });
    return NextResponse.json({ files });
  } catch (e: any) {
    console.error('List files error:', e);
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}

/**
 * Delete an NFR file record (does not delete files on disk).
 */
export async function DELETE(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const id = searchParams.get('id');
    if (!id) {
      return NextResponse.json({ error: 'Missing id' }, { status: 400 });
    }
    await db.nFRFile.delete({ where: { id } });
    return NextResponse.json({ ok: true });
  } catch (e: any) {
    console.error('Delete file error:', e);
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
