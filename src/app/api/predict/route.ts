import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';

export const runtime = 'nodejs';
export const maxDuration = 300;

const UPLOAD_DIR = path.join(process.cwd(), 'upload');
const ENGINE_PATH = path.join(process.cwd(), 'NFR-Compressor', 'nfr_v6_engine.py');
const PYTHON_BIN = process.env.NFR_PYTHON || '/home/z/.venv/bin/python';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File | null;
    const mode = (formData.get('mode') as string) || 'auto';

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    // Save upload to disk
    if (!existsSync(UPLOAD_DIR)) mkdirSync(UPLOAD_DIR, { recursive: true });

    const id = `nfr_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
    const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, '_');
    const inputPath = path.join(UPLOAD_DIR, `${id}__${safeName}`);

    const bytes = await file.arrayBuffer();
    writeFileSync(inputPath, Buffer.from(bytes));

    // Run predict subcommand to get instant ratio prediction
    let prediction: any = {
      type: 'unknown',
      original_size: file.size,
      predicted_ratio: 1.0,
      predicted_compressed_size: file.size,
      confidence: 'low',
    };
    try {
      const result = spawnSync(PYTHON_BIN, [ENGINE_PATH, 'predict', inputPath], {
        encoding: 'utf-8',
        timeout: 30000,
      });
      if (result.status === 0 && result.stdout.trim()) {
        prediction = JSON.parse(result.stdout.trim());
      }
    } catch (e) {
      console.error('Prediction failed:', e);
    }

    // Create DB record
    const record = await db.nFRFile.create({
      data: {
        id,
        originalName: file.name,
        originalSize: file.size,
        mode,
        detectedType: prediction.type || 'unknown',
        predictedRatio: prediction.predicted_ratio || 1.0,
        status: 'pending',
        inputPath,
      },
    });

    return NextResponse.json({
      id: record.id,
      originalName: record.originalName,
      originalSize: record.originalSize,
      detectedType: prediction.type,
      predictedRatio: prediction.predicted_ratio,
      predictedCompressedSize: prediction.predicted_compressed_size,
      confidence: prediction.confidence,
      entropy: prediction.entropy,
      width: prediction.width,
      height: prediction.height,
      frames: prediction.frames,
    });
  } catch (e: any) {
    console.error('Predict error:', e);
    return NextResponse.json({ error: e.message || 'Internal error' }, { status: 500 });
  }
}
