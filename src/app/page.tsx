'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, FileArchive, Zap, Download, Trash2, RefreshCw,
  Activity, TrendingDown, Sparkles, Clock, CheckCircle2,
  AlertCircle, FileType2, Image as ImageIcon, Video, FileText,
  ChevronRight, Gauge, Layers
} from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { toast } from 'sonner';

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------
type FileKind = 'video' | 'image' | 'binary' | 'unknown';

interface Prediction {
  id: string;
  originalName: string;
  originalSize: number;
  detectedType: string;
  predictedRatio: number;
  predictedCompressedSize: number;
  confidence: string;
  entropy?: number;
  width?: number;
  height?: number;
  frames?: number;
}

interface NFRFile {
  id: string;
  originalName: string;
  originalSize: number;
  compressedSize: number;
  ratio: number;
  mode: string;
  detectedType: string;
  status: string;
  predictedRatio: number;
  errorMessage: string | null;
  decompressedPath: string | null;
  createdAt: string;
}

interface CompressEvent {
  phase: string;
  progress?: number;
  current_ratio?: number;
  throughput_mbs?: number;
  loss?: number;
  input_size?: number;
  output_size?: number;
  ratio?: number;
  time_s?: number;
  message?: string;
  status?: string;
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
function fmtBytes(n: number): string {
  if (n === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(n) / Math.log(1024));
  return `${(n / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

function fmtTime(s: number): string {
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const r = (s % 60).toFixed(0);
  return `${m}m ${r}s`;
}

function typeIcon(type: string) {
  switch (type) {
    case 'video': return <Video className="w-4 h-4" />;
    case 'image': return <ImageIcon className="w-4 h-4" />;
    case 'binary': return <FileText className="w-4 h-4" />;
    default: return <FileType2 className="w-4 h-4" />;
  }
}

function typeColor(type: string): string {
  switch (type) {
    case 'video': return 'text-rose-300 bg-rose-500/10 border-rose-500/20';
    case 'image': return 'text-emerald-300 bg-emerald-500/10 border-emerald-500/20';
    case 'binary': return 'text-amber-300 bg-amber-500/10 border-amber-500/20';
    default: return 'text-slate-300 bg-slate-500/10 border-slate-500/20';
  }
}

// ----------------------------------------------------------------------------
// Main Page
// ----------------------------------------------------------------------------
export default function Home() {
  const [dragOver, setDragOver] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [compressing, setCompressing] = useState(false);
  const [events, setEvents] = useState<CompressEvent[]>([]);
  const [finalRatio, setFinalRatio] = useState<number | null>(null);
  const [files, setFiles] = useState<NFRFile[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);

  // ----- File list refresh
  const refreshFiles = useCallback(async () => {
    try {
      const res = await fetch('/api/files');
      const data = await res.json();
      setFiles(data.files || []);
    } catch (e) {
      console.error('Failed to refresh files:', e);
    }
  }, []);

  useEffect(() => {
    refreshFiles();
    const t = setInterval(refreshFiles, 3000);
    return () => clearInterval(t);
  }, [refreshFiles]);

  // ----- Drag & drop handlers
  const handleFile = useCallback(async (file: File) => {
    setPredicting(true);
    setPrediction(null);
    setEvents([]);
    setFinalRatio(null);

    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch('/api/predict', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
      const data = await res.json();
      setPrediction(data);
      toast.success('File analyzed', {
        description: `${data.detectedType} · predicted ${data.predictedRatio}x compression`,
      });
    } catch (e: any) {
      toast.error('Analysis failed', { description: e.message });
    } finally {
      setPredicting(false);
    }
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const onFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  // ----- Compress with SSE
  const startCompress = useCallback(async () => {
    if (!prediction) return;
    setCompressing(true);
    setEvents([]);
    setFinalRatio(null);

    try {
      // Start job
      const startRes = await fetch(`/api/compress?id=${prediction.id}&mode=auto`, {
        method: 'POST',
      });
      if (!startRes.ok) throw new Error(`Start failed: ${startRes.status}`);

      // Open SSE stream
      const evtSource = new EventSource(`/api/status/${prediction.id}`);
      evtSource.onmessage = (msg) => {
        try {
          const evt: CompressEvent = JSON.parse(msg.data);
          setEvents((prev) => [...prev, evt]);

          if (evt.phase === 'final' || evt.phase === 'done' || evt.phase === 'error') {
            evtSource.close();
            setCompressing(false);
            if (evt.ratio) setFinalRatio(evt.ratio);
            if (evt.status === 'done') {
              toast.success('Compression complete', {
                description: `Final ratio: ${(evt.ratio || 0).toFixed(2)}x`,
              });
            } else if (evt.status === 'error') {
              toast.error('Compression failed', { description: evt.message });
            }
            refreshFiles();
          }
        } catch {}
      };
      evtSource.onerror = () => {
        evtSource.close();
        setCompressing(false);
      };
    } catch (e: any) {
      setCompressing(false);
      toast.error('Compression failed', { description: e.message });
    }
  }, [prediction, refreshFiles]);

  // ----- Current event for animation
  const currentEvt = events.length > 0 ? events[events.length - 1] : null;
  const phase = currentEvt?.phase || 'idle';
  const progress = currentEvt?.progress || 0;
  const liveRatio = currentEvt?.current_ratio || 0;
  const throughput = currentEvt?.throughput_mbs || 0;
  const inputSize = currentEvt?.input_size || prediction?.originalSize || 0;
  const outputSize = currentEvt?.output_size || 0;

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-slate-100 relative overflow-x-hidden">
      {/* Animated background */}
      <div className="pointer-events-none fixed inset-0 z-0">
        <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-violet-600/20 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-emerald-600/15 rounded-full blur-[120px] animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] bg-rose-600/10 rounded-full blur-[100px] animate-pulse" style={{ animationDelay: '2s' }} />
      </div>

      {/* Grid overlay */}
      <div className="pointer-events-none fixed inset-0 z-0 opacity-[0.03]"
           style={{
             backgroundImage: 'linear-gradient(rgba(255,255,255,1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,1) 1px, transparent 1px)',
             backgroundSize: '40px 40px',
           }} />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        {/* Header */}
        <header className="mb-10 sm:mb-14">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
          >
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-br from-violet-500 to-emerald-500 rounded-xl blur-md opacity-60" />
                <div className="relative w-12 h-12 bg-gradient-to-br from-violet-500 to-emerald-500 rounded-xl flex items-center justify-center">
                  <Sparkles className="w-6 h-6 text-white" />
                </div>
              </div>
              <div>
                <h1 className="text-2xl sm:text-3xl font-bold tracking-tight bg-gradient-to-r from-white via-violet-200 to-emerald-200 bg-clip-text text-transparent">
                  NFR Studio
                </h1>
                <p className="text-sm text-slate-400">
                  Neural Fractal Reconstruction v6 · Universal Codec
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="border-violet-500/30 bg-violet-500/10 text-violet-300">
                <Layers className="w-3 h-3 mr-1" /> v6.0
              </Badge>
              <Badge variant="outline" className="border-emerald-500/30 bg-emerald-500/10 text-emerald-300">
                <Zap className="w-3 h-3 mr-1" /> NanoSiren v2
              </Badge>
            </div>
          </motion.div>
        </header>

        {/* Main grid */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* LEFT: Dropzone + compression animation (3 cols) */}
          <div className="lg:col-span-3 space-y-6">
            {/* Dropzone */}
            <motion.div
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.1 }}
            >
              <Card
                ref={dropZoneRef as any}
                onDragOver={(e: any) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                className={`relative overflow-hidden border-2 border-dashed transition-all duration-300 cursor-pointer ${
                  dragOver
                    ? 'border-violet-400 bg-violet-500/10 scale-[1.01]'
                    : 'border-slate-700 bg-slate-900/40 hover:border-slate-600 hover:bg-slate-900/60'
                } backdrop-blur-xl`}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  onChange={onFileInputChange}
                />
                <div className="p-8 sm:p-12 text-center">
                  <motion.div
                    animate={dragOver ? { scale: 1.15, rotate: 5 } : { scale: 1, rotate: 0 }}
                    transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                    className="mx-auto mb-4 w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500/20 to-emerald-500/20 flex items-center justify-center"
                  >
                    {predicting ? (
                      <RefreshCw className="w-7 h-7 text-violet-300 animate-spin" />
                    ) : (
                      <Upload className="w-7 h-7 text-violet-300" />
                    )}
                  </motion.div>
                  <h2 className="text-lg sm:text-xl font-semibold mb-1">
                    {predicting ? 'Analyzing...' : 'Drop a file to compress'}
                  </h2>
                  <p className="text-sm text-slate-400">
                    {predicting
                      ? 'Reading entropy, predicting ratio'
                      : 'Videos, images, or any binary file · instant prediction'}
                  </p>
                </div>

                {/* Drag overlay */}
                <AnimatePresence>
                  {dragOver && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute inset-0 bg-violet-500/20 backdrop-blur-sm flex items-center justify-center pointer-events-none"
                    >
                      <motion.div
                        animate={{ y: [0, -10, 0] }}
                        transition={{ repeat: Infinity, duration: 1.5 }}
                      >
                        <Upload className="w-12 h-12 text-violet-200" />
                      </motion.div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>
            </motion.div>

            {/* Prediction card */}
            <AnimatePresence>
              {prediction && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <Card className="p-6 bg-slate-900/60 backdrop-blur-xl border-slate-700">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <p className="text-xs uppercase tracking-wider text-slate-500 mb-1">File</p>
                        <p className="font-semibold text-sm truncate max-w-[280px] sm:max-w-md">
                          {prediction.originalName}
                        </p>
                        <div className="flex items-center gap-2 mt-2">
                          <Badge variant="outline" className={typeColor(prediction.detectedType)}>
                            {typeIcon(prediction.detectedType)}
                            <span className="ml-1 capitalize">{prediction.detectedType}</span>
                          </Badge>
                          <Badge variant="outline" className="text-slate-300 border-slate-600">
                            {fmtBytes(prediction.originalSize)}
                          </Badge>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-xs uppercase tracking-wider text-slate-500 mb-1">
                          Predicted
                        </p>
                        <p className="text-3xl font-bold bg-gradient-to-r from-violet-300 to-emerald-300 bg-clip-text text-transparent">
                          {prediction.predictedRatio.toFixed(2)}x
                        </p>
                        <p className="text-xs text-slate-400 mt-1">
                          → {fmtBytes(prediction.predictedCompressedSize)}
                        </p>
                      </div>
                    </div>

                    {/* Extra metadata */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4 text-xs">
                      {prediction.entropy !== undefined && (
                        <Stat label="Entropy" value={`${prediction.entropy.toFixed(2)} b/B`} />
                      )}
                      {prediction.width && (
                        <Stat label="Dimensions" value={`${prediction.width}×${prediction.height}`} />
                      )}
                      {prediction.frames && (
                        <Stat label="Frames" value={prediction.frames.toString()} />
                      )}
                      <Stat label="Confidence" value={prediction.confidence || 'medium'} />
                    </div>

                    <Button
                      onClick={startCompress}
                      disabled={compressing}
                      className="w-full bg-gradient-to-r from-violet-500 to-emerald-500 hover:from-violet-600 hover:to-emerald-600 text-white font-medium"
                      size="lg"
                    >
                      {compressing ? (
                        <>
                          <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                          Compressing...
                        </>
                      ) : (
                        <>
                          <Zap className="w-4 h-4 mr-2" />
                          Compress with NFR v6
                        </>
                      )}
                    </Button>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Live compression animation */}
            <AnimatePresence>
              {(compressing || finalRatio !== null) && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <Card className="p-6 bg-slate-900/60 backdrop-blur-xl border-slate-700 relative overflow-hidden">
                    {/* Animated bg */}
                    {compressing && (
                      <div className="absolute inset-0 pointer-events-none">
                        <motion.div
                          className="absolute inset-0 bg-gradient-to-r from-violet-500/10 via-transparent to-emerald-500/10"
                          animate={{ x: ['-100%', '100%'] }}
                          transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
                        />
                      </div>
                    )}

                    <div className="relative z-10">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                          <Activity className={`w-4 h-4 ${compressing ? 'text-violet-400 animate-pulse' : 'text-emerald-400'}`} />
                          <span className="text-sm font-medium">
                            {phase === 'train' && 'Training NanoSiren v2'}
                            {phase === 'scan' && 'Scanning entropy'}
                            {phase === 'encode' && 'Encoding stream'}
                            {phase === 'decode' && 'Decoding'}
                            {phase === 'reconstruct' && 'Reconstructing frames'}
                            {phase === 'done' && 'Complete'}
                            {phase === 'final' && 'Complete'}
                            {phase === 'idle' && 'Idle'}
                          </span>
                        </div>
                        {throughput > 0 && (
                          <Badge variant="outline" className="text-slate-300 border-slate-600">
                            <Gauge className="w-3 h-3 mr-1" />
                            {throughput.toFixed(1)} MB/s
                          </Badge>
                        )}
                      </div>

                      {/* Progress bar */}
                      <div className="space-y-2 mb-4">
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>Progress</span>
                          <span>{(progress * 100).toFixed(1)}%</span>
                        </div>
                        <div className="relative h-2 bg-slate-800 rounded-full overflow-hidden">
                          <motion.div
                            className="absolute inset-y-0 left-0 bg-gradient-to-r from-violet-500 to-emerald-500 rounded-full"
                            animate={{ width: `${progress * 100}%` }}
                            transition={{ duration: 0.3 }}
                          />
                          {compressing && (
                            <motion.div
                              className="absolute inset-y-0 w-20 bg-gradient-to-r from-transparent via-white/40 to-transparent"
                              animate={{ x: ['-80px', '120%'] }}
                              transition={{ repeat: Infinity, duration: 1.2, ease: 'linear' }}
                            />
                          )}
                        </div>
                      </div>

                      {/* Live ratio animation */}
                      <div className="grid grid-cols-3 gap-3">
                        <RatioStat
                          label="Original"
                          value={fmtBytes(inputSize)}
                          icon={<FileArchive className="w-3 h-3" />}
                        />
                        <RatioStat
                          label="Compressed"
                          value={outputSize > 0 ? fmtBytes(outputSize) : '—'}
                          icon={<TrendingDown className="w-3 h-3" />}
                          highlight={!!outputSize}
                        />
                        <RatioStat
                          label={finalRatio !== null ? 'Final ratio' : 'Live ratio'}
                          value={finalRatio !== null
                            ? `${finalRatio.toFixed(2)}x`
                            : liveRatio > 0 ? `${liveRatio.toFixed(2)}x` : '—'}
                          icon={<Zap className="w-3 h-3" />}
                          highlight={!!finalRatio || liveRatio > 0}
                        />
                      </div>

                      {/* Event log */}
                      {events.length > 1 && (
                        <div className="mt-4 pt-4 border-t border-slate-800">
                          <p className="text-xs uppercase tracking-wider text-slate-500 mb-2">
                            Event stream
                          </p>
                          <ScrollArea className="h-24">
                            <div className="space-y-1 font-mono text-xs text-slate-400">
                              {events.slice(-12).map((e, i) => (
                                <div key={i} className="flex gap-2">
                                  <span className="text-slate-600">›</span>
                                  <span>{e.phase}</span>
                                  {e.progress !== undefined && (
                                    <span className="text-slate-500">
                                      {(e.progress * 100).toFixed(0)}%
                                    </span>
                                  )}
                                  {e.current_ratio !== undefined && e.current_ratio > 0 && (
                                    <span className="text-emerald-400">
                                      {e.current_ratio.toFixed(2)}x
                                    </span>
                                  )}
                                  {e.loss !== undefined && (
                                    <span className="text-violet-400">
                                      loss={e.loss.toFixed(4)}
                                    </span>
                                  )}
                                  {e.time_s !== undefined && (
                                    <span className="text-slate-600">
                                      {fmtTime(e.time_s)}
                                    </span>
                                  )}
                                </div>
                              ))}
                            </div>
                          </ScrollArea>
                        </div>
                      )}
                    </div>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* RIGHT: File manager (2 cols) */}
          <div className="lg:col-span-2">
            <Card className="bg-slate-900/60 backdrop-blur-xl border-slate-700 h-full">
              <div className="p-4 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileArchive className="w-4 h-4 text-violet-300" />
                  <h3 className="text-sm font-semibold">Compression History</h3>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={refreshFiles}
                  className="h-7 w-7 p-0 text-slate-400 hover:text-slate-100"
                >
                  <RefreshCw className="w-3 h-3" />
                </Button>
              </div>

              <ScrollArea className="h-[calc(100vh-280px)] min-h-[400px]">
                {files.length === 0 ? (
                  <div className="p-8 text-center text-slate-500 text-sm">
                    No files yet. Drop a file on the left to start.
                  </div>
                ) : (
                  <div className="p-2 space-y-1">
                    <AnimatePresence>
                      {files.map((f) => (
                        <FileRow key={f.id} file={f} onDeleted={refreshFiles} />
                      ))}
                    </AnimatePresence>
                  </div>
                )}
              </ScrollArea>
            </Card>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 pt-6 border-t border-slate-800 text-xs text-slate-500 text-center">
          <p>
            NFR v6.0 · NanoSiren v2 + Context-Adaptive Arithmetic Coding + LZ77
          </p>
          <p className="mt-1">
            Proprietary codec · Bit-perfect reconstruction · GPU-accelerated
          </p>
        </footer>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------
// Sub-components
// ----------------------------------------------------------------------------
function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-2 rounded-md bg-slate-800/50 border border-slate-700/50">
      <p className="text-slate-500 mb-0.5">{label}</p>
      <p className="font-medium text-slate-200 truncate">{value}</p>
    </div>
  );
}

function RatioStat({
  label, value, icon, highlight,
}: { label: string; value: string; icon: React.ReactNode; highlight?: boolean }) {
  return (
    <div className={`p-3 rounded-lg border transition-all ${
      highlight
        ? 'bg-emerald-500/10 border-emerald-500/30'
        : 'bg-slate-800/50 border-slate-700/50'
    }`}>
      <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
        {icon}
        <span>{label}</span>
      </div>
      <p className={`text-lg font-bold ${highlight ? 'text-emerald-300' : 'text-slate-300'}`}>
        {value}
      </p>
    </div>
  );
}

function FileRow({ file, onDeleted }: { file: NFRFile; onDeleted: () => void }) {
  const [expanded, setExpanded] = useState(false);

  const handleDelete = async () => {
    try {
      // Delete DB record (will leave file on disk for simplicity)
      await fetch(`/api/files?id=${file.id}`, { method: 'DELETE' });
      onDeleted();
      toast.success('Removed from history');
    } catch (e) {
      toast.error('Failed to delete');
    }
  };

  const handleDecompress = async () => {
    try {
      const res = await fetch('/api/decompress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: file.id }),
      });
      if (!res.ok) throw new Error('Failed');
      toast.success('Decompression started', {
        description: 'Will appear in history when ready',
      });
    } catch (e) {
      toast.error('Decompression failed');
    }
  };

  const isDone = file.status === 'done';
  const isError = file.status === 'error';

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="rounded-lg border border-slate-800 bg-slate-900/40 hover:bg-slate-900/60 transition-colors overflow-hidden"
    >
      <div
        className="p-3 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <div className={`shrink-0 w-8 h-8 rounded-md flex items-center justify-center ${typeColor(file.detectedType)}`}>
              {typeIcon(file.detectedType)}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium truncate">{file.originalName}</p>
              <p className="text-xs text-slate-500">
                {fmtBytes(file.originalSize)}
                {isDone && ` → ${fmtBytes(file.compressedSize)}`}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isDone && (
              <Badge variant="outline" className="border-emerald-500/30 bg-emerald-500/10 text-emerald-300">
                {file.ratio.toFixed(2)}x
              </Badge>
            )}
            {file.status === 'compressing' && (
              <RefreshCw className="w-3 h-3 text-violet-400 animate-spin" />
            )}
            {isError && (
              <AlertCircle className="w-3 h-3 text-rose-400" />
            )}
            {isDone && (
              <CheckCircle2 className="w-3 h-3 text-emerald-400" />
            )}
            <ChevronRight className={`w-3 h-3 text-slate-500 transition-transform ${expanded ? 'rotate-90' : ''}`} />
          </div>
        </div>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-slate-800 overflow-hidden"
          >
            <div className="p-3 space-y-2 text-xs">
              <div className="grid grid-cols-2 gap-2">
                <Detail label="Type" value={file.detectedType} />
                <Detail label="Mode" value={file.mode} />
                <Detail label="Original" value={fmtBytes(file.originalSize)} />
                <Detail label="Compressed" value={isDone ? fmtBytes(file.compressedSize) : '—'} />
                <Detail label="Predicted" value={`${file.predictedRatio.toFixed(2)}x`} />
                <Detail label="Final ratio" value={isDone ? `${file.ratio.toFixed(2)}x` : '—'} />
              </div>

              {file.errorMessage && (
                <div className="p-2 rounded bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
                  {file.errorMessage}
                </div>
              )}

              <div className="flex flex-wrap gap-2 pt-1">
                {isDone && (
                  <>
                    <Button asChild size="sm" variant="outline"
                      className="h-7 text-xs border-slate-700 hover:bg-slate-800">
                      <a href={`/api/download/${file.id}?kind=compressed`} download>
                        <Download className="w-3 h-3 mr-1" />
                        .nfr
                      </a>
                    </Button>
                    <Button size="sm" variant="outline"
                      onClick={handleDecompress}
                      className="h-7 text-xs border-slate-700 hover:bg-slate-800">
                      <RefreshCw className="w-3 h-3 mr-1" />
                      Decompress
                    </Button>
                    {file.decompressedPath && (
                      <Button asChild size="sm" variant="outline"
                        className="h-7 text-xs border-emerald-700 hover:bg-emerald-900/30 text-emerald-300">
                        <a href={`/api/download/${file.id}?kind=decompressed`} download>
                          <Download className="w-3 h-3 mr-1" />
                          Decompressed
                        </a>
                      </Button>
                    )}
                  </>
                )}
                <Button asChild size="sm" variant="ghost"
                  className="h-7 text-xs text-slate-400 hover:text-slate-100">
                  <a href={`/api/download/${file.id}?kind=original`} download>
                    <Download className="w-3 h-3 mr-1" />
                    Original
                  </a>
                </Button>
                <Button size="sm" variant="ghost"
                  onClick={handleDelete}
                  className="h-7 text-xs text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 ml-auto">
                  <Trash2 className="w-3 h-3 mr-1" />
                  Remove
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-500">{label}</span>
      <span className="text-slate-200 font-medium capitalize">{value}</span>
    </div>
  );
}
