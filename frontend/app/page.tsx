'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Eraser, ScanSearch, PenLine, Upload, FolderOpen, RotateCcw,
  AlertTriangle, Loader2,
} from 'lucide-react';
import { GLSLHills } from '@/components/ui/glsl-hills';

// ── Types ─────────────────────────────────────────────────────────────────────

interface Prediction {
  label_id: number;
  predicted_character: string;
  confidence: number;
}

type Tab = 'draw' | 'upload';
type ResultState = 'idle' | 'loading' | 'content' | 'error';

// ── Constants ─────────────────────────────────────────────────────────────────

const API = 'http://localhost:8000';

const TAMIL_FONT: React.CSSProperties = { fontFamily: "'Mukta Malar', serif" };

// ── Component ─────────────────────────────────────────────────────────────────

export default function Home() {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [tab,          setTab]          = useState<Tab>('draw');
  const [brushSize,    setBrushSize]    = useState(12);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl,   setPreviewUrl]   = useState<string | null>(null);
  const [resultState,  setResultState]  = useState<ResultState>('idle');
  const [predictions,  setPredictions]  = useState<Prediction[]>([]);
  const [errorMsg,     setErrorMsg]     = useState('');
  const [apiOnline,    setApiOnline]    = useState<boolean | null>(null);
  const [apiLabel,     setApiLabel]     = useState('Connecting…');
  const [isDragging,   setIsDragging]   = useState(false);

  const drawingRef = useRef(false);
  const lastPosRef = useRef({ x: 0, y: 0 });

  // Fill canvas with solid black so toBlob() never produces a transparent PNG
  const fillBlack = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  useEffect(() => { fillBlack(); }, [fillBlack]);

  // ── API status ping ──────────────────────────────────────────────────────────
  const pingAPI = useCallback(async () => {
    try {
      const res  = await fetch(`${API}/`, { signal: AbortSignal.timeout(3000) });
      const data = await res.json();
      setApiOnline(true);
      setApiLabel(`Online · ${data.accuracy ?? ''}`);
    } catch {
      setApiOnline(false);
      setApiLabel('Offline');
    }
  }, []);

  useEffect(() => {
    pingAPI();
    const id = setInterval(pingAPI, 15000);
    return () => clearInterval(id);
  }, [pingAPI]);

  // ── Canvas drawing ───────────────────────────────────────────────────────────
  const getPos = (e: React.MouseEvent | React.TouchEvent, canvas: HTMLCanvasElement) => {
    const r  = canvas.getBoundingClientRect();
    const sx = canvas.width  / r.width;
    const sy = canvas.height / r.height;
    if ('touches' in e) {
      return { x: (e.touches[0].clientX - r.left) * sx, y: (e.touches[0].clientY - r.top) * sy };
    }
    return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy };
  };

  const startDraw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawingRef.current = true;
    const pos = getPos(e, canvas);
    lastPosRef.current = pos;
    const ctx = canvas.getContext('2d')!;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
  };

  const onDraw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (!drawingRef.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const pos = getPos(e, canvas);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth   = brushSize;
    ctx.lineCap     = 'round';
    ctx.lineJoin    = 'round';
    ctx.beginPath();
    ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    lastPosRef.current = pos;
  };

  const stopDraw = () => { drawingRef.current = false; };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.getContext('2d')!.clearRect(0, 0, canvas.width, canvas.height);
    fillBlack();
    setResultState('idle');
  };

  // ── Upload helpers ───────────────────────────────────────────────────────────
  const handleFile = (file: File) => {
    setUploadedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResultState('idle');
  };

  // ── Predict ──────────────────────────────────────────────────────────────────
  const predict = async () => {
    setResultState('loading');
    try {
      let blob: Blob;
      if (tab === 'draw') {
        const canvas = canvasRef.current;
        if (!canvas) throw new Error('Canvas not ready.');
        blob = await new Promise<Blob>((res, rej) =>
          canvas.toBlob((b) => (b ? res(b) : rej(new Error('Canvas is empty.'))), 'image/png')
        );
      } else {
        if (!uploadedFile) throw new Error('No file selected.');
        blob = uploadedFile;
      }

      const fd = new FormData();
      fd.append('file', blob, 'char.png');

      const res = await fetch(`${API}/predict`, { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
      }
      const data = await res.json();
      if (!data.predictions?.length) throw new Error('No predictions returned.');
      setPredictions(data.predictions);
      setResultState('content');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Unexpected error.');
      setResultState('error');
    }
  };

  // ── Helpers ──────────────────────────────────────────────────────────────────
  const top    = predictions[0];
  const topPct = top ? Math.round(top.confidence * 100) : 0;

  const confBarColor = (pct: number) =>
    pct >= 70 ? 'from-emerald-400 to-teal-400'
  : pct >= 40 ? 'from-amber-400 to-yellow-300'
  :             'from-rose-500 to-red-400';

  const confTextColor = (pct: number) =>
    pct >= 70 ? 'text-emerald-400' : pct >= 40 ? 'text-amber-400' : 'text-rose-400';

  // ── Shared class strings ─────────────────────────────────────────────────────
  const glass  = 'bg-black/40 backdrop-blur-2xl border border-white/[0.08] rounded-3xl';
  const panel  = `${glass} p-6 flex flex-col gap-5 shadow-2xl`;

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="relative min-h-screen bg-black text-white overflow-hidden">

      {/* ── GLSL Hills background ──────────────────────────────────────── */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <GLSLHills width="100%" height="100%" speed={0.35} cameraZ={135} planeSize={256} />
      </div>

      {/* ── Content ───────────────────────────────────────────────────── */}
      <div className="relative z-10 flex flex-col min-h-screen">

        {/* ── Header ────────────────────────────────────────────────────── */}
        <header className="sticky top-0 z-50 border-b border-white/[0.06] bg-black/30 backdrop-blur-xl">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">

            <div className="flex items-center gap-3">
              <div
                className="w-9 h-9 rounded-xl bg-sky-500/20 border border-sky-400/30 flex items-center justify-center text-sky-300 font-bold text-lg"
                style={TAMIL_FONT}
              >
                த
              </div>
              <span className="text-base font-semibold tracking-wide text-white">
                Tamil<span className="text-sky-400">Vision</span>
              </span>
            </div>

            <div className="flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${
                apiOnline === null ? 'bg-white/20'
              : apiOnline          ? 'bg-emerald-400'
              :                     'bg-rose-500'
              }`} />
              <span className="text-xs text-white/50">{apiLabel}</span>
            </div>
          </div>
        </header>

        {/* ── Main grid ─────────────────────────────────────────────────── */}
        <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-10 grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">

          {/* ── INPUT PANEL ──────────────────────────────────────────────── */}
          <section className={panel}>

            {/* Logo stamp above tabs */}
            <div className="flex items-center gap-3 pb-1">
              <div
                className="w-10 h-10 rounded-xl bg-sky-500/20 border border-sky-400/30 flex items-center justify-center text-sky-300 font-bold text-xl shrink-0"
                style={TAMIL_FONT}
              >
                த
              </div>
              <div>
                <p className="text-sm font-semibold text-white leading-tight">
                  Tamil<span className="text-sky-400">Vision</span>
                </p>
                <p className="text-[11px] text-white/40">Handwritten character recognition</p>
              </div>
            </div>

            {/* Segmented tab control */}
            <div className="relative flex bg-white/[0.05] rounded-2xl p-1 gap-1">
              {(['draw', 'upload'] as Tab[]).map((t) => (
                <button
                  key={t}
                  onClick={() => { setTab(t); setResultState('idle'); }}
                  className={`relative flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-xl text-sm font-medium transition-all duration-200
                    ${tab === t
                      ? 'bg-sky-500 text-white shadow-lg shadow-sky-500/20'
                      : 'text-white/40 hover:text-white/70'}`}
                >
                  {t === 'draw'
                    ? <><PenLine size={14} /><span>Draw</span></>
                    : <><Upload size={14} /><span>Upload</span></>}
                </button>
              ))}
            </div>

            {/* ── Draw mode ── */}
            {tab === 'draw' && (
              <div className="flex flex-col gap-4">

                {/* Canvas */}
                <div className="relative rounded-2xl overflow-hidden ring-1 ring-sky-400/30 shadow-[0_0_18px_2px_rgba(56,189,248,0.12)] shadow-inner">
                  <canvas
                    ref={canvasRef}
                    width={400}
                    height={400}
                    className="w-full aspect-square block cursor-crosshair touch-none bg-black"
                    onMouseDown={startDraw}
                    onMouseMove={onDraw}
                    onMouseUp={stopDraw}
                    onMouseLeave={stopDraw}
                    onTouchStart={startDraw}
                    onTouchMove={onDraw}
                    onTouchEnd={stopDraw}
                  />
                  <div className="absolute bottom-2 right-3 text-[10px] text-white/40 select-none pointer-events-none">
                    {brushSize}px
                  </div>
                </div>

                {/* Brush slider */}
                <div className="flex items-center gap-3">
                  <span className="text-xs text-white/50 w-8">Thin</span>
                  <input
                    type="range" min={4} max={40} value={brushSize}
                    onChange={(e) => setBrushSize(Number(e.target.value))}
                    className="flex-1 h-1 accent-sky-400 cursor-pointer"
                  />
                  <span className="text-xs text-white/50 w-8 text-right">Thick</span>
                </div>

                {/* Action buttons */}
                <div className="flex gap-2">
                  <button
                    onClick={clearCanvas}
                    className="flex items-center justify-center gap-2 flex-1 py-2.5 rounded-2xl border border-white/15 text-white/70 hover:text-white hover:bg-white/[0.08] text-sm font-medium transition-all duration-200"
                  >
                    <Eraser size={15} />
                    Clear
                  </button>
                  <button
                    onClick={predict}
                    className="flex items-center justify-center gap-2 flex-[2] py-2.5 rounded-2xl bg-sky-500 hover:bg-sky-400 text-white text-sm font-semibold transition-all duration-200 shadow-lg shadow-sky-500/25"
                  >
                    <ScanSearch size={15} />
                    Identify Character
                  </button>
                </div>
              </div>
            )}

            {/* ── Upload mode ── */}
            {tab === 'upload' && (
              <div className="flex flex-col gap-4">

                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault(); setIsDragging(false);
                    const f = e.dataTransfer.files[0];
                    if (f?.type.startsWith('image/')) handleFile(f);
                  }}
                  className={`flex flex-col items-center justify-center gap-3 rounded-2xl p-10 cursor-pointer border transition-all duration-200
                    ${isDragging
                      ? 'border-sky-400/60 bg-sky-500/10'
                      : 'border-dashed border-white/15 hover:border-white/30 hover:bg-white/[0.03]'}`}
                >
                  <div className="w-12 h-12 rounded-2xl bg-white/[0.05] border border-white/10 flex items-center justify-center text-white/30">
                    <FolderOpen size={22} />
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-white/70 font-medium">Drop an image here</p>
                    <p className="text-xs text-white/40 mt-0.5">PNG · JPG · BMP — or click to browse</p>
                  </div>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
                />

                {previewUrl && (
                  <div className="rounded-2xl overflow-hidden ring-1 ring-white/10">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={previewUrl} alt="preview" className="w-full object-contain max-h-56 bg-black" />
                  </div>
                )}

                <button
                  onClick={predict}
                  disabled={!uploadedFile}
                  className="flex items-center justify-center gap-2 w-full py-2.5 rounded-2xl bg-sky-500 hover:bg-sky-400 text-white text-sm font-semibold transition-all duration-200 shadow-lg shadow-sky-500/25 disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none"
                >
                  <ScanSearch size={15} />
                  Identify Character
                </button>
              </div>
            )}
          </section>

          {/* ── RESULTS PANEL ────────────────────────────────────────────── */}
          <section className={panel} style={{ minHeight: '420px' }}>

            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-white/50 uppercase tracking-widest">Results</span>
              {resultState === 'content' && (
                <button
                  onClick={() => setResultState('idle')}
                  className="flex items-center gap-1.5 text-xs text-white/50 hover:text-white/80 transition-colors"
                >
                  <RotateCcw size={11} /> Try again
                </button>
              )}
            </div>

            {/* Idle */}
            {resultState === 'idle' && (
              <div className="flex-1 flex flex-col items-center justify-center gap-4 py-8">
                <div
                  className="w-20 h-20 rounded-3xl bg-white/[0.06] border border-white/10 flex items-center justify-center"
                  style={TAMIL_FONT}
                >
                  <span className="text-4xl text-white/40">த</span>
                </div>
                <p className="text-sm text-white/50 text-center">
                  Draw or upload a Tamil character<br />then click Identify
                </p>
              </div>
            )}

            {/* Loading */}
            {resultState === 'loading' && (
              <div className="flex-1 flex flex-col items-center justify-center gap-4 py-8">
                <Loader2 size={36} className="text-sky-400 animate-spin" />
                <p className="text-sm text-white/50">Analysing character…</p>
              </div>
            )}

            {/* Content */}
            {resultState === 'content' && top && (
              <div className="flex flex-col gap-4">

                {/* Top prediction card */}
                <div className="rounded-2xl bg-white/[0.04] border border-white/[0.08] p-5 flex items-center gap-5">
                  <div
                    className="w-20 h-20 rounded-2xl bg-gradient-to-br from-sky-500/30 to-sky-600/10 border border-sky-400/20 flex items-center justify-center shrink-0 shadow-lg shadow-sky-500/10"
                    style={TAMIL_FONT}
                  >
                    <span className="text-4xl font-bold text-white">{top.predicted_character}</span>
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className="text-[10px] font-medium text-white/50 uppercase tracking-widest mb-0.5">
                      Best match
                    </p>
                    <p className="text-xs text-white/35 font-mono mb-3">label {top.label_id}</p>

                    <div className="flex items-center gap-3">
                      <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
                        <div
                          className={`h-full rounded-full bg-gradient-to-r transition-all duration-700 ${confBarColor(topPct)}`}
                          style={{ width: `${topPct}%` }}
                        />
                      </div>
                      <span className={`text-sm font-bold tabular-nums ${confTextColor(topPct)}`}>
                        {topPct}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Alternatives */}
                {predictions.length > 1 && (
                  <div className="flex flex-col gap-2">
                    <p className="text-[10px] font-medium text-white/45 uppercase tracking-widest px-1">
                      Alternatives
                    </p>
                    {predictions.slice(1).map((p) => {
                      const pct = Math.round(p.confidence * 100);
                      return (
                        <div
                          key={p.label_id}
                          className="flex items-center gap-4 px-4 py-3 rounded-2xl bg-white/[0.03] border border-white/[0.06] hover:bg-white/[0.05] transition-colors"
                        >
                          <div
                            className="w-9 h-9 rounded-xl bg-white/[0.06] border border-white/10 flex items-center justify-center shrink-0"
                            style={TAMIL_FONT}
                          >
                            <span className="text-lg text-white/70">{p.predicted_character}</span>
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex justify-between text-[11px] mb-1.5">
                              <span className="text-white/50 font-mono">label {p.label_id}</span>
                              <span className="text-white/55 tabular-nums">{pct}%</span>
                            </div>
                            <div className="h-1 rounded-full bg-white/10 overflow-hidden">
                              <div
                                className="h-full rounded-full bg-white/25"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* Error */}
            {resultState === 'error' && (
              <div className="flex-1 flex flex-col items-center justify-center gap-4 py-8">
                <div className="w-12 h-12 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
                  <AlertTriangle size={22} className="text-rose-400" />
                </div>
                <p className="text-sm text-white/60 text-center px-6">{errorMsg}</p>
                <button
                  onClick={() => setResultState('idle')}
                  className="text-xs text-white/45 hover:text-white/75 underline underline-offset-2 transition-colors"
                >
                  Dismiss
                </button>
              </div>
            )}

          </section>
        </main>

        {/* ── Footer ────────────────────────────────────────────────────── */}
        <footer className="text-center py-6">
          <p className="text-[11px] text-white/25 tracking-wide">
            TamilVision · MobileNetV3-Small · 156 Classes
          </p>
        </footer>

      </div>
    </div>
  );
}
