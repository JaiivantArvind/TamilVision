'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
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

// ── Component ─────────────────────────────────────────────────────────────────

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [tab, setTab] = useState<Tab>('draw');
  const [brushSize, setBrushSize] = useState(12);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultState, setResultState] = useState<ResultState>('idle');
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [errorMsg, setErrorMsg] = useState('');
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [apiLabel, setApiLabel] = useState('Connecting…');

  const drawingRef = useRef(false);
  const lastPosRef = useRef({ x: 0, y: 0 });

  // ── API status ping ──────────────────────────────────────────────────────────
  const pingAPI = useCallback(async () => {
    try {
      const res = await fetch(`${API}/`, { signal: AbortSignal.timeout(3000) });
      const data = await res.json();
      setApiOnline(true);
      setApiLabel(`Online · ${data.accuracy ?? ''}`);
    } catch {
      setApiOnline(false);
      setApiLabel('API Offline');
    }
  }, []);

  useEffect(() => {
    pingAPI();
    const id = setInterval(pingAPI, 15000);
    return () => clearInterval(id);
  }, [pingAPI]);

  // ── Canvas drawing ───────────────────────────────────────────────────────────
  const getPos = (e: React.MouseEvent | React.TouchEvent, canvas: HTMLCanvasElement) => {
    const r = canvas.getBoundingClientRect();
    const sx = canvas.width / r.width;
    const sy = canvas.height / r.height;
    if ('touches' in e) {
      return {
        x: (e.touches[0].clientX - r.left) * sx,
        y: (e.touches[0].clientY - r.top) * sy,
      };
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
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
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
  const confClass = (pct: number) =>
    pct >= 70
      ? 'from-green-500 to-emerald-400'
      : pct >= 40
      ? 'from-yellow-500 to-amber-400'
      : 'from-red-500 to-rose-400';

  const top = predictions[0];
  const topPct = top ? Math.round(top.confidence * 100) : 0;

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="relative min-h-screen bg-black text-gray-100 overflow-hidden">

      {/* ── GLSL Hills background ──────────────────────────────────────── */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <GLSLHills width="100%" height="100%" speed={0.4} cameraZ={130} planeSize={256} />
      </div>

      {/* ── Content ───────────────────────────────────────────────────── */}
      <div className="relative z-10 flex flex-col min-h-screen">

        {/* Header */}
        <header className="border-b border-white/10 bg-black/60 backdrop-blur-md sticky top-0 z-50">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-lg font-bold"
                   style={{ fontFamily: "'Mukta Malar', serif" }}>
                த
              </div>
              <h1 className="text-2xl font-bold tracking-wide"
                  style={{ color: '#38bdf8', textShadow: '0 0 10px #38bdf8, 0 0 30px #0ea5e9, 0 0 60px #0284c7' }}>
                TamilVision 156
              </h1>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span className={`w-2 h-2 rounded-full ${
                apiOnline === null ? 'bg-gray-500' : apiOnline ? 'bg-green-400' : 'bg-red-500'
              }`} />
              <span>{apiLabel}</span>
            </div>
          </div>
        </header>

        {/* Main grid */}
        <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8 grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* ── Input panel ──────────────────────────────────────────────── */}
          <section className="bg-gray-900/80 backdrop-blur-sm rounded-2xl p-6 flex flex-col gap-5 shadow-xl border border-white/10">
            <h2 className="text-lg font-semibold text-gray-200">Input</h2>

            {/* Tabs */}
            <div className="flex gap-2">
              {(['draw', 'upload'] as Tab[]).map((t) => (
                <button
                  key={t}
                  onClick={() => { setTab(t); setResultState('idle'); }}
                  className={`flex-1 py-2 px-4 rounded-lg border text-sm font-medium transition-all
                    ${tab === t
                      ? 'bg-blue-800 text-white border-blue-500'
                      : 'border-gray-600 text-gray-400 hover:text-white'}`}
                >
                  {t === 'draw' ? '✏️ Draw Mode' : '📁 Upload Mode'}
                </button>
              ))}
            </div>

            {/* Draw mode */}
            {tab === 'draw' && (
              <div className="flex flex-col gap-4">
                <div className="relative rounded-xl overflow-hidden border-2 border-gray-600 shadow-inner bg-black">
                  <canvas
                    ref={canvasRef}
                    width={400}
                    height={400}
                    className="w-full aspect-square block cursor-crosshair touch-none"
                    onMouseDown={startDraw}
                    onMouseMove={onDraw}
                    onMouseUp={stopDraw}
                    onMouseLeave={stopDraw}
                    onTouchStart={startDraw}
                    onTouchMove={onDraw}
                    onTouchEnd={stopDraw}
                  />
                </div>

                <div className="flex items-center gap-3 text-sm text-gray-400">
                  <span>Brush</span>
                  <input
                    type="range" min={4} max={40} value={brushSize}
                    onChange={(e) => setBrushSize(Number(e.target.value))}
                    className="flex-1 accent-blue-500"
                  />
                  <span className="w-10 text-right">{brushSize}px</span>
                </div>

                <div className="flex gap-3">
                  <button onClick={clearCanvas}
                    className="flex-1 py-2.5 rounded-xl bg-gray-700 hover:bg-gray-600 text-sm font-medium transition-colors">
                    🗑 Clear
                  </button>
                  <button onClick={predict}
                    className="flex-1 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-sm font-semibold transition-colors shadow-lg shadow-blue-900/50">
                    🔍 Predict
                  </button>
                </div>
              </div>
            )}

            {/* Upload mode */}
            {tab === 'upload' && (
              <div className="flex flex-col gap-4">
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    const f = e.dataTransfer.files[0];
                    if (f?.type.startsWith('image/')) handleFile(f);
                  }}
                  className="border-2 border-dashed border-gray-600 rounded-xl p-10 flex flex-col items-center justify-center gap-3 text-gray-500 cursor-pointer hover:border-blue-500 hover:text-gray-300 transition-all"
                >
                  <span className="text-4xl">📂</span>
                  <p className="text-sm font-medium">Drag & drop an image here</p>
                  <p className="text-xs">or click to browse — PNG, JPG, BMP</p>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
                />

                {previewUrl && (
                  <div className="rounded-xl overflow-hidden border border-gray-600">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={previewUrl} alt="preview" className="w-full object-contain max-h-64 bg-black" />
                  </div>
                )}

                <button
                  onClick={predict}
                  disabled={!uploadedFile}
                  className="w-full py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-sm font-semibold transition-colors shadow-lg shadow-blue-900/50 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  🔍 Predict
                </button>
              </div>
            )}
          </section>

          {/* ── Results panel ─────────────────────────────────────────────── */}
          <section className="bg-gray-900/80 backdrop-blur-sm rounded-2xl p-6 flex flex-col gap-6 shadow-xl border border-white/10">
            <h2 className="text-lg font-semibold text-gray-200">Results</h2>

            {/* Idle */}
            {resultState === 'idle' && (
              <div className="flex-1 flex flex-col items-center justify-center gap-3 text-gray-600 py-12">
                <span className="text-6xl" style={{ fontFamily: "'Mukta Malar', serif" }}>த</span>
                <p className="text-sm">Draw or upload a Tamil character and click Predict</p>
              </div>
            )}

            {/* Loading */}
            {resultState === 'loading' && (
              <div className="flex-1 flex flex-col items-center justify-center gap-4 py-12">
                <div className="w-16 h-16 rounded-full border-4 border-blue-500 border-t-transparent animate-spin" />
                <p className="text-sm text-gray-400 animate-pulse">Analysing character…</p>
              </div>
            )}

            {/* Content */}
            {resultState === 'content' && top && (
              <div className="flex flex-col gap-6">
                {/* Top prediction */}
                <div className="bg-black/40 rounded-xl p-6 flex items-center gap-6 border border-white/10">
                  <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-blue-700 to-blue-900 flex items-center justify-center shadow-lg shadow-blue-900/60 shrink-0">
                    <span className="text-5xl font-bold text-white" style={{ fontFamily: "'Mukta Malar', serif" }}>
                      {top.predicted_character}
                    </span>
                  </div>
                  <div className="flex-1">
                    <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">Top Prediction</p>
                    <p className="text-xs text-blue-400 font-mono mb-3">label {top.label_id}</p>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 bg-gray-700 rounded-full h-3 overflow-hidden">
                        <div
                          className={`h-3 rounded-full bg-gradient-to-r transition-all duration-700 ${confClass(topPct)}`}
                          style={{ width: `${topPct}%` }}
                        />
                      </div>
                      <span className={`text-sm font-semibold w-14 text-right ${
                        topPct >= 70 ? 'text-emerald-400' : topPct >= 40 ? 'text-amber-400' : 'text-red-400'
                      }`}>
                        {topPct}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Alternatives */}
                {predictions.length > 1 && (
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-widest mb-3">Alternatives</p>
                    <div className="flex flex-col gap-2">
                      {predictions.slice(1).map((p) => {
                        const pct = Math.round(p.confidence * 100);
                        return (
                          <div key={p.label_id}
                            className="bg-black/40 rounded-xl px-4 py-3 flex items-center gap-4 border border-white/10">
                            <div className="w-10 h-10 rounded-lg bg-gray-700 flex items-center justify-center shrink-0">
                              <span className="text-xl font-bold text-gray-200"
                                    style={{ fontFamily: "'Mukta Malar', serif" }}>
                                {p.predicted_character}
                              </span>
                            </div>
                            <div className="flex-1">
                              <div className="flex justify-between text-xs mb-1.5">
                                <span className="text-gray-400">
                                  label <span className="font-mono text-blue-400">{p.label_id}</span>
                                </span>
                                <span className="text-gray-400">{pct}%</span>
                              </div>
                              <div className="bg-gray-700 rounded-full h-1.5 overflow-hidden">
                                <div className="h-1.5 rounded-full bg-blue-500/70" style={{ width: `${pct}%` }} />
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                <button
                  onClick={() => setResultState('idle')}
                  className="mt-auto py-2 rounded-xl border border-gray-600 hover:bg-gray-700 text-sm text-gray-400 hover:text-white transition-colors"
                >
                  ↩ Try Again
                </button>
              </div>
            )}

            {/* Error */}
            {resultState === 'error' && (
              <div className="flex-1 flex flex-col items-center justify-center gap-3 text-red-400 py-12">
                <span className="text-4xl">⚠️</span>
                <p className="text-sm text-center px-4">{errorMsg}</p>
                <button
                  onClick={() => setResultState('idle')}
                  className="text-xs underline text-gray-500 hover:text-white"
                >
                  Try again
                </button>
              </div>
            )}
          </section>
        </main>

        {/* Footer */}
        <footer className="text-center text-xs text-gray-700 py-6">
          TamilVision 156 · MobileNetV3-Small · 156 Classes · GTX 1650
        </footer>
      </div>
    </div>
  );
}
