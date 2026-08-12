// 학습 지표를 차트가 먹는 형태로 옮기는 순수 변환.
//
// 숫자의 출처는 두 곳이다 — 끝난 에폭은 results.csv, 진행 중인 에폭은 SSE.
// 둘을 같은 Point 로 맞춰 하나의 차트에 그린다.

import type { TrainEpochEvent, TrainResultRow } from '../../api/client'

export const RUN_STATUS_COLOR: Record<string, string> = {
  running: 'blue',
  done: 'green',
  error: 'red',
  stopped: 'yellow',
  queued: 'gray',
}

export interface Point {
  epoch: number
  mAP50?: number
  'mAP50-95'?: number
  precision?: number
  recall?: number
  train_box?: number
  train_cls?: number
  train_dfl?: number
  val_box?: number
  val_cls?: number
  val_dfl?: number
  lr?: number
  time?: number
}

const num = (v: unknown): number | undefined =>
  typeof v === 'number' && Number.isFinite(v) ? v : undefined

/** Build a chart point from a results.csv row (keys may vary slightly by version). */
export function rowToPoint(row: TrainResultRow): Point {
  const g = (...keys: string[]): number | undefined => {
    for (const k of keys) if (row[k] != null) return num(row[k])
    return undefined
  }
  return {
    epoch: g('epoch') ?? 0,
    mAP50: g('metrics/mAP50(B)', 'metrics/mAP50'),
    'mAP50-95': g('metrics/mAP50-95(B)', 'metrics/mAP50-95'),
    precision: g('metrics/precision(B)', 'metrics/precision'),
    recall: g('metrics/recall(B)', 'metrics/recall'),
    train_box: g('train/box_loss'),
    train_cls: g('train/cls_loss'),
    train_dfl: g('train/dfl_loss'),
    val_box: g('val/box_loss'),
    val_cls: g('val/cls_loss'),
    val_dfl: g('val/dfl_loss'),
    lr: g('lr/pg0', 'lr/pg1', 'lr/pg2'),
    time: g('time'),
  }
}

/** Fallback point from a live SSE event (before results.csv is first flushed). */
export function eventToPoint(ev: TrainEpochEvent): Point | null {
  if (ev.phase !== 'epoch' || !ev.epoch || !ev.metrics) return null
  if (ev.epochs && ev.epoch > ev.epochs) return null
  const m = ev.metrics
  return {
    epoch: ev.epoch,
    mAP50: num(m['metrics/mAP50(B)']),
    'mAP50-95': num(m['metrics/mAP50-95(B)']),
    precision: num(m['metrics/precision(B)']),
    recall: num(m['metrics/recall(B)']),
    train_box: num(m['train/box_loss']),
    train_cls: num(m['train/cls_loss']),
    train_dfl: num(m['train/dfl_loss']),
    val_box: num(m['val/box_loss']),
    val_cls: num(m['val/cls_loss']),
    val_dfl: num(m['val/dfl_loss']),
    lr: num(m['lr/pg0']),
  }
}

export function formatDuration(seconds?: number): string {
  if (!seconds || seconds < 0) return '–'
  const s = Math.round(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h) return `${h}h ${m}m`
  if (m) return `${m}m ${sec}s`
  return `${sec}s`
}

export const PLOT_LABELS: { match: RegExp; label: string }[] = [
  { match: /^results\.png$/i, label: 'Training curves' },
  { match: /confusion_matrix_normalized/i, label: 'Confusion matrix (norm.)' },
  { match: /confusion_matrix/i, label: 'Confusion matrix' },
  { match: /PR_curve/i, label: 'Precision–Recall' },
  { match: /P_curve/i, label: 'Precision' },
  { match: /R_curve/i, label: 'Recall' },
  { match: /F1_curve/i, label: 'F1' },
  { match: /labels_correlogram/i, label: 'Label correlogram' },
  { match: /labels\.jpg/i, label: 'Label distribution' },
]

