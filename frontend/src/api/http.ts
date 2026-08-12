// HTTP 전송 계층 — 이 파일만 fetch/XHR 을 안다.
//
// 리소스 모듈은 여기서 `api` 래퍼만 가져다 쓴다. 진행률이 필요한 업로드는
// fetch 로 못 재므로 `xhrUpload` 를 쓴다 (학습 데이터셋 zip · 영상).

// 모든 경로의 접두사. 리소스 모듈은 경로만 적고 이 값을 앞에 붙이지 않는다
// (직접 URL 문자열을 만드는 다운로드 링크만 예외로 가져다 쓴다).
export const BASE = '/api/v1'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    const body = await res.text()
    throw new ApiError(res.status, body || res.statusText)
  }
  // 204 No Content (e.g. DELETE) has an empty body — nothing to parse
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    }),
  put: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  patch: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  delete: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'DELETE',
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    }),
  upload: <T>(path: string, form: FormData) =>
    request<T>(path, { method: 'POST', body: form }),
}

export interface UploadHandlers {
  onProgress?: (percent: number) => void // 0-100 network upload
  onUploaded?: () => void // request body fully sent; server now processing
  signal?: AbortSignal // abort the in-flight upload (cancel)
}

// XHR-based upload (fetch can't report upload %). Resolves with the parsed JSON
// body; rejects with ApiError on non-2xx, abort, or network failure.
export function xhrUpload<T>(
  path: string,
  form: FormData,
  handlers: UploadHandlers = {},
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}${path}`)
    if (handlers.signal) {
      handlers.signal.addEventListener('abort', () => xhr.abort(), { once: true })
    }
    xhr.onabort = () => reject(new ApiError(0, 'Cancelled'))
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) handlers.onProgress?.(Math.round((e.loaded / e.total) * 100))
    }
    xhr.upload.onload = () => handlers.onUploaded?.()
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(xhr.responseText ? JSON.parse(xhr.responseText) : (undefined as T))
        } catch {
          reject(new ApiError(xhr.status, 'Invalid server response'))
        }
      } else {
        reject(new ApiError(xhr.status, xhr.responseText || xhr.statusText))
      }
    }
    xhr.onerror = () => reject(new ApiError(0, 'Network error during upload'))
    xhr.send(form)
  })
}
