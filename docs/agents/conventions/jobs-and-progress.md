---
title: 잡과 진행률
scope:
  - backend/app/**/*.py
  - frontend/src/**/*.{ts,tsx}
applies_to: 오래 걸리는 작업을 만들거나 진행률·취소를 다룰 때
related:
  - ./layer-boundaries.md
  - ./frontend-data.md
  - ../data-layout.md
---

# 잡과 진행률

> 장시간 작업은 별도 프로세스에서 돌고, 파일로만 소통한다. 새 잡을 만들 때 읽는다.

## 파일 두 개가 전부다

```
DATA_DIR/jobs/{job_id}/
├── progress.jsonl    # 워커가 append, API 가 tail 해서 SSE 로 흘린다
└── CANCEL            # 존재하면 취소. 내용은 보지 않는다
```

잡을 제출하기 전에 **부모가 디렉터리와 빈 `progress.jsonl` 을 먼저 만든다**(그래야 구독이 바로 붙는다).
소켓·큐·공유메모리를 새로 도입하지 않는다. 파일이라서 **새로고침 후에도 처음부터 재생**된다.

## 진행 이벤트

한 줄에 JSON 하나. `ts` 와 `phase` 는 **필수**이고, 종료 phase 는 `done` · `error` · `cancelled` 셋뿐이다.
나머지 키는 잡 종류마다 다르다(`total`/`done`, `scanned`/`saved` 등) — **새 키를 늘리기 전에 프론트 매퍼가 그걸 쓸지 확인한다.**
실패하면 `{"phase": "error", "msg": str(e)}` 를 남기고 **예외를 다시 raise 한다.**

## 워커 쪽

계산 함수는 파일을 모른다. 콜백 두 개를 **인자로 받는다.**

```python
result = run_labeling(
    cfg,
    progress=lambda ev: _append_progress(progress_path, ev),
    cancel_check=cancel_path.exists,
)
```

취소를 감지하면 전용 예외(`JobCancelled` 등)를 던지고, 엔트리가 `{"status": "cancelled"}` 로 바꿔 돌려준다.

## 취소

아직 큐에 있으면 `future.cancel()` 로 끝난다. 이미 돌고 있으면 **`CANCEL` 파일을 touch 한다** —
프로세스를 kill 하지 않는다. 쓰다 만 파일이 남는다.

## API 쪽

- SSE 엔드포인트는 **바이트 오프셋으로 tail** 한다(`read_progress`). 마지막 줄이 잘려 있을 수 있어 `\n` 까지만 소비한다.
- 종료 phase 를 보면 스트림을 닫는다. 워커가 이벤트 없이 죽는 경우가 있어 유휴 폴링 중 DB 상태도 주기적으로 확인한다.

## 프론트 쪽

- `EventSource` 로 `progress` 이벤트를 듣고, 종료 phase 에서 `source.close()`.
- `onerror` 는 `readyState === EventSource.CLOSED` 일 때만 실패로 처리한다. 그 전은 자동 재접속 중이다.
- **페이지 이동에도 살아남아야 하는 잡은 컴포넌트가 아니라 `stores/jobStore.ts` 가 소유한다.** 서버 잡은 `refId` 를 localStorage 에 남겨 새로고침 후 다시 구독한다.
