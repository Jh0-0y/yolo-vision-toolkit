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

소켓·큐·공유메모리를 새로 도입하지 않는다. 파일이라서 **새로고침 후에도 처음부터 재생**된다.

**이 두 파일을 직접 조립하지 않는다.** `infra/jobs` 가 유일한 진입점이다.

```python
from infra import jobs

job = jobs.at(settings.jobs_dir, job_id)   # API 프로세스
job = jobs.at(Path(jobs_dir), job_id)      # 워커 (설정을 모르므로 루트를 인자로 받는다)
```

| 부를 것 | 언제 |
|---|---|
| `job.prepare()` | 잡을 제출하기 **전에** 부모가. 디렉터리·빈 진행률 파일 생성 + 묵은 `CANCEL` 제거 |
| `job.reset()` | 같은 대상을 재실행할 때(예: 프레임 재추출). 이전 이벤트를 지운다 |
| `job.emit(event)` | 진행 이벤트 한 줄 append |
| `job.read(offset)` | `(이벤트 목록, 다음 오프셋)` |
| `job.request_cancel()` / `job.cancelled()` | 취소 요청 / 확인 |

## 진행 이벤트

한 줄에 JSON 하나. `phase` 는 **필수**, 종료 phase 는 `done` · `error` · `cancelled` 셋뿐이다.
`ts` 는 `emit` 이 붙이므로 **호출자가 넣지 않는다.**

나머지 키는 잡 종류마다 다르다(`total`/`done`, `scanned`/`saved` 등) — **새 키를 늘리기 전에 프론트 매퍼가 그걸 쓸지 확인한다.**
`infra/jobs` 는 형식만 정하고 **무엇을 담을지는 각 기능이 정한다** — 공용 스키마를 억지로 만들지 않는다.

실패하면 `{"phase": "error", "msg": str(e)}` 를 남기고 **예외를 다시 raise 한다.**

## 순수 계산 쪽

`lib/` 의 계산 함수는 **파일도 잡도 모른다.** 콜백을 인자로 받는다.

```python
result = run_labeling(
    cfg,
    progress=job.emit,        # 어디에 기록할지는 호출자가 정한다
    cancel_check=job.cancelled,
)
```

취소를 감지하면 전용 예외(`JobCancelled` 등)를 던지고, 엔트리가 `{"status": "cancelled"}` 로 바꿔 돌려준다.

## 취소

아직 큐에 있으면 `future.cancel()` 로 끝난다. 이미 돌고 있으면 **`CANCEL` 파일을 touch 한다**(`job.request_cancel()`) —
프로세스를 kill 하지 않는다. 쓰다 만 파일이 남는다.

## API 쪽

- SSE 엔드포인트는 **바이트 오프셋으로 tail** 한다(`job.read`). 마지막 줄이 잘려 있을 수 있어 `\n` 까지만 소비한다.
- 종료 phase 를 보면 스트림을 닫는다. 워커가 이벤트 없이 죽는 경우가 있어 유휴 폴링 중 DB 상태도 주기적으로 확인한다.

## 파일과 DB 는 어긋난다

진행 중 상태는 **파일만** 알고, 끝난 사실은 **DB 에** 적힌다. 둘을 잇는 건 감시 스레드 하나뿐이라
API 가 재시작되면 끊긴다 — 그래서 학습은 부팅 시 `train_manager.reconcile_on_boot()` 로 정정한다.

**잡을 새로 만들면 이 복구를 같이 만들어야 한다.** 지금 복구가 있는 건 학습뿐이다.

## 프론트 쪽

- `EventSource` 로 `progress` 이벤트를 듣고, 종료 phase 에서 `source.close()`.
- `onerror` 는 `readyState === EventSource.CLOSED` 일 때만 실패로 처리한다. 그 전은 자동 재접속 중이다.
- **페이지 이동에도 살아남아야 하는 잡은 컴포넌트가 아니라 `stores/jobStore.ts` 가 소유한다.** 서버 잡은 `refId` 를 localStorage 에 남겨 새로고침 후 다시 구독한다.
