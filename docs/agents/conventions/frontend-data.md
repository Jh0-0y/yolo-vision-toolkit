---
title: 프론트 데이터 접근
scope: frontend/src/**/*.{ts,tsx}
applies_to: 서버 데이터를 읽고 쓰거나 상태를 어디 둘지 정할 때
related:
  - ./backend-api-route.md
  - ./jobs-and-progress.md
---

# 프론트 데이터 접근

> 서버 통신은 `client.ts` 하나만 거친다. 데이터를 가져오거나 상태를 만들 때 읽는다.

## `src/api/client.ts` 가 유일한 출입구

컴포넌트에서 `fetch` 를 직접 부르지 **않는다.** 화면은 언제나 `from '../api/client'`
하나만 알면 된다.

`client.ts` 자체는 **재수출만** 한다. 실제 구현은 리소스별 파일에 있다:

```
api/http.ts        BASE · ApiError · api 래퍼 · xhrUpload   (fetch/XHR 을 아는 유일한 파일)
api/<리소스>.ts     datasets · projects · labels · classes · jobs · training · labs · labCrops · models
api/test/          predict · annotate · live · compare      (백엔드 endpoints/predict 와 같은 갈래)
api/client.ts      전부 재수출하는 배럴
```

새 경로는 해당 리소스 파일에 함수와 응답 인터페이스를 함께 두고, 새 리소스라면
파일을 만들어 `client.ts` 에 `export *` 한 줄을 더한다.

- **이름이 겹치면 안 된다.** `export *` 는 두 파일이 같은 이름을 내보내면 그 이름을
  조용히 빼버린다. 나눈 뒤에는 `npm run build` 로 확인한다.

```ts
export const cancelAutoLabel = (jobId: string) => api.post(`/jobs/${jobId}/cancel`)
```

- `BASE` 는 `/api/v1`. 각 함수는 **리소스 경로만** 적는다.
- 래퍼는 `api.get` · `post` · `put` · `patch` · `delete` · `upload`. 204 는 `undefined` 로 돌아온다.
- 실패는 `ApiError`(`status` 를 가진다)로 던져진다. `status` 로 분기할 게 아니면 그냥 위로 흘린다.

## 서버 상태는 TanStack Query

- 조회는 `useQuery`, 변경은 `useMutation`.
- `queryKey` 는 배열이고 첫 요소가 리소스명이다. 프로젝트 스코프면 `projectId` 를 포함한다: `['images', projectId, filterQuery]`.
- 변경 후에는 **영향받는 키를 전부 `invalidateQueries` 한다.** 한 번의 변경이 여러 화면 수치를 바꾸는 경우가 많다(이미지 삭제 → `images` · `image-names` · `stats`).
- 서버에서 오는 값을 `useState` 에 복사해 두지 않는다. 두 벌이 되면 갈라진다.

## 상태를 어디 둘까

| 상태 | 자리 |
|---|---|
| 서버에서 오는 데이터 | TanStack Query |
| 한 화면 안에서만 쓰는 UI 상태 | `useState` |
| 화면을 넘나드는 공유 상태 | `stores/` 의 zustand 스토어 |
| 페이지를 떠나도 계속 도는 작업 | **`stores/jobStore.ts`** — 컴포넌트에 두지 않는다 |

`jobStore` 는 SSE 구독과 업로드 XHR 을 직접 소유한다. 진행률 UI 를 페이지에 새로 만들지 말고
잡을 스토어에 등록한다 → [잡과 진행률](jobs-and-progress.md)

## 컴포넌트

- UI 는 Mantine 9 를 쓴다. 다른 UI 라이브러리를 새로 들이지 않는다.
- 라우트 하나 = `pages/<이름>Page.tsx` 하나. `App.tsx` 에 라우트를 추가한다.
- 특정 화면에서만 쓰는 컴포넌트·훅은 `components/<영역>/` 아래 둔다. 여러 화면이 쓰게 되면 `components/` 바로 아래로 올린다.
- 페이지가 300줄을 넘고 안에 **서브 컴포넌트가 눌러앉아 있으면** `components/<영역>/` 으로 뺀다.
  `pages/TrainRunDetailPage.tsx` + `components/train/` 이 그 예다. 순수 변환 함수는 `.ts` 로
  따로 두면 컴포넌트 없이 읽을 수 있다.
