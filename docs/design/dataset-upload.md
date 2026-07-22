# 설계: 외부 데이터셋 zip → 학습 입력

> 상태: 설계 (미구현) · 최종 수정 2026-07-22

## 목표

프로젝트/오토라벨링과 **무관하게** YOLO 포맷 데이터셋 zip을 업로드해 바로
학습 소스로 등록한다. 출력은 학습된 `.pt`.

학습 데이터의 출처는 오토라벨링 루프만이 아니다 — 공개 데이터셋(COCO,
Roboflow export), 다른 팀/도구의 산출물, 과거에 뽑아둔 zip 등. 이런
데이터셋은 거의 항상 zip으로 유통되므로 **직접 투입 진입점**이 필요하다.
오토라벨링 export는 학습 데이터를 만드는 여러 방법 중 하나일 뿐이다.

## 핵심 통찰 — 소스가 여럿, 학습기는 하나

```
학습 데이터 소스
├── ① 내부 오토라벨링 export   (현재 구현됨)
└── ② 외부 데이터셋 zip 업로드  (본 문서)
              │  둘 다 같은 형태(YOLO: images/ + labels/ + data.yaml)로 수렴
              ▼
          학습 (training) → 출력 .pt
```

zip 업로드는 "새 학습기"가 아니라 **기존 학습기에 데이터셋 소스를 하나 더
꽂는 것**이다.

## 전제 — GPU를 쓰지 않는다

zip 압축 해제·검증은 CPU/IO 작업이다. GPU 직렬화 게이트를 타지 않고
**별도 백그라운드(threadpool)** 로 처리한다.

## 저장 구조

프로젝트 밖 신규 최상위 폴더를 둔다. (특정 프로젝트 소유가 아니라 **전역
학습 자산**이므로)

```
data/
├── datasets/                    # 신규 (config에 datasets_dir 추가)
│   └── {dataset_id}/
│       ├── images/ labels/ ...  # 정규화된 YOLO 구조
│       ├── data.yaml            # 경로를 절대화해 재작성
│       └── meta.json            # {name, source:"upload", train, val, classes, names}
└── projects/                    # 기존 (오토라벨링 export는 여기 유지)
```

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/api/datasets` | zip 업로드 + 검증 시작 (비동기) → `dataset_id` |
| `GET` | `/api/datasets` | 업로드된 데이터셋 목록 |
| `GET` | `/api/datasets/{id}/events` | 검증 진행률 SSE |
| `DELETE` | `/api/datasets/{id}` | 삭제 |

## 검증 / 정규화 단계 (핵심)

1. **안전한 압축 해제** — zip-slip 방어(절대경로·`..` 금지), `datasets/{id}/`로 풀기
2. **`data.yaml` 탐색** — 루트 또는 중첩 폴더에서 찾음
3. **파싱** — `names`, `nc`, `train`, `val` 경로 추출
4. **구조 검증** — train 이미지/라벨 폴더 존재 + 비어있지 않음 확인, 개수 카운트
5. **라벨 포맷 샘플 체크** — `.txt` 몇 개를 열어 `cls cx cy w h`(정규화 0~1) 형태 확인
6. **경로 절대화** — `data.yaml`의 train/val을 절대경로로 재작성 (학습기가 바로 찾도록)
7. **`meta.json` 저장**

## 지원 레이아웃 (유연하게)

현실의 zip은 형태가 제각각이라 둘 다 흡수해야 한다.

```
# Ultralytics 표준            # Roboflow 스타일
images/train/ images/val/     train/images/ train/labels/
labels/train/ labels/val/     valid/images/ valid/labels/
data.yaml                     data.yaml
```

`data.yaml`을 기준점으로 상대경로를 해석한다. `data.yaml`이 없으면 **422 +
명확한 안내** (또는 사용자가 클래스명을 직접 입력하는 폴백 — 선택).

## 기존 코드 연결점

- `app/api/training.py:79` `list_datasets` — 현재 프로젝트 export만 스캔 →
  **`datasets_dir`도 스캔해 통합 목록**으로. 각 항목에
  `source: "export" | "upload"` 필드 추가
- `app/api/training.py:107` `create_run` — 현재
  `projects/{pid}/exports/{export_id}` 경로를 조립 → **통합 `dataset_id`로
  폴더를 resolve**하도록 일반화. 그 폴더에 `data.yaml`만 있으면 학습 로직은
  그대로 동작 (export든 upload든 무관)
- 즉 **학습기 자체는 수정하지 않고**, 데이터셋 소스를 {export} →
  {export, upload}로 확장하는 것이 전부

## 출력 (.pt) — 이미 구현됨, 그대로 연결

- `app/api/training.py:245` `download_weights` → 학습된 `.pt` 다운로드
- `app/api/training.py:259` `register_weights` → `.pt`를 모델 레지스트리에
  등록 → 그 모델로 다시 오토라벨링 가능 (선순환)

## config 변경

`app/config.py`에 추가:

```python
@property
def datasets_dir(self) -> Path:
    return self.data_dir / "datasets"
# ensure_dirs()에도 datasets_dir 추가
```

## 선순환 (전체 그림)

```
외부 zip → 검증/정규화 → datasets/ → 학습 → .pt
                                        │
                          모델 레지스트리 등록 ← ─┘
                                        │
                          그 모델로 오토라벨링 → 더 좋은 데이터 → 재학습
```

## 열린 질문

- `data.yaml` 부재 시: 거부(422) vs 클래스명 수동 입력 폴백
- 데이터셋 메타 관리: 파일 기반 `meta.json` 스캔(현재 export 방식과 일관)
  vs DB 테이블 신설
- 업로드 데이터셋의 클래스 스키마와 모델 레지스트리 클래스의 정합성 검사 여부
