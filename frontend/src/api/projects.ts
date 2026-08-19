// 프로젝트 — **껍데기**다. 이미지·라벨·클래스·검수는 전부 데이터셋 안에 있고
// (`./datasets`), 여기 남는 것은 프로젝트 자체와 데이터셋이 함께 쓰는 형(型)뿐이다.

export interface ProjectOut {
  id: string
  name: string
  created_at: string
}

export interface LabelBox {
  id?: string | null
  cls: number
  xyxy_n: [number, number, number, number]
  score?: number | null
  status?: string | null
  reason?: string | null
  sources?: { model: string; score: number }[] | null
}

export interface ImageItem {
  name: string
  stem: string
  thumb: string
  url: string
  labeled: boolean
  reviewed: boolean
  boxes: LabelBox[]
  created_at: number
}

