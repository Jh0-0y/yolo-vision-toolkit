// 이미지 한 장의 라벨 박스 읽기·쓰기.
import { api } from './http'
import type { LabelBox } from './projects'


export interface LabelDetail {
  stem: string
  name: string
  image_url: string
  boxes: LabelBox[]
  classes: { id: number; name: string; sources: string[] }[]
  reviewed: boolean
}

export const getLabels = (projectId: string, stem: string) =>
  api.get<LabelDetail>(`/projects/${projectId}/labels/${encodeURIComponent(stem)}`)

export const putLabels = (projectId: string, stem: string, boxes: LabelBox[]) =>
  api.put(`/projects/${projectId}/labels/${encodeURIComponent(stem)}`, { boxes })
