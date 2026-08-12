// 프로젝트 클래스 목록 CRUD.
import { api } from './http'


export interface ProjectClass {
  id: number
  name: string
  sources: string[]
}

export const listClasses = (projectId: string) =>
  api.get<ProjectClass[]>(`/projects/${projectId}/classes`)

export const addClass = (projectId: string, name: string) =>
  api.post<ProjectClass>(`/projects/${projectId}/classes`, { name })

export const renameClass = (projectId: string, classId: number, name: string) =>
  api.patch<ProjectClass>(`/projects/${projectId}/classes/${classId}`, { name })

export const deleteClass = (projectId: string, classId: number) =>
  api.delete<{ ok: boolean; removed_boxes: number; classes: ProjectClass[] }>(
    `/projects/${projectId}/classes/${classId}`,
  )
