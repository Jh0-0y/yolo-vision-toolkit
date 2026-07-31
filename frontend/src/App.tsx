import { Navigate, Route, Routes } from 'react-router-dom'
import JobIndicator from './components/JobIndicator'
import ProjectLayout from './layouts/ProjectLayout'
import ClassesPage from './pages/ClassesPage'
import DatasetPage from './pages/DatasetPage'
import ExportsPage from './pages/ExportsPage'
import CropDrawPage from './pages/CropDrawPage'
import CropResultPage from './pages/CropResultPage'
import HomePage from './pages/HomePage'
import LabelEditorPage from './pages/LabelEditorPage'
import ModelsPage from './pages/ModelsPage'
import TrainPage from './pages/TrainPage'
import TrainingHistoryPage from './pages/TrainingHistoryPage'
import TrainRunDetailPage from './pages/TrainRunDetailPage'
import UploadPage from './pages/UploadPage'

export default function App() {
  return (
    <>
      <JobIndicator />
      <Routes>
        <Route path="/" element={<HomePage />} />
      <Route path="/projects/:projectId" element={<ProjectLayout />}>
        <Route index element={<Navigate to="dataset" replace />} />
        <Route path="upload" element={<UploadPage />} />
        <Route path="dataset" element={<DatasetPage />} />
        <Route path="dataset/label/:stem" element={<LabelEditorPage />} />
        <Route path="classes" element={<ClassesPage />} />
        <Route path="exports" element={<ExportsPage />} />
        <Route path="train" element={<TrainPage />} />
        <Route path="history" element={<TrainingHistoryPage />} />
        <Route path="history/:runId" element={<TrainRunDetailPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="lab/crop-result" element={<CropResultPage />} />
        <Route path="lab/crop-draw" element={<CropDrawPage />} />
        {/* 옛 Test 경로 → 그리기 도구로 리다이렉트 (링크 깨짐 방지) */}
        <Route path="test" element={<Navigate to="../lab/crop-draw" replace />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}
