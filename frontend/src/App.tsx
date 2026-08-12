import { Navigate, Route, Routes } from 'react-router-dom'
import JobIndicator from './components/JobIndicator'
import ProjectLayout from './layouts/ProjectLayout'
import ClassesPage from './pages/ClassesPage'
import DatasetPage from './pages/DatasetPage'
import ExportsPage from './pages/ExportsPage'
import CropPage from './pages/CropPage'
import CropRunsPage from './pages/CropRunsPage'
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
        <Route path="lab/crop" element={<CropPage />} />
        <Route path="lab/crops" element={<CropRunsPage />} />
        {/* 옛 경로 → 통합 Crop 페이지로 리다이렉트 (링크 깨짐 방지) */}
        <Route path="lab/crop-result" element={<Navigate to="../lab/crop" replace />} />
        <Route path="lab/crop-draw" element={<Navigate to="../lab/crop" replace />} />
        <Route path="test" element={<Navigate to="../lab/crop" replace />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}
