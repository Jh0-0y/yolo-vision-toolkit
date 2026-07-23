import { Navigate, Route, Routes } from 'react-router-dom'
import ProjectLayout from './layouts/ProjectLayout'
import ClassesPage from './pages/ClassesPage'
import DatasetPage from './pages/DatasetPage'
import ExportsPage from './pages/ExportsPage'
import HomePage from './pages/HomePage'
import LabelEditorPage from './pages/LabelEditorPage'
import ModelsPage from './pages/ModelsPage'
import TestPage from './pages/TestPage'
import TrainPage from './pages/TrainPage'
import TrainingHistoryPage from './pages/TrainingHistoryPage'
import TrainRunDetailPage from './pages/TrainRunDetailPage'
import UploadPage from './pages/UploadPage'

export default function App() {
  return (
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
        <Route path="test" element={<TestPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
