// 헬스체크와 디바이스 정보.

export interface Health {
  status: string
  data_dir: string
}

export interface DeviceInfo {
  device: string
  requested: string
  accelerator: 'cuda' | 'mps' | 'cpu'
  accelerated: boolean
  device_name: string
  device_label: string
  cuda_available: boolean
  mps_available: boolean
  torch_version: string
  platform: string
  vram_total_mb: number | null
  vram_used_mb: number | null
}
