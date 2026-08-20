// 모델을 레지스트리에 들이는 두 가지 방법을 버튼 하나로 모은다.
//
// 헤더와 빈 상태 **두 곳에서** 쓰이기 때문에 컴포넌트로 뺐다. 다른 목록 화면과 같은
// 규칙이다 — 주 버튼은 하나뿐이고 제목 줄 오른쪽에 선다.
import { useRef } from 'react'
import { Button, Menu } from '@mantine/core'
import { IconChevronDown, IconCloudDownload, IconPlus, IconUpload } from '@tabler/icons-react'

interface Props {
  onDownload: () => void
  onUpload: (file: File) => void
  uploading?: boolean
  /** 빈 상태에서는 무게를 낮춰 쓴다 — 그 자리에서는 카드 전체가 이미 안내다. */
  variant?: 'filled' | 'light'
}

export default function AddModelMenu({
  onDownload,
  onUpload,
  uploading,
  variant = 'filled',
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null)

  return (
    <>
      <Menu position="bottom-end" withinPortal>
        <Menu.Target>
          <Button
            variant={variant}
            leftSection={<IconPlus size={16} />}
            rightSection={<IconChevronDown size={14} />}
            loading={uploading}
          >
            Add model
          </Button>
        </Menu.Target>
        <Menu.Dropdown>
          <Menu.Item leftSection={<IconCloudDownload size={14} />} onClick={onDownload}>
            Download official model
          </Menu.Item>
          <Menu.Item
            leftSection={<IconUpload size={14} />}
            onClick={() => fileRef.current?.click()}
          >
            Upload .pt
          </Menu.Item>
        </Menu.Dropdown>
      </Menu>

      {/* 파일 입력은 **드롭다운 밖**에 둔다. 안에 두면 항목을 누르는 순간 메뉴가 닫히면서
          입력도 함께 사라져, 열리려던 파일 선택창이 취소된다. */}
      <input
        ref={fileRef}
        type="file"
        accept=".pt"
        hidden
        onChange={(e) => {
          const file = e.currentTarget.files?.[0]
          // 같은 파일을 두 번 고를 수 있게 값을 비운다 — 안 비우면 change 가 안 뜬다
          e.currentTarget.value = ''
          if (file) onUpload(file)
        }}
      />
    </>
  )
}
