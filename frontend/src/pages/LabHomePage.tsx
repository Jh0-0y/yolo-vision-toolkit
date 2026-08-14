// 연구실 목록 — 지금은 하나다.
//
// **읽기 전용이다.** 만들지도 지우지도 이름을 바꾸지도 않는다 — 연구실은 런타임에
// 추가하는 것이 아니라 코드로 들어온다. 야구를 하게 되면 `baseball-adaptive-crop` 이
// 생기는데, 패키지부터 다르고 파이프라인이 아예 달라질 수 있어서 여기서 "추가" 로
// 만들어 낼 수 있는 물건이 아니다.
//
// 그래서 이 화면이 하는 일은 **무슨 연구인지 알려주고 들여보내는 것**이다.
import {
  Anchor,
  Card,
  Container,
  Group,
  Stack,
  Text,
  ThemeIcon,
  Title,
  UnstyledButton,
} from '@mantine/core'
import { IconChevronLeft, IconChevronRight, IconCrop, IconFlask, IconVideo } from '@tabler/icons-react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getLab } from '../api/client'
import StatTile from '../components/StatTile'

export default function LabHomePage() {
  const lab = useQuery({ queryKey: ['lab'], queryFn: getLab })

  return (
    <Container size={1000} py="xl">
      <Stack gap="xl">
        <div>
          <Anchor component={Link} to="/" size="sm" c="dimmed">
            <Group gap={4}>
              <IconChevronLeft size={14} />
              Home
            </Group>
          </Anchor>
        </div>

        <Group gap="md">
          <ThemeIcon size={52} radius="md" variant="light" color="grape">
            <IconFlask size={30} stroke={1.6} />
          </ThemeIcon>
          <div>
            <Title order={2}>Lab</Title>
            <Text c="dimmed" size="sm">
              Each lab is one line of research with its own pipeline. New ones arrive with the code,
              not from this screen.
            </Text>
          </div>
        </Group>

        {lab.data && (
          <UnstyledButton component={Link} to="/lab/crop">
            <Card withBorder radius="md" padding="lg">
              <Group justify="space-between" align="flex-start" wrap="nowrap">
                <div style={{ minWidth: 0 }}>
                  <Text fw={700} size="lg">
                    {lab.data.name}
                  </Text>
                  <Text size="sm" c="dimmed" mt={2}>
                    {lab.data.description}
                  </Text>
                </div>
                <IconChevronRight
                  size={18}
                  stroke={1.6}
                  style={{ flexShrink: 0, marginTop: 6 }}
                />
              </Group>

              <Group gap="sm" mt="md" grow>
                <StatTile
                  label="Videos"
                  value={lab.data.video_count}
                  color="gray.7"
                  icon={<IconVideo size={13} />}
                />
                <StatTile
                  label="Crop runs"
                  value={lab.data.run_count}
                  color="grape"
                  icon={<IconCrop size={13} />}
                />
              </Group>
            </Card>
          </UnstyledButton>
        )}
      </Stack>
    </Container>
  )
}
