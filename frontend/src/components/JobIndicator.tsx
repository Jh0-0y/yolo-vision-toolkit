import { useEffect, useRef } from 'react'
import { Anchor, CloseButton, Group, Paper, Progress, Text } from '@mantine/core'
import { IconBrush, IconCrop, IconFileExport, IconFileZip, IconMovie, IconWand } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { useJobStore, type Job } from '../stores/jobStore'

// Mounted once at the app root: renders every background job's phased progress
// so it stays visible across page navigation, reconnects server jobs on load,
// and centralizes completion side effects (cache invalidation + toasts).
export default function JobIndicator() {
  const jobs = useJobStore((s) => s.jobs)
  const order = useJobStore((s) => s.order)
  const dismiss = useJobStore((s) => s.dismiss)
  const cancel = useJobStore((s) => s.cancel)
  const hydrate = useJobStore((s) => s.hydrate)
  const queryClient = useQueryClient()
  const handled = useRef<Set<string>>(new Set())

  // reconnect any video extractions that were running before a reload
  useEffect(() => {
    hydrate()
  }, [hydrate])

  // react to each terminal transition exactly once
  useEffect(() => {
    for (const id of order) {
      const j = jobs[id]
      if (!j) continue
      if (j.status === 'done') {
        const sig = `done:${id}:${j.seq}`
        if (handled.current.has(sig)) continue
        handled.current.add(sig)
        if (j.kind === 'dataset') {
          queryClient.invalidateQueries({ queryKey: ['train-datasets'] })
          notifications.show({ message: `Dataset uploaded: ${j.title}`, color: 'green' })
        } else if (j.kind === 'video') {
          queryClient.invalidateQueries({ queryKey: ['images', j.projectId] })
          queryClient.invalidateQueries({ queryKey: ['stats', j.projectId] })
          notifications.show({ message: `Frames extracted: ${j.title}`, color: 'green' })
        } else if (j.kind === 'autolabel') {
          queryClient.invalidateQueries({ queryKey: ['images', j.projectId] })
          queryClient.invalidateQueries({ queryKey: ['stats', j.projectId] })
          notifications.show({ message: `Auto-labeling done: ${j.title}`, color: 'green' })
        } else if (j.kind === 'crop') {
          queryClient.invalidateQueries({ queryKey: ['crop-runs', j.projectId] })
          notifications.show({ message: `Crop done: ${j.title} — saved to Crop Runs`, color: 'green' })
        } else if (j.kind === 'live') {
          notifications.show({ message: `Crop Draw ready: ${j.title}`, color: 'green' })
        } else {
          // export: finished export now appears in the list; download on the Exports page
          queryClient.invalidateQueries({ queryKey: ['exports', j.projectId] })
          queryClient.invalidateQueries({ queryKey: ['train-datasets'] })
          notifications.show({ message: `Export ready: ${j.title} — download on the Exports page`, color: 'green' })
        }
        // a sticky card is the handle to its own result — the user closes it
        if (!j.sticky) setTimeout(() => dismiss(id), 4000)
      } else if (j.status === 'error') {
        const sig = `error:${id}`
        if (handled.current.has(sig)) continue
        handled.current.add(sig)
        notifications.show({ message: j.error ?? 'Job failed', color: 'red' })
        if (!j.sticky) setTimeout(() => dismiss(id), 5000)
      }
    }
  }, [jobs, order, dismiss, queryClient])

  // Only guard reloads that would actually lose work: the browser owns the
  // bytes during a client upload. Pure server jobs re-attach on their own
  // (crop runs from the server's own list, the rest from localStorage), so
  // blocking those would be a lie about what a refresh costs.
  const blocking = order.some((id) => {
    const j = jobs[id]
    if (!j || j.status !== 'running') return false
    return j.kind === 'dataset' || (j.kind === 'video' && j.phaseIndex === 0)
  })
  useEffect(() => {
    if (!blocking) return
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault()
      e.returnValue = ''
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [blocking])

  const visible = order.map((id) => jobs[id]).filter(Boolean) as Job[]
  if (visible.length === 0) return null

  return (
    <div
      style={{
        position: 'fixed',
        top: 72,
        right: 16,
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        width: 320,
      }}
    >
      {visible.map((job) => (
        <JobCard
          key={job.id}
          job={job}
          onDismiss={() => dismiss(job.id)}
          onCancel={() => cancel(job.id)}
        />
      ))}
    </div>
  )
}

function JobCard({
  job,
  onDismiss,
  onCancel,
}: {
  job: Job
  onDismiss: () => void
  onCancel: () => void
}) {
  const cur = job.phases[job.phaseIndex]
  const done = job.status === 'done'
  const error = job.status === 'error'
  const running = job.status === 'running'
  const spinning = running && cur?.indeterminate

  return (
    <Paper withBorder shadow="md" radius="md" p="sm">
      <Group justify="space-between" wrap="nowrap" mb={6}>
        <Group gap="xs" wrap="nowrap" style={{ minWidth: 0 }}>
          {job.kind === 'video' ? (
            <IconMovie size={16} stroke={1.5} />
          ) : job.kind === 'autolabel' ? (
            <IconWand size={16} stroke={1.5} />
          ) : job.kind === 'export' ? (
            <IconFileExport size={16} stroke={1.5} />
          ) : job.kind === 'crop' ? (
            <IconCrop size={16} stroke={1.5} />
          ) : job.kind === 'live' ? (
            <IconBrush size={16} stroke={1.5} />
          ) : (
            <IconFileZip size={16} stroke={1.5} />
          )}
          <Text size="sm" fw={500} truncate>
            {job.title}
          </Text>
        </Group>
        {running ? (
          <CloseButton size="sm" title="Cancel" aria-label="Cancel" onClick={onCancel} />
        ) : (
          <CloseButton size="sm" title="Dismiss" aria-label="Dismiss" onClick={onDismiss} />
        )}
      </Group>
      <Progress
        value={done ? 100 : spinning ? 100 : (cur?.value ?? 0)}
        animated={spinning}
        striped={spinning}
        color={error ? 'red' : done ? 'green' : 'blue'}
      />
      <Text size="xs" c="dimmed" mt={4} lineClamp={2}>
        {error
          ? (job.error ?? 'Failed')
          : done
            ? 'Done'
            : `${job.phaseIndex + 1}/${job.phases.length} · ${cur?.label ?? ''}${
                cur?.detail ? ` — ${cur.detail}` : cur && !cur.indeterminate ? ` (${cur.value}%)` : ''
              }`}
      </Text>
      {!running && job.resultHref && (
        <Anchor component={Link} to={job.resultHref} size="xs" mt={4} display="block" onClick={onDismiss}>
          {job.resultLabel ?? 'Open result'}
        </Anchor>
      )}
    </Paper>
  )
}
