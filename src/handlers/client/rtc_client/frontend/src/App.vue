<script setup lang="ts">
import WebcamPermission from '@/components/WebcamPermission.vue'
import { antdLocale, locale } from '@/langs'
import VideoChat from '@/views/VideoChat/index.vue'
import { ConfigProvider } from 'ant-design-vue'
import { useVideoChatStore } from './store'
import { ref, onMounted } from 'vue'

const videoChatState = useVideoChatStore()
videoChatState.init()

type Theme = 'dark' | 'light' | 'system'
const theme = ref<Theme>('dark')

function getEffectiveTheme(t: Theme): 'dark' | 'light' {
  if (t !== 'system') return t
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

function applyTheme(t: Theme) {
  document.documentElement.setAttribute('data-theme', t)
  localStorage.setItem('theme', t)
  theme.value = t
  // Update 3D avatar background to match theme
  const effective = getEffectiveTheme(t)
  const bgColor = effective === 'dark' ? '0x0a0a0f' : '0xf4f4f8'
  const renderer = videoChatState.localAvatarRenderer
  if (renderer && typeof renderer.setBackgroundColor === 'function') {
    renderer.setBackgroundColor(bgColor)
  }
}

function toggleTheme() {
  const next: Record<Theme, Theme> = { dark: 'light', light: 'system', system: 'dark' }
  applyTheme(next[theme.value])
}

onMounted(() => {
  const saved = localStorage.getItem('theme') as Theme | null
  applyTheme(saved || 'dark')
})
</script>
<template>
  <ConfigProvider :locale="antdLocale[locale]">
    <div class="wrap">
      <WebcamPermission v-if="!videoChatState.webcamAccessed" />
      <VideoChat />
    </div>
    <button class="theme-toggle" @click="toggleTheme" :title="`Theme: ${theme}`">
      <!-- sun icon for light theme -->
      <svg v-if="theme === 'light'" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M2 12h2M20 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" stroke="currentColor" stroke-width="2" stroke-linecap="round" fill="none"/></svg>
      <!-- moon icon for dark theme -->
      <svg v-else-if="theme === 'dark'" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      <!-- monitor icon for system theme -->
      <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
    </button>
  </ConfigProvider>
</template>
<style lang="less" scoped>
.wrap {
  background: var(--bg-primary);
  height: calc(max(80vh, 100%));
  position: relative;
  *::-webkit-scrollbar {
    display: none;
  }
}

.theme-toggle {
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 1000;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: 1px solid var(--border-color);
  background: var(--bg-surface);
  backdrop-filter: blur(8px);
  color: var(--text-primary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  padding: 0;
  transition: all 0.2s ease;

  &:hover {
    opacity: 0.8;
    transform: scale(1.05);
  }
}
</style>
