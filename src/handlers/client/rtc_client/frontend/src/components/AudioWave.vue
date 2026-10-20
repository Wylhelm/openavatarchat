<script setup lang="ts">
import { StreamState } from '@/interface/voiceChat';
import { computed, onUnmounted, watch } from 'vue';

const props = withDefaults(
  defineProps<{
    streamState: StreamState;
    // onStartChat: any;
    audioSourceCallback: () => MediaStream | null;
    numBars?: number;
    icon?: string;
    iconButtonColor?: string;
    pulseColor?: string;
    waveColor?: string;
    pulseScale?: number;
  }>(),
  {
    streamState: StreamState.closed,
    numBars: 16,
    iconButtonColor: 'var(--color-accent)',
    pulseColor: 'var(--color-accent)',
    waveColor: 'var(--color-accent)',
    pulseScale: 1,
  },
);
const emit = defineEmits([]);

let animationId: number | null = null;
let phase = 0;

const containerWidth = computed(() => {
  return props.icon ? '128px' : `calc((var(--boxSize) + var(--gutter)) * ${props.numBars} + 80px)`;
});

function stopVisualization() {
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
}

watch(
  () => props.streamState,
  () => {
    stopVisualization();
    if (props.streamState === 'open') {
      updateVisualization();
    }
  },
  { immediate: true },
);

onUnmounted(() => {
  stopVisualization();
});

// Animate wave bars without creating a second AudioContext or calling
// createMediaStreamSource(localStream), which causes iOS WebKit CoreAudio
// clock-domain conflicts and crackling when WebRTC mic capture is active.
function updateVisualization() {
  phase += 0.12;
  const bars = document.querySelectorAll('.gradio-webrtc-waveContainer .gradio-webrtc-box');
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i] as HTMLDivElement;
    const wave = 0.22 + 0.18 * Math.sin(phase + i * 0.45);
    bar.style.transform = `scaleY(${wave.toFixed(2)})`;
    bar.style.background = props.waveColor;
    bar.style.opacity = '0.5';
  }
  animationId = requestAnimationFrame(updateVisualization);
}
</script>

<template>
  <div class="gradio-webrtc-waveContainer">
    <div class="gradio-webrtc-boxContainer" :style="{ width: containerWidth }">
      <template v-for="(_, index) in Array(numBars / 2)" :key="index">
        <div class="gradio-webrtc-box"></div>
      </template>
      <div class="split-container"></div>
      <template v-for="(_, index) in Array(numBars / 2)" :key="index">
        <div class="gradio-webrtc-box"></div>
      </template>
    </div>
  </div>
</template>

<style scoped lang="less">
.gradio-webrtc-waveContainer {
  position: relative;
  display: flex;
  min-height: 100px;
  max-height: 128px;
  justify-content: center;
  align-items: center;
}

.gradio-webrtc-boxContainer {
  display: flex;
  justify-content: space-between;
  height: 64px;
  --boxSize: 4px;
  --gutter: 4px;
}

.split-container {
  width: 80px;
}

.gradio-webrtc-box {
  height: 100%;
  width: var(--boxSize);
  background: var(--color-accent);
  border-radius: 8px;
  transition: transform 0.05s ease;
}

.gradio-webrtc-icon-container {
  position: relative;
  width: 128px;
  height: 128px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.gradio-webrtc-icon {
  position: relative;
  width: 48px;
  height: 48px;
  border-radius: 50%;
  transition: transform 0.1s ease;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 2;
}

.icon-image {
  width: 32px;
  height: 32px;
  object-fit: contain;
  filter: brightness(0) invert(1);
}

.pulse-ring {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 48px;
  height: 48px;
  border-radius: 50%;
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  opacity: 0.5;
}

@keyframes pulse {
  0% {
    transform: translate(-50%, -50%) scale(1);
    opacity: 0.5;
  }

  100% {
    transform: translate(-50%, -50%) scale(var(--max-scale, 3));
    opacity: 0;
  }
}

.dots {
  display: flex;
  gap: 8px;
  align-items: center;
  height: 64px;
}

.dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  opacity: 0.5;
  animation: pulse 1.5s infinite;
}

.dot:nth-child(2) {
  animation-delay: 0.2s;
}

.dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 0.4;
    transform: scale(1);
  }

  50% {
    opacity: 1;
    transform: scale(1.1);
  }
}
</style>
