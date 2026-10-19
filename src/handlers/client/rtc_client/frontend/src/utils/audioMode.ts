// Keep WebRTC echo cancellation enabled by default on both desktop and mobile
// so the microphone stays open while the avatar speaks and vocal barge-in works.
// Pass ?aec=0 in the URL only if a device explicitly needs half-duplex mic muting.
const aecParam = new URLSearchParams(window.location.search).get('aec')

export const isCoarsePointer =
  window.matchMedia?.('(pointer: coarse)').matches ?? false

export const noEchoCancel = aecParam === '0'

