// Android routes all page audio through its phone-call path while an
// echo-cancelled mic is open, and the avatar voice crackles there. Phones
// therefore open the mic without echo cancellation and mute it while the
// avatar talks so it can't hear itself. ?aec=1 / ?aec=0 force either mode.
const aecParam = new URLSearchParams(window.location.search).get('aec')

export const noEchoCancel =
  aecParam === '0' ||
  (aecParam !== '1' && (window.matchMedia?.('(pointer: coarse)').matches ?? false))
