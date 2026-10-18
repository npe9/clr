// Lithe co-scheduling for rocclr (lithe-docker docs/PLAN-gpu-and-network-locality.md 4.2, 4.2a).
// Weak references only: in a process without parlib/liblithe everything reports "not cooperative" and
// rocclr behaves as upstream. In a lithified process, a caller that is a Lithe context and may yield
// (parlib_can_cooperatively_yield(), GAPS B13) yields its hart instead of sched_yield / sem_wait, and
// amd::Thread is created as a Lithe progress context. pthread_create is not intercepted.
//   ROCCLR_LITHE=0 disables at run time; ROCCLR_LITHE_THREADS=0 keeps pthreads for amd::Thread.
#pragma once
#include <cstddef>
#include <cstdlib>

extern "C" {
int parlib_can_cooperatively_yield(void) __attribute__((weak));
void lithe_context_yield(void) __attribute__((weak));
void* lithe_sched_current(void) __attribute__((weak));
void* lithe_fork_join_context_create_progress(size_t stack_size, void (*start)(void*), void* arg)
    __attribute__((weak));
}

namespace amd {
inline bool LitheEnabled() {
  static int on = -1;
  if (on < 0) {
    const char* e = getenv("ROCCLR_LITHE");
    on = (e && *e) ? (atoi(e) != 0) : 1;
  }
  return on && parlib_can_cooperatively_yield && lithe_context_yield && lithe_sched_current;
}
inline bool LitheCoop() {
  return LitheEnabled() && lithe_sched_current() != nullptr && parlib_can_cooperatively_yield();
}
inline void LitheYield() { lithe_context_yield(); }
inline bool LitheThreads() {
  const char* e = getenv("ROCCLR_LITHE_THREADS");
  return !(e && *e && atoi(e) == 0) && lithe_fork_join_context_create_progress != nullptr;
}
}  // namespace amd
