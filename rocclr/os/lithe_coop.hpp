// Lithe co-scheduling for rocclr (lithe-docker docs/PLAN-gpu-and-network-locality.md 4.2, 4.2a).
// Weak references only: in a process without parlib/liblithe everything reports "not cooperative" and
// rocclr behaves as upstream. In a lithified process, a caller that is a Lithe context and may yield
// (parlib_can_cooperatively_yield(), GAPS B13) yields its hart instead of sched_yield / sem_wait, and
// amd::Thread is created as a Lithe progress context. pthread_create is not intercepted.
//   ROCCLR_LITHE=0 disables at run time; ROCCLR_LITHE_THREADS=0 keeps pthreads for amd::Thread.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <sched.h>

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
// Lithe locks only (ROCCLR_LITHE_LOCKS=0 keeps blocking locks while waits stay cooperative; ablation).
inline bool LitheLockCoop() {
  static int on = -1;
  if (on < 0) {
    const char* e = getenv("ROCCLR_LITHE_LOCKS");
    on = (e && *e) ? (atoi(e) != 0) : 1;
  }
  return on && LitheCoop();
}
inline bool LitheThreads() {
  const char* e = getenv("ROCCLR_LITHE_THREADS");
  return !(e && *e && atoi(e) == 0) && lithe_fork_join_context_create_progress != nullptr;
}
// Lithe locks: drop-ins for std::mutex / std::shared_mutex under lock_guard, unique_lock and
// shared_lock. A Lithe context that may yield never blocks its hart on them (try, yield, retry);
// every other caller locks normally.
class LitheMutex {
 public:
  void lock() {
    if (LitheLockCoop()) { while (!m_.try_lock()) LitheYield(); return; }
    m_.lock();
  }
  bool try_lock() { return m_.try_lock(); }
  void unlock() { m_.unlock(); }
 private:
  std::mutex m_;
};
class LitheSharedMutex {
 public:
  void lock() {
    if (LitheLockCoop()) { while (!m_.try_lock()) LitheYield(); return; }
    m_.lock();
  }
  bool try_lock() { return m_.try_lock(); }
  void unlock() { m_.unlock(); }
  void lock_shared() {
    if (LitheLockCoop()) { while (!m_.try_lock_shared()) LitheYield(); return; }
    m_.lock_shared();
  }
  bool try_lock_shared() { return m_.try_lock_shared(); }
  void unlock_shared() { m_.unlock_shared(); }
 private:
  std::shared_mutex m_;
};
// Lithe once: replaces std::call_once where the initializer can wait (HIP's own init, heap setup).
// std::call_once parks later callers on a futex, holding their hart while the initializer's context may
// itself be parked; here later callers yield (Lithe context) or sched_yield (kernel thread).
struct LitheOnceFlag {
  std::atomic<int> state{0};  // 0 not run, 1 running, 2 done
};
template <class F> inline void LitheCallOnce(LitheOnceFlag& flag, F&& f) {
  if (flag.state.load(std::memory_order_acquire) == 2) return;
  int expect = 0;
  if (flag.state.compare_exchange_strong(expect, 1, std::memory_order_acq_rel)) {
    f();
    flag.state.store(2, std::memory_order_release);
    return;
  }
  while (flag.state.load(std::memory_order_acquire) != 2) {
    if (LitheLockCoop()) LitheYield();
    else sched_yield();
  }
}
}  // namespace amd
