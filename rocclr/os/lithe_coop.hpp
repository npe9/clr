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
#include <stdint.h>
#include <time.h>

extern "C" {
int parlib_can_cooperatively_yield(void) __attribute__((weak));
void lithe_context_yield(void) __attribute__((weak));
void* lithe_sched_current(void) __attribute__((weak));
void* lithe_fork_join_context_create_progress(size_t stack_size, void (*start)(void*), void* arg)
    __attribute__((weak));// Progress contexts live as long as the runtime: keep them out of join_all (GAPS B26).
void lithe_fork_join_context_set_daemon(void* ctx) __attribute__((weak));
int futex_wait(int* uaddr, int val) __attribute__((weak));
int futex_wait_timeout(int* uaddr, int val, const struct timespec* rel) __attribute__((weak));
int futex_wake_all(int* uaddr) __attribute__((weak));
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
// De-spin (2026-10-24, lithe-docker .build_logs/spin_census/DESPIN_RULES.md). ROCCLR_LITHE_NOSPIN=1
// (default 0): the Lithe "try, yield, retry" waits (LitheMutex, LitheSharedMutex, LitheCallOnce,
// Semaphore::wait/timedWait, Monitor's yield phase) park on a Lithe futex word instead; the release
// side wakes. A yield-waiter stays runnable and round-robins for harts against real work, and a yield
// never requests a hart (GAPS B56). 64 hashed buckets {waiters, seq}: the waiter counts in, samples
// seq, retries, parks; the releaser bumps seq and wakes only if someone counts (one load otherwise).
bool LitheNoSpin();
struct LitheParkBucket {
  std::atomic<int> waiters;
  int seq;
  char pad[56];
};
extern LitheParkBucket g_lithe_park[64];
inline LitheParkBucket& LitheParkOf(const void* a) {
  return g_lithe_park[((reinterpret_cast<uintptr_t>(a) >> 4) * 0x9E3779B97F4A7C15ull) >> 58];
}
inline int LitheParkPrep(const void* a) {
  LitheParkBucket& b = LitheParkOf(a);
  b.waiters.fetch_add(1, std::memory_order_seq_cst);
  return __atomic_load_n(&b.seq, __ATOMIC_SEQ_CST);
}
inline void LitheParkCancel(const void* a) {
  LitheParkOf(a).waiters.fetch_sub(1, std::memory_order_seq_cst);
}
inline void LitheParkWait(const void* a, int seq, long long rel_ns = -1) {
  LitheParkBucket& b = LitheParkOf(a);
  if (rel_ns < 0) {
    futex_wait(&b.seq, seq);
  } else {
    struct timespec ts = {time_t(rel_ns / 1000000000ll), long(rel_ns % 1000000000ll)};
    futex_wait_timeout(&b.seq, seq, &ts);
  }
  b.waiters.fetch_sub(1, std::memory_order_seq_cst);
}
inline void LitheParkWake(const void* a) {
  LitheParkBucket& b = LitheParkOf(a);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (b.waiters.load(std::memory_order_seq_cst) == 0) return;
  __atomic_fetch_add(&b.seq, 1, __ATOMIC_SEQ_CST);
  futex_wake_all(&b.seq);
}
template <class F> inline void LitheWaitUntil(const void* addr, F&& done) {
  while (!done()) {
    if (!LitheNoSpin()) { LitheYield(); continue; }
    int s = LitheParkPrep(addr);
    if (done()) { LitheParkCancel(addr); return; }
    LitheParkWait(addr, s);
  }
}
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
    if (LitheLockCoop()) { LitheWaitUntil(this, [this] { return m_.try_lock(); }); return; }
    m_.lock();
  }
  bool try_lock() { return m_.try_lock(); }
  void unlock() { m_.unlock(); if (LitheNoSpin()) LitheParkWake(this); }
 private:
  std::mutex m_;
};
class LitheSharedMutex {
 public:
  void lock() {
    if (LitheLockCoop()) { LitheWaitUntil(this, [this] { return m_.try_lock(); }); return; }
    m_.lock();
  }
  bool try_lock() { return m_.try_lock(); }
  void unlock() { m_.unlock(); if (LitheNoSpin()) LitheParkWake(this); }
  void lock_shared() {
    if (LitheLockCoop()) { LitheWaitUntil(this, [this] { return m_.try_lock_shared(); }); return; }
    m_.lock_shared();
  }
  bool try_lock_shared() { return m_.try_lock_shared(); }
  void unlock_shared() { m_.unlock_shared(); if (LitheNoSpin()) LitheParkWake(this); }
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
    if (LitheNoSpin()) LitheParkWake(&flag);
    return;
  }
  while (flag.state.load(std::memory_order_acquire) != 2) {
    if (LitheLockCoop()) {
      LitheWaitUntil(&flag, [&flag] { return flag.state.load(std::memory_order_acquire) == 2; });
      return;
    }
    sched_yield();
  }
}
}  // namespace amd
