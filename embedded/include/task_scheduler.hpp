#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

enum class Priority : int { HIGH = 0, MED = 1, LOW = 2 };

struct Task {
    Priority               priority;
    uint64_t               deadline_ns;   // monotonic ns; earlier = more urgent
    std::function<void()>  fn;
    uint64_t               enqueue_ns;    // for WCET tracking

    bool operator>(const Task& o) const {
        if (priority != o.priority)
            return static_cast<int>(priority) > static_cast<int>(o.priority);
        return deadline_ns > o.deadline_ns;
    }
};

// Worst-case execution time statistics per priority level.
struct WcetStats {
    uint64_t count     = 0;
    uint64_t sum_ns    = 0;
    uint64_t max_ns    = 0;
    uint64_t p99_ns    = 0;  // approximate: updated via running sort on recent window
};

class TaskScheduler {
public:
    explicit TaskScheduler(std::size_t n_workers = 3)
        : running_(true) {
        stats_.resize(3);
        for (std::size_t i = 0; i < n_workers; ++i)
            workers_.emplace_back(&TaskScheduler::worker_loop, this);
    }

    ~TaskScheduler() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            running_ = false;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

    void submit(Priority p, uint64_t deadline_ns, std::function<void()> fn) {
        uint64_t now = now_ns();
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push({p, deadline_ns, std::move(fn), now});
        }
        cv_.notify_one();
    }

    // Convenience: submit with implicit deadline = now + budget_ns
    void submit(Priority p, std::function<void()> fn, uint64_t budget_ns = 0) {
        submit(p, now_ns() + budget_ns, std::move(fn));
    }

    const WcetStats& stats(Priority p) const {
        return stats_[static_cast<int>(p)];
    }

    static uint64_t now_ns() {
        using namespace std::chrono;
        return static_cast<uint64_t>(
            duration_cast<nanoseconds>(
                steady_clock::now().time_since_epoch()).count());
    }

private:
    void worker_loop() {
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return !queue_.empty() || !running_; });
                if (!running_ && queue_.empty()) return;
                task = queue_.top();
                queue_.pop();
            }

            uint64_t start = now_ns();
            task.fn();
            uint64_t elapsed = now_ns() - start;

            auto& s = stats_[static_cast<int>(task.priority)];
            ++s.count;
            s.sum_ns += elapsed;
            if (elapsed > s.max_ns) s.max_ns = elapsed;
        }
    }

    std::priority_queue<Task, std::vector<Task>, std::greater<Task>> queue_;
    std::mutex              mu_;
    std::condition_variable cv_;
    std::atomic<bool>       running_;
    std::vector<std::thread> workers_;
    std::vector<WcetStats>  stats_;
};
