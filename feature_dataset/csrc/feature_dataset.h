#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <torch/torch.h>
#include <zstd.h>

namespace feature_dataset {

constexpr size_t NUM_FEATURES = 24;
constexpr size_t RECORD_SIZE = 54;

#pragma pack(push, 1)
struct ReversiRecord {
    float score;
    uint16_t features[NUM_FEATURES];
    uint8_t mobility;
    uint8_t ply;
};
#pragma pack(pop)

static_assert(sizeof(ReversiRecord) == RECORD_SIZE,
              "ReversiRecord must be exactly 54 bytes");

constexpr int64_t FEATURE_CUM_OFFSETS[NUM_FEATURES] = {
    0,      6561,   13122,  19683,  26244,  32805,  39366,  45927,
    52488,  59049,  65610,  72171,  78732,  85293,  91854,  98415,
    104976, 111537, 118098, 124659, 131220, 150903, 170586, 190269};

class ZstdStreamReader {
  public:
    explicit ZstdStreamReader(const std::string &filepath);
    ~ZstdStreamReader();

    ZstdStreamReader(const ZstdStreamReader &) = delete;
    ZstdStreamReader &operator=(const ZstdStreamReader &) = delete;

    size_t read(uint8_t *buffer, size_t max_bytes);
    bool eof() const { return eof_; }

  private:
    FILE *file_ = nullptr;
    ZSTD_DStream *dstream_ = nullptr;
    std::vector<uint8_t> input_buffer_;
    ZSTD_inBuffer zstd_in_ = {nullptr, 0, 0};
    bool eof_ = false;
};

/// Thread-safe bounded queue for producer-consumer pattern.
/// Blocks on push when full and on pop when empty.
template <typename T> class BoundedQueue {
  public:
    explicit BoundedQueue(size_t max_size) : max_size_(max_size) {
        if (max_size == 0) {
            throw std::invalid_argument("max_size must be at least 1");
        }
    }

    // Push an item, blocking if queue is full.
    // Returns true if item was enqueued, false if queue was closed.
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock,
                       [this] { return queue_.size() < max_size_ || closed_; });
        if (closed_) {
            return false;
        }
        queue_.push(std::move(item));
        lock.unlock();
        not_empty_.notify_one();
        return true;
    }

    // Pop an item, returns nullopt if queue is closed and empty
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || closed_; });
        if (queue_.empty()) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        not_full_.notify_one();
        return item;
    }

    // Close the queue, waking up all waiting threads
    void close() {
        std::unique_lock<std::mutex> lock(mutex_);
        closed_ = true;
        lock.unlock();
        not_full_.notify_all();
        not_empty_.notify_all();
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

  private:
    std::queue<T> queue_;
    const size_t max_size_;
    mutable std::mutex mutex_;  // mutable for const method is_closed()
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    bool closed_ = false;
};

using BatchTuple =
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

// Per-worker context for independent decompression
struct WorkerContext {
    std::unique_ptr<ZstdStreamReader> reader;
    std::vector<uint8_t> buffer;
    size_t buffer_offset = 0;

    WorkerContext() = default;
    WorkerContext(const WorkerContext &) = delete;
    WorkerContext &operator=(const WorkerContext &) = delete;
    WorkerContext(WorkerContext &&) = default;
    WorkerContext &operator=(WorkerContext &&) = default;
};

class FeatureDatasetReader {
  public:
    FeatureDatasetReader(std::vector<std::string> filepaths, size_t batch_size,
                         double file_usage_ratio, bool shuffle,
                         size_t num_workers = 2, size_t prefetch_depth = 4);
    ~FeatureDatasetReader();

    // Non-copyable, non-movable due to threads
    FeatureDatasetReader(const FeatureDatasetReader &) = delete;
    FeatureDatasetReader &operator=(const FeatureDatasetReader &) = delete;
    FeatureDatasetReader(FeatureDatasetReader &&) = delete;
    FeatureDatasetReader &operator=(FeatureDatasetReader &&) = delete;

    std::optional<BatchTuple> next();
    void set_worker_info(int worker_id, int num_workers);

  private:
    void process_records(const uint8_t *data, size_t num_records,
                         float *scores_out, int64_t *features_out,
                         int64_t *mobility_out, int64_t *ply_out);

    // Worker thread methods
    void start_workers();
    void stop_workers();
    void worker_loop(size_t worker_id);
    std::optional<BatchTuple> produce_batch(WorkerContext &ctx);
    std::optional<std::string> get_next_file();
    bool fill_worker_buffer(WorkerContext &ctx);

    // Configuration
    std::vector<std::string> filepaths_;
    size_t batch_size_;
    double file_usage_ratio_;
    bool shuffle_;
    size_t num_decompress_workers_;
    size_t prefetch_depth_;

    // DataLoader worker info
    int worker_id_ = 0;
    int num_dataloader_workers_ = 1;

    // Threading state
    std::vector<std::thread> workers_;
    std::unique_ptr<BoundedQueue<BatchTuple>> batch_queue_;
    std::queue<std::string> file_queue_;
    std::mutex file_queue_mutex_;
    std::atomic<size_t> active_workers_{0};
    std::atomic<bool> shutdown_{false};
    bool workers_started_ = false;

    std::mt19937 rng_;
};

} // namespace feature_dataset
