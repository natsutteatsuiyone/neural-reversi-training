#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

namespace dataset {

constexpr size_t NUM_FEATURES = 24;
constexpr size_t BIN_RECORD_SIZE = 24;

// GameRecord: raw binary record from .bin files (24 bytes)
#pragma pack(push, 1)
struct GameRecord {
    uint64_t player;     // 8 bytes: player's bitboard
    uint64_t opponent;   // 8 bytes: opponent's bitboard
    float score;         // 4 bytes: evaluation score
    int8_t game_score;   // 1 byte: final game score
    uint8_t ply;         // 1 byte: move number
    uint8_t is_random;   // 1 byte: random move flag
    uint8_t move;        // 1 byte: move (unused)
};
#pragma pack(pop)

static_assert(sizeof(GameRecord) == BIN_RECORD_SIZE,
              "GameRecord must be exactly 24 bytes");

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

/// Returns a reasonable default number of decompression workers.
/// Uses hardware concurrency, capped at a reasonable maximum.
inline size_t default_decompress_workers() {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 2;  // Fallback if detection fails
    return std::min(static_cast<size_t>(hw), static_cast<size_t>(4));
}

// BinStreamReader: reads raw binary .bin files
class BinStreamReader {
  public:
    explicit BinStreamReader(const std::string &filepath);
    ~BinStreamReader();

    BinStreamReader(const BinStreamReader &) = delete;
    BinStreamReader &operator=(const BinStreamReader &) = delete;

    size_t read(GameRecord *buffer, size_t max_records);
    bool eof() const { return eof_; }

  private:
    FILE *file_ = nullptr;
    bool eof_ = false;
};

// Per-worker context for BinDatasetReader
struct BinWorkerContext {
    std::unique_ptr<BinStreamReader> reader;
    std::vector<GameRecord> buffer;
    size_t data_start = 0;
    size_t data_end = 0;
    std::mt19937 rng;  // Per-worker RNG for symmetry selection

    explicit BinWorkerContext(uint64_t seed) : rng(seed) {}
    BinWorkerContext(const BinWorkerContext &) = delete;
    BinWorkerContext &operator=(const BinWorkerContext &) = delete;
    BinWorkerContext(BinWorkerContext &&) = default;
    BinWorkerContext &operator=(BinWorkerContext &&) = default;

    size_t available_records() const { return data_end - data_start; }
    GameRecord* data_ptr() { return buffer.data() + data_start; }

    void compact() {
        if (data_start > 0 && data_end > data_start) {
            size_t len = data_end - data_start;
            std::memmove(buffer.data(), buffer.data() + data_start,
                         len * sizeof(GameRecord));
            data_start = 0;
            data_end = len;
        } else if (data_start == data_end) {
            data_start = 0;
            data_end = 0;
        }
    }

    void consume(size_t records) {
        assert(records <= available_records() &&
               "consume() called with more records than available");
        data_start += records;
    }
};

// BinDatasetReader: reads .bin files and generates features in real-time
class BinDatasetReader {
  public:
    BinDatasetReader(std::vector<std::string> filepaths, size_t batch_size,
                     double file_usage_ratio, bool shuffle,
                     size_t num_workers = 0, size_t prefetch_depth = 4,
                     uint8_t ply_min = 0, uint8_t ply_max = 59,
                     uint64_t seed = 0);
    ~BinDatasetReader();

    BinDatasetReader(const BinDatasetReader &) = delete;
    BinDatasetReader &operator=(const BinDatasetReader &) = delete;
    BinDatasetReader(BinDatasetReader &&) = delete;
    BinDatasetReader &operator=(BinDatasetReader &&) = delete;

    std::optional<BatchTuple> next();
    void set_worker_info(int worker_id, int num_workers);

  private:
    void process_game_record(const GameRecord &record, std::mt19937 &rng,
                             float *score_out, int64_t *features_out,
                             int64_t *mobility_out, int64_t *ply_out);

    void start_workers();
    void stop_workers();
    void worker_loop(size_t worker_id);
    std::optional<BatchTuple> produce_batch(BinWorkerContext &ctx);
    std::optional<std::string> get_next_file();
    bool fill_worker_buffer(BinWorkerContext &ctx);

    // Configuration
    std::vector<std::string> filepaths_;
    size_t batch_size_;
    double file_usage_ratio_;
    bool shuffle_;
    size_t num_decompress_workers_;
    size_t prefetch_depth_;
    uint8_t ply_min_;
    uint8_t ply_max_;
    uint64_t seed_;

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

    // Worker exception handling
    std::mutex exception_mutex_;
    std::exception_ptr worker_exception_;

    std::mt19937 rng_;
};

} // namespace dataset
