#include "dataset.h"
#include "bitboard.h"
#include "patterns.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dataset {

// BinStreamReader implementation

BinStreamReader::BinStreamReader(const std::string &filepath) {
    file_ = fopen(filepath.c_str(), "rb");
    if (!file_) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
}

BinStreamReader::~BinStreamReader() {
    if (file_) {
        fclose(file_);
    }
}

size_t BinStreamReader::read(GameRecord *buffer, size_t max_records) {
    if (eof_) {
        return 0;
    }

    size_t records_read = fread(buffer, sizeof(GameRecord), max_records, file_);
    if (records_read < max_records) {
        if (feof(file_)) {
            eof_ = true;
        } else if (ferror(file_)) {
            throw std::runtime_error("Error reading from file");
        }
    }
    return records_read;
}

// BinDatasetReader implementation

BinDatasetReader::BinDatasetReader(std::vector<std::string> filepaths,
                                   size_t batch_size, double file_usage_ratio,
                                   bool shuffle, size_t num_workers,
                                   size_t prefetch_depth, uint8_t ply_min,
                                   uint8_t ply_max, uint64_t seed)
    : filepaths_(std::move(filepaths)), batch_size_(batch_size),
      file_usage_ratio_(file_usage_ratio), shuffle_(shuffle),
      num_decompress_workers_(num_workers == 0 ? default_decompress_workers()
                                               : num_workers),
      prefetch_depth_(prefetch_depth), ply_min_(ply_min), ply_max_(ply_max),
      seed_(seed == 0 ? std::random_device{}() : seed),
      rng_(seed_) {

    if (file_usage_ratio_ <= 0.0 || file_usage_ratio_ > 1.0) {
        throw std::invalid_argument(
            "file_usage_ratio must be in the range (0.0, 1.0]");
    }

    if (prefetch_depth_ == 0) {
        throw std::invalid_argument("prefetch_depth must be at least 1");
    }

    if (ply_min_ > ply_max_) {
        throw std::invalid_argument("ply_min must be <= ply_max");
    }
}

BinDatasetReader::~BinDatasetReader() { stop_workers(); }

void BinDatasetReader::set_worker_info(int worker_id, int num_workers) {
    if (workers_started_) {
        throw std::logic_error(
            "set_worker_info must be called before iteration starts");
    }
    worker_id_ = worker_id;
    num_dataloader_workers_ = num_workers;
}

void BinDatasetReader::process_game_record(const GameRecord &record,
                                           std::mt19937 &rng, float *score_out,
                                           int64_t *features_out,
                                           int64_t *mobility_out,
                                           int64_t *ply_out) {
    // Apply score calculation logic from feature.rs
    float score = record.score;
    if (record.ply <= 1) {
        score = 0.0f;
    } else if (!record.is_random) {
        score = static_cast<float>(record.game_score);
    }
    *score_out = score;

    // Apply random symmetry transformation
    uint64_t player = record.player;
    uint64_t opponent = record.opponent;
    int symmetry = static_cast<int>(rng() % 8);
    apply_symmetry(player, opponent, symmetry);

    // Extract features using base-3 encoding
    uint16_t features[NUM_FEATURES];
    extract_features(player, opponent, features);

    for (size_t i = 0; i < NUM_FEATURES; ++i) {
        features_out[i] = static_cast<int64_t>(features[i]);
    }

    // Calculate mobility (after symmetry transformation)
    *mobility_out = static_cast<int64_t>(count_mobility(player, opponent));

    // Store ply
    *ply_out = static_cast<int64_t>(record.ply);
}

std::optional<std::string> BinDatasetReader::get_next_file() {
    std::lock_guard<std::mutex> lock(file_queue_mutex_);
    if (file_queue_.empty()) {
        return std::nullopt;
    }
    std::string filepath = std::move(file_queue_.front());
    file_queue_.pop();
    return filepath;
}

bool BinDatasetReader::fill_worker_buffer(BinWorkerContext &ctx) {
    constexpr size_t CHUNK_RECORDS = 32768; // Read 32K records at a time
    const size_t required_records = batch_size_;

    // Try to get a new file if we don't have a reader
    while (!ctx.reader) {
        auto filepath = get_next_file();
        if (!filepath) {
            return false;
        }
        try {
            ctx.reader = std::make_unique<BinStreamReader>(*filepath);
        } catch (const std::exception &e) {
            fprintf(stderr, "Warning: Failed to open %s: %s\n",
                    filepath->c_str(), e.what());
            continue;
        }
    }

    // Compact buffer when data_start is past halfway
    if (ctx.data_start > ctx.buffer.size() / 2) {
        ctx.compact();
    }

    // Fill buffer until we have enough for a batch
    while (ctx.available_records() < required_records) {
        size_t needed_capacity = ctx.data_end + CHUNK_RECORDS;
        if (ctx.buffer.size() < needed_capacity) {
            ctx.buffer.resize(needed_capacity);
        }

        size_t records_read;
        try {
            records_read = ctx.reader->read(ctx.buffer.data() + ctx.data_end,
                                            CHUNK_RECORDS);
        } catch (const std::exception &e) {
            fprintf(stderr, "Warning: Read error, skipping file: %s\n",
                    e.what());
            ctx.reader.reset();
            auto filepath = get_next_file();
            if (!filepath) {
                break;
            }
            try {
                ctx.reader = std::make_unique<BinStreamReader>(*filepath);
            } catch (const std::exception &e2) {
                fprintf(stderr, "Warning: Failed to open %s: %s\n",
                        filepath->c_str(), e2.what());
                ctx.reader.reset();
                continue;
            }
            continue;
        }
        ctx.data_end += records_read;

        if (records_read == 0 || ctx.reader->eof()) {
            ctx.reader.reset();
            auto filepath = get_next_file();
            if (!filepath) {
                break;
            }
            try {
                ctx.reader = std::make_unique<BinStreamReader>(*filepath);
            } catch (const std::exception &e) {
                fprintf(stderr, "Warning: Failed to open %s: %s\n",
                        filepath->c_str(), e.what());
                continue;
            }
        }
    }

    return ctx.available_records() >= required_records;
}

std::optional<BatchTuple> BinDatasetReader::produce_batch(BinWorkerContext &ctx) {
    if (!fill_worker_buffer(ctx)) {
        return std::nullopt;
    }

    if (ctx.available_records() < batch_size_) {
        return std::nullopt;
    }

    // Allocate output tensors
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32);
    auto options_int64 = torch::TensorOptions().dtype(torch::kInt64);

    torch::Tensor scores =
        torch::empty({static_cast<int64_t>(batch_size_), 1}, options_float);
    torch::Tensor features = torch::empty(
        {static_cast<int64_t>(batch_size_), static_cast<int64_t>(NUM_FEATURES)},
        options_int64);
    torch::Tensor mobility =
        torch::empty({static_cast<int64_t>(batch_size_), 1}, options_int64);
    torch::Tensor ply =
        torch::empty({static_cast<int64_t>(batch_size_), 1}, options_int64);

    float *scores_ptr = scores.data_ptr<float>();
    int64_t *features_ptr = features.data_ptr<int64_t>();
    int64_t *mobility_ptr = mobility.data_ptr<int64_t>();
    int64_t *ply_ptr = ply.data_ptr<int64_t>();

    // Process records with ply filtering
    GameRecord *records = ctx.data_ptr();
    size_t out_idx = 0;
    size_t in_idx = 0;

    while (out_idx < batch_size_ && in_idx < ctx.available_records()) {
        const GameRecord &record = records[in_idx++];

        // Apply ply filter
        if (record.ply < ply_min_ || record.ply > ply_max_) {
            continue;
        }

        process_game_record(record, ctx.rng,
                            scores_ptr + out_idx,
                            features_ptr + out_idx * NUM_FEATURES,
                            mobility_ptr + out_idx,
                            ply_ptr + out_idx);
        ++out_idx;
    }

    ctx.consume(in_idx);

    // If we didn't fill the batch due to ply filtering, try to get more data
    if (out_idx < batch_size_) {
        // Recursively try to fill more (simplified approach: just return partial)
        // For production, we would loop until batch is full or no more data
        if (out_idx == 0) {
            return std::nullopt;
        }
        // Resize tensors to actual size
        scores = scores.slice(0, 0, static_cast<int64_t>(out_idx));
        features = features.slice(0, 0, static_cast<int64_t>(out_idx));
        mobility = mobility.slice(0, 0, static_cast<int64_t>(out_idx));
        ply = ply.slice(0, 0, static_cast<int64_t>(out_idx));
    }

    return std::make_tuple(scores, features, mobility, ply);
}

void BinDatasetReader::worker_loop(size_t worker_id) {
    active_workers_++;

    try {
        // Each worker gets a unique seed derived from the base seed
        BinWorkerContext ctx(seed_ + worker_id);
        constexpr size_t CHUNK_RECORDS = 32768;
        ctx.buffer.resize(batch_size_ * 2 + CHUNK_RECORDS);

        while (!shutdown_) {
            auto batch = produce_batch(ctx);
            if (!batch) {
                break;
            }
            if (shutdown_) {
                break;
            }
            if (!batch_queue_->push(std::move(*batch))) {
                break;
            }
        }
    } catch (...) {
        // Store the first worker exception for propagation to Python
        std::lock_guard<std::mutex> lock(exception_mutex_);
        if (!worker_exception_) {
            worker_exception_ = std::current_exception();
        }
    }

    if (active_workers_.fetch_sub(1) == 1) {
        batch_queue_->close();
    }
}

void BinDatasetReader::start_workers() {
    if (workers_started_) {
        return;
    }

    if (shuffle_) {
        std::shuffle(filepaths_.begin(), filepaths_.end(), rng_);
    }

    // Select files for this DataLoader worker
    std::vector<std::string> worker_files;
    for (size_t i = static_cast<size_t>(worker_id_); i < filepaths_.size();
         i += static_cast<size_t>(num_dataloader_workers_)) {
        worker_files.push_back(filepaths_[i]);
    }
    filepaths_ = std::move(worker_files);

    // Apply file_usage_ratio
    size_t n_use = std::max(
        static_cast<size_t>(1),
        static_cast<size_t>(std::round(static_cast<double>(filepaths_.size()) *
                                       file_usage_ratio_)));
    filepaths_.resize(n_use);

    // Check if we have any files to process
    if (filepaths_.empty()) {
        throw std::runtime_error(
            "No data files available for this worker after file distribution");
    }

    // Populate file queue
    for (const auto &filepath : filepaths_) {
        file_queue_.push(filepath);
    }

    batch_queue_ = std::make_unique<BoundedQueue<BatchTuple>>(prefetch_depth_);

    shutdown_ = false;
    active_workers_ = 0;

    try {
        for (size_t i = 0; i < num_decompress_workers_; ++i) {
            workers_.emplace_back(&BinDatasetReader::worker_loop, this, i);
        }
    } catch (...) {
        shutdown_ = true;
        batch_queue_->close();
        for (auto &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
        throw;
    }

    workers_started_ = true;
}

void BinDatasetReader::stop_workers() {
    if (!workers_started_) {
        return;
    }

    shutdown_ = true;
    if (batch_queue_) {
        batch_queue_->close();
    }

    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
    workers_started_ = false;
}

std::optional<BatchTuple> BinDatasetReader::next() {
    if (!workers_started_) {
        start_workers();
    }

    auto result = batch_queue_->pop();

    // Check for worker exceptions and propagate to Python
    if (!result) {
        std::lock_guard<std::mutex> lock(exception_mutex_);
        if (worker_exception_) {
            std::rethrow_exception(worker_exception_);
        }
    }

    return result;
}

} // namespace dataset
