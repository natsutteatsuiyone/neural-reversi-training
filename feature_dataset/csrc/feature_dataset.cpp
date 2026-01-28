#include "feature_dataset.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace feature_dataset {

// ZstdStreamReader implementation

ZstdStreamReader::ZstdStreamReader(const std::string &filepath)
    : input_buffer_(ZSTD_DStreamInSize()) {
    file_ = fopen(filepath.c_str(), "rb");
    if (!file_) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    dstream_ = ZSTD_createDStream();
    if (!dstream_) {
        fclose(file_);
        throw std::runtime_error("Failed to create ZSTD decompression stream");
    }

    size_t init_result = ZSTD_initDStream(dstream_);
    if (ZSTD_isError(init_result)) {
        ZSTD_freeDStream(dstream_);
        fclose(file_);
        throw std::runtime_error("Failed to initialize ZSTD stream: " +
                                 std::string(ZSTD_getErrorName(init_result)));
    }
}

ZstdStreamReader::~ZstdStreamReader() {
    if (dstream_) {
        ZSTD_freeDStream(dstream_);
    }
    if (file_) {
        fclose(file_);
    }
}

size_t ZstdStreamReader::read(uint8_t *buffer, size_t max_bytes) {
    if (eof_) {
        return 0;
    }

    ZSTD_outBuffer zstd_out = {buffer, max_bytes, 0};

    while (zstd_out.pos < max_bytes) {
        if (zstd_in_.pos >= zstd_in_.size) {
            size_t bytes_read =
                fread(input_buffer_.data(), 1, input_buffer_.size(), file_);
            if (bytes_read == 0) {
                if (feof(file_)) {
                    eof_ = true;
                    break;
                }
                throw std::runtime_error("Error reading from file");
            }
            zstd_in_.src = input_buffer_.data();
            zstd_in_.size = bytes_read;
            zstd_in_.pos = 0;
        }

        size_t result = ZSTD_decompressStream(dstream_, &zstd_out, &zstd_in_);
        if (ZSTD_isError(result)) {
            throw std::runtime_error("ZSTD decompression error: " +
                                     std::string(ZSTD_getErrorName(result)));
        }

        if (result == 0 && zstd_in_.pos >= zstd_in_.size) {
            eof_ = true;
            break;
        }
    }

    return zstd_out.pos;
}

// FeatureDatasetReader implementation

FeatureDatasetReader::FeatureDatasetReader(std::vector<std::string> filepaths,
                                           size_t batch_size,
                                           double file_usage_ratio,
                                           bool shuffle, size_t num_workers,
                                           size_t prefetch_depth)
    : filepaths_(std::move(filepaths)), batch_size_(batch_size),
      file_usage_ratio_(file_usage_ratio), shuffle_(shuffle),
      num_decompress_workers_(num_workers), prefetch_depth_(prefetch_depth),
      rng_(std::random_device{}()) {

    if (file_usage_ratio_ <= 0.0 || file_usage_ratio_ > 1.0) {
        throw std::invalid_argument(
            "file_usage_ratio must be in the range (0.0, 1.0]");
    }

    if (num_decompress_workers_ == 0) {
        throw std::invalid_argument("num_workers must be at least 1");
    }

    if (prefetch_depth_ == 0) {
        throw std::invalid_argument("prefetch_depth must be at least 1");
    }
}

FeatureDatasetReader::~FeatureDatasetReader() { stop_workers(); }

void FeatureDatasetReader::set_worker_info(int worker_id, int num_workers) {
    worker_id_ = worker_id;
    num_dataloader_workers_ = num_workers;
}

void FeatureDatasetReader::process_records(const uint8_t *data,
                                           size_t num_records,
                                           float *scores_out,
                                           int64_t *features_out,
                                           int64_t *mobility_out,
                                           int64_t *ply_out) {
    for (size_t i = 0; i < num_records; ++i) {
        const auto *rec =
            reinterpret_cast<const ReversiRecord *>(data + i * RECORD_SIZE);

        scores_out[i] = rec->score;

        for (size_t f = 0; f < NUM_FEATURES; ++f) {
            features_out[i * NUM_FEATURES + f] =
                static_cast<int64_t>(rec->features[f]) + FEATURE_CUM_OFFSETS[f];
        }

        mobility_out[i] = static_cast<int64_t>(rec->mobility);
        ply_out[i] = static_cast<int64_t>(rec->ply);
    }
}

std::optional<std::string> FeatureDatasetReader::get_next_file() {
    std::lock_guard<std::mutex> lock(file_queue_mutex_);
    if (file_queue_.empty()) {
        return std::nullopt;
    }
    std::string filepath = std::move(file_queue_.front());
    file_queue_.pop();
    return filepath;
}

bool FeatureDatasetReader::fill_worker_buffer(WorkerContext &ctx) {
    constexpr size_t CHUNK_SIZE = 1 << 20; // 1 MB

    // Try to get a new file if we don't have a reader (loop instead of recursion)
    while (!ctx.reader) {
        auto filepath = get_next_file();
        if (!filepath) {
            return false;
        }
        try {
            ctx.reader = std::make_unique<ZstdStreamReader>(*filepath);
        } catch (const std::exception &e) {
            fprintf(stderr, "Warning: Failed to open %s: %s\n",
                    filepath->c_str(), e.what());
            // Continue loop to try next file
            continue;
        }
    }

    // Compact buffer
    size_t old_size = ctx.buffer.size() - ctx.buffer_offset;
    if (ctx.buffer_offset > 0 && old_size > 0) {
        memmove(ctx.buffer.data(), ctx.buffer.data() + ctx.buffer_offset,
                old_size);
    }
    ctx.buffer.resize(old_size);
    ctx.buffer_offset = 0;

    // Fill buffer until we have enough for a batch
    while (ctx.buffer.size() < batch_size_ * RECORD_SIZE) {
        size_t read_start = ctx.buffer.size();
        ctx.buffer.resize(read_start + CHUNK_SIZE);

        size_t bytes_read;
        try {
            bytes_read =
                ctx.reader->read(ctx.buffer.data() + read_start, CHUNK_SIZE);
        } catch (const std::exception &e) {
            // Read or decompression error - skip this file
            fprintf(stderr, "Warning: Read error, skipping file: %s\n",
                    e.what());
            ctx.buffer.resize(read_start); // Restore buffer size
            ctx.reader.reset();
            auto filepath = get_next_file();
            if (!filepath) {
                break;
            }
            try {
                ctx.reader = std::make_unique<ZstdStreamReader>(*filepath);
            } catch (const std::exception &e2) {
                fprintf(stderr, "Warning: Failed to open %s: %s\n",
                        filepath->c_str(), e2.what());
            }
            continue;
        }
        ctx.buffer.resize(read_start + bytes_read);

        if (bytes_read == 0 || ctx.reader->eof()) {
            ctx.reader.reset();
            auto filepath = get_next_file();
            if (!filepath) {
                break;
            }
            try {
                ctx.reader = std::make_unique<ZstdStreamReader>(*filepath);
            } catch (const std::exception &e) {
                fprintf(stderr, "Warning: Failed to open %s: %s\n",
                        filepath->c_str(), e.what());
                continue;
            }
        }
    }

    return ctx.buffer.size() >= batch_size_ * RECORD_SIZE;
}

std::optional<BatchTuple> FeatureDatasetReader::produce_batch(WorkerContext &ctx) {
    if (!fill_worker_buffer(ctx)) {
        return std::nullopt;
    }

    size_t available_bytes = ctx.buffer.size() - ctx.buffer_offset;
    size_t available_records = available_bytes / RECORD_SIZE;

    if (available_records < batch_size_) {
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

    // Process records directly into tensor storage
    process_records(ctx.buffer.data() + ctx.buffer_offset, batch_size_,
                    scores.data_ptr<float>(), features.data_ptr<int64_t>(),
                    mobility.data_ptr<int64_t>(), ply.data_ptr<int64_t>());

    ctx.buffer_offset += batch_size_ * RECORD_SIZE;

    return std::make_tuple(scores, features, mobility, ply);
}

void FeatureDatasetReader::worker_loop(size_t worker_id) {
    // Register immediately before any code that can throw, to ensure
    // the "last worker closes queue" logic works even if all workers fail
    active_workers_++;

    try {
        WorkerContext ctx;
        ctx.buffer.reserve(batch_size_ * RECORD_SIZE * 2);

        while (!shutdown_) {
            auto batch = produce_batch(ctx);
            if (!batch) {
                break;
            }
            if (shutdown_) {
                break;
            }
            // Stop if queue was closed (push returns false)
            if (!batch_queue_->push(std::move(*batch))) {
                break;
            }
        }
    } catch (const std::exception &e) {
        fprintf(stderr, "Error: Worker %zu failed with exception: %s\n",
                worker_id, e.what());
    } catch (...) {
        fprintf(stderr, "Error: Worker %zu failed with unknown exception\n",
                worker_id);
    }

    // Decrement active workers and close queue if last worker
    // fetch_sub returns the value BEFORE decrement, so == 1 means we were the last
    if (active_workers_.fetch_sub(1) == 1) {
        batch_queue_->close();
    }
}

void FeatureDatasetReader::start_workers() {
    if (workers_started_) {
        return;
    }

    // Apply shuffling if enabled
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

    // Populate file queue
    for (const auto &filepath : filepaths_) {
        file_queue_.push(filepath);
    }

    // Create batch queue
    batch_queue_ = std::make_unique<BoundedQueue<BatchTuple>>(prefetch_depth_);

    // Start worker threads with rollback on failure
    shutdown_ = false;
    active_workers_ = 0;

    try {
        for (size_t i = 0; i < num_decompress_workers_; ++i) {
            workers_.emplace_back(&FeatureDatasetReader::worker_loop, this, i);
        }
    } catch (...) {
        // Rollback: stop any threads we did start
        shutdown_ = true;
        batch_queue_->close();
        for (auto &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
        throw; // Re-throw the original exception
    }

    // Only mark as started after ALL threads are successfully created
    workers_started_ = true;
}

void FeatureDatasetReader::stop_workers() {
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

std::optional<BatchTuple> FeatureDatasetReader::next() {
    // Start workers on first call
    if (!workers_started_) {
        start_workers();
    }

    // Get next batch from queue
    return batch_queue_->pop();
}

} // namespace feature_dataset

// pybind11 module definition

PYBIND11_MODULE(_C, m) {
    m.doc() = "C++ extension for fast Reversi feature dataset loading";

    py::class_<feature_dataset::FeatureDatasetReader>(m, "FeatureDatasetReader")
        .def(py::init<std::vector<std::string>, size_t, double, bool, size_t,
                      size_t>(),
             py::arg("filepaths"), py::arg("batch_size"),
             py::arg("file_usage_ratio"), py::arg("shuffle"),
             py::arg("num_workers") = 2, py::arg("prefetch_depth") = 4)
        .def("next", &feature_dataset::FeatureDatasetReader::next,
             py::call_guard<py::gil_scoped_release>())
        .def("set_worker_info",
             &feature_dataset::FeatureDatasetReader::set_worker_info,
             py::arg("worker_id"), py::arg("num_workers"));
}
