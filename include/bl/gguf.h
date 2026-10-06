// GGUF reader: memory-maps every shard of a split model and exposes its metadata and tensor table.
// Only the header is parsed eagerly; tensor data is touched only through Tensor::data.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace bl {

// ggml type ids used by the Qwen3.8-Flash-Next GSQ-RCO files (ggml.h at llama.cpp 3cf0325)
enum class GgmlType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
    BF16 = 30, Q2_0 = 42,
};

struct TypeGeometry {
    const char * name;
    uint32_t     block_elems;   // values per block
    uint32_t     block_bytes;   // bytes per block
};

// nullptr geometry name for a type this reader does not know
TypeGeometry type_geometry(uint32_t type_id);
size_t       row_bytes(uint32_t type_id, int64_t n_elems);   // throws when n_elems is not a whole number of blocks

using MetaArray = std::vector<std::variant<int64_t, uint64_t, double, bool, std::string>>;
using MetaValue = std::variant<int64_t, uint64_t, double, bool, std::string, MetaArray>;

struct Tensor {
    std::string          name;
    uint32_t             type_id = 0;
    std::vector<int64_t> shape;        // ggml order: shape[0] is the contiguous (row) dimension
    uint64_t             offset  = 0;  // relative to the shard's data section
    uint32_t             shard   = 0;
    size_t               nbytes  = 0;
    const uint8_t *      data    = nullptr;   // nullptr when the bytes lie past the end of a partial file
    int64_t elements() const;
};

class GgufModel {
public:
    // path: shard 1 (`...-00001-of-0000N.gguf`); the other shards are found next to it
    explicit GgufModel(const std::string & path);
    ~GgufModel();
    GgufModel(const GgufModel &) = delete;
    GgufModel & operator=(const GgufModel &) = delete;

    const std::map<std::string, MetaValue> & metadata() const { return meta_; }
    const std::vector<Tensor> &              tensors() const { return tensors_; }
    const Tensor * find(const std::string & name) const;   // nullptr when absent
    const Tensor & at(const std::string & name) const;     // throws when absent

    std::string               get_str(const std::string & key) const;
    int64_t                   get_int(const std::string & key) const;
    double                    get_float(const std::string & key) const;
    std::vector<int64_t>      get_int_array(const std::string & key) const;
    bool                      has(const std::string & key) const { return meta_.count(key) != 0; }

    size_t n_shards() const { return shards_.size(); }
    const std::string & shard_path(uint32_t i) const { return shards_.at(i).path; }
    uint64_t file_offset(const Tensor & t) const { return shards_.at(t.shard).data_start + t.offset; }   // in its shard
    void   advise_willneed(const Tensor & t) const;   // start reading the tensor's pages in the background
    bool   complete() const;   // every tensor's bytes are inside its mapped file

private:
    struct Shard {
        std::string path;
        int         fd = -1;
        uint8_t *   base = nullptr;
        size_t      size = 0;
        size_t      data_start = 0;
    };
    void parse_shard(uint32_t index);

    std::vector<Shard>               shards_;
    std::map<std::string, MetaValue> meta_;
    std::vector<Tensor>              tensors_;
    std::map<std::string, size_t>    by_name_;
};

}  // namespace bl
