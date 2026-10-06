#include "bl/gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <regex>
#include <stdexcept>

namespace bl {

namespace {

// block geometry from ggml-common.h's static_asserts (llama.cpp 3cf0325)
constexpr TypeGeometry kTypes[] = {
    {"F32", 1, 4}, {"F16", 1, 2}, {"Q4_0", 32, 18}, {"Q8_0", 32, 34},
    {"Q4_K", 256, 144}, {"Q5_K", 256, 176}, {"Q6_K", 256, 210},
    {"IQ2_XXS", 256, 66}, {"IQ2_XS", 256, 74}, {"IQ3_XXS", 256, 98}, {"IQ4_NL", 32, 18},
    {"IQ3_S", 256, 110}, {"IQ2_S", 256, 82}, {"IQ4_XS", 256, 136}, {"BF16", 1, 2}, {"Q2_0", 64, 18},
};
constexpr uint32_t kTypeIds[] = {0, 1, 2, 8, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 30, 42};

enum : uint32_t {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32, GGUF_F32, GGUF_BOOL, GGUF_STRING,
    GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64,
};

class Cursor {
public:
    Cursor(const uint8_t * p, size_t n, const std::string & path) : p_(p), end_(p + n), path_(path) {}
    template <typename T> T get() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p_, sizeof(T));
        p_ += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        need(n);
        std::string s(reinterpret_cast<const char *>(p_), n);
        p_ += n;
        return s;
    }
    size_t pos(const uint8_t * base) const { return static_cast<size_t>(p_ - base); }

private:
    void need(uint64_t n) const {
        if (n > static_cast<uint64_t>(end_ - p_)) throw std::runtime_error(path_ + ": truncated GGUF header");
    }
    const uint8_t * p_;
    const uint8_t * end_;
    std::string     path_;
};

using Scalar = std::variant<int64_t, uint64_t, double, bool, std::string>;

Scalar read_scalar(Cursor & c, uint32_t type) {
    switch (type) {
        case GGUF_U8:     return static_cast<uint64_t>(c.get<uint8_t>());
        case GGUF_I8:     return static_cast<int64_t>(c.get<int8_t>());
        case GGUF_U16:    return static_cast<uint64_t>(c.get<uint16_t>());
        case GGUF_I16:    return static_cast<int64_t>(c.get<int16_t>());
        case GGUF_U32:    return static_cast<uint64_t>(c.get<uint32_t>());
        case GGUF_I32:    return static_cast<int64_t>(c.get<int32_t>());
        case GGUF_F32:    return static_cast<double>(c.get<float>());
        case GGUF_BOOL:   return c.get<uint8_t>() != 0;
        case GGUF_STRING: return c.str();
        case GGUF_U64:    return c.get<uint64_t>();
        case GGUF_I64:    return c.get<int64_t>();
        case GGUF_F64:    return c.get<double>();
        default: throw std::runtime_error("GGUF: unknown metadata type " + std::to_string(type));
    }
}

MetaValue read_value(Cursor & c, uint32_t type) {
    if (type != GGUF_ARRAY) {
        return std::visit([](auto && v) -> MetaValue { return v; }, read_scalar(c, type));
    }
    const uint32_t item = c.get<uint32_t>();
    const uint64_t n    = c.get<uint64_t>();
    MetaArray arr;
    arr.reserve(n);
    for (uint64_t i = 0; i < n; ++i) arr.push_back(read_scalar(c, item));
    return arr;
}

int64_t as_int(const Scalar & v, const std::string & key) {
    if (auto p = std::get_if<int64_t>(&v)) return *p;
    if (auto p = std::get_if<uint64_t>(&v)) return static_cast<int64_t>(*p);
    if (auto p = std::get_if<bool>(&v)) return *p;
    throw std::runtime_error("GGUF: " + key + " is not an integer");
}

}  // namespace

TypeGeometry type_geometry(uint32_t type_id) {
    for (size_t i = 0; i < std::size(kTypeIds); ++i)
        if (kTypeIds[i] == type_id) return kTypes[i];
    return {nullptr, 0, 0};
}

size_t row_bytes(uint32_t type_id, int64_t n_elems) {
    const TypeGeometry g = type_geometry(type_id);
    if (!g.name) throw std::runtime_error("GGUF: unsupported tensor type " + std::to_string(type_id));
    if (n_elems % g.block_elems) throw std::runtime_error(std::string("GGUF: row not a whole number of ") + g.name + " blocks");
    return static_cast<size_t>(n_elems / g.block_elems) * g.block_bytes;
}

int64_t Tensor::elements() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

GgufModel::GgufModel(const std::string & path) {
    // the shard list: shard 1 names the count, its siblings differ only in the index
    std::vector<std::string> paths{path};
    static const std::regex split(R"((.*)-00001-of-(\d{5})\.gguf$)");
    std::smatch m;
    if (std::regex_match(path, m, split)) {
        const int n = std::stoi(m[2]);
        for (int i = 2; i <= n; ++i) {
            char idx[12];
            std::snprintf(idx, sizeof idx, "%05d", i);
            paths.push_back(m[1].str() + "-" + idx + "-of-" + m[2].str() + ".gguf");
        }
    }
    for (const auto & p : paths) {
        Shard s;
        s.path = p;
        s.fd   = ::open(p.c_str(), O_RDONLY);
        if (s.fd < 0) {
            if (shards_.empty()) throw std::runtime_error("cannot open " + p);
            continue;   // a shard not downloaded yet: its tensors are reported missing
        }
        struct stat st {};
        ::fstat(s.fd, &st);
        s.size = static_cast<size_t>(st.st_size);
        void * base = ::mmap(nullptr, s.size, PROT_READ, MAP_SHARED, s.fd, 0);
        if (base == MAP_FAILED) throw std::runtime_error("mmap failed: " + p);
        s.base = static_cast<uint8_t *>(base);
        shards_.push_back(s);
        parse_shard(static_cast<uint32_t>(shards_.size() - 1));
    }
}

GgufModel::~GgufModel() {
    for (auto & s : shards_) {
        if (s.base) ::munmap(s.base, s.size);
        if (s.fd >= 0) ::close(s.fd);
    }
}

void GgufModel::parse_shard(uint32_t index) {
    Shard & s = shards_[index];
    Cursor c(s.base, s.size, s.path);
    if (c.get<uint32_t>() != 0x46554747u) throw std::runtime_error(s.path + ": not a GGUF file");
    const uint32_t version = c.get<uint32_t>();
    if (version != 3) throw std::runtime_error(s.path + ": GGUF version " + std::to_string(version) + " (need 3)");
    const uint64_t n_tensors = c.get<uint64_t>();
    const uint64_t n_kv      = c.get<uint64_t>();

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key  = c.str();
        const uint32_t t = c.get<uint32_t>();
        MetaValue v      = read_value(c, t);
        if (index == 0 || !meta_.count(key)) meta_[key] = std::move(v);   // shard 1 is authoritative
    }

    const size_t first = tensors_.size();
    for (uint64_t i = 0; i < n_tensors; ++i) {
        Tensor t;
        t.name = c.str();
        const uint32_t nd = c.get<uint32_t>();
        for (uint32_t d = 0; d < nd; ++d) t.shape.push_back(static_cast<int64_t>(c.get<uint64_t>()));
        t.type_id = c.get<uint32_t>();
        t.offset  = c.get<uint64_t>();
        t.shard   = index;
        if (!t.shape.empty()) t.nbytes = row_bytes(t.type_id, t.shape[0]) * static_cast<size_t>(t.elements() / t.shape[0]);
        tensors_.push_back(std::move(t));
    }

    uint64_t align = 32;
    if (auto it = meta_.find("general.alignment"); it != meta_.end()) align = static_cast<uint64_t>(get_int("general.alignment"));
    const size_t hdr = c.pos(s.base);
    s.data_start     = (hdr + align - 1) / align * align;

    for (size_t i = first; i < tensors_.size(); ++i) {
        Tensor & t      = tensors_[i];
        const size_t at = s.data_start + t.offset;
        t.data          = (at + t.nbytes <= s.size) ? s.base + at : nullptr;
        if (by_name_.count(t.name)) throw std::runtime_error("GGUF: tensor " + t.name + " appears twice");
        by_name_[t.name] = i;
    }
}

const Tensor * GgufModel::find(const std::string & name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

const Tensor & GgufModel::at(const std::string & name) const {
    if (const Tensor * t = find(name)) return *t;
    throw std::runtime_error("GGUF: no tensor " + name);
}

void GgufModel::advise_willneed(const Tensor & t) const {
    if (!t.data) return;
    const long page = sysconf(_SC_PAGESIZE);
    const auto a = reinterpret_cast<uintptr_t>(t.data) & ~static_cast<uintptr_t>(page - 1);
    const size_t len = reinterpret_cast<uintptr_t>(t.data) + t.nbytes - a;
    ::madvise(reinterpret_cast<void *>(a), len, MADV_WILLNEED);
}

bool GgufModel::complete() const {
    for (const auto & t : tensors_)
        if (!t.data) return false;
    return true;
}

std::string GgufModel::get_str(const std::string & key) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) throw std::runtime_error("GGUF: no key " + key);
    if (auto p = std::get_if<std::string>(&it->second)) return *p;
    throw std::runtime_error("GGUF: " + key + " is not a string");
}

int64_t GgufModel::get_int(const std::string & key) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) throw std::runtime_error("GGUF: no key " + key);
    return std::visit([&](auto && v) -> int64_t {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, MetaArray> || std::is_same_v<T, std::string> || std::is_same_v<T, double>)
            throw std::runtime_error("GGUF: " + key + " is not an integer");
        else
            return static_cast<int64_t>(v);
    }, it->second);
}

double GgufModel::get_float(const std::string & key) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) throw std::runtime_error("GGUF: no key " + key);
    if (auto p = std::get_if<double>(&it->second)) return *p;
    return static_cast<double>(get_int(key));
}

std::vector<int64_t> GgufModel::get_int_array(const std::string & key) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) throw std::runtime_error("GGUF: no key " + key);
    auto p = std::get_if<MetaArray>(&it->second);
    if (!p) throw std::runtime_error("GGUF: " + key + " is not an array");
    std::vector<int64_t> out;
    out.reserve(p->size());
    for (const auto & v : *p) out.push_back(as_int(v, key));
    return out;
}

}  // namespace bl
