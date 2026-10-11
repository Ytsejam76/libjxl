// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/memory_manager_internal.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>     // memcpy
#include <hwy/base.h>  // kMaxVectorSize
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/simd_util.h"

namespace jxl {

namespace {

// Large buffers (image planes, coefficient images) are allocated and freed
// many times during an encode. malloc serves them with fresh mmap()s and
// unmaps them on free, so each reuse page-faults and zeroes the memory again,
// which costs up to a third of the CPU time of fast encodes. The default
// manager therefore keeps a bounded cache of freed large blocks and hands
// them out again. Applications with their own JxlMemoryManager are not
// affected.
constexpr size_t kBlockHeaderSize = 64;  // Preserves malloc's alignment.
constexpr size_t kMinCachedBlockSize = size_t{1} << 20;
constexpr size_t kMaxCachedBytes = size_t{256} << 20;

struct BlockCache {
  std::mutex mutex;
  std::vector<std::pair<size_t, void*>> blocks;  // (capacity, base pointer)
  size_t cached_bytes = 0;
};

BlockCache& GetBlockCache() {
  // Intentionally leaked: blocks may be freed during static destruction.
  static BlockCache* cache = new BlockCache();
  return *cache;
}

void* MemoryManagerDefaultAlloc(void* opaque, size_t size) {
  if (size >= kMinCachedBlockSize) {
    BlockCache& cache = GetBlockCache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    // Best fit among the cached blocks that are at most twice as large.
    size_t best = cache.blocks.size();
    for (size_t i = 0; i < cache.blocks.size(); ++i) {
      const size_t capacity = cache.blocks[i].first;
      if (capacity >= size && capacity <= 2 * size &&
          (best == cache.blocks.size() ||
           capacity < cache.blocks[best].first)) {
        best = i;
      }
    }
    if (best != cache.blocks.size()) {
      void* base = cache.blocks[best].second;
      cache.cached_bytes -= cache.blocks[best].first;
      cache.blocks[best] = cache.blocks.back();
      cache.blocks.pop_back();
      return static_cast<uint8_t*>(base) + kBlockHeaderSize;
    }
  }
  if (size > std::numeric_limits<size_t>::max() - kBlockHeaderSize) {
    return nullptr;
  }
  void* base = malloc(size + kBlockHeaderSize);
  if (base == nullptr) return nullptr;
  memcpy(base, &size, sizeof(size));
  return static_cast<uint8_t*>(base) + kBlockHeaderSize;
}

void MemoryManagerDefaultFree(void* opaque, void* address) {
  if (address == nullptr) return;
  void* base = static_cast<uint8_t*>(address) - kBlockHeaderSize;
  size_t size;
  memcpy(&size, base, sizeof(size));
  if (size >= kMinCachedBlockSize) {
    BlockCache& cache = GetBlockCache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.cached_bytes + size <= kMaxCachedBytes) {
      cache.blocks.emplace_back(size, base);
      cache.cached_bytes += size;
      return;
    }
  }
  free(base);
}

}  // namespace

void* MemoryManagerAlloc(const JxlMemoryManager* memory_manager, size_t size) {
  return memory_manager->alloc(memory_manager->opaque, size);
}

void MemoryManagerFree(const JxlMemoryManager* memory_manager, void* address) {
  memory_manager->free(memory_manager->opaque, address);
}

Status MemoryManagerInit(JxlMemoryManager* self,
                         const JxlMemoryManager* memory_manager) {
  if (memory_manager) {
    *self = *memory_manager;
  } else {
    memset(self, 0, sizeof(*self));
  }
  bool is_default_alloc = (self->alloc == nullptr);
  bool is_default_free = (self->free == nullptr);
  if (is_default_alloc != is_default_free) {
    return false;
  }
  if (is_default_alloc) self->alloc = jxl::MemoryManagerDefaultAlloc;
  if (is_default_free) self->free = jxl::MemoryManagerDefaultFree;

  return true;
}

StatusOr<size_t> BytesPerRow(const size_t xsize, const size_t sizeof_t) {
  // Special case: we don't allow any ops -> don't need extra padding/
  if (xsize == 0) {
    return 0;
  }

  const size_t vec_size = MaxVectorSize();
  size_t valid_bytes;
  if (!SafeMul(xsize, sizeof_t, valid_bytes)) {
    return JXL_FAILURE("Image dimensions are too large");
  }

  // Allow unaligned accesses starting at the last valid value.
  // NB: in scalar build `vec_size` is `4` (one `float`); still worth checking.
  if (vec_size != 0) {
    // NB: in scalar build when T is "double" we "extra" would be negative.
    JXL_ENSURE(vec_size >= sizeof_t);
    size_t extra = vec_size - sizeof_t;
    if (!SafeAdd(valid_bytes, extra, valid_bytes)) {
      return JXL_FAILURE("Image dimensions are too large");
    }
  }

  // Round up to vector and cache line size.
  const size_t align = std::max(vec_size, memory_manager_internal::kAlignment);
  JXL_ENSURE((align & (align - 1)) == 0);
  size_t bytes_per_row;
  if (!SafeRoundUpTo(valid_bytes, align, bytes_per_row)) {
    return JXL_FAILURE("Image dimensions are too large");
  }

  // During the lengthy window before writes are committed to memory, CPUs
  // guard against read after write hazards by checking the address, but
  // only the lower 11 bits. We avoid a false dependency between writes to
  // consecutive rows by ensuring their sizes are not multiples of 2 KiB.
  // Avoid2K prevents the same problem for the planes of an Image3.
  if (bytes_per_row % memory_manager_internal::kAlias == 0) {
    if (!SafeAdd(bytes_per_row, align, bytes_per_row)) {
      return JXL_FAILURE("Image dimensions are too large");
    }
  }

  JXL_DASSERT(bytes_per_row % align == 0);
  return bytes_per_row;
}

StatusOr<AlignedMemory> AlignedMemory::Create(JxlMemoryManager* memory_manager,
                                              size_t size, size_t pre_padding) {
  JXL_ENSURE(pre_padding <= memory_manager_internal::kAlias);
  size_t allocation_size;
  if (!SafeAdd<size_t>(size, pre_padding, allocation_size) ||
      !SafeAdd(allocation_size, memory_manager_internal::kAlias,
               allocation_size)) {
    return JXL_FAILURE("Requested allocation is too large");
  }
  JXL_ENSURE(memory_manager);
  void* allocated =
      memory_manager->alloc(memory_manager->opaque, allocation_size);
  if (allocated == nullptr) {
    return JXL_FAILURE("Allocation failed");
  }
  return AlignedMemory(memory_manager, allocated, pre_padding);
}

AlignedMemory::AlignedMemory(JxlMemoryManager* memory_manager, void* allocation,
                             size_t pre_padding)
    : allocation_(allocation), memory_manager_(memory_manager) {
  // Congruence to `offset` (mod kAlias) reduces cache conflicts and load/store
  // stalls, especially with large allocations that would otherwise have similar
  // alignments.
  static std::atomic<uint32_t> next_group{0};
  size_t group =
      static_cast<size_t>(next_group.fetch_add(1, std::memory_order_relaxed));
  group &= (memory_manager_internal::kNumAlignmentGroups - 1);
  size_t offset = memory_manager_internal::kAlignment * group;

  // Actual allocation.
  uintptr_t address = reinterpret_cast<uintptr_t>(allocation) + pre_padding;

  // Aligned address, but might land before allocation (50%/50%) or not have
  // enough pre-padding.
  uintptr_t aligned_address =
      (address & ~(memory_manager_internal::kAlias - 1)) + offset;
  if (aligned_address < address)
    aligned_address += memory_manager_internal::kAlias;

  address_ = reinterpret_cast<void*>(aligned_address);  // NOLINT
}

AlignedMemory::AlignedMemory(AlignedMemory&& other) noexcept {
  allocation_ = other.allocation_;
  memory_manager_ = other.memory_manager_;
  address_ = other.address_;
  other.memory_manager_ = nullptr;
}

AlignedMemory& AlignedMemory::operator=(AlignedMemory&& other) noexcept {
  if (this == &other) return *this;
  if (memory_manager_ && allocation_) {
    memory_manager_->free(memory_manager_->opaque, allocation_);
  }
  allocation_ = other.allocation_;
  memory_manager_ = other.memory_manager_;
  address_ = other.address_;
  other.memory_manager_ = nullptr;
  return *this;
}

AlignedMemory::~AlignedMemory() {
  if (memory_manager_ == nullptr) return;
  memory_manager_->free(memory_manager_->opaque, allocation_);
}

}  // namespace jxl
