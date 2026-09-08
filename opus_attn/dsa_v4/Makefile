OPUS_INCLUDE_DIR ?= /path/to/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx1250

ifeq ($(wildcard $(ARCH)/mla_v4_traits.h),)
$(error unknown ARCH '$(ARCH)'; available: $(patsubst %/mla_v4_traits.h,%,$(wildcard */mla_v4_traits.h)))
endif

ARCH_MACRO := MLA_V4_ARCH_$(shell echo $(ARCH) | tr '[:lower:]' '[:upper:]')

# The gfx1250 kernels depend on the expert mode and llvm scheduler
SCHED_FLAGS_gfx1250 = -mllvm -amdgpu-expert-scheduling-mode \
                      -mllvm -amdgpu-sched-strategy=coexec \
                      -mllvm -amdgpu-anti-hints-for-va-vdst

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -pthread -O3 -Wall -D$(ARCH_MACRO) --offload-arch=$(ARCH)
KERNEL_FLAGS = -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage --no-offload-new-driver $(SCHED_FLAGS_$(ARCH))
LINK_FLAGS   = --offload-arch=$(ARCH) -pthread

BUILD = build/$(ARCH)
OUT   = $(BUILD)/mla_v4_prefill.exe

KERNEL_SRCS = $(wildcard $(ARCH)/mla_v4_prefill_*_kernel.cc)
HOST_SRC    = common/mla_v4_host.cc
COMMON_HDRS = common/mla_v4_kargs.h common/mla_v4_parallel.h
SHARED_HDRS = $(filter-out %_template.hpp,$(wildcard $(ARCH)/*.h) $(wildcard $(ARCH)/*.hpp)) $(COMMON_HDRS)
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(notdir $(KERNEL_SRCS)))
KERNEL_ASMS = $(patsubst %.cc,$(BUILD)/%.s,$(notdir $(KERNEL_SRCS)))
HOST_OBJ    = $(BUILD)/mla_v4_host.o

.PHONY: all asm clean verify
.DELETE_ON_ERROR:

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

# Kernel objects (heavy — device+host passes, parallelizable via make -j)
# -D__HIPCC_RTC__ skips implicit __clang_hip_runtime_wrapper.h on host pass
$(BUILD)/%_kernel.o: $(ARCH)/%_kernel.cc $(ARCH)/%_template.hpp $(SHARED_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) -c -o $@

# Device assembly.
$(BUILD)/%_kernel.s: $(ARCH)/%_kernel.cc $(ARCH)/%_template.hpp $(SHARED_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) --offload-device-only -S -o $@

asm: $(KERNEL_ASMS)

# Host object (no -D__HIPCC_RTC__ — needs full HIP runtime for <<<>>> syntax).
$(HOST_OBJ): $(HOST_SRC) $(ARCH)/mla_v4_traits.h $(COMMON_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

# Link
$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ $(LINK_FLAGS) -o $@

VERIFY_HQ     ?= 16 32 64 128
VERIFY_DTYPES ?= bf16 fp8

verify: $(OUT)
	@for dtype in $(VERIFY_DTYPES); do for hq in $(VERIFY_HQ); do \
	    echo "=== $(ARCH) $$dtype h_q=$$hq --verify ==="; $(OUT) -dtype $$dtype -h_q $$hq -n 256 --verify || exit 1; \
	done; done

clean:
	rm -rf build
