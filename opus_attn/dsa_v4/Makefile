OPUS_INCLUDE_DIR ?= /path/to/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx1250

ARCH_MACRO = PA_ARCH_$(shell echo $(ARCH) | tr '[:lower:]' '[:upper:]')
COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -pthread -O3 -Wall -D$(ARCH_MACRO) --offload-arch=$(ARCH)
KERNEL_FLAGS = -ffast-math -mllvm -amdgpu-expert-scheduling-mode -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage --no-offload-new-driver -save-temps=obj

BUILD = build/$(ARCH)
OUT   = $(BUILD)/pa_prefill.exe

KERNEL_SRCS = $(wildcard $(ARCH)/pa_prefill_*_kernel.cc)
HOST_SRC    = common/pa_host.cc
ARCH_HDRS   = $(wildcard $(ARCH)/*.h) $(wildcard $(ARCH)/*.hpp) common/pa_kargs.h common/pa_parallel.h
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(notdir $(KERNEL_SRCS)))
HOST_OBJ    = $(BUILD)/pa_host.o

.PHONY: all clean benchmark

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

# Kernel objects (heavy — device+host passes, parallelizable via make -j)
# -D__HIPCC_RTC__ skips implicit __clang_hip_runtime_wrapper.h on host pass (~250ms saved)
$(BUILD)/%.o: $(ARCH)/%.cc $(ARCH_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) -c -o $@

# Host object (no -D__HIPCC_RTC__ — needs full HIP runtime for <<<>>> syntax)
$(HOST_OBJ): $(HOST_SRC) $(ARCH_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

# Link
$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -pthread -o $@

benchmark: $(OUT)
	@echo "=== h_q=16 (16mx1_16nx4) ===" && $(OUT) -h_q 16
	@echo "=== h_q=128 (16mx8_32nx1) ===" && $(OUT) -h_q 128

clean:
	rm -rf build
