OPUS_INCLUDE_DIR ?= /path/to/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx950

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -fopenmp -O3 -Wall --offload-arch=$(ARCH)
KERNEL_FLAGS = -ffast-math -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj

BUILD = build
OUT   = $(BUILD)/dsa_v32_decode.exe

KERNEL_SRCS = dsa_v32_decode_16mx8_32nx1_fp8_kernel.cc
HOST_SRC    = dsa_v32_host.cc
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(KERNEL_SRCS))
HOST_OBJ    = $(BUILD)/dsa_v32_host.o

.PHONY: all clean benchmark

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

# Kernel objects (heavy — device+host passes, parallelizable via make -j)
# -D__HIPCC_RTC__ skips implicit __clang_hip_runtime_wrapper.h on host pass (~250ms saved)
$(BUILD)/%.o: %.cc dsa_v32_decode_16mx8_32nx1_fp8_template.hpp dsa_v32_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) -c -o $@

# Host object (no -D__HIPCC_RTC__ — needs full HIP runtime for <<<>>> syntax)
$(HOST_OBJ): $(HOST_SRC) dsa_v32_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

# Link
$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp -o $@

benchmark: $(OUT)
	@echo "=== h_q=128 (16mx8_32nx1_fp8) ===" && $(OUT) -h_q 128

clean:
	rm -rf $(BUILD)
