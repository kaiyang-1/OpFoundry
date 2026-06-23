OPUS_INCLUDE_DIR ?= /path/to/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx950

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -fopenmp -O3 -Wall --offload-arch=$(ARCH)
KERNEL_FLAGS = -ffast-math -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj

BUILD = build
OUT   = $(BUILD)/dsa_v32_decode.exe

KERNEL_SRCS = get_mla_metadata.cc dsa_v32_splitkv_kernel.cc mla_combine.cc
HOST_SRC    = host.cc
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(KERNEL_SRCS))
HOST_OBJ    = $(BUILD)/host.o

.PHONY: all clean benchmark

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: %.cc | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) -c -o $@

$(BUILD)/get_mla_metadata.o:       get_mla_metadata.hpp defs.h
$(BUILD)/dsa_v32_splitkv_kernel.o: dsa_v32_splitkv.hpp  defs.h
$(BUILD)/mla_combine.o:            mla_combine.hpp      defs.h

$(HOST_OBJ): $(HOST_SRC) defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp -o $@

benchmark: $(OUT)
	@echo "=== h_q=128 (16mx8_32nx1_fp8) ===" && $(OUT) -h_q 128

clean:
	rm -rf $(BUILD)
