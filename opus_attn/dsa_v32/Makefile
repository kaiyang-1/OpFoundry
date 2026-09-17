OPUS_INCLUDE_DIR ?= /path/to/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx950

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -pthread -O3 -Wall --offload-arch=$(ARCH)
KERNEL_FLAGS = -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage
LINK_FLAGS   = --offload-arch=$(ARCH) -pthread --rtlib=compiler-rt --unwindlib=libgcc

BUILD = build
OUT   = $(BUILD)/mla_decode.exe

KERNEL_SRCS = $(wildcard *_kernel.cc)
HOST_SRC    = host.cc
SHARED_HDRS = defs.h global_load.hpp parallel.h
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(KERNEL_SRCS))
KERNEL_ASMS = $(patsubst %.cc,$(BUILD)/%.s,$(KERNEL_SRCS))
HOST_OBJ    = $(BUILD)/host.o

.PHONY: all asm clean verify
.DELETE_ON_ERROR:

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

.SECONDEXPANSION:
$(BUILD)/%_kernel.o: %_kernel.cc $$(wildcard %_template.hpp) $$(wildcard %.hpp) $(SHARED_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) -c -o $@

$(BUILD)/%_kernel.s: %_kernel.cc $$(wildcard %_template.hpp) $$(wildcard %.hpp) $(SHARED_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) --offload-device-only -S -o $@

asm: $(KERNEL_ASMS)

$(HOST_OBJ): $(HOST_SRC) $(SHARED_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ $(LINK_FLAGS) -o $@

VERIFY_HQ ?= 64 128 192

verify: $(OUT)
	@for hq in $(VERIFY_HQ); do \
	    echo "=== fp8 16mx8_32nx1 h_q=$$hq --verify ==="; \
	    $(OUT) -dtype fp8 -h_q $$hq -b 32 --verify || exit 1; \
	done

benchmark: $(OUT)
	@echo "=== h_q=128 (a8w8_16mx8_32nx1) ===" && $(OUT) -h_q 128

clean:
	rm -rf $(BUILD)
