OPUS_INCLUDE_DIR ?= /path/to/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx950

ifeq ($(wildcard $(ARCH)/traits.h),)
$(error unknown ARCH '$(ARCH)'; available: $(patsubst %/traits.h,%,$(wildcard */traits.h)))
endif

ARCH_MACRO := MLA_DECODE_ARCH_$(shell echo $(ARCH) | tr '[:lower:]' '[:upper:]')

COMMON_FLAGS     = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -pthread -O3 -Wall -D$(ARCH_MACRO) --offload-arch=$(ARCH)
KERNEL_FLAGS     = -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage
KERNEL_OBJ_FLAGS = -fno-offload-lto
LINK_FLAGS       = --offload-arch=$(ARCH) -pthread --rtlib=compiler-rt --unwindlib=libgcc

BUILD = build/$(ARCH)
OUT   = $(BUILD)/mla_decode.exe

KERNEL_SRCS = $(wildcard common/*_kernel.cc) $(wildcard $(ARCH)/*_kernel.cc)
HOST_SRC    = common/host.cc
COMMON_HDRS = common/defs.h common/parallel.h common/get_mla_metadata.hpp common/mla_reduce.hpp
SHARED_HDRS = $(filter-out %_template.hpp,$(wildcard $(ARCH)/*.h) $(wildcard $(ARCH)/*.hpp)) $(COMMON_HDRS)
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(notdir $(KERNEL_SRCS)))
KERNEL_ASMS = $(patsubst %.cc,$(BUILD)/%.s,$(notdir $(KERNEL_SRCS)))
HOST_OBJ    = $(BUILD)/host.o

.PHONY: all asm clean
.DELETE_ON_ERROR:

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%_kernel.o: $(ARCH)/%_kernel.cc $(ARCH)/%_template.hpp $(SHARED_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) $(KERNEL_OBJ_FLAGS) -c -o $@

$(BUILD)/%_kernel.o: common/%_kernel.cc $(SHARED_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) $(KERNEL_OBJ_FLAGS) -c -o $@

$(BUILD)/%_kernel.s: $(ARCH)/%_kernel.cc $(ARCH)/%_template.hpp $(SHARED_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) --offload-device-only -S -o $@

$(BUILD)/%_kernel.s: common/%_kernel.cc $(SHARED_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) --offload-device-only -S -o $@

asm: $(KERNEL_ASMS)

$(HOST_OBJ): $(HOST_SRC) $(SHARED_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ $(LINK_FLAGS) -o $@

clean:
	rm -rf build
