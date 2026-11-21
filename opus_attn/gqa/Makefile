OPUS_INCLUDE_DIR ?= $(HOME)/develop/aiter/csrc/include
HIPCC            ?= /opt/rocm/bin/hipcc
ARCH             ?= gfx1250

ifeq ($(wildcard $(ARCH)/gqa_traits.h),)
$(error unknown ARCH '$(ARCH)'; available: $(patsubst %/gqa_traits.h,%,$(wildcard */gqa_traits.h)))
endif

ARCH_MACRO := GQA_ARCH_$(shell echo $(ARCH) | tr '[:lower:]' '[:upper:]')

# gfx1250 builds with the public LLVM (upstream main) through HIP_CLANG_PATH, gfx950 with the compiler of HIPCC.
HIP_CLANG_PATH_gfx1250 ?= $(HOME)/develop/llvm-project/build/bin
ifneq ($(HIP_CLANG_PATH_$(ARCH)),)
HIP_CLANG_PATH ?= $(HIP_CLANG_PATH_$(ARCH))
endif
ifneq ($(HIP_CLANG_PATH),)
export HIP_CLANG_PATH
endif

# gfx1250: the co-execution scheduler in expert scheduling mode (the kernel sets SCHED_MODE 2; the compiler inserts the
# waits between VALU and VMEM/LDS instructions), and no offload LTO, so that `make asm` matches the built objects.
# gfx950: the kernels are tuned with fast math and offload LTO, whose code differs from `make asm` in VGPR allocation.
KERNEL_FLAGS_gfx1250     = -mllvm -amdgpu-sched-strategy=coexec -mllvm -amdgpu-expert-scheduling-mode
KERNEL_OBJ_FLAGS_gfx1250 = -fno-offload-lto
KERNEL_FLAGS_gfx950      = -ffast-math

COMMON_FLAGS     = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -pthread -O3 -Wall -Wno-deprecated-attributes -D$(ARCH_MACRO) \
                   --offload-arch=$(ARCH)
KERNEL_FLAGS     = -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage $(KERNEL_FLAGS_$(ARCH)) $(EXTRA)
KERNEL_OBJ_FLAGS = $(KERNEL_OBJ_FLAGS_$(ARCH))
LINK_FLAGS       = --offload-arch=$(ARCH) -pthread

BUILD = build/$(ARCH)
OUT   = $(BUILD)/gqa_fwd.exe

KERNEL_SRCS = $(wildcard $(ARCH)/*_kernel_*.cc)
HOST_SRC    = common/gqa_host.cc
COMMON_HDRS = common/gqa_kargs.h
ARCH_HDRS   = $(wildcard $(ARCH)/*.h) $(wildcard $(ARCH)/*.hpp)
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(notdir $(KERNEL_SRCS)))
KERNEL_ASMS = $(patsubst %.cc,$(BUILD)/%.s,$(notdir $(KERNEL_SRCS)))
KERNEL_DIS  = $(patsubst %.cc,$(BUILD)/%.dis.s,$(notdir $(KERNEL_SRCS)))
HOST_OBJ    = $(BUILD)/gqa_host.o

.PHONY: all asm dis clean
.DELETE_ON_ERROR:

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

# Kernel objects. -D__HIPCC_RTC__ skips the implicit __clang_hip_runtime_wrapper.h on the host pass. Each kernel .cc
# includes only its own template; depending on all of the arch's headers keeps the rule simple.
$(BUILD)/%.o: $(ARCH)/%.cc $(ARCH_HDRS) $(COMMON_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) $(KERNEL_OBJ_FLAGS) -c -o $@

$(BUILD)/%.s: $(ARCH)/%.cc $(ARCH_HDRS) $(COMMON_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) --offload-device-only -S -o $@

asm: $(KERNEL_ASMS)

# Disassembly of the device code objects.
OBJDUMP ?= $(if $(HIP_CLANG_PATH),$(HIP_CLANG_PATH),$(dir $(HIPCC))../lib/llvm/bin)/llvm-objdump
dis: $(KERNEL_DIS)
$(BUILD)/%.dis.s: $(ARCH)/%.cc $(ARCH_HDRS) $(COMMON_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(KERNEL_FLAGS) --offload-device-only --no-gpu-bundle-output -c -o $(BUILD)/$*.co
	$(OBJDUMP) -d --mcpu=$(ARCH) $(BUILD)/$*.co > $@

# Host object: no -D__HIPCC_RTC__ (the <<<>>> launches need the full HIP runtime), no fast math (the checks test for
# inf and NaN).
$(HOST_OBJ): $(HOST_SRC) $(ARCH)/gqa_traits.h $(COMMON_HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ $(LINK_FLAGS) -o $@

clean:
	rm -rf build
