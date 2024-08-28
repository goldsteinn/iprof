#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

extern "C" {
#include "qemu-plugin.h"
}

#define IPROF_ENV_GET_LLVM_MTRIPLE "IPROF_LLVM_MTRIPLE"
#define IPROF_ENV_GET_LLVM_MCPU "IPROF_LLVM_MCPU"
#define IPROF_ENV_GET_LLVM_MATTRS "IPROF_LLVM_MATTRS"

#define iprof_assert(expr) assert(expr)

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

namespace iprof {
namespace detail::wyhash {

static inline void mum(uint64_t *a, uint64_t *b) {
#if defined(__SIZEOF_INT128__)
  __uint128_t r = *a;
  r *= *b;
  *a = static_cast<uint64_t>(r);
  *b = static_cast<uint64_t>(r >> 64U);
#elif defined(_MSC_VER) && defined(_M_X64)
  *a = _umul128(*a, *b, b);
#else
  uint64_t ha = *a >> 32U;
  uint64_t hb = *b >> 32U;
  uint64_t la = static_cast<uint32_t>(*a);
  uint64_t lb = static_cast<uint32_t>(*b);
  uint64_t hi{};
  uint64_t lo{};
  uint64_t rh = ha * hb;
  uint64_t rm0 = ha * lb;
  uint64_t rm1 = hb * la;
  uint64_t rl = la * lb;
  uint64_t t = rl + (rm0 << 32U);
  auto c = static_cast<uint64_t>(t < rl);
  lo = t + (rm1 << 32U);
  c += static_cast<uint64_t>(lo < t);
  hi = rh + (rm0 >> 32U) + (rm1 >> 32U) + c;
  *a = lo;
  *b = hi;
#endif
}

// multiply and xor mix function, aka MUM
static inline uint64_t mix(uint64_t a, uint64_t b) {
  mum(&a, &b);
  return a ^ b;
}

// read functions. WARNING: we don't care about endianness, so results are
// different on big endian!
static inline uint64_t r8(const uint8_t *p) {
  uint64_t v{};
  std::memcpy(&v, p, 8U);
  return v;
}

static inline uint64_t r4(const uint8_t *p) {
  uint32_t v{};
  std::memcpy(&v, p, 4);
  return v;
}

// reads 1, 2, or 3 bytes
static inline uint64_t r3(const uint8_t *p, size_t k) {
  return (static_cast<uint64_t>(p[0]) << 16U) |
         (static_cast<uint64_t>(p[k >> 1U]) << 8U) | p[k - 1];
}

static inline uint64_t hash(void const *key, size_t len) {
  static constexpr uint64_t secret[4] = {
      UINT64_C(0xa0761d6478bd642f), UINT64_C(0xe7037ed1a0b428db),
      UINT64_C(0x8ebc6af09c88c6e3), UINT64_C(0x589965cc75374cc3)};

  auto const *p = static_cast<uint8_t const *>(key);
  uint64_t seed = secret[0];
  uint64_t a{};
  uint64_t b{};
  if (len <= 16) {
    if (len >= 4) {
      a = (r4(p) << 32U) | r4(p + ((len >> 3U) << 2U));
      b = (r4(p + len - 4) << 32U) | r4(p + len - 4 - ((len >> 3U) << 2U));
    } else if (len > 0) {
      a = r3(p, len);
      b = 0;
    } else {
      a = 0;
      b = 0;
    }
  } else {
    size_t i = len;
    if (i > 48) {
      uint64_t see1 = seed;
      uint64_t see2 = seed;
      do {
        seed = mix(r8(p) ^ secret[1], r8(p + 8) ^ seed);
        see1 = mix(r8(p + 16) ^ secret[2], r8(p + 24) ^ see1);
        see2 = mix(r8(p + 32) ^ secret[3], r8(p + 40) ^ see2);
        p += 48;
        i -= 48;
      } while (i > 48);
      seed ^= see1 ^ see2;
    }
    while (i > 16) {
      seed = mix(r8(p) ^ secret[1], r8(p + 8) ^ seed);
      i -= 16;
      p += 16;
    }
    a = r8(p + i - 16);
    b = r8(p + i - 8);
  }

  return mix(secret[1] ^ len, mix(a ^ secret[1], b ^ seed));
}

} // namespace detail::wyhash

static const char *gArchName = NULL;
static constexpr size_t kMaxInstLength = 16;
struct InstDataT {
  uint8_t Bytes_[kMaxInstLength];

  InstDataT *self() { return this; }
  const InstDataT *self() const { return this; }
  const uint8_t *bytes() const { return &Bytes_[0]; }
  uint8_t *bytes() { return &Bytes_[0]; }
  size_t size() const { return Bytes_[kMaxInstLength - 1]; }
  void setSize(size_t Size) {
    iprof_assert(Size < kMaxInstLength && Size != 0);
    Bytes_[kMaxInstLength - 1] = static_cast<uint8_t>(Size);
  }

  struct Equals {
    bool operator()(const InstDataT &LHS, const InstDataT &RHS) const {
      return std::memcmp(LHS.self(), RHS.self(), sizeof(InstDataT));
    }
  };

  struct Hasher {
    uint64_t operator()(const InstDataT &Item) const {
      return detail::wyhash::hash(Item.self(), sizeof(InstDataT));
    }
  };
};

static std::unordered_set<InstDataT, InstDataT::Hasher, InstDataT::Equals>
    AllInst{};



    
static std ::vector<const InstDataT *> AsExecuted{};

static void IProfOnExec(unsigned int vcpu_index, void *userdata) {
  const InstDataT *InstData = reinterpret_cast<const InstDataT *>(userdata);
  AsExecuted.emplace_back(InstData);
  iprof_assert(InstData->size() != 0 && InstData->size() < kMaxInstLength);
  (void)vcpu_index;
  (void)userdata;
}

static void IProfOnTranslation(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
  size_t NumInst = qemu_plugin_tb_n_insns(tb);
  for (size_t I = 0; I < NumInst; ++I) {
    struct qemu_plugin_insn *Inst = qemu_plugin_tb_get_insn(tb, I);
    InstDataT InstData{};
    size_t InstSize =
        qemu_plugin_insn_data(Inst, InstData.bytes(), kMaxInstLength);
    InstData.setSize(InstSize);
    //    InstData.setAddr(qemu_plugin_insn_vaddr(Inst));
    //    InstData.setDisasm(strdup(qemu_plugin_insn_disas(Inst)));
    iprof_assert(InstSize != 0);

    auto Existing = AllInst.emplace(InstData);
    qemu_plugin_register_vcpu_insn_exec_cb(
        Inst, iprof::IProfOnExec, QEMU_PLUGIN_CB_NO_REGS,
        const_cast<InstDataT *>(Existing.first->self()));
  }
  (void)id;
}

static const llvm::Target *IProfGetTarget(const std::string &TripleName,
                                          const char *ArchName) {
  llvm::Triple TheTriple(TripleName);
  std::string Error;
  const llvm::Target *TheTarget =
      llvm::TargetRegistry::lookupTarget(ArchName, TheTriple, Error);

  if (TheTarget == nullptr) {
    return nullptr;
  }
  return TheTarget;
}

static std::string IProfGetMTriple(void) {
  std::string TripleName{};
  const char *EnvTriple = getenv(IPROF_ENV_GET_LLVM_MTRIPLE);
  if (EnvTriple == nullptr) {
    TripleName = llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());
  } else {
    TripleName = EnvTriple;
  }
  return TripleName;
}

static std::string IProfGetMCPU(void) {
  const char *EnvCPU = getenv(IPROF_ENV_GET_LLVM_MCPU);
  if (EnvCPU == nullptr) {
    EnvCPU = "native";
  }
  std::string CPUName{};
  if (EnvCPU == nullptr || std::strcmp(EnvCPU, "native") == 0) {
    CPUName = std::string(llvm::sys::getHostCPUName());
  } else {
    CPUName = EnvCPU;
  }
  return CPUName;
}

static std::string IProfGetMAttrs(void) {
  const char *EnvAttrs = getenv(IPROF_ENV_GET_LLVM_MATTRS);
  if (EnvAttrs == nullptr) {
    return "";
  }
  return EnvAttrs;
}

static const char *IProfQEMUArchToLLVMArch(const char *QemuArch) {
  if (std::strcmp(QemuArch, "aarch64") == 0) {
    return "aarch64";
  } else if (std::strcmp(QemuArch, "aarch64_be") == 0) {
    return "aarch64_be";
  } else if (std::strcmp(QemuArch, "arm") == 0) {
    return "arm";
  } else if (std::strcmp(QemuArch, "armeb") == 0) {
    return "armeb";
  } else if (std::strcmp(QemuArch, "i386") == 0) {
    return "x86";
  } else if (std::strcmp(QemuArch, "hexagon") == 0) {
    return "hexagon";
  } else if (std::strcmp(QemuArch, "loongarch64") == 0) {
    return "loongarch64";
  } else if (std::strcmp(QemuArch, "m68k") == 0) {
    return "m68k";
  } else if (std::strcmp(QemuArch, "mips") == 0) {
    return "mips";
  } else if (std::strcmp(QemuArch, "mips64") == 0) {
    return "mips64";
  } else if (std::strcmp(QemuArch, "mips64el") == 0) {
    return "mips64el";
  } else if (std::strcmp(QemuArch, "mipsel") == 0) {
    return "mipsel";
  } else if (std::strcmp(QemuArch, "ppc") == 0) {
    return "ppc32";
  } else if (std::strcmp(QemuArch, "ppc64") == 0) {
    return "ppc64";
  } else if (std::strcmp(QemuArch, "ppc64le") == 0) {
    return "ppc64le";
  } else if (std::strcmp(QemuArch, "riscv32") == 0) {
    return "riscv32";
  } else if (std::strcmp(QemuArch, "riscv64") == 0) {
    return "riscv64";
  } else if (std::strcmp(QemuArch, "sparc") == 0) {
    return "sparc";
  } else if (std::strcmp(QemuArch, "x86_64") == 0) {
    return "x86-64";
  } else if (std::strcmp(QemuArch, "xtensa") == 0) {
    return "xtensa";
  } else {
    return nullptr;
  }
}

static void IProfOnExit(qemu_plugin_id_t id, void *userdata) {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllTargetMCAs();
  llvm::InitializeAllDisassemblers();

  std::string MTriple = IProfGetMTriple();
  std::string MCPU = IProfGetMCPU();
  std::string MAttrs = IProfGetMAttrs();
  const llvm::Target *TheTarget = IProfGetTarget(MTriple, gArchName);

  assert(TheTarget != nullptr);
  std::unique_ptr<llvm::MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(MTriple, MCPU, MAttrs));

  assert(STI);
  assert(STI->getSchedModel().hasInstrSchedModel());

  std::unique_ptr<llvm::MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(MTriple));
  assert(MRI);

  std::unique_ptr<llvm::MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
  assert(MCII);

  std::unique_ptr<llvm::MCInstrAnalysis> MCIA(
      TheTarget->createMCInstrAnalysis(MCII.get()));
  assert(MCIA);

  llvm::MCTargetOptions MCOptions{};
  std::unique_ptr<llvm::MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, MTriple, MCOptions));
  assert(MAI);

  llvm::MCContext ACtx(llvm::Triple(MTriple), MAI.get(), MRI.get(), STI.get());

  std::unique_ptr<llvm::MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*STI.get(), ACtx));
  assert(DisAsm);

  std::unique_ptr<llvm::MCInstPrinter> MIP(TheTarget->createMCInstPrinter(
      llvm::Triple(MTriple), MAI->getAssemblerDialect(), *MAI.get(),
      *MCII.get(), *MRI.get()));
  assert(MIP);
#if 0
  FILE *fp = fopen("collected.txt", "w+");
  for (const InstDataT *InstData : AsExecuted) {
    llvm::MCInst Inst;
    uint64_t Size;
    bool Disassembled = DisAsm->getInstruction(
        Inst, Size, llvm::ArrayRef<uint8_t>{InstData->bytes(), kMaxInstLength},
        0, llvm::nulls());

    std::string asm_str{};
    llvm::raw_string_ostream os(asm_str);

    fprintf(fp, "%s\n%zu vs %zu%s\n", asm_str.c_str(), Size, InstData->size(),
            Size == InstData->size() ? "" : " !!!!!!!!!!!!");
    (void)Disassembled;
  }

  fclose(fp);
#endif
  (void)id;
  (void)userdata;
}
} // namespace iprof

int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc,
                        char **argv) {
  (void)info;
  (void)argc;
  (void)argv;

  iprof::gArchName = iprof::IProfQEMUArchToLLVMArch(info->target_name);
  if (iprof::gArchName == nullptr) {
    fprintf(stderr, "Unable to detect correct LLVM arch from '%s'\n",
            info->target_name);
    return -1;
  }

  qemu_plugin_register_vcpu_tb_trans_cb(id, iprof::IProfOnTranslation);
  qemu_plugin_register_atexit_cb(id, iprof::IProfOnExit, nullptr);
  return 0;
}
