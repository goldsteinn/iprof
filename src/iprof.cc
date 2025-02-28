#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/random.h>
#include <sys/time.h>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

#include <unistd.h>

extern "C" {
#include "qemu-plugin.h"
}

#define IPROF_UNLIKELY(cond) __builtin_expect(cond, 0)

#define IPROF_ENV_GET_LLVM_MTRIPLE "IPROF_LLVM_MTRIPLE"
#define IPROF_ENV_GET_LLVM_MCPU "IPROF_LLVM_MCPU"
#define IPROF_ENV_GET_LLVM_MATTRS "IPROF_LLVM_MATTRS"

#define iprof_assert(expr) assert(expr)
#define iprof_unreachable(msg)                                                 \
  fprintf(stderr, "%s\n", msg);                                                \
  std::abort()

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

namespace iprof {
static const llvm::Target *IProfGetTargetAndTriple(const char *ArchName,
                                                   llvm::Triple *TheTripleOut) {
  std::string TripleName{};
  const char *EnvTriple = getenv(IPROF_ENV_GET_LLVM_MTRIPLE);
  if (EnvTriple == nullptr) {
    TripleName = llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());
  } else {
    TripleName = EnvTriple;
  }
  llvm::Triple TheTriple(TripleName);
  std::string Error;
  const llvm::Target *TheTarget =
      llvm::TargetRegistry::lookupTarget(ArchName, TheTriple, Error);

  if (TheTarget == nullptr) {
    return nullptr;
  }
  *TheTripleOut = TheTriple;
  return TheTarget;
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

static constexpr size_t kMaxInstLength = 16;

class BinUtilsT {
  llvm::Triple MTriple_;
  std::string MCPU_;
  std::string MAttrs_;
  const llvm::Target *MTarget_;

  std::unique_ptr<llvm::MCSubtargetInfo> SubtargetInfo_;
  std::unique_ptr<llvm::MCRegisterInfo> MRegInfo_;
  std::unique_ptr<llvm::MCInstrInfo> MIInfo_;
  std::unique_ptr<llvm::MCAsmInfo> AsmInfo_;
  std::unique_ptr<llvm::MCContext> AsmCtx_;
  std::unique_ptr<llvm::MCDisassembler> DisAsm_;

public:
  static void preInit(void) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllDisassemblers();
  }

  BinUtilsT(const char *ArchName) {
    MCPU_ = IProfGetMCPU();
    MAttrs_ = IProfGetMAttrs();
    MTarget_ = IProfGetTargetAndTriple(ArchName, &MTriple_);

    SubtargetInfo_.reset(
        MTarget_->createMCSubtargetInfo(MTriple_.str(), MCPU_, MAttrs_));
    iprof_assert(SubtargetInfo_);
    iprof_assert(SubtargetInfo_->getSchedModel().hasInstrSchedModel());
    MRegInfo_.reset(MTarget_->createMCRegInfo(MTriple_.str()));
    iprof_assert(MRegInfo_);
    MIInfo_.reset(MTarget_->createMCInstrInfo());
    iprof_assert(MIInfo_);
    llvm::MCTargetOptions MCOptions{};
    AsmInfo_.reset(
        MTarget_->createMCAsmInfo(*MRegInfo_.get(), MTriple_.str(), MCOptions));
    iprof_assert(AsmInfo_);
    AsmCtx_ = std::make_unique<llvm::MCContext>(
        MTriple_, AsmInfo_.get(), MRegInfo_.get(), SubtargetInfo_.get());
    iprof_assert(AsmCtx_);
    DisAsm_.reset(
        MTarget_->createMCDisassembler(*SubtargetInfo_.get(), *AsmCtx_.get()));
    iprof_assert(DisAsm_);
  }

  unsigned getNumOpcodes() const { return MIInfo_->getNumOpcodes(); }
  const char *getOpcName(unsigned Opc) const {
    return MIInfo_->getName(Opc).data();
  }
  unsigned getOpcode(const uint8_t *Bytes, size_t Length) const {
    llvm::MCInst MInst;
    uint64_t Size;
    bool Disassembled = DisAsm_->getInstruction(
        MInst, Size, llvm::ArrayRef<uint8_t>{Bytes, Length}, 0, llvm::nulls());
    return MInst.getOpcode();
    (void)Disassembled;
  }
};

static std::unique_ptr<BinUtilsT> GBinUtils{};
static struct qemu_plugin_scoreboard *GScoreBoard = NULL;

static void IProfOnTranslation(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
  static constexpr size_t kMaxInsnLength = 64;
  size_t NumInst = qemu_plugin_tb_n_insns(tb);
  uint8_t Bytes[kMaxInsnLength] = {0};
  llvm::SmallVector<unsigned> Opcodes;
  Opcodes.reserve(NumInst);
  for (size_t I = 0; I < NumInst; ++I) {
    struct qemu_plugin_insn *Inst = qemu_plugin_tb_get_insn(tb, I);
    size_t InstSize = qemu_plugin_insn_data(Inst, Bytes, kMaxInsnLength);
    Opcodes.emplace_back(GBinUtils->getOpcode(Bytes, InstSize));
  }

  std::sort(Opcodes.begin(), Opcodes.end());
  unsigned LastOpc = Opcodes[0];
  unsigned Cnt = 1;
  for (unsigned I = 1, E = Opcodes.size(); I < E; ++I) {
    unsigned CurOpc = Opcodes[I];
    if (CurOpc == LastOpc) {
      ++Cnt;
      continue;
    }
    qemu_plugin_u64 Score = {GScoreBoard, LastOpc * sizeof(uint64_t)};
    qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu(
        tb, QEMU_PLUGIN_INLINE_ADD_U64, Score, Cnt);
    LastOpc = CurOpc;
    Cnt = 1;
  }
  qemu_plugin_u64 Score = {GScoreBoard, LastOpc * sizeof(uint64_t)};
  qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu(
      tb, QEMU_PLUGIN_INLINE_ADD_U64, Score, Cnt);

  (void)id;
}

static void IProfOnExit(qemu_plugin_id_t id, void *userdata) {
  uint64_t Total = 0;
  size_t NumOpc = GBinUtils->getNumOpcodes();
  char Path[512] = "";
  uint64_t RandomBytes[2] = {0};
  struct timeval TV;
  gettimeofday(&TV, NULL);
  getrandom(RandomBytes, sizeof(RandomBytes), 0);
  sprintf(Path, "insn-counts-%lu-%lu-%lx%lx.txt", TV.tv_sec, id, RandomBytes[0],
          RandomBytes[1]);
  FILE *FPCounts = fopen(Path, "w+");
  iprof_assert(FPCounts);
  std::vector<std::pair<const char *, uint64_t>> OpcCounts{};
  for (uint64_t Off = 0; Off < NumOpc; ++Off) {
    qemu_plugin_u64 Score = {GScoreBoard, Off * sizeof(uint64_t)};
    uint64_t Sum = qemu_plugin_u64_sum(Score);
    if (Sum) {
      OpcCounts.emplace_back(GBinUtils->getOpcName(Off), Sum);
      Total += Sum;
    }
  }
  std::sort(OpcCounts.begin(), OpcCounts.end(),
            [](const std::pair<const char *, uint64_t> &Lhs,
               const std::pair<const char *, uint64_t> &Rhs) {
              return Lhs.second < Rhs.second;
            });
  for (unsigned I = 0, E = OpcCounts.size(); I < E; ++I) {
    fprintf(FPCounts, "%-32s -> %-16lu\n", OpcCounts[I].first,
            OpcCounts[I].second);
  }
  fprintf(FPCounts, "TotalInsn: %lu\n", Total);
  fclose(FPCounts);

  qemu_plugin_scoreboard_free(GScoreBoard);
  (void)id;
  (void)userdata;
}

} // namespace iprof

int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc,
                        char **argv) {
  (void)info;
  (void)argc;
  (void)argv;
  fprintf(stderr, "id: %lu\n", id);
  const char *ArchName = iprof::IProfQEMUArchToLLVMArch(info->target_name);
  if (ArchName == nullptr) {
    fprintf(stderr, "Unable to detect correct LLVM arch from '%s'\n",
            info->target_name);
    return -1;
  }

  iprof::BinUtilsT::preInit();
  iprof::GBinUtils = std::make_unique<iprof::BinUtilsT>(ArchName);
  iprof_assert(iprof::GBinUtils);
  iprof::GScoreBoard = qemu_plugin_scoreboard_new(
      iprof::GBinUtils->getNumOpcodes() * sizeof(uint64_t));
  iprof_assert(iprof::GScoreBoard);

  qemu_plugin_register_vcpu_tb_trans_cb(id, iprof::IProfOnTranslation);
  qemu_plugin_register_atexit_cb(id, iprof::IProfOnExit, nullptr);
  return 0;
}
