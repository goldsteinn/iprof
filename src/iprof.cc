#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MCA/Context.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/IncrementalSourceMgr.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/MCA/SourceMgr.h"
#include "llvm/MCA/View.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

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

namespace llvm_ext {

class StreamingSourceMgr : public llvm::mca::SourceMgr {
  std::deque<UniqueInst> Staging;
  /// Current instruction index.
  unsigned TotalCounter = 0U;

  /// End-of-stream flag.
  bool EOS = false;

public:
  StreamingSourceMgr() = default;
  bool hasNext() const override { return !Staging.empty(); }
  bool isEnd() const override { return EOS; }
  llvm::ArrayRef<UniqueInst> getInstructions() const override {
    iprof_unreachable("Invalid in this context");
  }
  llvm::mca::SourceRef peekNext() const override {
    iprof_assert(hasNext());
    return llvm::mca::SourceRef(TotalCounter, *Staging.front());
  }

  /// Add a new instruction.
  void addInst(UniqueInst &&Inst) { Staging.push_back(std::move(Inst)); }
  void updateNext() override {
    iprof_assert(hasNext());
    ++TotalCounter;
    Staging.front().release();
    Staging.pop_front();
  }

  /// Mark the end of instruction stream.
  void endOfStream() { EOS = true; }
};

/// A view that collects and prints a few performance numbers.
class SummaryView : public llvm::mca::View {
  const llvm::MCSchedModel &SM;

public:
  uint64_t TotalInsts;
  const unsigned DispatchWidth;
  unsigned LastInstructionIdx;
  unsigned TotalCycles;
  // The total number of micro opcodes contributed by a block of instructions.
  unsigned NumMicroOps;

  struct DisplayValues {
    unsigned Instructions;
    unsigned Iterations;
    unsigned TotalInstructions;
    unsigned TotalCycles;
    unsigned DispatchWidth;
    unsigned TotalUOps;
    double IPC;
    double UOpsPerCycle;
    double BlockRThroughput;
  };

  // For each processor resource, this vector stores the cumulative number of
  // resource cycles consumed by the analyzed code block.
  llvm::SmallVector<unsigned, 8> ProcResourceUsage;

  // Each processor resource is associated with a so-called processor resource
  // mask. This vector allows to correlate processor resource IDs with processor
  // resource masks. There is exactly one element per each processor resource
  // declared by the scheduling model.
  llvm::SmallVector<uint64_t, 8> ProcResourceMasks;

  // Used to map resource indices to actual processor resource IDs.
  llvm::SmallVector<unsigned, 8> ResIdx2ProcResID;

  /// Compute the data we want to print out in the object DV.
  void collectData(DisplayValues &DV) const {
    DV.Instructions = TotalInsts;
    DV.Iterations = (LastInstructionIdx / DV.Instructions) + 1;
    DV.TotalInstructions = DV.Instructions * DV.Iterations;
    DV.TotalCycles = TotalCycles;
    DV.DispatchWidth = DispatchWidth;
    DV.TotalUOps = NumMicroOps * DV.Iterations;
    DV.UOpsPerCycle = (double)DV.TotalUOps / TotalCycles;
    DV.IPC = (double)DV.TotalInstructions / TotalCycles;
    DV.BlockRThroughput = llvm::mca::computeBlockRThroughput(
        SM, DispatchWidth, NumMicroOps, ProcResourceUsage);
  }

  SummaryView(const llvm::MCSchedModel &Model, unsigned Width)
      : SM(Model), TotalInsts(0),
        DispatchWidth(Width ? Width : Model.IssueWidth), LastInstructionIdx(0),
        TotalCycles(0), NumMicroOps(0),
        ProcResourceUsage(Model.getNumProcResourceKinds(), 0),
        ProcResourceMasks(Model.getNumProcResourceKinds()),
        ResIdx2ProcResID(Model.getNumProcResourceKinds(), 0) {
    llvm::mca::computeProcResourceMasks(SM, ProcResourceMasks);
    for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I) {
      unsigned Index = llvm::mca::getResourceStateIndex(ProcResourceMasks[I]);
      ResIdx2ProcResID[Index] = I;
    }
  }
  void onCycleEnd() override { ++TotalCycles; }
  void addInst(void) { ++TotalInsts; }

  void onEvent(const llvm::mca::HWInstructionEvent &Event) override {
    if (Event.Type == llvm::mca::HWInstructionEvent::Dispatched)
      LastInstructionIdx = Event.IR.getSourceIndex();

    // We are only interested in the "instruction retired" events generated by
    // the retire stage for instructions that are part of iteration #0.
    if (Event.Type != llvm::mca::HWInstructionEvent::Retired ||
        Event.IR.getSourceIndex() >= TotalInsts)
      return;

    // Update the cumulative number of resource cycles based on the processor
    // resource usage information available from the instruction descriptor.
    // We need to compute the cumulative number of resource cycles for every
    // processor resource which is consumed by an instruction of the block.
    const llvm::mca::Instruction &Inst = *Event.IR.getInstruction();
    const llvm::mca::InstrDesc &Desc = Inst.getDesc();
    NumMicroOps += Desc.NumMicroOps;
    for (const std::pair<uint64_t, llvm::mca::ResourceUsage> &RU :
         Desc.Resources) {
      if (RU.second.size()) {
        unsigned ProcResID =
            ResIdx2ProcResID[llvm::mca::getResourceStateIndex(RU.first)];
        ProcResourceUsage[ProcResID] += RU.second.size();
      }
    }
  }
  void printView(llvm::raw_ostream &OS) const override { (void)OS; }
  llvm::StringRef getNameAsString() const override { return "SummaryView"; }
  llvm::json::Value toJSON() const override {
    DisplayValues DV;
    collectData(DV);
    llvm::json::Object JO({{"Iterations", DV.Iterations},
                           {"Instructions", DV.TotalInstructions},
                           {"TotalCycles", DV.TotalCycles},
                           {"TotaluOps", DV.TotalUOps},
                           {"DispatchWidth", DV.DispatchWidth},
                           {"uOpsPerCycle", DV.UOpsPerCycle},
                           {"IPC", DV.IPC},
                           {"BlockRThroughput", DV.BlockRThroughput}});
    return JO;
  }
};
} // namespace llvm_ext

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
struct InstDataT {
  static constexpr uintptr_t kMaskBits = 8;
  uint8_t Bytes_[sizeof(void *)];

  uintptr_t asRawPtrInt() const {
    uintptr_t Val;
    std::memcpy(&Val, Bytes_, sizeof(void *));
    return Val;
  }

  uintptr_t asPtrInt() const { return asRawPtrInt() >> kMaskBits; }
  bool isSmall() const {
    return static_cast<uint8_t>(asRawPtrInt()) < sizeof(void *);
  }
  bool isLarge() const { return !isSmall(); }

  static InstDataT create(uintptr_t PtrInt) {
    InstDataT InstData;
    std::memcpy(InstData.Bytes_, &PtrInt, sizeof(void *));
    return InstData;
  }

  static InstDataT create(size_t Size, const uint8_t *Bytes) {
    InstDataT InstData;
    iprof_assert(Size < kMaxInstLength && Size != 0);
    if (IPROF_UNLIKELY(Size >= sizeof(void *))) {
      uint8_t *Storage = reinterpret_cast<uint8_t *>(malloc(kMaxInstLength));
      assert(Storage != nullptr);

      std::memcpy(Storage, Bytes, kMaxInstLength);
      Storage[kMaxInstLength - 1] = static_cast<uint8_t>(Size);
      uintptr_t PtrInt =
          (reinterpret_cast<uintptr_t>(Storage) << kMaskBits) | Size;

      InstData = InstDataT::create(PtrInt);
      iprof_assert(InstData.isLarge());
    } else {
      std::memcpy(&InstData.Bytes_[1], Bytes, sizeof(void *));
      InstData.Bytes_[0] = static_cast<uint8_t>(Size);
      iprof_assert(InstData.isSmall());
    }
    return InstData;
  }

  const uint8_t *bytes() const {
    if (IPROF_UNLIKELY(isLarge())) {
      return reinterpret_cast<uint8_t *>(asPtrInt());
    } else {
      return &Bytes_[1];
    }
  }
  size_t size() const { return Bytes_[0]; }
};

class MCAPipeline {
  struct MCAPipelineComponents {
    std::unique_ptr<llvm::MCSubtargetInfo> SubtargetInfo_;
    std::unique_ptr<llvm::MCRegisterInfo> MRegInfo_;
    std::unique_ptr<llvm::MCInstrInfo> MIInfo_;
    std::unique_ptr<llvm::MCInstrAnalysis> MIAnalysis_;
    std::unique_ptr<llvm::MCAsmInfo> AsmInfo_;
    std::unique_ptr<llvm::MCContext> AsmCtx_;
    std::unique_ptr<llvm::MCDisassembler> DisAsm_;
    std::unique_ptr<llvm::mca::InstrumentManager> InstrumentMgr_;
    std::unique_ptr<llvm::mca::InstrBuilder> InstBuilder_;
    std::unique_ptr<llvm_ext::StreamingSourceMgr> SrcMgr_;
    std::unique_ptr<llvm::mca::CustomBehaviour> CustomPipelineBehavior_;
    std::unique_ptr<llvm::mca::Pipeline> Pipeline_;
    std::unique_ptr<llvm::mca::Context> MCCtx_;
    std::unique_ptr<llvm_ext::SummaryView> SummaryView_;

    MCAPipelineComponents(const llvm::Triple &MTriple, const std::string &MCPU,
                          const std::string &MAttrs,
                          const llvm::Target *MTarget) {
      SubtargetInfo_.reset(
          MTarget->createMCSubtargetInfo(MTriple.str(), MCPU, MAttrs));
      iprof_assert(SubtargetInfo_);
      iprof_assert(SubtargetInfo_->getSchedModel().hasInstrSchedModel());
      MRegInfo_.reset(MTarget->createMCRegInfo(MTriple.str()));
      iprof_assert(MRegInfo_);
      MIInfo_.reset(MTarget->createMCInstrInfo());
      iprof_assert(MIInfo_);
      MIAnalysis_.reset(MTarget->createMCInstrAnalysis(MIInfo_.get()));
      iprof_assert(MIAnalysis_);
      llvm::MCTargetOptions MCOptions{};
      AsmInfo_.reset(
          MTarget->createMCAsmInfo(*MRegInfo_.get(), MTriple.str(), MCOptions));
      iprof_assert(AsmInfo_);
      AsmCtx_ = std::make_unique<llvm::MCContext>(
          MTriple, AsmInfo_.get(), MRegInfo_.get(), SubtargetInfo_.get());
      iprof_assert(AsmCtx_);
      DisAsm_.reset(
          MTarget->createMCDisassembler(*SubtargetInfo_.get(), *AsmCtx_.get()));
      iprof_assert(DisAsm_);
      InstrumentMgr_ = std::make_unique<llvm::mca::InstrumentManager>(
          *SubtargetInfo_.get(), *MIInfo_.get());
      iprof_assert(InstrumentMgr_);
      InstBuilder_ = std::make_unique<llvm::mca::InstrBuilder>(
          *SubtargetInfo_.get(), *MIInfo_.get(), *MRegInfo_.get(),
          MIAnalysis_.get(), *InstrumentMgr_.get(),
          /*CallLatency=*/3);
      iprof_assert(InstBuilder_);
      MCCtx_ = std::make_unique<llvm::mca::Context>(*MRegInfo_.get(),
                                                    *SubtargetInfo_.get());
      iprof_assert(MCCtx_);
      SrcMgr_ = std::make_unique<llvm_ext::StreamingSourceMgr>();
      iprof_assert(SrcMgr_);
      CustomPipelineBehavior_ = std::make_unique<llvm::mca::CustomBehaviour>(
          *SubtargetInfo_.get(), *SrcMgr_.get(), *MIInfo_.get());
      iprof_assert(CustomPipelineBehavior_);
      llvm::mca::PipelineOptions PO(/*MicroOpQueue=*/0, /*DecoderThroughput=*/0,
                                    /*DispatchWidth=*/0,
                                    /*RegisterFileSize=*/0,
                                    /*LoadQueueSize=*/0, /*StoreQueueSize=*/0,
                                    /*AssumeNoAlias=*/true,
                                    /*EnableBottleneckAnalysis=*/false);
      Pipeline_ = MCCtx_->createDefaultPipeline(PO, *SrcMgr_.get(),
                                                *CustomPipelineBehavior_.get());
      iprof_assert(Pipeline_);
      SummaryView_ = std::make_unique<llvm_ext::SummaryView>(
          SubtargetInfo_->getSchedModel(), PO.DispatchWidth);
      iprof_assert(SummaryView_);
      Pipeline_->addEventListener(SummaryView_.get());
    }

    void handleInst(InstDataT InstData) {
      llvm::MCInst MInst;
      uint64_t Size;
      bool Disassembled = DisAsm_->getInstruction(
          MInst, Size,
          llvm::ArrayRef<uint8_t>{InstData.bytes(), InstData.size()}, 0,
          llvm::nulls());
      SummaryView_->addInst();

      const llvm::SmallVector<llvm::mca::Instrument *> Instruments;
      llvm::Expected<std::unique_ptr<llvm::mca::Instruction>> ProfInst =
          InstBuilder_->createInstruction(MInst, Instruments);
      iprof_assert(ProfInst);
      SrcMgr_->addInst(std::move(ProfInst.get()));
      llvm::Expected<unsigned> Cycles = Pipeline_->run();
      iprof_assert(Cycles || Cycles.errorIsA<llvm::mca::InstStreamPause>());
      (void)Disassembled;
    }
    void finish(void) {
      SrcMgr_->endOfStream();
      llvm::Expected<unsigned> Cycles = Pipeline_->run();
      assert(Cycles);
    }
  };

  llvm::Triple MTriple_;
  std::string MCPU_;
  std::string MAttrs_;
  const llvm::Target *MTarget_;
  std::vector<MCAPipelineComponents> Pipelines_;

public:
  static void preInit(void) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllTargetMCAs();
    llvm::InitializeAllDisassemblers();
  }

  MCAPipeline(const char *ArchName, unsigned NumVCPU) {
    if (NumVCPU == 0)
      NumVCPU = 1;
    MCPU_ = IProfGetMCPU();
    MAttrs_ = IProfGetMAttrs();
    MTarget_ = IProfGetTargetAndTriple(ArchName, &MTriple_);
    for (unsigned I = 0; I < NumVCPU; ++I) {
      Pipelines_.push_back(
          MCAPipelineComponents(MTriple_, MCPU_, MAttrs_, MTarget_));
    }
  }

  void handleInst(unsigned VCPU, InstDataT InstData) {
    iprof_assert(VCPU < Pipelines_.size());
    Pipelines_[VCPU].handleInst(InstData);
  }

  void finish(void) {
    for (unsigned I = 0; I < Pipelines_.size(); ++I) {
      Pipelines_[I].finish();
    }
  }

  void summarize(void) const {
    FILE *fp = fopen("result4.json", "w+");
    for (unsigned I = 0; I < Pipelines_.size(); ++I) {
      llvm::json::Value Result = Pipelines_[I].SummaryView_->toJSON();
      std::string ResStr;
      llvm::raw_string_ostream ROS(ResStr);
      ROS << Result;
      fprintf(fp, "%s\n", ResStr.c_str());
    }
    fclose(fp);
  }
};

static const char *gArchName = NULL;
static std::unique_ptr<MCAPipeline> Pipeline{};
static std::vector<InstDataT> AsExecuted{};

static void IProfOnExec(unsigned int vcpu_index, void *userdata) {
  InstDataT InstData = InstDataT::create(reinterpret_cast<uintptr_t>(userdata));
  AsExecuted.emplace_back(InstData);
  iprof_assert(InstData.size() != 0 && InstData.size() < kMaxInstLength);
  Pipeline->handleInst(vcpu_index, InstData);
  (void)vcpu_index;
  (void)userdata;
}

static void IProfOnTranslation(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
  size_t NumInst = qemu_plugin_tb_n_insns(tb);
  for (size_t I = 0; I < NumInst; ++I) {
    struct qemu_plugin_insn *Inst = qemu_plugin_tb_get_insn(tb, I);
    uint8_t Bytes[kMaxInstLength] = {0};
    size_t InstSize = qemu_plugin_insn_data(Inst, Bytes, kMaxInstLength);
    InstDataT InstData = InstDataT::create(InstSize, Bytes);

    iprof_assert(InstSize != 0);
    iprof_assert(InstSize == InstData.size());
    qemu_plugin_register_vcpu_insn_exec_cb(
        Inst, iprof::IProfOnExec, QEMU_PLUGIN_CB_NO_REGS,
        reinterpret_cast<void *>(InstData.asRawPtrInt()));
  }
  (void)id;
}

static void IProfOnExit(qemu_plugin_id_t id, void *userdata) {
#if 0
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllTargetMCAs();
  llvm::InitializeAllDisassemblers();
#endif

  Pipeline->finish();
  Pipeline->summarize();

  llvm::Triple MTriple;
  std::string MCPU = IProfGetMCPU();
  std::string MAttrs = IProfGetMAttrs();
  const llvm::Target *TheTarget = IProfGetTargetAndTriple(gArchName, &MTriple);

  assert(TheTarget != nullptr);
  std::unique_ptr<llvm::MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(MTriple.str(), MCPU, MAttrs));

  assert(STI);
  assert(STI->getSchedModel().hasInstrSchedModel());

  std::unique_ptr<llvm::MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(MTriple.str()));
  assert(MRI);

  std::unique_ptr<llvm::MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
  assert(MCII);

  std::unique_ptr<llvm::MCInstrAnalysis> MCIA(
      TheTarget->createMCInstrAnalysis(MCII.get()));
  assert(MCIA);

  llvm::MCTargetOptions MCOptions{};
  std::unique_ptr<llvm::MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, MTriple.str(), MCOptions));
  assert(MAI);

  llvm::MCContext ACtx(llvm::Triple(MTriple), MAI.get(), MRI.get(), STI.get());

  std::unique_ptr<llvm::MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*STI.get(), ACtx));
  assert(DisAsm);

  std::vector<llvm::MCInst> Insts;
  Insts.reserve(AsExecuted.size());
  for (const InstDataT &InstData : AsExecuted) {
    llvm::MCInst Inst;
    uint64_t Size;
    bool Disassembled = DisAsm->getInstruction(
        Inst, Size, llvm::ArrayRef<uint8_t>{InstData.bytes(), InstData.size()},
        0, llvm::nulls());
    Insts.emplace_back(Inst);
    (void)Disassembled;
  }

  const llvm::SmallVector<llvm::mca::Instrument *> Instruments;
  {
    FILE *fp = fopen("result.json", "w+");
    std::unique_ptr<llvm::mca::InstrumentManager> IM =
        std::make_unique<llvm::mca::InstrumentManager>(*STI, *MCII);
    llvm::mca::InstrBuilder IB(*STI, *MCII, *MRI, MCIA.get(), *IM,
                               /*CallLatency=*/3);
    std::vector<std::unique_ptr<llvm::mca::Instruction>> ReadyInsts;
    for (const llvm::MCInst &I : Insts) {
      llvm::Expected<std::unique_ptr<llvm::mca::Instruction>> Inst =
          IB.createInstruction(I, Instruments);
      assert(Inst);
      ReadyInsts.emplace_back(std::move(Inst.get()));
    }

    llvm::mca::CircularSourceMgr SrcMgr(ReadyInsts, 1);

    llvm::mca::PipelineOptions PO(/*MicroOpQueue=*/0, /*DecoderThroughput=*/0,
                                  /*DispatchWidth=*/0,
                                  /*RegisterFileSize=*/0,
                                  /*LoadQueueSize=*/0, /*StoreQueueSize=*/0,
                                  /*AssumeNoAlias=*/true,
                                  /*EnableBottleneckAnalysis=*/false);

    std::unique_ptr<llvm::mca::CustomBehaviour> CB =
        std::make_unique<llvm::mca::CustomBehaviour>(*STI, SrcMgr, *MCII);
    llvm::mca::Context MCtx(*MRI, *STI);
    auto P = MCtx.createDefaultPipeline(PO, SrcMgr, *CB);

    std::unique_ptr<llvm_ext::SummaryView> SV =
        std::make_unique<llvm_ext::SummaryView>(STI->getSchedModel(),
                                                PO.DispatchWidth);
    SV->TotalInsts = Insts.size();
    P->addEventListener(SV.get());
    llvm::Expected<unsigned> Cycles = P->run();
    assert(Cycles);
    llvm::json::Value Result = SV->toJSON();
    std::string ResStr;
    llvm::raw_string_ostream ROS(ResStr);
    ROS << Result;
    fprintf(fp, "%s\n", ResStr.c_str());

    fclose(fp);
  }

  {
    FILE *fp = fopen("result2.json", "w+");
    std::unique_ptr<llvm::mca::InstrumentManager> IM =
        std::make_unique<llvm::mca::InstrumentManager>(*STI, *MCII);
    llvm::mca::InstrBuilder IB(*STI, *MCII, *MRI, MCIA.get(), *IM,
                               /*CallLatency=*/3);

    llvm::mca::IncrementalSourceMgr SrcMgr;
    llvm::mca::PipelineOptions PO(/*MicroOpQueue=*/0, /*DecoderThroughput=*/0,
                                  /*DispatchWidth=*/0,
                                  /*RegisterFileSize=*/0,
                                  /*LoadQueueSize=*/0, /*StoreQueueSize=*/0,
                                  /*AssumeNoAlias=*/true,
                                  /*EnableBottleneckAnalysis=*/false);

    std::unique_ptr<llvm::mca::CustomBehaviour> CB =
        std::make_unique<llvm::mca::CustomBehaviour>(*STI, SrcMgr, *MCII);
    llvm::mca::Context MCtx(*MRI, *STI);
    auto P = MCtx.createDefaultPipeline(PO, SrcMgr, *CB);

    std::unique_ptr<llvm_ext::SummaryView> SV =
        std::make_unique<llvm_ext::SummaryView>(STI->getSchedModel(),
                                                PO.DispatchWidth);
    SV->TotalInsts = Insts.size();
    P->addEventListener(SV.get());

    for (const llvm::MCInst &I : Insts) {
      llvm::Expected<std::unique_ptr<llvm::mca::Instruction>> Inst =
          IB.createInstruction(I, Instruments);
      assert(Inst);
      SrcMgr.addInst(std::move(Inst.get()));
      llvm::Expected<unsigned> Cycles = P->run();
      assert(Cycles || Cycles.errorIsA<llvm::mca::InstStreamPause>());
    }

    SrcMgr.endOfStream();
    llvm::Expected<unsigned> Cycles = P->run();

    assert(Cycles);
    llvm::json::Value Result = SV->toJSON();
    std::string ResStr;
    llvm::raw_string_ostream ROS(ResStr);
    ROS << Result;
    fprintf(fp, "%s\n", ResStr.c_str());

    fclose(fp);
  }

  {
    FILE *fp = fopen("result3.json", "w+");
    std::unique_ptr<llvm::mca::InstrumentManager> IM =
        std::make_unique<llvm::mca::InstrumentManager>(*STI, *MCII);
    llvm::mca::InstrBuilder IB(*STI, *MCII, *MRI, MCIA.get(), *IM,
                               /*CallLatency=*/3);

    llvm_ext::StreamingSourceMgr SrcMgr;
    llvm::mca::PipelineOptions PO(/*MicroOpQueue=*/0, /*DecoderThroughput=*/0,
                                  /*DispatchWidth=*/0,
                                  /*RegisterFileSize=*/0,
                                  /*LoadQueueSize=*/0, /*StoreQueueSize=*/0,
                                  /*AssumeNoAlias=*/true,
                                  /*EnableBottleneckAnalysis=*/false);

    std::unique_ptr<llvm::mca::CustomBehaviour> CB =
        std::make_unique<llvm::mca::CustomBehaviour>(*STI, SrcMgr, *MCII);
    llvm::mca::Context MCtx(*MRI, *STI);
    auto P = MCtx.createDefaultPipeline(PO, SrcMgr, *CB);

    std::unique_ptr<llvm_ext::SummaryView> SV =
        std::make_unique<llvm_ext::SummaryView>(STI->getSchedModel(),
                                                PO.DispatchWidth);
    SV->TotalInsts = Insts.size();
    P->addEventListener(SV.get());

    for (const llvm::MCInst &I : Insts) {
      llvm::Expected<std::unique_ptr<llvm::mca::Instruction>> Inst =
          IB.createInstruction(I, Instruments);
      assert(Inst);
      SrcMgr.addInst(std::move(Inst.get()));
      llvm::Expected<unsigned> Cycles = P->run();
      assert(Cycles || Cycles.errorIsA<llvm::mca::InstStreamPause>());
    }

    SrcMgr.endOfStream();
    llvm::Expected<unsigned> Cycles = P->run();

    assert(Cycles);
    llvm::json::Value Result = SV->toJSON();
    std::string ResStr;
    llvm::raw_string_ostream ROS(ResStr);
    ROS << Result;
    fprintf(fp, "%s\n", ResStr.c_str());

    fclose(fp);
  }

  {
    std::unique_ptr<llvm::MCInstPrinter> MIP(TheTarget->createMCInstPrinter(
        llvm::Triple(MTriple), MAI->getAssemblerDialect(), *MAI.get(),
        *MCII.get(), *MRI.get()));
    assert(MIP);

    FILE *fp_disasm = fopen("collected.txt", "w+");
    for (const InstDataT &InstData : AsExecuted) {
      llvm::MCInst Inst;
      uint64_t Size;
      bool Disassembled = DisAsm->getInstruction(
          Inst, Size,
          llvm::ArrayRef<uint8_t>{InstData.bytes(), InstData.size()}, 0,
          llvm::nulls());
      std::string asm_str{};
      llvm::raw_string_ostream os(asm_str);
      MIP->printInst(&Inst, 0, "", *STI.get(), os);
      fprintf(fp_disasm, "%s\t#%zu vs %zu%s\n", asm_str.c_str(), Size,
              InstData.size(), Size == InstData.size() ? "" : " !!!!!!!!!!!!");
      (void)Disassembled;
    }

    fclose(fp_disasm);
  }

  (void)id;
  (void)userdata;
}

} // namespace iprof

int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc,
                        char **argv) {
  (void)info;
  (void)argc;
  (void)argv;

  const char *ArchName = iprof::IProfQEMUArchToLLVMArch(info->target_name);
  if (ArchName == nullptr) {
    fprintf(stderr, "Unable to detect correct LLVM arch from '%s'\n",
            info->target_name);
    return -1;
  }

  iprof::gArchName = ArchName;
  iprof::MCAPipeline::preInit();

  fprintf(stderr, "%d / %d / %d\n", qemu_plugin_num_vcpus(),
          info->system.max_vcpus, info->system.smp_vcpus);
  iprof::Pipeline =
      std::make_unique<iprof::MCAPipeline>(ArchName, info->system.max_vcpus);
  iprof_assert(iprof::Pipeline);

  qemu_plugin_register_vcpu_tb_trans_cb(id, iprof::IProfOnTranslation);
  qemu_plugin_register_atexit_cb(id, iprof::IProfOnExit, nullptr);
  return 0;
}
