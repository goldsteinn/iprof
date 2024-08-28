#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

#define IPROF_UNREACHABLE(msg)                                                 \
  fprintf(stderr, msg);                                                        \
  std::abort();

namespace iprof {

static constexpr int kIProfOkay = 0;
static constexpr int kIProfErr = -1;

static pid_t ptraceInitChild(char **Argv, char **Envp) {
  pid_t Pid = fork();
  // Error.
  if (Pid < 0) {
    fprintf(stderr, "Error forking: %s\n", strerror(errno));
    exit(kIProfErr);
  }
  // Child case.
  if (Pid == 0) {
    // Setup child to be traced
    if (ptrace(PTRACE_TRACEME, /*pid=*/0, /*addr=*/NULL, /*data=*/NULL) ==
        -1L) {
      fprintf(stderr, "Error setting up child ptrace: %s\n", strerror(errno));
      exit(kIProfErr);
    }

    printf("Running: ");
    char **Arg = Argv;
    printf("%s", *Arg);
    for (++Arg; *Arg != NULL; ++Arg) {
      printf(" %s", *Arg);
    }
    printf("\n");

    execvpe(Argv[0], Argv, Envp);
    IPROF_UNREACHABLE("execve() should never return");
  }

  return Pid;
}

static int ptraceSingleStep(pid_t Pid) {
  if (ptrace(PTRACE_SINGLESTEP, Pid, /*addr=*/NULL, /*data=*/NULL) == -1L) {
    fprintf(stderr, "Error performing single-step ptrace: %s\n",
            strerror(errno));
    return kIProfErr;
  }
  return kIProfOkay;
}

static int ptraceGetRip(pid_t Pid, long *RipOut) {
  struct user_regs_struct Registers;
  if (ptrace(PTRACE_GETREGS, Pid, /*addr=*/NULL,
             reinterpret_cast<void *>(&Registers)) == -1L) {
    fprintf(stderr, "Error getting register info from ptrace: %s\n",
            strerror(errno));
    return kIProfErr;
  }
  *RipOut = Registers.rip;
  return kIProfOkay;
}

static void ptraceCleanup(pid_t Pid) {
  ptrace(PTRACE_DETACH, Pid, /*addr=*/NULL, /*data=*/NULL);
}

static void ptraceCleanupErr(pid_t Pid) {
  // Ensure we cleanup child process
  kill(Pid, SIGKILL);
  (void)iprof::ptraceCleanup(Pid);
}

static int getPidStatus(pid_t Pid, int *StatusOut) {
  pid_t Res = waitpid(Pid, StatusOut, /*options=*/0);
  if (Res != Pid) {
    if (Res < 0) {
      fprintf(stderr, "Error waiting for child to stop: %s\n", strerror(errno));
    } else {
      fprintf(stderr, "Received signal from unexpected child\n");
    }
    return kIProfErr;
  }
  return kIProfOkay;
}

static bool checkPidStatus(int Status) { return WIFSTOPPED(Status) != 0; }

static const char *signalToStr(int Sig) {
  switch (Sig) {
  case SIGABRT:
    return "Aborted";
  case SIGALRM:
    return "Alarm clock";
  case SIGBUS:
    return "Bus error";
  case SIGCHLD:
    return "Child exited";
  case SIGCONT:
    return "Continued";
  case SIGFPE:
    return "Floating point exception";
  case SIGHUP:
    return "Hangup";
  case SIGILL:
    return "Illegal instruction";
  case SIGINT:
    return "Interrupt";
  case SIGKILL:
    return "Killed";
  case SIGPIPE:
    return "Broken pipe";
  case SIGPOLL:
    return "I/O possible";
  case SIGPROF:
    return "Profiling timer expired";
  case SIGQUIT:
    return "Quit";
  case SIGSEGV:
    return "Segmentation fault";
  case SIGSTOP:
    return "Stopped (signal)";
  case SIGSYS:
    return "Bad system call";
  case SIGTERM:
    return "Terminated";
  case SIGTRAP:
    return "Trace/breakpoint trap";
  case SIGTSTP:
    return "Stopped";
  case SIGTTIN:
    return "Stopped (tty input)";
  case SIGTTOU:
    return "Stopped (tty output)";
  case SIGURG:
    return "Urgent I/O condition";
  case SIGUSR1:
    return "User defined signal 1";
  case SIGUSR2:
    return "User defined signal 2";
  case SIGXCPU:
    return "CPU time limit exceeded";
  case SIGXFSZ:
    return "File size limit exceeded";
  default:
    return "Unknown signal";
  }
}

static void printPidStatus(int Status) {
  if (WIFSTOPPED(Status)) {
    printf("Child Stopped: [%d] -> %s\n", WSTOPSIG(Status),
           iprof::signalToStr(WSTOPSIG(Status)));
  }
  if (WIFEXITED(Status)) {
    printf("Child Exited: %d\n", WEXITSTATUS(Status));
  }
  if (WIFSIGNALED(Status)) {
    printf("Child Signaled: [%d] -> %s\n", WTERMSIG(Status),
           iprof::signalToStr(WTERMSIG(Status)));
  }
  if (WCOREDUMP(Status)) {
    printf("Child Core dumped.\n");
  }
}

} // namespace iprof

int main(int argc, char **argv, char **envp) {
  if (argc < 2) {
    printf("Usage: %s elffile [args...]\n", argv[0]);
    return 0;
  }

  pid_t ChildPid = iprof::ptraceInitChild(&argv[1], envp);
  for (;;) {
    int ChildStatus;
    if (iprof::getPidStatus(ChildPid, &ChildStatus) != iprof::kIProfOkay) {
      iprof::ptraceCleanupErr(ChildPid);
      return -1;
    }
    if (!iprof::checkPidStatus(ChildStatus)) {
      iprof::printPidStatus(ChildStatus);
      break;
    }

    long Rip;
    if (iprof::ptraceGetRip(ChildPid, &Rip) != iprof::kIProfOkay) {
      iprof::ptraceCleanupErr(ChildPid);
      return -1;
    }
    //    printf("RIP: %lx\n", Rip);

    if (iprof::ptraceSingleStep(ChildPid) != iprof::kIProfOkay) {
      iprof::ptraceCleanupErr(ChildPid);
      return -1;
    }
  }

  iprof::ptraceCleanup(ChildPid);
  return 0;
}
