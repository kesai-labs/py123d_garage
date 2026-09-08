const fs = require("fs");
const { spawn } = require("child_process");

const cgroup = "/sys/fs/cgroup" + fs.readFileSync("/proc/self/cgroup", "utf8").trim().split(":")[2];
const peakFile = `${process.env.RUNNER_TEMP}/memory_peak_bytes`;
const sampler = spawn(
  "bash",
  ["-c", `peak=0; while :; do current=$(cat "$0/memory.current" 2>/dev/null || echo 0); if ((current > peak)); then peak=$current; echo "$peak" > "$1"; fi; sleep 1; done`, cgroup, peakFile],
  { detached: true, stdio: "ignore" },
);
sampler.unref();
fs.appendFileSync(process.env.GITHUB_STATE, `sampler_pid=${sampler.pid}\ncgroup=${cgroup}\npeak_file=${peakFile}\n`);
