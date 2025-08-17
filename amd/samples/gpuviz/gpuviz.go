package main

import (
	"flag"
	"os"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/heteromark/fir"
	"github.com/sarchlab/mgpusim/v4/amd/samples/runner"
)

var (
	defaultPath = flag.String("gpuVizDumpPath", os.Getenv("PWD"), "Path where gpuViz-related data will be dumped")
	usegpuviz   = flag.Bool("use-gpuviz", false, "use gpuViz to visualize")
	numData     = flag.Int("length", 32768, "The number of samples to filter.")
)

func main() {
	flag.Parse()

	runner := new(runner.Runner).Init()

	benchmark := fir.NewBenchmark(runner.Driver())
	benchmark.Length = *numData

	runner.AddBenchmark(benchmark)

	runner.Run()

	if *usegpuviz {
		runner.DumpGpuViz(*defaultPath)
	}
}
