package runner

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sarchlab/akita/v4/sim"
)

// Define a struct specifically for JSON dumping
type ComponentDump struct {
	Name  string     `json:"name"`
	Ports []PortDump `json:"ports"`
}

type PortDump struct {
	Name         string           `json:"name"`
	IncomingPort []sim.RemotePort `json:"incomingPort"`
	OutgoingPort []sim.RemotePort `json:"outgoingPort"`
}

// Define the SystemDump structure that holds components
type VizDump struct {
	Components []ComponentDump `json:"components"`
}

func (r *Runner) DumpGpuViz(method ...string) {
	selectedMethod := "sqlite" // default
	if len(method) > 0 && method[0] != "" {
		selectedMethod = method[0]
	}

	switch strings.ToLower(selectedMethod) {
	case "sqlite":
		r.DumpGpuVizSqlite()
	case "json":
		r.DumpGpuVizJson()
	case "both":
		r.DumpGpuVizSqlite()
		r.DumpGpuVizJson()
	default:
		fmt.Printf("Error: unsupported GPU viz dump method: %s\n", selectedMethod)
		return
	}
}

func (r *Runner) DumpGpuVizSqlite() {
	r.simulation.GetMsgTracer().AddTopologyPortMap(r.simulation.Components())
}

func (r *Runner) CreateGPUDumpStruct() *VizDump {

	vizDump := VizDump{
		Components: []ComponentDump{},
	}

	for _, component := range r.simulation.Components() {
		var componentName string = component.Name()
		ports := component.Ports()

		portDumps := []PortDump{}
		for _, port := range ports {
			incoming := port.GetIncomingPorts()
			outgoing := port.GetOutgoingPorts()
			portDumps = append(portDumps, PortDump{
				Name:         port.Name(),
				IncomingPort: incoming,
				OutgoingPort: outgoing,
			})
		}
		vizDump.Components = append(vizDump.Components, ComponentDump{
			Name:  componentName,
			Ports: portDumps,
		})
	}
	return &vizDump
}

func (r *Runner) DumpGpuVizJson() {
	var path, _ = os.Getwd()
	// Create and fulfill the dump structure using the new method
	vizDump := r.CreateGPUDumpStruct()
	// Output dump to the path
	jsonData, err := json.MarshalIndent(vizDump, "", "  ")
	if err != nil {
		fmt.Printf("Error marshaling JSON: %v\n", err)
		return
	}

	// Create the directory if it doesn't exist
	err = os.MkdirAll(path, 0755)
	if err != nil {
		fmt.Printf("Error creating directory: %v\n", err)
		return
	}

	// Generate filename with topology, component topology map, and current date/time
	// Example: component_topology_map_20250818_1530.json
	dateStr := time.Now().Format("20060102_1504")
	fileName := "component_topology_map_" + dateStr + ".json"
	filePath := filepath.Join(path, fileName)
	err = os.WriteFile(filePath, jsonData, 0644)
	if err != nil {
		fmt.Printf("Error writing file: %v\n", err)
		return
	}
}
