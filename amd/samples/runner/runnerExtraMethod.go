package runner

import (
	"encoding/json"
	"os"
	"path/filepath"
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

func (r *Runner) DumpGpuViz(path string) {
	//create dump
	vizDump := VizDump{
		Components: []ComponentDump{}, // Start with an empty slice for components
	}
	//fulfill dump
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
	//output dump to the path
	jsonData, err := json.MarshalIndent(vizDump, "", "  ")
	if err != nil {
		panic(err)
	}

	// Create the directory if it doesn't exist
	err = os.MkdirAll(path, 0755)
	if err != nil {
		panic(err)
	}

	// Generate filename with topology, component topology map, and current date
	// Example: component_topology_map_20250818.json
	dateStr := time.Now().Format("20060102")
	fileName := "component_topology_map_" + dateStr + ".json"
	filePath := filepath.Join(path, fileName)
	err = os.WriteFile(filePath, jsonData, 0644)
	if err != nil {
		panic(err)
	}
}
