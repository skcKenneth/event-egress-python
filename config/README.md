# Network configurations

All configurations are synthetic queue-network benchmarks. They are not calibrated venue layouts and must not be interpreted as operational evacuation models.

| File | Purpose |
|---|---|
| `default.json` | Four-zone, three-corridor asymmetric base benchmark used for the confirmatory held-out validation. |
| `topology_symmetric.json` | Symmetric path-capacity and exit-capacity family with heterogeneous travel times. |
| `topology_path_bottleneck.json` | Path-limited family with deliberately high downstream exit capacities. |
| `topology_exit_bottleneck.json` | Exit-limited family with deliberately high upstream path capacities. |
| `topology_expanded_venue.json` | Larger five-zone, four-corridor synthetic family. |

The topology-transfer experiment applies the same persistent capacity uncertainty and mean compliance to each family and compares the same center-only conditioned and unconditioned controllers with paired seeds.
