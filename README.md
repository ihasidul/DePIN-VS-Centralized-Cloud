# A Comparative Analysis of Sustained GPU Throughput and Latency Stability on Decentralized Physical Infrastructure Network(DePIN) Platform vs Centralized Cloud Platforms
## Architecture
```
```
                ┌─────────────────────────┐
                │ AWS Monitoring Instance │
                │-------------------------│
                │ Prometheus             │
                │ Grafana                │
                │ Pushgateway           │
                └──────────┬────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
        ┌───────▼────────┐   ┌────────▼───────┐
        │ AWS p5.4xlarge │   │ Akash H100     │
        │----------------│   │----------------│
        │ trainer        │   │ trainer        │
        │ dcgm-exporter  │   │ dcgm-exporter  │
        │ node-exporter  │   │ node-exporter  │
        └────────────────┘   └────────────────┘
```
```

## Tools

### Infrastructure:
- AWS monitor node
- AWS p5.4xlarge
- Akash H100

### Monitoring:
- Prometheus
- DCGM exporter
- Node exporter
- Grafana

### Training:
- Docker
- Accelerate
- HuggingFace Trainer


