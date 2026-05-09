provider "aws" {
  region = "eu-west-2"
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ---------------------------
# REFERENCE EXISTING SECURITY GROUP
# ---------------------------
data "aws_security_group" "existing_gpu_sg" {
  name = "benchmark-gpu-sg"
}

# ---------------------------
# GPU INSTANCE
# ---------------------------
resource "aws_instance" "gpu" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "p5.4xlarge"

  key_name = "benchmark-key"

  vpc_security_group_ids = [data.aws_security_group.existing_gpu_sg.id]

  root_block_device {
    volume_size = 200
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    repo_url      = "https://github.com/ihasidul/DePIN-VS-Centralized-Cloud.git"
    repo_dir      = "/home/ubuntu/DePIN-VS-Centralized-Cloud"
    platform      = "aws"
    prometheus_port = "8000"
  }))

  tags = {
    Name = "benchmark-aws-h100"
  }
}

output "gpu_ip" {
  value = aws_instance.gpu.public_ip
}