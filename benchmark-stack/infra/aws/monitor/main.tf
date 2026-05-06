provider "aws" {
  region = "eu-west-2"
}

resource "aws_key_pair" "key" {
  key_name   = "benchmark-key"
  public_key = file("~/.ssh/id_ed25519.pub")
}

resource "aws_security_group" "monitor_sg" {
  name = "monitor-sg"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 9091
    to_port     = 9091
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 9100
    to_port     = 9500
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical's official AWS account ID

  filter {
    name   = "name"
    # "noble" is the codename for Ubuntu 24.04
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "monitor" {
  ami           = data.aws_ami.ubuntu.id # Ubuntu
  instance_type = "t3.small"
  key_name      = aws_key_pair.key.key_name

  vpc_security_group_ids = [aws_security_group.monitor_sg.id]

  tags = {
    Name = "benchmark-monitor"
  }
}

output "monitor_ip" {
  value = aws_instance.monitor.public_ip
}
