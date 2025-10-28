#!/usr/bin/env python3
"""
Diagrama de Arquitectura para AI/ML Workloads en AWS con Amazon S3 y EKS
Basado en el CloudFormation template: Onboarding & optimizing AI_ML workloads on AWS
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EKS, EC2, ECS, Fargate
from diagrams.aws.network import VPC, InternetGateway, NATGateway, ELB, Route53, CloudFront
from diagrams.aws.storage import S3, EBS
from diagrams.aws.database import RDS
from diagrams.aws.security import IAM, KMS, SecretsManager
from diagrams.aws.management import Cloudwatch, SystemsManager, Cloudtrail
from diagrams.aws.integration import SNS
from diagrams.aws.analytics import Kinesis
from diagrams.k8s.compute import Pod, Deployment
from diagrams.k8s.network import Service, Ingress
from diagrams.k8s.group import Namespace
from diagrams.k8s.clusterconfig import HPA
from diagrams.k8s.controlplane import API
from diagrams.k8s.podconfig import Secret
from diagrams.k8s.storage import PV, PVC, StorageClass
from diagrams.k8s.others import CRD
from diagrams.onprem.compute import Server
from diagrams.onprem.monitoring import Grafana, Prometheus
from diagrams.programming.framework import React
from diagrams.generic.storage import Storage
from diagrams.generic.network import Firewall
import os

# Configuración para el diagrama
graph_attr = {
    "fontsize": "16",
    "bgcolor": "white",
    "rankdir": "TB",
    "splines": "ortho"
}

def create_ai_ml_architecture():
    """Crea el diagrama de arquitectura completo para AI/ML workloads"""
    
    with Diagram(
        "AI/ML Workloads on AWS with Amazon S3 and EKS",
        filename="ai_ml_architecture",
        show=False,
        direction="TB",
        graph_attr=graph_attr
    ):
        # Internet y usuarios externos
        with Cluster("External Access"):
            internet = InternetGateway("Internet")
            users = Server("Data Scientists\n(JupyterHub Users)")
        
        # VPC y Networking
        with Cluster("VPC - 10.0.0.0/16", graph_attr={"bgcolor": "lightblue", "style": "rounded"}):
            
            # Subnets Públicas
            with Cluster("Public Subnets", graph_attr={"bgcolor": "lightgreen"}):
                with Cluster("Public Subnet 1\n10.0.0.0/23\nus-west-2a"):
                    nat_gateway = NATGateway("NAT Gateway")
                    nlb_jupyter = ELB("JupyterHub NLB\n(Internet-facing)")
                
                with Cluster("Public Subnet 2\n10.0.2.0/23\nus-west-2b"):
                    nlb_component2 = ELB("NLB Component")
                
                with Cluster("Public Subnet 3\n10.0.4.0/23\nus-west-2c"):
                    nlb_component3 = ELB("NLB Component")
            
            # Subnets Privadas
            with Cluster("Private Subnets", graph_attr={"bgcolor": "lightyellow"}):
                with Cluster("Private Subnet 1\n10.0.6.0/23\nus-west-2a"):
                    # EC2 Instance para JupyterHub
                    jupyter_instance = EC2("JupyterHub Instance\nc5.4xlarge")
                    
                    # EBS Volumes
                    ebs_root = EBS("Root Volume\ngp3 - 32GB")
                    ebs_data = EBS("Data Volume\ngp3 - 46GB")
                
                with Cluster("Private Subnet 2\n10.0.8.0/23\nus-west-2b"):
                    # EKS Cluster Components
                    eks_node2 = EC2("EKS Worker Node")
                
                with Cluster("Private Subnet 3\n10.0.10.0/23\nus-west-2c"):
                    eks_node3 = EC2("EKS Worker Node")
                
                # EKS Cluster
                with Cluster("EKS Cluster - AI/ML Workshop", graph_attr={"bgcolor": "lavender"}):
                    eks_control_plane = EKS("Control Plane")
                    
                    # Namespaces y aplicaciones
                    with Cluster("Kubernetes Namespaces"):
                        # Namespace: kube-system
                        with Cluster("kube-system"):
                            core_dns = Pod("CoreDNS")
                            kube_proxy = Pod("kube-proxy")
                            vpc_cni = Pod("VPC CNI")
                            ebs_csi = Pod("EBS CSI Driver")
                            s3_csi = Pod("S3 CSI Driver")
                            alb_controller = Pod("ALB Controller")
                            karpenter_controller = Pod("Karpenter")
                        
                        # Namespace: monitoring
                        with Cluster("monitoring"):
                            prometheus = Prometheus("Prometheus")
                            grafana = Grafana("Grafana")
                            otel_operator = Pod("OTEL Operator")
                            otel_collector = Pod("OTEL Collector")
                            metrics_server = Pod("Metrics Server")
                        
                        # Namespace: ray
                        with Cluster("ray"):
                            kuberay_operator = Pod("KubeRay Operator")
                            ray_head = Pod("Ray Head")
                            ray_workers = Pod("Ray Workers\n(Auto-scaling)")
                            ray_dashboard = Pod("Ray Dashboard")
                    
                    # Kubernetes Services
                    with Cluster("Kubernetes Services"):
                        ray_head_svc = Service("ray-head-svc")
                        ray_dashboard_svc = Service("ray-dashboard-svc")
                        grafana_svc = Service("grafana-svc")
                
                # Fargate Profiles
                with Cluster("Fargate Profiles"):
                    fargate_system = Fargate("System\n(kube-system)")
                    fargate_monitoring = Fargate("Monitoring")
                    fargate_ray = Fargate("Ray")
        
        # Servicios AWS fuera de VPC
        with Cluster("AWS Services", graph_attr={"bgcolor": "lightcoral"}):
            # S3 Buckets
            workshop_bucket = S3("Workshop Bucket\n${WorkshopName}-${Region}-${AccountId}")
            assets_bucket = S3("Assets Bucket\nws-assets-prod-*")
            
            # IAM Roles
            with Cluster("IAM Roles"):
                instance_role = IAM("JupyterHub\nInstance Role")
                cluster_role = IAM("EKS Cluster Role")
                fargate_role = IAM("Fargate Pod\nExecution Role")
                ray_worker_role = IAM("Ray Worker Role")
                alb_controller_role = IAM("ALB Controller Role")
                karpenter_role = IAM("Karpenter Role")
                otel_collector_role = IAM("OTEL Collector Role")
            
            # KMS
            kms_key = KMS("Workshop KMS Key")
            
            # SSM
            ssm_document = SystemsManager("Workshop Bootstrap\nSSM Document")
        
        # Security Groups
        with Cluster("Security Groups"):
            shared_sg = Firewall("Shared SG")
            nlb_sg = Firewall("NLB SG")
        
        # ========== CONEXIONES Y FLUJOS ==========
        
        # Flujo de usuarios a JupyterHub
        users >> Edge(label="HTTP/80", color="blue") >> internet
        internet >> Edge(label="TCP/80", color="blue") >> nlb_jupyter
        nlb_jupyter >> Edge(label="TCP/80", color="blue") >> jupyter_instance
        
        # Conexiones de red dentro de VPC
        jupyter_instance >> Edge(label="Internal Traffic", color="green") >> [eks_node2, eks_node3]
        
        # Conexiones EKS a servicios AWS
        eks_control_plane >> Edge(label="Cluster Management", color="purple") >> cluster_role
        eks_control_plane >> Edge(label="Encryption", color="orange") >> kms_key
        
        # Fargate connections
        [fargate_system, fargate_monitoring, fargate_ray] >> Edge(label="Pod Execution", color="brown") >> fargate_role
        
        # Storage connections
        jupyter_instance >> Edge(label="Root Volume", color="red") >> ebs_root
        jupyter_instance >> Edge(label="Data Volume", color="red") >> ebs_data
        
        # S3 Access
        jupyter_instance >> Edge(label="Read/Write", color="darkgreen") >> workshop_bucket
        jupyter_instance >> Edge(label="Assets Download", color="darkgreen") >> assets_bucket
        
        # CSI Drivers to S3
        s3_csi >> Edge(label="S3 Mountpoint", color="darkgreen") >> workshop_bucket
        ebs_csi >> Edge(label="EBS Provisioning", color="red") >> EBS("EBS Volumes")
        
        # Ray Cluster access
        ray_workers >> Edge(label="S3 Access", color="darkgreen") >> ray_worker_role
        ray_worker_role >> Edge(label="Permissions", color="orange") >> workshop_bucket
        
        # Monitoring and Observability
        [prometheus, grafana, otel_collector] >> Edge(label="Metrics & Logs", color="purple") >> Cloudwatch("CloudWatch")
        otel_collector >> Edge(label="Assume Role", color="brown") >> otel_collector_role
        
        # ALB Controller
        alb_controller >> Edge(label="Load Balancer Management", color="blue") >> alb_controller_role
        
        # Karpenter Auto-scaling
        karpenter_controller >> Edge(label="Node Management", color="orange") >> karpenter_role
        
        # SSM Bootstrap
        jupyter_instance >> Edge(label="Bootstrap Script", color="purple") >> ssm_document
        
        # Internet access via NAT
        [jupyter_instance, eks_node2, eks_node3] >> Edge(label="Outbound Internet", color="gray") >> nat_gateway
        nat_gateway >> Edge(label="Internet Access", color="gray") >> internet
        
        # Security Groups
        jupyter_instance >> Edge(label="Security Group", color="red") >> shared_sg
        nlb_jupyter >> Edge(label="Security Group", color="red") >> nlb_sg
        
        # Internal Kubernetes services exposure
        ray_head_svc >> Edge(label="NLB Exposure", color="blue") >> ELB("Ray Head NLB")
        ray_dashboard_svc >> Edge(label="NLB Exposure", color="blue") >> ELB("Ray Dashboard NLB")
        grafana_svc >> Edge(label="NLB Exposure", color="blue") >> ELB("Grafana NLB")

def create_simplified_architecture():
    """Crea una versión simplificada del diagrama"""
    
    with Diagram(
        "AI/ML Workshop - Simplified Architecture",
        filename="ai_ml_simplified",
        show=False,
        direction="TB",
        graph_attr=graph_attr
    ):
        
        # Internet & Users
        internet = InternetGateway("Internet")
        users = Server("Data Scientists")
        
        # Main VPC
        with Cluster("Workshop VPC"):
            # Load Balancers
            nlb = ELB("JupyterHub NLB")
            
            # Compute
            with Cluster("Private Subnets"):
                jupyter = EC2("JupyterHub\n(c5.4xlarge)")
                eks_cluster = EKS("AI/ML EKS Cluster")
                
                with Cluster("Kubernetes Workloads"):
                    ray_cluster = Pod("Ray Cluster")
                    monitoring = Pod("Monitoring Stack")
                    kubernetes_services = Pod("K8s Services")
            
            # Networking
            nat = NATGateway("NAT Gateway")
        
        # AWS Services
        s3_bucket = S3("Workshop S3")
        iam_roles = IAM("IAM Roles")
        kms = KMS("KMS Key")
        
        # Connections
        users >> internet >> nlb >> jupyter
        jupyter >> eks_cluster
        eks_cluster >> ray_cluster
        eks_cluster >> monitoring
        eks_cluster >> kubernetes_services
        
        # Storage & Security
        jupyter >> s3_bucket
        ray_cluster >> s3_bucket
        [jupyter, eks_cluster, ray_cluster] >> iam_roles
        eks_cluster >> kms
        
        # Internet Access
        jupyter >> nat >> internet

def create_data_flow_diagram():
    """Diagrama específico de flujo de datos para AI/ML"""
    
    with Diagram(
        "AI/ML Data Flow - S3 to Ray Cluster",
        filename="data_flow",
        show=False,
        direction="LR",
        graph_attr=graph_attr
    ):
        
        # Data Sources
        with Cluster("Data Sources"):
            s3_raw = S3("Raw Datasets")
            s3_processed = S3("Processed Data")
            s3_models = S3("Trained Models")
        
        # Processing
        with Cluster("Data Processing"):
            jupyter = EC2("JupyterHub\n(Data Prep)")
            ray_head = Pod("Ray Head\n(Orchestration)")
            ray_workers = Pod("Ray Workers\n(Distributed Processing)")
        
        # Storage
        with Cluster("Storage"):
            ebs_local = EBS("Local EBS\n(Fast I/O)")
            s3_output = S3("Output Bucket")
        
        # Monitoring
        with Cluster("Monitoring"):
            prometheus = Prometheus("Metrics")
            grafana = Grafana("Dashboards")
        
        # Data Flow
        s3_raw >> Edge(label="1. Load Data", color="blue") >> jupyter
        jupyter >> Edge(label="2. Pre-process", color="green") >> s3_processed
        s3_processed >> Edge(label="3. Distributed Training", color="red") >> ray_head
        ray_head >> Edge(label="4. Schedule Tasks", color="purple") >> ray_workers
        ray_workers >> Edge(label="5. Model Training", color="orange") >> s3_models
        ray_workers >> Edge(label="6. Cache Data", color="brown") >> ebs_local
        s3_models >> Edge(label="7. Store Results", color="darkgreen") >> s3_output
        
        # Monitoring Flow
        [jupyter, ray_head, ray_workers] >> Edge(label="Metrics", color="gray", style="dashed") >> prometheus
        prometheus >> Edge(label="Visualization", color="gray", style="dashed") >> grafana

if __name__ == "__main__":
    print("Generando diagramas de arquitectura AI/ML...")
    
    # Crear directorio de salida si no existe
    os.makedirs("diagrams", exist_ok=True)
    
    # Generar diagramas
    create_ai_ml_architecture()
    print("✓ Diagrama principal generado: ai_ml_architecture.png")
    
    create_simplified_architecture()
    print("✓ Diagrama simplificado generado: ai_ml_simplified.png")
    
    create_data_flow_diagram()
    print("✓ Diagrama de flujo de datos generado: data_flow.png")
    
    print("\n🎯 Todos los diagramas han sido generados exitosamente!")
    print("📁 Los archivos se encuentran en el directorio actual")