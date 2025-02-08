# Infrastructure Documentation

## Cisco VM & Cloudflare Tunnel
SSH Access: `ssh ubuntu@<VM_IP> -i ~/.ssh/<KEY_NAME>`

The VM hosts the Cloudflare tunnel for routing traffic to Modal deployments.

### Accessing VM & Tunnel Logs
1. SSH into the VM
2. Cloudflare tunnel logs:
   ```bash
   # View live tunnel logs
   sudo cloudflared tunnel run <TUNNEL_ID>

   # View system logs for cloudflared service
   sudo journalctl -u cloudflared.service
   ```

### Tunnel Management
- View tunnel details in Cloudflare Dashboard: https://dash.cloudflare.com
  - Navigate to: Access > Tunnels
  - Find the tunnel named "llm-server"

## Modal Deployment

### Accessing Modal Logs
1. Command Line:
   ```bash
   # View logs for specific deployment
   modal logs --deployment llm-server-staging-fastapi-app
   modal logs --deployment llm-server-production-fastapi-app
   ```

2. Modal Dashboard:
   - URL: https://modal.com/apps
   - Navigate to your deployment (llm-server-staging or llm-server-production)
   - Click on "Logs" tab

### Application Logs
Application logs are stored in two locations:
1. Modal's built-in logging:
   - Accessible via Modal dashboard or CLI
2. Volume-persisted logs:
   - Located in `/data/logs` in the Modal container
   - Configured in `deploy_modal_app.py`

## Quick Troubleshooting Guide

1. API Not Responding:
   - Check Modal deployment status
   - Check Cloudflare tunnel status
   - Verify tunnel configuration in `action.yml`

2. Tunnel Issues:
   - SSH into Cisco VM
   - Check tunnel logs
   - Verify tunnel is running with correct ID

3. Application Errors:
   - Check Modal logs for application errors
   - Check volume-persisted logs for detailed error traces

## Key Storage
Private keys are stored in macOS Keychain. Contact system administrator for access details.

## Relevant Files
- Tunnel Configuration: `.github/actions/update_tunnel/action.yml`
- Modal Deployment: `deploy_modal_app.py`
- Application Logging: `app/core/logging.py`