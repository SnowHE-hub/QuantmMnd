' QuantMind WSL supervisor launcher (T-035 P0; user-approved 2026-10-21). Copied into the Startup folder.
' Runs D:\qm_ops\wsl_supervisor.ps1 hidden at logon.
CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""D:\qm_ops\wsl_supervisor.ps1""", 0, False
