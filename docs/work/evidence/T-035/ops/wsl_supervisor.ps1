# QuantMind WSL supervisor (T-035 P0; user-approved 2026-10-21: keepalive plan A + B).
# Keeps one wsl.exe session attached so the Ubuntu instance (cron, downloads, pipeline) stays up while the user is
# logged in, restarts it if it exits, checks cron every 5 minutes and writes a heartbeat for the pipeline's catch-up check.
# Started at logon by qm_wsl_supervisor.vbs in the Startup folder. Single instance via a named mutex.
# This does not survive shutdown, sleep or logoff; the pipeline catches up data and research after a restart.
$ErrorActionPreference = 'Continue'
$Distro = 'Ubuntu'
$Root = 'D:\qm_ops'
$LogDir = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force $LogDir | Out-Null

$mutex = New-Object System.Threading.Mutex($false, 'Local\QM_WSL_Supervisor')
if (-not $mutex.WaitOne(0)) { exit 0 }

function Write-Log([string]$msg) {
    $line = '{0:yyyy-MM-dd HH:mm:ss} {1}' -f (Get-Date), $msg
    Add-Content -Path (Join-Path $LogDir ('wsl_supervisor_{0:yyyyMMdd}.log' -f (Get-Date))) -Value $line -Encoding utf8
}

function Start-Keepalive {
    Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d', $Distro, '--exec', '/bin/sleep', 'infinity') `
        -WindowStyle Hidden -PassThru
}

Write-Log 'supervisor start'
$keep = Start-Keepalive
Write-Log ('keepalive started pid={0}' -f $keep.Id)

while ($true) {
    Start-Sleep -Seconds 300
    if ($keep.HasExited) {
        Write-Log ('keepalive exited code={0}; restarting' -f $keep.ExitCode)
        Start-Sleep -Seconds 10
        $keep = Start-Keepalive
        Write-Log ('keepalive restarted pid={0}' -f $keep.Id)
    }
    $cron = (& wsl.exe -d $Distro --exec /bin/systemctl is-active cron 2>$null | Out-String).Trim()
    if ($cron -ne 'active') {
        Write-Log ("cron state='{0}'; trying to start it" -f $cron)
        & wsl.exe -d $Distro -u root --exec /bin/systemctl start cron 2>$null | Out-Null
    }
    $hb = @{ utc = (Get-Date).ToUniversalTime().ToString('o'); keepalive_pid = $keep.Id; cron = $cron } |
        ConvertTo-Json -Compress
    Set-Content -Path (Join-Path $LogDir 'heartbeat.json') -Value $hb -Encoding utf8
}
