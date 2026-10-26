# Set up Windows Task Scheduler to run run_daily.bat every day at 11 AM and 6 PM (this computer's time).
# Run once from the repo folder:  powershell -ExecutionPolicy Bypass -File schedule_daily.ps1
# Running it again replaces the task. Each run's output is appended to daily.log in this folder.

$name = "NBA daily predictions"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c run_daily.bat >> daily.log 2>&1" `
    -WorkingDirectory $PSScriptRoot
$triggers = @(
    New-ScheduledTaskTrigger -Daily -At "11:00"  # morning: overnight results and the first injury report
    New-ScheduledTaskTrigger -Daily -At "18:00"  # about an hour before most tip-offs (Eastern)
)
# Wake the computer from sleep if needed, run on battery too, and give up after an hour.
$settings = New-ScheduledTaskSettingsSet -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask -TaskName $name -Action $action -Trigger $triggers -Settings $settings `
    -Description "Refresh NBA data, read the injury report, predict today's games and push them." -Force | Out-Null
$next = (Get-ScheduledTaskInfo -TaskName $name).NextRunTime
Write-Output "Scheduled '$name' in $PSScriptRoot. Next run: $next"
