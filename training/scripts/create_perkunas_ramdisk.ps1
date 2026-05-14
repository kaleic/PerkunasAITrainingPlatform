param(
    [string]$AimCliPath = "D:\Chrome\Downloads\Arsenal-Image-Mounter-v3.12.344\Arsenal-Image-Mounter-v3.12.344\aim_cli.exe",
    [string]$DriveLetter = "R",
    [string]$DiskSize = "16G",
    [string]$Label = "PERK_RAM",
    [string]$ActiveRunDirName = "Perkunas_v2.6_50m_run4_active"
)

$ErrorActionPreference = "Stop"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Convert-SizeToBytes {
    param([string]$Value)
    if ($Value -notmatch '^\s*(\d+(?:\.\d+)?)\s*([KMGTP]?)\s*$') {
        throw "Unsupported size format: $Value"
    }
    $number = [double]$Matches[1]
    $suffix = $Matches[2].ToUpperInvariant()
    $multiplier = switch ($suffix) {
        "K" { 1KB }
        "M" { 1MB }
        "G" { 1GB }
        "T" { 1TB }
        "P" { 1PB }
        default { 1 }
    }
    return [int64]($number * $multiplier)
}

if (-not (Test-IsAdministrator)) {
    throw "Run this script from PowerShell as Administrator."
}

if (-not (Test-Path -LiteralPath $AimCliPath)) {
    throw "AIM CLI was not found: $AimCliPath"
}

$DriveLetter = $DriveLetter.TrimEnd(":").ToUpperInvariant()
if (Get-PSDrive -Name $DriveLetter -ErrorAction SilentlyContinue) {
    throw "Drive $DriveLetter`: already exists. Pick another drive letter or dismount the existing one first."
}

$requestedBytes = Convert-SizeToBytes $DiskSize
$os = Get-CimInstance Win32_OperatingSystem
$freeBytes = [int64]$os.FreePhysicalMemory * 1KB
if ($requestedBytes -gt ($freeBytes * 0.85)) {
    Write-Warning ("Requested RAM disk is {0:N2} GiB, current free RAM is {1:N2} GiB. Close apps or choose a smaller disk if Windows starts paging." -f ($requestedBytes / 1GB), ($freeBytes / 1GB))
}

$beforeDiskNumbers = @(Get-Disk | ForEach-Object { $_.Number })

Write-Host "Creating Arsenal RAM disk: size=$DiskSize"
& $AimCliPath --ramdisk "--disksize=$DiskSize"
if ($LASTEXITCODE -ne 0) {
    throw "AIM CLI failed with exit code $LASTEXITCODE."
}

Start-Sleep -Seconds 2
Update-HostStorageCache

$newDisks = @(
    Get-Disk |
        Where-Object { $beforeDiskNumbers -notcontains $_.Number } |
        Sort-Object Number -Descending
)

if (-not $newDisks) {
    $requestedGiB = $requestedBytes / 1GB
    $newDisks = @(
        Get-Disk |
            Where-Object {
                $_.PartitionStyle -eq "RAW" -and
                [math]::Abs(($_.Size / 1GB) - $requestedGiB) -lt 1.0
            } |
            Sort-Object Number -Descending
    )
}

if (-not $newDisks) {
    throw "Could not identify the new RAM disk. Run aim_cli.exe --list and Disk Management to inspect the mounted device."
}

$disk = $newDisks[0]
Write-Host "Formatting Disk $($disk.Number) as $DriveLetter`: ($([math]::Round($disk.Size / 1GB, 2)) GiB)"

if ($disk.PartitionStyle -eq "RAW") {
    Initialize-Disk -Number $disk.Number -PartitionStyle GPT
}

$partition = New-Partition -DiskNumber $disk.Number -DriveLetter $DriveLetter -UseMaximumSize
Format-Volume -Partition $partition -FileSystem NTFS -NewFileSystemLabel $Label -Confirm:$false | Out-Null

$activeRunDir = "${DriveLetter}:\$ActiveRunDirName"
New-Item -ItemType Directory -Force -Path $activeRunDir | Out-Null

Write-Host ""
Write-Host "RAM disk ready: $activeRunDir"
Write-Host "Use this training argument:"
Write-Host "  --active-run-dir $activeRunDir ``"
Write-Host ""
Write-Host "To dismount later:"
Write-Host "  & `"$AimCliPath`" --dismount=\\?\PhysicalDrive$($disk.Number) --force"
