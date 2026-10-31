<#
.SYNOPSIS
    Manages a local (or remote) Ollama install: updates, inventory, disk usage, cleanup and backups.
.DESCRIPTION
    Covers multi-step tasks the ollama CLI does not do in one command:
      update       Pull only the models whose registry digest changed (cloud models are skipped).
      outdated     List models that have a newer version on the registry. Downloads manifests only.
      list         Rich inventory: size, age, family, parameters, quantization, context, capabilities.
      stop         Unload every running model (frees VRAM).
      usage        Real disk usage: shared vs unique blobs, orphaned blobs, partial downloads.
      prune        Delete models by age, pattern or duplicate alias; delete orphaned blobs.
      export-list  Write the model list to a JSON file.
      import-list  Pull every model from an exported list that is missing locally.
      backup       Copy models (manifests + blobs) to a folder. Incremental.
      restore      Copy backed-up models back, optionally under a different tag.
      doctor       Health report: server, CLI, latest release, config, models folder, registry.
      bench        Tokens/s throughput test.
    Run without an action to print a usage summary.

    The models folder (usage, prune -Orphans, backup, restore, free-space checks) is resolved from
    -ModelsPath, then OLLAMA_MODELS in the running server's server.log, then the OLLAMA_MODELS
    environment variable, then ~\.ollama\models.

    Exit codes: 0 success / all current; 1 partial failure or (outdated) updates available;
    2 server not reachable; 3 some registry checks failed; 4 invalid arguments or failed precondition.
.PARAMETER Action
    The action to run. Omit it to print the usage summary.
.PARAMETER Model
    Wildcard patterns selecting models (qwen*, *:latest). A pattern without a wildcard and without
    ':' means ':latest' (gemma4 = gemma4:latest). Default: all models.
.PARAMETER Exclude
    Wildcard patterns to skip. Exclude wins over -Model.
.PARAMETER OllamaHost
    Server URL. Default: $env:OLLAMA_HOST, then http://127.0.0.1:11434. 'gpu-box' becomes
    http://gpu-box:11434; 0.0.0.0 becomes 127.0.0.1.
.PARAMETER ModelsPath
    Models folder (the one holding 'manifests' and 'blobs'). Overrides automatic detection.
.PARAMETER Path
    JSON file for export-list/import-list; folder for backup/restore.
.PARAMETER All
    outdated: show every checked model with its status, not only outdated ones.
.PARAMETER Deep
    list: also check the registry, mark unknown models as 'custom' and add a Status column.
.PARAMETER Sort
    list: Name (default), Size or Modified. Size and Modified sort descending.
.PARAMETER Capability
    list: only models that have all of these capabilities (tools, vision, thinking, embedding).
.PARAMETER OlderThanDays
    prune: models pulled or updated more than this many days ago. Ollama does not record when a
    model was last *used*; modified_at is the time it was last pulled or updated.
.PARAMETER Duplicates
    prune: remove alias tags that share a digest with another tag. Keeps one name per digest
    (a non-'latest' tag, then the shortest name, then alphabetical). Frees no disk space.
.PARAMETER Orphans
    prune: delete blobs no manifest references. Never touches *-partial* files.
.PARAMETER AsTag
    restore: restore a single model under this tag instead of overwriting its own tag.
.PARAMETER Prompt
    bench: prompt to use instead of the built-in one.
.PARAMETER Runs
    bench: number of measured runs per model (default 3).
.PARAMETER KeepLoaded
    bench: leave benchmarked models loaded afterwards.
.PARAMETER PassThru
    Emit objects instead of formatted tables.
.PARAMETER Help
    Print the usage summary and exit. Also: -h, --help, help, /?.
.PARAMETER Rest
    Catches unrecognized arguments (e.g. --help). Anything other than a help token is an error.
.EXAMPLE
    .\ollamas.ps1
    Prints the usage summary.
.EXAMPLE
    .\ollamas.ps1 update -Model qwen* -Exclude *:2b
    Pulls the qwen models that changed on the registry, except the 2b tags.
.EXAMPLE
    .\ollamas.ps1 outdated -All
    Shows the registry status of every local model without downloading anything.
.EXAMPLE
    .\ollamas.ps1 list -Capability vision -Sort Size
    Lists vision models, largest first.
.EXAMPLE
    .\ollamas.ps1 stop
    Unloads all running models.
.EXAMPLE
    .\ollamas.ps1 usage
    Shows real disk usage per model, shared blobs, orphans and partial downloads.
.EXAMPLE
    .\ollamas.ps1 prune -OlderThanDays 120 -WhatIf
    Shows which models would be deleted.
.EXAMPLE
    .\ollamas.ps1 export-list -Path D:\models.json
    Writes the model list to a JSON file.
.EXAMPLE
    .\ollamas.ps1 import-list -Path D:\models.json
    Pulls every listed model that is missing locally.
.EXAMPLE
    .\ollamas.ps1 backup -Model gemma4:26b -Path E:\ollama-backup
    Copies the model's manifest and blobs to E:\ollama-backup.
.EXAMPLE
    .\ollamas.ps1 restore -Path E:\ollama-backup -Model gemma4:26b -AsTag 26b-old
    Restores the backed-up gemma4:26b as gemma4:26b-old.
.EXAMPLE
    .\ollamas.ps1 doctor
    Prints a health report.
.EXAMPLE
    .\ollamas.ps1 bench -Model gemma4*,qwen3.5*
    Measures load time and tokens/s.
.EXAMPLE
    .\ollamas.ps1 outdated -OllamaHost gpu-box
    Checks the models on http://gpu-box:11434.
#>
#Requires -Version 7.2
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'Medium')]
param(
    # Validated in the script body instead of ValidateSet, so 'help' and '/?' reach the help check.
    [Parameter(Position = 0)]
    [ArgumentCompleter({
            param($cmd, $param, $word)
            'update', 'outdated', 'list', 'stop', 'usage', 'prune', 'export-list', 'import-list',
            'backup', 'restore', 'doctor', 'bench' | Where-Object { $_ -like "$word*" }
        })]
    [string]$Action,

    [Parameter(Position = 1)]
    [string[]]$Model,

    [string[]]$Exclude,
    [string]$OllamaHost,
    [string]$ModelsPath,
    [string]$Path,
    [switch]$All,
    [switch]$Deep,

    [ValidateSet('Name', 'Size', 'Modified')]
    [string]$Sort = 'Name',

    [ValidateSet('tools', 'vision', 'thinking', 'embedding')]
    [string[]]$Capability,

    [ValidateRange(1, 36500)]
    [int]$OlderThanDays,

    [switch]$Duplicates,
    [switch]$Orphans,
    [string]$AsTag,
    [string]$Prompt,

    [ValidateRange(1, 100)]
    [int]$Runs = 3,

    [switch]$KeepLoaded,
    [switch]$PassThru,

    [Alias('h')]
    [switch]$Help,
    # Catches tokens PowerShell will not bind to a parameter, e.g. --help
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

#region Constants

$script:DefaultRegistry = 'registry.ollama.ai'
$script:DefaultBenchPrompt = 'Write a short story of about 200 words about a lighthouse keeper.'
$script:ApiTimeoutSec = 30
$script:GenerateTimeoutSec = 600
$script:RegistryTimeoutSec = 10

# Action-specific parameters each action accepts (2.3). Everything else except the global ones is an error.
$script:ActionParameters = [ordered]@{
    'update'      = @('Model', 'Exclude')
    'outdated'    = @('Model', 'Exclude', 'All')
    'list'        = @('Model', 'Exclude', 'Sort', 'Capability', 'Deep')
    'stop'        = @('Model', 'Exclude')
    'usage'       = @('Model', 'Exclude', 'ModelsPath')
    'prune'       = @('Model', 'Exclude', 'OlderThanDays', 'Duplicates', 'Orphans', 'ModelsPath')
    'export-list' = @('Model', 'Exclude', 'Path')
    'import-list' = @('Model', 'Exclude', 'Path')
    'backup'      = @('Model', 'Exclude', 'Path', 'ModelsPath')
    'restore'     = @('Model', 'Exclude', 'Path', 'AsTag', 'ModelsPath')
    'doctor'      = @()
    'bench'       = @('Model', 'Exclude', 'Prompt', 'Runs', 'KeepLoaded')
}
$script:RequiredParameters = @{
    'export-list' = @('Path')
    'import-list' = @('Path')
    'backup'      = @('Model', 'Path')
    'restore'     = @('Path')
}
$script:GlobalParameters = @('Action', 'OllamaHost', 'PassThru', 'WhatIf', 'Confirm') +
    [System.Management.Automation.PSCmdlet]::CommonParameters

# Actions that work without a running server (section 3).
$script:OfflineActions = @('usage', 'backup', 'restore', 'doctor')

#endregion

#region Small helpers

function Write-Fail {
    param([string]$Message)
    $Host.UI.WriteErrorLine($Message)
}

function ConvertTo-BlobFileName {
    # 'sha256:abc' -> 'sha256-abc'
    param([string]$Digest)
    $Digest -replace '^sha256:', 'sha256-'
}

function Get-ShortDigest {
    param([string]$Digest)
    if (-not $Digest) { return '' }
    $d = $Digest -replace '^sha256[:-]', ''
    $d.Substring(0, [Math]::Min(12, $d.Length))
}

function Get-Sha256Hex {
    param([byte[]]$Bytes)
    [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
}

function Get-FileSha256Hex {
    param([string]$LiteralPath)
    (Get-FileHash -LiteralPath $LiteralPath -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Format-Size {
    # GB with 1 GB = 10^9 bytes to match `ollama list`.
    param([Nullable[long]]$Bytes)
    if ($null -eq $Bytes) { return '?' }
    if ($Bytes -ge 1e9) { return '{0:N1} GB' -f ($Bytes / 1e9) }
    if ($Bytes -ge 1e6) { return '{0:N0} MB' -f ($Bytes / 1e6) }
    if ($Bytes -ge 1e3) { return '{0:N0} KB' -f ($Bytes / 1e3) }
    return "$Bytes B"
}

function Format-Age {
    param([datetimeoffset]$Time)
    $span = [datetimeoffset]::Now - $Time
    $rel = if ($span.TotalMinutes -lt 1) { 'just now' }
    elseif ($span.TotalHours -lt 1) { '{0} min ago' -f [int][Math]::Floor($span.TotalMinutes) }
    elseif ($span.TotalDays -lt 1) { '{0} h ago' -f [int][Math]::Floor($span.TotalHours) }
    else { '{0} d ago' -f [int][Math]::Floor($span.TotalDays) }
    '{0} ({1:yyyy-MM-dd HH:mm})' -f $rel, $Time.ToLocalTime()
}

function Get-ExitCode {
    # Highest code wins (4.8).
    param([int[]]$Codes)
    $max = 0
    foreach ($c in $Codes) { if ($c -gt $max) { $max = $c } }
    $max
}

function Get-OllamaExe {
    $cmd = Get-Command ollama -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { $cmd.Source } else { $null }
}

function Get-ModelTag {
    param([string]$Name)
    $slash = $Name.LastIndexOf('/')
    $colon = $Name.LastIndexOf(':')
    if ($colon -gt $slash) { $Name.Substring($colon + 1) } else { 'latest' }
}

function Get-FreeSpace {
    # Free bytes on the drive holding $Directory, or $null when unknown (e.g. UNC path).
    param([string]$Directory)
    try {
        $root = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Directory))
        if (-not $root -or $root.StartsWith('\\')) { return $null }
        [long]([IO.DriveInfo]::new($root).AvailableFreeSpace)
    }
    catch { $null }
}

#endregion

#region Host, CLI and API

function Resolve-OllamaHost {
    <#
    Normalizes an OLLAMA_HOST-style value (2.2) and decides whether it points at this machine.
    Returns Url (no trailing slash), HostName and IsLocal.
    #>
    param([string]$Value, [switch]$SkipLocalCheck)

    $v = if ([string]::IsNullOrWhiteSpace($Value)) { 'http://127.0.0.1:11434' } else { $Value.Trim() }
    if ($v -notmatch '^[A-Za-z][A-Za-z0-9+.-]*://') { $v = "http://$v" }
    $scheme, $rest = $v -split '://', 2
    $authority, $pathPart = $rest -split '/', 2

    # Split authority into host and port; IPv6 hosts are bracketed.
    if ($authority -match '^\[(?<h>[^\]]*)\](:(?<p>\d+))?$' -or $authority -match '^(?<h>[^:]*)(:(?<p>\d+))?$') {
        $hostName = $Matches['h']
        $port = $Matches['p']
    }
    else {
        throw "Invalid Ollama host '$Value'."
    }
    $isV6 = $authority.StartsWith('[')
    if (-not $hostName) { $hostName = '127.0.0.1' }
    if ($hostName -eq '0.0.0.0') { $hostName = '127.0.0.1' }
    if ($isV6 -and ($hostName -eq '::' -or $hostName -eq '::0')) { $hostName = '::1' }
    if (-not $port) { $port = '11434' }

    $hostPart = if ($isV6) { "[$hostName]" } else { $hostName }
    $url = '{0}://{1}:{2}' -f $scheme.ToLowerInvariant(), $hostPart, $port
    if ($pathPart) { $url = "$url/$pathPart" }
    $url = $url.TrimEnd('/')

    [pscustomobject]@{
        Url      = $url
        HostName = $hostName
        IsLocal  = if ($SkipLocalCheck) { $false } else { Test-LocalHostName $hostName }
    }
}

function Test-LocalHostName {
    param([string]$HostName)
    if ($HostName -in @('localhost', '127.0.0.1', '::1') -or $HostName -like '127.*') { return $true }
    if ($HostName -ieq $env:COMPUTERNAME -or $HostName -ieq [Net.Dns]::GetHostName()) { return $true }
    try {
        $own = [Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() |
            ForEach-Object { $_.GetIPProperties().UnicastAddresses } |
            ForEach-Object { $_.Address.ToString() -replace '%.*$', '' }
        $addresses = [Net.Dns]::GetHostAddresses($HostName) | ForEach-Object { $_.ToString() -replace '%.*$', '' }
        foreach ($a in $addresses) {
            if ($a -eq '127.0.0.1' -or $a -eq '::1' -or $a -in $own) { return $true }
        }
    }
    catch {
        Write-Verbose "Could not resolve '$HostName': $($_.Exception.Message)"
    }
    $false
}

function Invoke-OllamaApi {
    <#
    Calls the Ollama REST API on the resolved host. Throws a readable error on failure.
    #>
    param(
        [Parameter(Mandatory)][string]$Endpoint,
        [string]$Method = 'GET',
        [object]$Body,
        [int]$TimeoutSec = $script:ApiTimeoutSec
    )
    $uri = "$($script:Server.Url)$Endpoint"
    $params = @{ Uri = $uri; Method = $Method; TimeoutSec = $TimeoutSec; ErrorAction = 'Stop'; Verbose = $false }
    if ($null -ne $Body) {
        $params.Body = ($Body | ConvertTo-Json -Depth 10 -Compress)
        $params.ContentType = 'application/json'
    }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        $result = Invoke-RestMethod @params
        Write-Verbose ('{0} {1} -> 200 ({2} ms)' -f $Method, $uri, $sw.ElapsedMilliseconds)
        $result
    }
    catch {
        $status = $_.Exception.Response.StatusCode.value__ 2>$null
        $detail = $_.ErrorDetails.Message
        if ($detail) {
            try { $detail = ($detail | ConvertFrom-Json).error } catch { }
        }
        if (-not $detail) { $detail = $_.Exception.Message }
        Write-Verbose ('{0} {1} -> {2} ({3} ms)' -f $Method, $uri, ($status ?? 'failed'), $sw.ElapsedMilliseconds)
        throw "$Method $Endpoint failed: $detail"
    }
}

function Test-OllamaServer {
    try { $null = Invoke-OllamaApi -Endpoint '/api/version' -TimeoutSec 5; $true } catch { $false }
}

function Invoke-OllamaCli {
    <#
    Runs the ollama CLI with OLLAMA_HOST set to the resolved host for the child process only.
    -Interactive leaves the console attached so native progress bars show.
    #>
    param([Parameter(Mandatory)][string[]]$Arguments, [switch]$Interactive)

    $exe = Get-OllamaExe
    if (-not $exe) { throw 'The ollama CLI is not on PATH.' }
    $psi = [Diagnostics.ProcessStartInfo]::new($exe)
    foreach ($a in $Arguments) { $psi.ArgumentList.Add($a) }
    $psi.UseShellExecute = $false
    $psi.Environment['OLLAMA_HOST'] = $script:Server.Url
    if (-not $Interactive) {
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
    }
    Write-Verbose "CLI: ollama $($Arguments -join ' ') (OLLAMA_HOST=$($script:Server.Url))"
    $p = [Diagnostics.Process]::Start($psi)
    try {
        $out = $null; $err = $null
        if (-not $Interactive) {
            $out = $p.StandardOutput.ReadToEndAsync()
            $err = $p.StandardError.ReadToEndAsync()
        }
        # Short waits keep Ctrl+C responsive.
        while (-not $p.WaitForExit(200)) { }
        $p.WaitForExit()
        [pscustomobject]@{
            ExitCode = $p.ExitCode
            Output   = if ($out) { ($out.Result + $err.Result).Trim() } else { $null }
        }
    }
    finally {
        if (-not $p.HasExited) { try { $p.Kill($true) } catch { } }
        $p.Dispose()
    }
}

function Get-OllamaModels {
    # Local models from /api/tags, with an IsCloud flag.
    $tags = Invoke-OllamaApi -Endpoint '/api/tags'
    foreach ($m in @($tags.models)) {
        if ($null -eq $m) { continue }
        $isCloud = [bool]($m.PSObject.Properties['remote_host'] -and $m.remote_host)
        $m | Add-Member -NotePropertyName IsCloud -NotePropertyValue $isCloud -Force -PassThru
    }
}

function Get-LoadedModels {
    $ps = Invoke-OllamaApi -Endpoint '/api/ps'
    @($ps.models) | Where-Object { $null -ne $_ }
}

function Stop-OllamaModel {
    param([string]$Name)
    $null = Invoke-OllamaApi -Endpoint '/api/generate' -Method POST -Body @{ model = $Name; keep_alive = 0 } -TimeoutSec $script:GenerateTimeoutSec
}

#endregion

#region Model names and matching

function Split-ModelName {
    <#
    Splits a model name into registry host, namespace, name and tag.
    gemma4 -> registry.ollama.ai / library / gemma4 / latest
    #>
    param([Parameter(Mandatory)][string]$Name)
    $tag = 'latest'
    $base = $Name
    $slash = $Name.LastIndexOf('/')
    $colon = $Name.LastIndexOf(':')
    if ($colon -gt $slash) {
        $tag = $Name.Substring($colon + 1)
        $base = $Name.Substring(0, $colon)
    }
    $parts = $base -split '/'
    switch ($parts.Count) {
        1 { $registry = 'registry.ollama.ai'; $ns = 'library'; $model = $parts[0] }
        2 { $registry = 'registry.ollama.ai'; $ns = $parts[0]; $model = $parts[1] }
        default {
            $registry = $parts[0]
            $ns = ($parts[1..($parts.Count - 2)]) -join '/'
            $model = $parts[-1]
        }
    }
    [pscustomobject]@{ Registry = $registry; Namespace = $ns; Name = $model; Tag = $tag }
}

function ConvertTo-ModelName {
    # Inverse of Split-ModelName, in the short form /api/tags uses.
    param([string]$Registry, [string]$Namespace, [string]$Name, [string]$Tag)
    if ($Registry -eq $script:DefaultRegistry) {
        if ($Namespace -eq 'library') { return "${Name}:$Tag" }
        return "$Namespace/${Name}:$Tag"
    }
    "$Registry/$Namespace/${Name}:$Tag"
}

function ConvertTo-NamePattern {
    # A pattern without wildcard and without ':' means ':latest' (4.4).
    param([string]$Pattern)
    if ([WildcardPattern]::ContainsWildcardCharacters($Pattern)) { return $Pattern }
    $slash = $Pattern.LastIndexOf('/')
    if ($Pattern.LastIndexOf(':') -gt $slash) { return $Pattern }
    "${Pattern}:latest"
}

function Select-ModelsByPattern {
    <#
    Applies -Model / -Exclude patterns to items that have a name property.
    Returns Items, UnmatchedPatterns and NoneMatched (true when -Model was given and no pattern matched).
    #>
    param(
        [object[]]$Items,
        [string[]]$Model,
        [string[]]$Exclude,
        [string]$NameProperty = 'name'
    )
    $include = @($Model | Where-Object { $_ } | ForEach-Object { ConvertTo-NamePattern $_ })
    $excl = @($Exclude | Where-Object { $_ } | ForEach-Object { ConvertTo-NamePattern $_ })
    $matched = @{}
    $selected = foreach ($it in @($Items)) {
        if ($null -eq $it) { continue }
        $n = [string]$it.$NameProperty
        if ($include.Count) {
            $hit = $false
            foreach ($p in $include) { if ($n -like $p) { $matched[$p] = $true; $hit = $true } }
            if (-not $hit) { continue }
        }
        $skip = $false
        foreach ($p in $excl) { if ($n -like $p) { $skip = $true; break } }
        if (-not $skip) { $it }
    }
    $unmatched = @($include | Where-Object { -not $matched.ContainsKey($_) })
    [pscustomobject]@{
        Items             = @($selected)
        UnmatchedPatterns = $unmatched
        NoneMatched       = ($include.Count -gt 0 -and $unmatched.Count -eq $include.Count)
    }
}

function Select-Models {
    <#
    Select-ModelsByPattern plus the warnings of 4.4. Returns $null when no pattern matched (exit 4).
    #>
    param([object[]]$Items, [string[]]$Model, [string[]]$Exclude, [string]$NameProperty = 'name')
    $r = Select-ModelsByPattern -Items $Items -Model $Model -Exclude $Exclude -NameProperty $NameProperty
    foreach ($p in $r.UnmatchedPatterns) { Write-Warning "Pattern '$p' matches no model." }
    if ($r.NoneMatched) {
        Write-Fail 'No -Model pattern matched any model.'
        return $null
    }
    , $r.Items
}

function Test-ActionParameters {
    <#
    Validates the parameter combination for an action (2.3). Returns an error message or $null.
    #>
    param([string]$Action, [System.Collections.IDictionary]$BoundParameters)
    $allowed = $script:ActionParameters[$Action]
    foreach ($k in $BoundParameters.Keys) {
        if ($k -in $script:GlobalParameters -or $k -in $allowed) { continue }
        return "Parameter -$k is not valid for action '$Action'."
    }
    foreach ($req in @($script:RequiredParameters[$Action])) {
        if ($req -and -not $BoundParameters.ContainsKey($req)) {
            return "Action '$Action' requires -$req."
        }
    }
    if ($Action -eq 'prune' -and -not ($BoundParameters.Keys | Where-Object { $_ -in 'OlderThanDays', 'Duplicates', 'Orphans', 'Model' })) {
        return 'prune needs at least one of -OlderThanDays, -Duplicates, -Orphans or -Model.'
    }
    $null
}

#endregion

#region Models directory

function ConvertFrom-ServerConfigLine {
    <#
    Parses the env="map[K:V K2:V2 ...]" part of a server.log 'server config' line into an ordered
    dictionary. Values may contain spaces (paths, lists); keys are upper-case identifiers.
    #>
    param([string]$Line)
    $result = [ordered]@{}
    if ($Line -notmatch 'env="map\[(?<m>.*)\]"\s*$') { return $result }
    $inner = $Matches['m']
    $rx = [regex]'(?:^|\s)(?<k>[A-Z][A-Z0-9_]*):(?<v>.*?)(?=\s[A-Z][A-Z0-9_]*:|$)'
    foreach ($m in $rx.Matches($inner)) {
        $result[$m.Groups['k'].Value] = $m.Groups['v'].Value.Replace('\\', '\')
    }
    $result
}

function Get-ServerConfig {
    # The most recent 'server config' line of server.log, parsed. $null when unavailable.
    param([string]$LogPath = (Join-Path $env:LOCALAPPDATA 'Ollama\server.log'))
    if (-not $LogPath -or -not (Test-Path -LiteralPath $LogPath)) { return $null }
    $line = Select-String -LiteralPath $LogPath -Pattern 'msg="server config"' -SimpleMatch:$false |
        Select-Object -Last 1
    if (-not $line) { return $null }
    $time = if ($line.Line -match '^time=(\S+)') { $Matches[1] } else { $null }
    [pscustomobject]@{ Time = $time; Settings = (ConvertFrom-ServerConfigLine $line.Line); LogPath = $LogPath }
}

function Test-ModelsDirectory {
    param([string]$Directory, [switch]$AllowEmpty)
    if (-not $Directory -or -not (Test-Path -LiteralPath $Directory -PathType Container)) { return $false }
    $man = Join-Path $Directory 'manifests'
    $blobs = Join-Path $Directory 'blobs'
    if (-not (Test-Path -LiteralPath $man -PathType Container) -or -not (Test-Path -LiteralPath $blobs -PathType Container)) {
        return $false
    }
    if ($AllowEmpty) { return $true }
    [bool](Get-ChildItem -LiteralPath $man -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Resolve-ModelsPath {
    <#
    Finds the models directory (4.3). Returns Path (or $null), Source and the list of Tried locations.
    An explicit -ModelsPath that is not valid is never skipped in favor of another location.
    #>
    param(
        [string]$ExplicitPath,
        [bool]$IsLocal = $true,
        [switch]$AllowEmpty,
        [string]$ServerLogPath = (Join-Path ($env:LOCALAPPDATA ?? '') 'Ollama\server.log'),
        [string]$HomeDirectory = $HOME
    )
    $tried = [Collections.Generic.List[string]]::new()
    if ($ExplicitPath) {
        $tried.Add("-ModelsPath: $ExplicitPath")
        $ok = Test-ModelsDirectory $ExplicitPath -AllowEmpty:$AllowEmpty
        return [pscustomobject]@{ Path = if ($ok) { (Resolve-Path -LiteralPath $ExplicitPath).ProviderPath } else { $null }; Source = '-ModelsPath'; Tried = $tried }
    }

    $candidates = [Collections.Generic.List[object]]::new()
    if ($IsLocal) {
        $cfg = Get-ServerConfig -LogPath $ServerLogPath
        $fromLog = if ($cfg) { $cfg.Settings['OLLAMA_MODELS'] } else { $null }
        $candidates.Add(@('server.log', $fromLog, "server.log ($ServerLogPath)"))
    }
    foreach ($scope in 'Process', 'User', 'Machine') {
        $v = [Environment]::GetEnvironmentVariable('OLLAMA_MODELS', $scope)
        $candidates.Add(@("OLLAMA_MODELS ($scope)", $v, "OLLAMA_MODELS ($scope)"))
    }
    $default = Join-Path $HomeDirectory '.ollama\models'
    $candidates.Add(@('default', $default, $default))

    foreach ($c in $candidates) {
        $source, $dir, $label = $c
        if ([string]::IsNullOrWhiteSpace($dir)) { $tried.Add("${label}: not set"); continue }
        $tried.Add("${label}: $dir")
        if (Test-ModelsDirectory $dir -AllowEmpty:$AllowEmpty) {
            Write-Verbose "Models directory: $dir (from $source)"
            return [pscustomobject]@{ Path = $dir; Source = $source; Tried = $tried }
        }
    }
    [pscustomobject]@{ Path = $null; Source = $null; Tried = $tried }
}

function Get-RequiredModelsPath {
    # Resolve-ModelsPath for actions that cannot run without it. Prints the tried list on failure.
    param([string]$ExplicitPath, [switch]$AllowEmpty)
    if (-not $script:Server.IsLocal) {
        Write-Fail "This action works on the local disk, but -OllamaHost $($script:Server.Url) is not this machine."
        return $null
    }
    $r = Resolve-ModelsPath -ExplicitPath $ExplicitPath -IsLocal $true -AllowEmpty:$AllowEmpty
    if (-not $r.Path) {
        Write-Fail ("Could not find the Ollama models directory. Tried:`n  " + ($r.Tried -join "`n  ") +
            "`nPass it with -ModelsPath.")
        return $null
    }
    $r.Path
}

function Get-OptionalModelsPath {
    # Models directory for estimates and free-space checks; $null for remote hosts or when unknown.
    if (-not $script:Server.IsLocal) { return $null }
    (Resolve-ModelsPath -IsLocal $true).Path
}

#endregion

#region Manifests and blobs on disk

function Get-ManifestIndex {
    <#
    Reads every manifest under <ModelsDir>\manifests (all registry hosts).
    Each entry: Name, Registry, Namespace, Model, Tag, Path, Digest, Blobs (blob file names), BlobSizes,
    IsCloud, Manifest.
    #>
    param([Parameter(Mandatory)][string]$ModelsDir)
    $root = Join-Path $ModelsDir 'manifests'
    $blobsDir = Join-Path $ModelsDir 'blobs'
    foreach ($f in Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue) {
        $rel = [IO.Path]::GetRelativePath($root, $f.FullName) -split '[\\/]'
        if ($rel.Count -lt 4) { Write-Verbose "Skipping unexpected file $($f.FullName)"; continue }
        $registry = $rel[0]; $tag = $rel[-1]; $name = $rel[-2]
        $ns = ($rel[1..($rel.Count - 3)]) -join '/'
        try {
            $bytes = [IO.File]::ReadAllBytes($f.FullName)
            $m = [Text.Encoding]::UTF8.GetString($bytes) | ConvertFrom-Json
        }
        catch {
            Write-Warning "Cannot read manifest $($f.FullName): $($_.Exception.Message)"
            continue
        }
        $blobs = [Collections.Generic.List[string]]::new()
        $sizes = @{}
        $configBlob = $null
        if ($m.PSObject.Properties['config'] -and $m.config -and $m.config.digest) {
            $configBlob = ConvertTo-BlobFileName $m.config.digest
            $blobs.Add($configBlob); $sizes[$configBlob] = [long]$m.config.size
        }
        $layerCount = 0
        if ($m.PSObject.Properties['layers']) {
            foreach ($l in @($m.layers)) {
                if ($null -eq $l -or -not $l.digest) { continue }
                $layerCount++
                $b = ConvertTo-BlobFileName $l.digest
                if (-not $sizes.ContainsKey($b)) { $blobs.Add($b) }
                $sizes[$b] = [long]$l.size
            }
        }
        # Cloud models: no layers, config blob carries remote_host.
        $isCloud = $false
        if ($layerCount -eq 0 -and $configBlob) {
            $cfgPath = Join-Path $blobsDir $configBlob
            try {
                $cfg = Get-Content -LiteralPath $cfgPath -Raw -ErrorAction Stop | ConvertFrom-Json
                $isCloud = [bool]($cfg.PSObject.Properties['remote_host'] -and $cfg.remote_host)
            }
            catch { }
        }
        [pscustomobject]@{
            Name      = ConvertTo-ModelName -Registry $registry -Namespace $ns -Name $name -Tag $tag
            Registry  = $registry
            Namespace = $ns
            Model     = $name
            Tag       = $tag
            Path      = $f.FullName
            RelPath   = [IO.Path]::GetRelativePath($root, $f.FullName)
            Digest    = Get-Sha256Hex $bytes
            Blobs     = [string[]]$blobs
            BlobSizes = $sizes
            IsCloud   = $isCloud
            Manifest  = $m
        }
    }
}

function Get-BlobInventory {
    # Classifies files in the blobs folder: real blobs (sha256-<64 hex>) and partial downloads.
    param([Parameter(Mandatory)][string]$BlobsDir)
    $blobs = @{}
    $partial = [Collections.Generic.List[object]]::new()
    foreach ($f in Get-ChildItem -LiteralPath $BlobsDir -File -Force -ErrorAction SilentlyContinue) {
        if ($f.Name -match '^sha256-[0-9a-f]{64}$') { $blobs[$f.Name] = $f }
        elseif ($f.Name -like '*-partial*') { $partial.Add($f) }
    }
    [pscustomobject]@{ Blobs = $blobs; Partial = $partial }
}

function Get-DiskUsage {
    <#
    Computes real disk usage (3.5). Per model: Size, SharedSize, UniqueSize. Also Orphans (blob files no
    manifest references), Partial (*-partial* files), TotalOnDisk and SumOfModelSizes.
    #>
    param([Parameter(Mandatory)][string]$ModelsDir, [object[]]$Index)
    if (-not $PSBoundParameters.ContainsKey('Index')) { $Index = @(Get-ManifestIndex -ModelsDir $ModelsDir) }
    $inv = Get-BlobInventory -BlobsDir (Join-Path $ModelsDir 'blobs')
    $refCount = @{}
    foreach ($e in $Index) {
        foreach ($b in $e.Blobs) { $refCount[$b] = 1 + ($refCount[$b] ?? 0) }
    }
    $rows = foreach ($e in $Index) {
        $size = 0L; $shared = 0L
        foreach ($b in $e.Blobs) {
            $len = if ($inv.Blobs.ContainsKey($b)) { $inv.Blobs[$b].Length } else { $e.BlobSizes[$b] }
            $size += $len
            if ($refCount[$b] -gt 1) { $shared += $len }
        }
        $missing = @($e.Blobs | Where-Object { -not $inv.Blobs.ContainsKey($_) })
        [pscustomobject]@{
            Name         = $e.Name
            Size         = [long]$size
            SharedSize   = [long]$shared
            UniqueSize   = [long]($size - $shared)
            MissingBlobs = $missing.Count
        }
    }
    $orphans = @($inv.Blobs.Values | Where-Object { -not $refCount.ContainsKey($_.Name) } | Sort-Object Name)
    $referenced = 0L
    foreach ($b in $refCount.Keys) { if ($inv.Blobs.ContainsKey($b)) { $referenced += $inv.Blobs[$b].Length } }
    $orphanBytes = [long](($orphans | Measure-Object Length -Sum).Sum ?? 0)
    $partialBytes = [long](($inv.Partial | Measure-Object Length -Sum).Sum ?? 0)
    [pscustomobject]@{
        Models          = @($rows)
        Orphans         = $orphans
        OrphanBytes     = $orphanBytes
        Partial         = @($inv.Partial)
        PartialBytes    = $partialBytes
        ReferencedBytes = $referenced
        TotalOnDisk     = $referenced + $orphanBytes + $partialBytes
        SumOfModelSizes = [long](($rows | Measure-Object Size -Sum).Sum ?? 0)
    }
}

function Test-PullInProgress {
    # A pull is in progress when a *-partial* file changed in the last 60 s.
    param([string]$BlobsDir)
    $limit = (Get-Date).AddSeconds(-60)
    [bool](Get-ChildItem -LiteralPath $BlobsDir -File -Filter '*-partial*' -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -gt $limit } | Select-Object -First 1)
}

function Copy-BlobFile {
    # Copies one file; robocopy /Z when available (resumable), otherwise Copy-Item.
    param([string]$SourceDir, [string]$DestDir, [string]$FileName)
    $null = New-Item -ItemType Directory -Path $DestDir -Force
    $robocopy = Get-Command robocopy.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($robocopy) {
        Write-Verbose "robocopy $SourceDir -> $DestDir $FileName"
        $null = & $robocopy.Source $SourceDir $DestDir $FileName /Z /R:2 /W:2 /NJH /NJS /NDL /NC /NS /NP
        if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE for $FileName" }
    }
    else {
        Copy-Item -LiteralPath (Join-Path $SourceDir $FileName) -Destination (Join-Path $DestDir $FileName) -Force
    }
}

function Copy-ModelFiles {
    <#
    Copies the blobs of one manifest entry from SourceDir to DestDir (both models-style folders) and
    verifies each copied blob. Blobs already in the destination with the right size (or for restore,
    at all) are skipped. Returns BlobsCopied, BytesCopied, Error.
    #>
    param(
        [Parameter(Mandatory)][object]$Entry,
        [Parameter(Mandatory)][string]$SourceDir,
        [Parameter(Mandatory)][string]$DestDir,
        [switch]$SkipExistingAnySize,
        # Remove blobs copied by this call when the model fails (restore: they would be orphans).
        [switch]$CleanupOnError
    )
    $srcBlobs = Join-Path $SourceDir 'blobs'
    $dstBlobs = Join-Path $DestDir 'blobs'
    $copied = 0; $bytes = 0L
    $newFiles = [Collections.Generic.List[string]]::new()
    $fail = {
        param($msg)
        if ($CleanupOnError) { foreach ($n in $newFiles) { Remove-Item -LiteralPath $n -Force -ErrorAction SilentlyContinue } }
        [pscustomobject]@{ BlobsCopied = $copied; BytesCopied = $bytes; Error = $msg }
    }
    foreach ($b in $Entry.Blobs) {
        $dst = Join-Path $dstBlobs $b
        $src = Join-Path $srcBlobs $b
        if (Test-Path -LiteralPath $dst) {
            if ($SkipExistingAnySize) { continue }
            $srcItem = Get-Item -LiteralPath $src -ErrorAction SilentlyContinue
            if ($srcItem -and (Get-Item -LiteralPath $dst).Length -eq $srcItem.Length) { continue }
        }
        if (-not (Test-Path -LiteralPath $src)) {
            return (& $fail "blob $b is missing in $srcBlobs")
        }
        try { Copy-BlobFile -SourceDir $srcBlobs -DestDir $dstBlobs -FileName $b }
        catch { return (& $fail $_.Exception.Message) }
        $newFiles.Add($dst)
        $copied++
        $bytes += (Get-Item -LiteralPath $dst).Length
        $hash = Get-FileSha256Hex $dst
        if ("sha256-$hash" -ne $b) {
            Remove-Item -LiteralPath $dst -Force -ErrorAction SilentlyContinue
            return (& $fail "checksum mismatch for $b (got $(Get-ShortDigest $hash))")
        }
    }
    [pscustomobject]@{ BlobsCopied = $copied; BytesCopied = $bytes; Error = $null }
}

#endregion

#region Registry

function Invoke-RegistryRequest {
    <#
    One registry request with retries for 429/5xx (Retry-After, else 2 s then 5 s).
    Returns StatusCode, Headers, Bytes; or StatusCode 0 with Error and Unreachable on network failure.
    #>
    param(
        [Parameter(Mandatory)][string]$Uri,
        [ValidateSet('HEAD', 'GET')][string]$Method = 'HEAD',
        [int]$TimeoutSec = 10,
        [int[]]$RetryDelays = @(2, 5)
    )
    $headers = @{ Accept = 'application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json' }
    for ($attempt = 0; ; $attempt++) {
        $sw = [Diagnostics.Stopwatch]::StartNew()
        try {
            $r = Invoke-WebRequest -Uri $Uri -Method $Method -Headers $headers -TimeoutSec $TimeoutSec `
                -SkipHttpErrorCheck -ErrorAction Stop -Verbose:$false
        }
        catch {
            $ex = $_.Exception
            $isTimeout = $false
            for ($e = $ex; $e; $e = $e.InnerException) {
                if ($e -is [TimeoutException] -or $e -is [Threading.Tasks.TaskCanceledException]) { $isTimeout = $true }
            }
            Write-Verbose ('{0} {1} -> {2} ({3} ms)' -f $Method, $Uri, ($isTimeout ? 'timeout' : 'failed'), $sw.ElapsedMilliseconds)
            if ($isTimeout) {
                return [pscustomobject]@{ StatusCode = 0; Headers = $null; Bytes = $null; Error = "timeout after $TimeoutSec s"; Unreachable = $false }
            }
            return [pscustomobject]@{ StatusCode = 0; Headers = $null; Bytes = $null; Error = 'registry unreachable'; Unreachable = $true; Detail = $ex.Message }
        }
        $code = [int]$r.StatusCode
        Write-Verbose ('{0} {1} -> {2} ({3} ms)' -f $Method, $Uri, $code, $sw.ElapsedMilliseconds)
        if (($code -eq 429 -or $code -ge 500) -and $attempt -lt $RetryDelays.Count) {
            $delay = $RetryDelays[$attempt]
            $ra = if ($r.Headers) { @($r.Headers['Retry-After'])[0] } else { $null }
            $sec = 0
            if ($ra -and [int]::TryParse([string]$ra, [ref]$sec)) { $delay = [Math]::Min([Math]::Max($sec, 1), 30) }
            Start-Sleep -Seconds $delay
            continue
        }
        $bytes = $null
        if ($Method -eq 'GET' -and $r.PSObject.Properties['RawContentStream'] -and $r.RawContentStream) {
            $bytes = $r.RawContentStream.ToArray()
        }
        return [pscustomobject]@{ StatusCode = $code; Headers = $r.Headers; Bytes = $bytes; Error = $null; Unreachable = $false }
    }
}

function Get-RegistryDigest {
    <#
    Checks one model against its registry (1.3). Returns Name, Status (current / outdated /
    not-in-registry / error), RemoteDigest, Manifest (remote, for outdated models),
    Error, Unreachable.
    Self-contained apart from Split-ModelName and Invoke-RegistryRequest, so it can run in
    ForEach-Object -Parallel.
    #>
    param(
        [Parameter(Mandatory)][string]$Name,
        [string]$LocalDigest,
        [int]$TimeoutSec = 10,
        [int[]]$RetryDelays = @(2, 5)
    )
    $p = Split-ModelName $Name
    $uri = 'https://{0}/v2/{1}/{2}/manifests/{3}' -f $p.Registry, $p.Namespace, $p.Name, $p.Tag
    $local = ($LocalDigest ?? '') -replace '^sha256:', ''
    $result = [ordered]@{ Name = $Name; Status = 'error'; RemoteDigest = $null; Manifest = $null; Error = $null; Unreachable = $false }

    function Set-HttpError($resp) {
        if ($resp.StatusCode -eq 0) {
            $result.Error = $resp.Error; $result.Unreachable = $resp.Unreachable
        }
        elseif ($resp.StatusCode -in 401, 403) { $result.Error = "access denied (HTTP $($resp.StatusCode))" }
        else { $result.Error = "HTTP $($resp.StatusCode)" }
    }

    $head = Invoke-RegistryRequest -Uri $uri -Method HEAD -TimeoutSec $TimeoutSec -RetryDelays $RetryDelays
    if ($head.StatusCode -eq 404) { $result.Status = 'not-in-registry'; return [pscustomobject]$result }
    if ($head.StatusCode -ne 200) { Set-HttpError $head; return [pscustomobject]$result }

    $headerDigest = @($head.Headers['ollama-content-digest'])[0]
    if ($headerDigest) {
        $result.RemoteDigest = ([string]$headerDigest) -replace '^sha256:', ''
        if ($result.RemoteDigest -eq $local) { $result.Status = 'current'; return [pscustomobject]$result }
    }

    # Outdated (need the manifest for the download estimate) or no digest header (need the body hash).
    $get = Invoke-RegistryRequest -Uri $uri -Method GET -TimeoutSec $TimeoutSec -RetryDelays $RetryDelays
    if ($get.StatusCode -eq 404) { $result.Status = 'not-in-registry'; return [pscustomobject]$result }
    if ($get.StatusCode -ne 200 -or -not $get.Bytes) {
        if ($headerDigest) {
            # Outdated is certain from the header; only the estimate is lost.
            $result.Status = 'outdated'; return [pscustomobject]$result
        }
        Set-HttpError $get; if (-not $result.Error) { $result.Error = 'empty manifest response' }
        return [pscustomobject]$result
    }
    try { $result.Manifest = [Text.Encoding]::UTF8.GetString($get.Bytes) | ConvertFrom-Json } catch { }
    if ($headerDigest) { $result.Status = 'outdated'; return [pscustomobject]$result }

    # Ollama stores the manifest bytes as received, so the body hash equals the local digest when
    # current. Verified for registry.ollama.ai and hf.co.
    $bodyDigest = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($get.Bytes)).ToLowerInvariant()
    $result.RemoteDigest = $bodyDigest
    $result.Status = ($bodyDigest -eq $local) ? 'current' : 'outdated'
    [pscustomobject]$result
}

function Get-DownloadEstimate {
    <#
    Bytes to download for a remote manifest: config + layers whose blob file is not already local.
    Exact is $false when the blobs folder is unknown (then all layers are counted).
    #>
    param([object]$Manifest, [string]$BlobsDir)
    if (-not $Manifest) { return $null }
    $items = @()
    if ($Manifest.PSObject.Properties['config'] -and $Manifest.config) { $items += $Manifest.config }
    if ($Manifest.PSObject.Properties['layers']) { $items += @($Manifest.layers | Where-Object { $_ }) }
    $total = 0L
    foreach ($i in $items) {
        $file = ConvertTo-BlobFileName $i.digest
        if ($BlobsDir -and (Test-Path -LiteralPath (Join-Path $BlobsDir $file))) { continue }
        $total += [long]$i.size
    }
    [pscustomobject]@{ Bytes = $total; Exact = [bool]$BlobsDir }
}

function Invoke-RegistryCheck {
    <#
    Runs Get-RegistryDigest for non-cloud models, concurrently (throttle 8), and adds the local data
    and download estimate. Returns one object per model (schema of 4.6 'outdated' plus extras).
    #>
    param([object[]]$Models, [string]$ModelsDir, [switch]$Sequential)
    $Models = @($Models | Where-Object { $_ -and -not $_.IsCloud })
    if (-not $Models.Count) { return @() }

    $jobs = foreach ($m in $Models) {
        [pscustomobject]@{ Name = $m.name; Digest = $m.digest }
    }
    $timeout = $script:RegistryTimeoutSec
    if ($Sequential -or $jobs.Count -eq 1) {
        $raw = foreach ($j in $jobs) { Get-RegistryDigest -Name $j.Name -LocalDigest $j.Digest -TimeoutSec $timeout }
    }
    else {
        $defs = @{}
        foreach ($n in 'Split-ModelName', 'Invoke-RegistryRequest', 'Get-RegistryDigest') {
            $defs[$n] = (Get-Item "function:$n").ScriptBlock.ToString()
        }
        $vp = $VerbosePreference
        $raw = $jobs | ForEach-Object -ThrottleLimit 8 -Parallel {
            foreach ($k in ($using:defs).Keys) { Set-Item -Path "function:$k" -Value ([scriptblock]::Create(($using:defs)[$k])) }
            $VerbosePreference = $using:vp
            Get-RegistryDigest -Name $_.Name -LocalDigest $_.Digest -TimeoutSec $using:timeout
        }
    }
    $byName = @{}
    foreach ($r in @($raw)) { $byName[$r.Name] = $r }
    $blobsDir = if ($ModelsDir) { Join-Path $ModelsDir 'blobs' } else { $null }
    foreach ($m in $Models) {
        $r = $byName[$m.name]
        $est = if ($r.Status -eq 'outdated') { Get-DownloadEstimate -Manifest $r.Manifest -BlobsDir $blobsDir } else { $null }
        [pscustomobject]@{
            Name             = $m.name
            Status           = $r.Status
            LocalDigest      = $m.digest
            RemoteDigest     = $r.RemoteDigest
            Size             = [long]$m.size
            DownloadEstimate = if ($est) { $est.Bytes } else { $null }
            ModifiedAt       = [datetimeoffset]$m.modified_at
            Error            = $r.Error
            EstimateExact    = if ($est) { $est.Exact } else { $false }
            Unreachable      = [bool]$r.Unreachable
        }
    }
}

function Write-CheckWarnings {
    # Warnings for models whose check did not succeed; one summary line when the registry is unreachable.
    param([object[]]$Results)
    $unreach = @($Results | Where-Object Unreachable)
    if ($unreach.Count) {
        Write-Warning ("Registry unreachable: {0} model(s) could not be checked ({1})." -f $unreach.Count, (($unreach.Name) -join ', '))
    }
    foreach ($r in $Results | Where-Object { -not $_.Unreachable }) {
        switch ($r.Status) {
            'not-in-registry' { Write-Warning "$($r.Name): not in registry (created locally or removed upstream)." }
            'error' { Write-Warning "$($r.Name): check failed: $($r.Error)." }
        }
    }
}

function Get-CheckExitCode {
    # outdated semantics: 0 all current, 1 some outdated, 3 some checks failed.
    param([object[]]$Results)
    $codes = @(0)
    if ($Results | Where-Object Status -EQ 'outdated') { $codes += 1 }
    if ($Results | Where-Object { $_.Status -eq 'error' }) { $codes += 3 }
    Get-ExitCode $codes
}

function Format-Estimate {
    param([object]$Row)
    if ($null -eq $Row.DownloadEstimate) { return '?' }
    ($Row.EstimateExact ? '' : '≤') + (Format-Size $Row.DownloadEstimate)
}

#endregion

#region Pull

function Invoke-ApiPull {
    # /api/pull streaming fallback when the CLI is missing. 60 s without progress fails the pull.
    param([string]$Name)
    $client = [Net.Http.HttpClient]::new()
    $client.Timeout = [Threading.Timeout]::InfiniteTimeSpan
    $reader = $null
    try {
        $req = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post, "$($script:Server.Url)/api/pull")
        $req.Content = [Net.Http.StringContent]::new((@{ model = $Name; stream = $true } | ConvertTo-Json -Compress), [Text.Encoding]::UTF8, 'application/json')
        Write-Verbose "POST $($script:Server.Url)/api/pull ($Name)"
        $resp = $client.SendAsync($req, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        if (-not $resp.IsSuccessStatusCode) { return "HTTP $([int]$resp.StatusCode)" }
        $reader = [IO.StreamReader]::new($resp.Content.ReadAsStream())
        $lastProgress = [DateTime]::UtcNow
        $lastKey = ''
        while ($true) {
            $task = $reader.ReadLineAsync()
            while (-not $task.Wait(500)) {
                if (([DateTime]::UtcNow - $lastProgress).TotalSeconds -ge 60) {
                    $client.CancelPendingRequests()
                    return 'stalled (no progress for 60 s)'
                }
            }
            $line = $task.Result
            if ($null -eq $line) { return 'stream ended before success' }
            if (-not $line.Trim()) { continue }
            $j = $line | ConvertFrom-Json
            if ($j.PSObject.Properties['error']) { return $j.error }
            $completed = if ($j.PSObject.Properties['completed']) { [long]$j.completed } else { 0 }
            $key = "$($j.status)|$completed"
            if ($key -ne $lastKey) { $lastProgress = [DateTime]::UtcNow; $lastKey = $key }
            if ($j.PSObject.Properties['total'] -and $j.total) {
                $pct = [Math]::Min(100, [int](100 * $completed / $j.total))
                Write-Progress -Activity "Pulling $Name" -Status "$($j.status) $(Format-Size $completed) / $(Format-Size ([long]$j.total))" -PercentComplete $pct
            }
            else {
                Write-Progress -Activity "Pulling $Name" -Status $j.status
            }
            if ($j.status -eq 'success') { return $null }
        }
    }
    catch { return $_.Exception.Message }
    finally {
        Write-Progress -Activity "Pulling $Name" -Completed
        if ($reader) { $reader.Dispose() }
        $client.Dispose()
    }
}

function Invoke-ModelPull {
    # Pulls one model; returns $null on success or an error message.
    param([string]$Name)
    if (Get-OllamaExe) {
        Write-Host "`n>>> ollama pull $Name" -ForegroundColor Cyan
        $r = Invoke-OllamaCli -Arguments 'pull', $Name -Interactive
        if ($r.ExitCode -ne 0) { return "ollama pull exited with code $($r.ExitCode)" }
        return $null
    }
    Write-Host "`n>>> pull $Name (API)" -ForegroundColor Cyan
    Invoke-ApiPull -Name $Name
}

function Test-FreeSpace {
    # 3.1 step 5. Returns $true when the download fits or the check cannot run.
    param([Nullable[long]]$Required, [string]$ModelsDir)
    if (-not $ModelsDir) {
        Write-Warning 'Models directory unknown; skipping the free-space check.'
        return $true
    }
    $free = Get-FreeSpace $ModelsDir
    if ($null -eq $free) {
        Write-Warning "Cannot read free space for $ModelsDir; skipping the free-space check."
        return $true
    }
    if ($null -ne $Required -and $Required -gt $free) {
        Write-Fail ("Not enough free space in {0}: about {1} needed, {2} free." -f $ModelsDir, (Format-Size $Required), (Format-Size $free))
        return $false
    }
    $true
}

function Invoke-PullLoop {
    <#
    Pulls each item one at a time, continuing after failures. The summary always prints, also after
    Ctrl+C (the model being pulled is then marked 'interrupted'). Returns the result rows.
    Items: Name, OldDigest, OldSize, ExportDigest (import only).
    #>
    param(
        [object[]]$Items,
        [ValidateSet('update', 'import')][string]$Mode = 'update',
        [object[]]$ExtraRows = @(),
        [string]$CountLine
    )
    $results = [Collections.Generic.List[object]]::new()
    $current = $null
    try {
        foreach ($it in $Items) {
            if (-not $script:Cmdlet.ShouldProcess($it.Name, 'Pull')) { continue }
            $current = $it
            $err = Invoke-ModelPull -Name $it.Name
            $results.Add([pscustomobject]@{
                    Name = $it.Name; Status = ($err ? 'failed' : 'updated'); OldDigest = $it.OldDigest; NewDigest = $null
                    OldSize = $it.OldSize; NewSize = $null; Error = $err; ExportDigest = $it.ExportDigest
                })
            $current = $null
        }
    }
    finally {
        if ($current) {
            $results.Add([pscustomobject]@{
                    Name = $current.Name; Status = 'interrupted'; OldDigest = $current.OldDigest; NewDigest = $null
                    OldSize = $current.OldSize; NewSize = $null; Error = 'interrupted'; ExportDigest = $current.ExportDigest
                })
        }
        $after = @{}
        if ($results.Count) {
            try { foreach ($m in Get-OllamaModels) { $after[$m.name] = $m } }
            catch { Write-Warning "Could not re-read /api/tags: $_" }
        }
        foreach ($r in $results) {
            $a = $after[$r.Name]
            if ($a) { $r.NewDigest = $a.digest; $r.NewSize = [long]$a.size }
            if ($Mode -eq 'import' -and $r.Status -eq 'updated') {
                $r.Status = if ($r.ExportDigest -and $r.NewDigest -and $r.NewDigest -ne $r.ExportDigest) { 'pulled (newer than export)' } else { 'pulled' }
            }
        }
        $table = @($results) + @($ExtraRows)
        if ($table.Count) {
            Write-Host ''
            $table | Select-Object Name, Status,
            @{ n = 'Old digest'; e = { Get-ShortDigest $_.OldDigest } },
            @{ n = 'New digest'; e = { Get-ShortDigest $_.NewDigest } },
            @{ n = 'Size change'; e = {
                    if ($null -ne $_.NewSize -and $null -ne $_.OldSize) {
                        $d = $_.NewSize - $_.OldSize; ($d -ge 0 ? '+' : '-') + (Format-Size ([Math]::Abs($d)))
                    }
                    elseif ($null -ne $_.NewSize) { '+' + (Format-Size $_.NewSize) } else { '' }
                }
            } | Format-Table -AutoSize | Out-Host
        }
        if ($CountLine) { Write-Host $CountLine }
    }
    $results | ForEach-Object { $_ | Select-Object Name, Status, OldDigest, NewDigest, OldSize, NewSize, Error }
}

#endregion

#region Confirmation

function Confirm-Destructive {
    <#
    ShouldProcess (honors -WhatIf/-Confirm) plus a default prompt for destructive operations (4.7).
    -Confirm:$false skips the prompt unless -Always.
    #>
    param([string]$Target, [string]$Operation, [string]$Query, [switch]$Always)
    if (-not $script:Cmdlet.ShouldProcess($Target, $Operation)) { return $false }
    if ($script:ConfirmMode -eq 'on') { return $true }
    if ($script:ConfirmMode -eq 'off' -and -not $Always) { return $true }
    if ($script:YesToAll -and -not $Always) { return $true }
    $yes = $false; $no = $false
    $ok = $script:Cmdlet.ShouldContinue($Query, 'Confirm', [ref]$yes, [ref]$no)
    if ($yes -and -not $Always) { $script:YesToAll = $true }
    $ok
}

#endregion

#region Actions

function Invoke-Outdated {
    param($o)
    $models = @(Get-OllamaModels)
    $sel = Select-Models -Items $models -Model $o.Model -Exclude $o.Exclude
    if ($null -eq $sel) { $script:ExitCode = 4; return }
    $cloud = @($sel | Where-Object IsCloud)
    $results = @(Invoke-RegistryCheck -Models $sel -ModelsDir (Get-OptionalModelsPath))
    $outdated = @($results | Where-Object Status -EQ 'outdated')

    if ($o.PassThru) {
        ($o.All ? $results : $outdated) | Select-Object Name, Status, LocalDigest, RemoteDigest, Size, DownloadEstimate, ModifiedAt, Error
    }
    elseif ($o.All) {
        $results | Sort-Object Name | Select-Object @{ n = 'Model'; e = { $_.Name } },
        @{ n = 'Status'; e = { $_.Status -eq 'error' ? "error: $($_.Error)" : $_.Status } },
        @{ n = 'Size'; e = { Format-Size $_.Size } },
        @{ n = 'Modified'; e = { Format-Age $_.ModifiedAt } },
        @{ n = 'Download (est.)'; e = { $_.Status -eq 'outdated' ? (Format-Estimate $_) : '' } },
        @{ n = 'Local digest'; e = { Get-ShortDigest $_.LocalDigest } },
        @{ n = 'Remote digest'; e = { Get-ShortDigest $_.RemoteDigest } } | Format-Table -AutoSize | Out-Host
    }
    elseif ($outdated.Count) {
        Write-OutdatedTable $outdated
    }
    if (-not $outdated.Count -and -not $o.All) { Write-Host 'All models are up to date.' -ForegroundColor Green }
    Write-CheckWarnings $results
    $current = @($results | Where-Object Status -EQ 'current').Count
    Write-Host ("{0} outdated, {1} current, {2} cloud skipped, {3} not checked" -f $outdated.Count, $current, $cloud.Count,
        @($results | Where-Object { $_.Status -notin 'current', 'outdated' }).Count)
    $script:ExitCode = Get-CheckExitCode $results
}

function Write-OutdatedTable {
    param([object[]]$Rows)
    $Rows | Sort-Object Name | Select-Object @{ n = 'Model'; e = { $_.Name } },
    @{ n = 'Size'; e = { Format-Size $_.Size } },
    @{ n = 'Modified'; e = { Format-Age $_.ModifiedAt } },
    @{ n = 'Download (est.)'; e = { Format-Estimate $_ } },
    @{ n = 'Local digest'; e = { Get-ShortDigest $_.LocalDigest } },
    @{ n = 'Remote digest'; e = { Get-ShortDigest $_.RemoteDigest } } | Format-Table -AutoSize | Out-Host
}

function Invoke-Update {
    param($o)
    $models = @(Get-OllamaModels)
    $sel = Select-Models -Items $models -Model $o.Model -Exclude $o.Exclude
    if ($null -eq $sel) { $script:ExitCode = 4; return }
    $cloud = @($sel | Where-Object IsCloud)
    $modelsDir = Get-OptionalModelsPath
    $results = @(Invoke-RegistryCheck -Models $sel -ModelsDir $modelsDir)
    $checkCode = Get-CheckExitCode $results
    $checkFailed = [int]($checkCode -eq 3) * 3
    $outdated = @($results | Where-Object Status -EQ 'outdated' | Sort-Object Name)
    Write-CheckWarnings $results

    $rest = @(
        '{0} current' -f @($results | Where-Object Status -EQ 'current').Count
        if ($cloud.Count) { '{0} cloud skipped' -f $cloud.Count }
        $nir = @($results | Where-Object Status -EQ 'not-in-registry').Count
        if ($nir) { '{0} not in registry' -f $nir }
        $failed = @($results | Where-Object { $_.Status -eq 'error' }).Count
        if ($failed) { '{0} check failed' -f $failed }
    ) -join ', '

    if (-not $outdated.Count) {
        Write-Host 'All models are up to date.' -ForegroundColor Green
        Write-Host $rest
        $script:ExitCode = $checkFailed
        return
    }

    if ($WhatIfPreference) {
        Write-OutdatedTable $outdated
    }
    $estimate = [long](($outdated | Measure-Object DownloadEstimate -Sum).Sum ?? 0)
    $exact = -not ($outdated | Where-Object { $null -eq $_.DownloadEstimate -or -not $_.EstimateExact })
    Write-Host ("{0} model(s) to update, estimated download {1}{2}." -f $outdated.Count, ($exact ? '' : '≤'), (Format-Size $estimate))
    if (-not $WhatIfPreference -and -not (Test-FreeSpace -Required ($exact ? $estimate : $null) -ModelsDir $modelsDir)) {
        $script:ExitCode = 4; return
    }

    $items = foreach ($r in $outdated) { [pscustomobject]@{ Name = $r.Name; OldDigest = $r.LocalDigest; OldSize = $r.Size; ExportDigest = $null } }
    $script:ExitCode = $checkFailed
    $rows = @(Invoke-PullLoop -Items $items -Mode update -CountLine $rest)
    $pullFailed = [bool]($rows | Where-Object { $_.Status -in 'failed', 'interrupted' })
    $script:ExitCode = Get-ExitCode @($checkFailed, [int]$pullFailed)
    if ($o.PassThru) { $rows }
}

function Invoke-List {
    param($o)
    $models = @(Get-OllamaModels)
    $sel = Select-Models -Items $models -Model $o.Model -Exclude $o.Exclude
    if ($null -eq $sel) { $script:ExitCode = 4; return }
    foreach ($c in @($o.Capability)) {
        if ($c) { $sel = @($sel | Where-Object { @($_.capabilities) -contains $c }) }
    }
    $status = @{}
    if ($o.Deep) {
        $results = @(Invoke-RegistryCheck -Models $sel -ModelsDir (Get-OptionalModelsPath))
        foreach ($r in $results) { $status[$r.Name] = $r }
        Write-CheckWarnings @($results | Where-Object Status -NE 'not-in-registry')
        $script:ExitCode = [int]([bool]($results | Where-Object { $_.Status -eq 'error' })) * 3
    }
    $rows = foreach ($m in $sel) {
        $d = $m.details
        $r = $status[$m.name]
        $type = if ($m.IsCloud) { 'cloud' } elseif ($r -and $r.Status -eq 'not-in-registry') { 'custom' } else { 'local' }
        $st = if ($m.IsCloud) { 'cloud' } elseif ($r) { $r.Status -eq 'error' ? "error: $($r.Error)" : $r.Status } else { $null }
        [pscustomobject]@{
            Name          = $m.name
            Type          = $type
            Size          = [long]$m.size
            ModifiedAt    = [datetimeoffset]$m.modified_at
            Family        = $d.family
            Parameters    = $d.parameter_size
            Quantization  = $d.quantization_level
            ContextLength = if ($d.PSObject.Properties['context_length']) { $d.context_length } else { $null }
            Capabilities  = [string[]]@($m.capabilities | Where-Object { $_ })
            Digest        = $m.digest
            Status        = $st
        }
    }
    $rows = switch ($o.Sort) {
        'Size' { $rows | Sort-Object Size -Descending }
        'Modified' { $rows | Sort-Object ModifiedAt -Descending }
        default { $rows | Sort-Object Name }
    }
    if ($o.PassThru) {
        if ($o.Deep) { $rows } else { $rows | Select-Object * -ExcludeProperty Status }
        return
    }
    if (-not $rows) { Write-Host 'No models match.'; return }
    $cols = @(
        'Name'
        'Type'
        if ($o.Deep) { 'Status' }
        @{ n = 'Size'; e = { $_.Type -eq 'cloud' ? '-' : (Format-Size $_.Size) } }
        @{ n = 'Modified'; e = { Format-Age $_.ModifiedAt } }
        'Family'
        @{ n = 'Params'; e = { $_.Parameters } }
        @{ n = 'Quant'; e = { $_.Quantization } }
        @{ n = 'Context'; e = { $_.ContextLength } }
        @{ n = 'Capabilities'; e = { ($_.Capabilities | Where-Object { $_ -ne 'completion' }) -join ',' } }
    )
    $rows | Select-Object $cols | Format-Table -AutoSize | Out-Host
    Write-Host ('{0} model(s), {1} total' -f @($rows).Count, (Format-Size ([long](($rows | Where-Object Type -NE 'cloud' | Measure-Object Size -Sum).Sum ?? 0))))
}

function Invoke-Stop {
    param($o)
    $loaded = @(Get-LoadedModels)
    if (-not $loaded.Count) { Write-Host 'No models are loaded.'; return }
    $sel = Select-Models -Items $loaded -Model $o.Model -Exclude $o.Exclude
    if ($null -eq $sel) { $script:ExitCode = 4; return }
    $rows = foreach ($m in $sel) {
        if (-not $script:Cmdlet.ShouldProcess($m.name, 'Unload')) { continue }
        try {
            Stop-OllamaModel -Name $m.name
            [pscustomobject]@{ Name = $m.name; Status = 'unloaded' }
        }
        catch {
            Write-Warning "$($m.name): $_"
            [pscustomobject]@{ Name = $m.name; Status = 'failed' }
        }
    }
    if ($rows | Where-Object Status -EQ 'failed') { $script:ExitCode = 1 }
    if ($o.PassThru) { $rows }
    elseif ($rows) { $rows | Format-Table -AutoSize | Out-Host }
}

function Invoke-Usage {
    param($o)
    $dir = Get-RequiredModelsPath -ExplicitPath $o.ModelsPath
    if (-not $dir) { $script:ExitCode = 4; return }
    Write-Host "Models directory: $dir"
    $u = Get-DiskUsage -ModelsDir $dir
    $sel = Select-Models -Items $u.Models -Model $o.Model -Exclude $o.Exclude -NameProperty Name
    if ($null -eq $sel) { $script:ExitCode = 4; return }
    foreach ($r in $sel | Where-Object MissingBlobs) { Write-Warning "$($r.Name): $($r.MissingBlobs) referenced blob(s) missing on disk." }
    if ($o.PassThru) {
        $sel | Select-Object Name, Size, SharedSize, UniqueSize
        [pscustomobject]@{ Name = '(orphans)'; Size = $u.OrphanBytes; SharedSize = 0L; UniqueSize = $u.OrphanBytes }
        [pscustomobject]@{ Name = '(partial)'; Size = $u.PartialBytes; SharedSize = 0L; UniqueSize = $u.PartialBytes }
        return
    }
    $sel | Sort-Object Size -Descending | Select-Object Name,
    @{ n = 'Size'; e = { Format-Size $_.Size } },
    @{ n = 'Shared'; e = { Format-Size $_.SharedSize } },
    @{ n = 'Unique (freed by rm)'; e = { Format-Size $_.UniqueSize } } | Format-Table -AutoSize | Out-Host
    Write-Host ('Referenced blobs   {0,12}' -f (Format-Size $u.ReferencedBytes))
    Write-Host ('Orphaned blobs     {0,12}  ({1} file(s); remove with: prune -Orphans)' -f (Format-Size $u.OrphanBytes), $u.Orphans.Count)
    Write-Host ('Partial downloads  {0,12}  ({1} file(s))' -f (Format-Size $u.PartialBytes), $u.Partial.Count)
    Write-Host ('Total on disk      {0,12}' -f (Format-Size $u.TotalOnDisk)) -ForegroundColor Cyan
    Write-Host ('Sum of model sizes {0,12}  (what ollama list adds up to; shared blobs counted once per model)' -f (Format-Size $u.SumOfModelSizes))
}

function Get-DuplicateRemovals {
    <#
    For tags sharing a digest, returns the names to remove (Name, KeptName). Keep rule (3.6):
    a non-'latest' tag, then the shortest full name, then alphabetical.
    #>
    param([object[]]$Models)
    foreach ($g in @($Models | Where-Object { $_ -and $_.digest } | Group-Object digest | Where-Object Count -GT 1)) {
        $ordered = @($g.Group | Sort-Object @{ e = { (Get-ModelTag $_.name) -eq 'latest' ? 1 : 0 } },
            @{ e = { $_.name.Length } }, @{ e = { $_.name } })
        $keep = $ordered[0].name
        foreach ($m in $ordered | Select-Object -Skip 1) {
            [pscustomobject]@{ Name = $m.name; KeptName = $keep }
        }
    }
}

function Remove-OllamaModel {
    param([string]$Name)
    if (Get-OllamaExe) {
        $r = Invoke-OllamaCli -Arguments 'rm', $Name
        if ($r.ExitCode -ne 0) { throw ($r.Output ? $r.Output : "ollama rm exited with code $($r.ExitCode)") }
    }
    else {
        $null = Invoke-OllamaApi -Endpoint '/api/delete' -Method DELETE -Body @{ model = $Name }
    }
}

function Invoke-Prune {
    param($o)
    $hasModelCriteria = $o.OlderThanDays -or $o.Duplicates -or $o.Model
    if (-not $hasModelCriteria -and -not $o.Orphans) {
        Write-Fail 'prune needs at least one of -OlderThanDays, -Duplicates, -Orphans or -Model.'
        $script:ExitCode = 4; return
    }
    $rows = [Collections.Generic.List[object]]::new()
    $failed = $false

    if ($hasModelCriteria) {
        if (-not (Test-OllamaServer)) {
            Write-Fail "Ollama server not reachable at $($script:Server.Url) — is it running?"
            $script:ExitCode = 2; return
        }
        $models = @(Get-OllamaModels)
        $sel = Select-Models -Items $models -Model $o.Model -Exclude $o.Exclude
        if ($null -eq $sel) { $script:ExitCode = 4; return }
        $reasons = @{}
        $candidates = $sel
        if ($o.OlderThanDays) {
            $limit = [datetimeoffset]::Now.AddDays(-$o.OlderThanDays)
            $candidates = @($candidates | Where-Object { -not $_.IsCloud -and [datetimeoffset]$_.modified_at -lt $limit })
            foreach ($c in $candidates) { $reasons[$c.name] = @("modified $(Format-Age ([datetimeoffset]$c.modified_at))") }
        }
        if ($o.Duplicates) {
            $dups = @{}
            foreach ($d in Get-DuplicateRemovals -Models $models) { $dups[$d.Name] = $d.KeptName }
            $candidates = @($candidates | Where-Object { $dups.ContainsKey($_.name) })
            foreach ($c in $candidates) { $reasons[$c.name] = @($reasons[$c.name]) + "alias of $($dups[$c.name])" | Where-Object { $_ } }
        }
        foreach ($c in $candidates) { if (-not $reasons[$c.name]) { $reasons[$c.name] = @('matches -Model') } }

        # Freed bytes: blobs not referenced by any other manifest (local host only).
        $modelsDir = Get-OptionalModelsPath
        if (-not $candidates.Count) { Write-Host 'No models match the prune criteria.' }
        foreach ($c in $candidates | Sort-Object name) {
            $reason = $reasons[$c.name] -join '; '
            $freed = 0L
            if ($modelsDir -and -not $o.Duplicates) {
                $idx = @(Get-ManifestIndex -ModelsDir $modelsDir)
                $u = Get-DiskUsage -ModelsDir $modelsDir -Index $idx
                $freed = [long](($u.Models | Where-Object Name -EQ $c.name | Select-Object -First 1).UniqueSize ?? 0)
            }
            $q = "Delete model $($c.name) ($reason)?"
            if (-not (Confirm-Destructive -Target $c.name -Operation 'Delete model' -Query $q)) {
                $st = $WhatIfPreference ? 'would delete' : 'skipped'
                $rows.Add([pscustomobject]@{ Name = $c.name; Reason = $reason; Status = $st; FreedBytes = ($WhatIfPreference ? $freed : 0L) }); continue
            }
            try {
                Remove-OllamaModel -Name $c.name
                $rows.Add([pscustomobject]@{ Name = $c.name; Reason = $reason; Status = 'deleted'; FreedBytes = $freed })
            }
            catch {
                $failed = $true
                Write-Warning "$($c.name): $_"
                $rows.Add([pscustomobject]@{ Name = $c.name; Reason = $reason; Status = 'failed'; FreedBytes = 0L })
            }
        }
    }

    if ($o.Orphans) {
        $dir = Get-RequiredModelsPath -ExplicitPath $o.ModelsPath
        if (-not $dir) { $script:ExitCode = 4; return }
        $blobsDir = Join-Path $dir 'blobs'
        $u = Get-DiskUsage -ModelsDir $dir
        if (-not $u.Orphans.Count) { Write-Host 'No orphaned blobs.' }
        else {
            $force = $false
            if ((Test-OllamaServer) -and (Test-PullInProgress $blobsDir)) {
                Write-Warning 'A pull seems to be in progress (a *-partial* file changed in the last 60 s). Its finished layers can look orphaned until the pull completes.'
                $force = $true
            }
            foreach ($f in $u.Orphans) {
                $q = "Delete orphaned blob $($f.Name) ($(Format-Size $f.Length))?"
                if (-not (Confirm-Destructive -Target $f.FullName -Operation 'Delete orphaned blob' -Query $q -Always:$force)) {
                    $st = $WhatIfPreference ? 'would delete' : 'skipped'
                    $rows.Add([pscustomobject]@{ Name = $f.Name; Reason = 'orphaned blob'; Status = $st; FreedBytes = ($WhatIfPreference ? [long]$f.Length : 0L) }); continue
                }
                # Re-read every manifest right before deleting (4.7).
                $refs = @{}
                foreach ($e in Get-ManifestIndex -ModelsDir $dir) { foreach ($b in $e.Blobs) { $refs[$b] = $true } }
                if ($refs.ContainsKey($f.Name)) {
                    $rows.Add([pscustomobject]@{ Name = $f.Name; Reason = 'orphaned blob'; Status = 'skipped (now referenced)'; FreedBytes = 0L }); continue
                }
                try {
                    Remove-Item -LiteralPath $f.FullName -Force -ErrorAction Stop
                    $rows.Add([pscustomobject]@{ Name = $f.Name; Reason = 'orphaned blob'; Status = 'deleted'; FreedBytes = [long]$f.Length })
                }
                catch {
                    $failed = $true
                    Write-Warning "$($f.Name): $($_.Exception.Message)"
                    $rows.Add([pscustomobject]@{ Name = $f.Name; Reason = 'orphaned blob'; Status = 'failed'; FreedBytes = 0L })
                }
            }
        }
    }

    if ($failed) { $script:ExitCode = 1 }
    if ($o.PassThru) { $rows; return }
    if ($rows.Count) {
        $rows | Select-Object Name, Reason, Status, @{ n = 'Freed'; e = { Format-Size $_.FreedBytes } } | Format-Table -AutoSize | Out-Host
        $total = [long](($rows | Measure-Object FreedBytes -Sum).Sum ?? 0)
        Write-Host (($WhatIfPreference ? 'Would free {0}.' : 'Freed {0}.') -f (Format-Size $total))
    }
}

function Invoke-ExportList {
    param($o)
    $models = @(Get-OllamaModels)
    $sel = Select-Models -Items $models -Model $o.Model -Exclude $o.Exclude
    if ($null -eq $sel) { $script:ExitCode = 4; return }
    $results = @(Invoke-RegistryCheck -Models $sel -ModelsDir (Get-OptionalModelsPath))
    $status = @{}
    foreach ($r in $results) { $status[$r.Name] = $r.Status }
    if ($results | Where-Object Unreachable) {
        Write-Warning 'Registry unreachable: custom (locally created) models cannot be detected; they are exported as type "local".'
    }
    $failedChecks = @($results | Where-Object { $_.Status -eq 'error' -and -not $_.Unreachable })
    foreach ($r in $failedChecks) { Write-Warning "$($r.Name): check failed ($($r.Error)); exported as type 'local'." }
    $version = try { (Invoke-OllamaApi -Endpoint '/api/version').version } catch { $null }
    $doc = [ordered]@{
        schemaVersion = 1
        exportedAt    = [datetimeoffset]::Now.ToString('yyyy-MM-ddTHH:mm:sszzz')
        sourceHost    = $script:Server.Url
        ollamaVersion = $version
        models        = @(foreach ($m in $sel | Sort-Object name) {
                [ordered]@{
                    name   = $m.name
                    digest = $m.digest
                    type   = if ($m.IsCloud) { 'cloud' } elseif ($status[$m.name] -eq 'not-in-registry') { 'custom' } else { 'local' }
                    size   = if ($m.IsCloud) { 0 } else { [long]$m.size }
                }
            })
    }
    $full = [IO.Path]::GetFullPath($o.Path, (Get-Location).ProviderPath)
    if ($script:Cmdlet.ShouldProcess($full, "Write list of $($doc.models.Count) model(s)")) {
        $dirName = Split-Path -Parent $full
        if ($dirName -and -not (Test-Path -LiteralPath $dirName)) { $null = New-Item -ItemType Directory -Path $dirName -Force }
        [IO.File]::WriteAllText($full, ($doc | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
        Write-Host ('Exported {0} model(s) to {1}' -f $doc.models.Count, $full)
    }
    $script:ExitCode = ($results | Where-Object { $_.Status -eq 'error' }) ? 3 : 0
    if ($o.PassThru) { $doc.models | ForEach-Object { [pscustomobject]$_ } }
}

function Invoke-ImportList {
    param($o)
    $full = [IO.Path]::GetFullPath($o.Path, (Get-Location).ProviderPath)
    try { $doc = Get-Content -LiteralPath $full -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop }
    catch { Write-Fail "Cannot read ${full}: $($_.Exception.Message)"; $script:ExitCode = 4; return }
    if (-not $doc.PSObject.Properties['schemaVersion'] -or $doc.schemaVersion -ne 1) {
        Write-Fail "Unsupported schemaVersion '$($doc.schemaVersion)' in $full (expected 1)."
        $script:ExitCode = 4; return
    }
    $listed = Select-Models -Items @($doc.models) -Model $o.Model -Exclude $o.Exclude
    if ($null -eq $listed) { $script:ExitCode = 4; return }
    $local = @{}
    foreach ($m in Get-OllamaModels) { $local[$m.name] = $m }

    $toPull = [Collections.Generic.List[object]]::new()
    $extra = [Collections.Generic.List[object]]::new()
    $same = 0; $custom = 0
    foreach ($m in $listed) {
        if ($m.type -eq 'custom') { Write-Warning "$($m.name): custom model, skipped (it cannot be pulled)."; $custom++; continue }
        $have = $local[$m.name]
        if ($have) {
            if ($m.digest -and $have.digest -ne $m.digest) {
                $extra.Add([pscustomobject]@{ Name = $m.name; Status = 'exists (different version)'; OldDigest = $have.digest; NewDigest = $m.digest; OldSize = $null; NewSize = $null; Error = $null })
            }
            else { $same++ }
            continue
        }
        $toPull.Add([pscustomobject]@{ Name = $m.name; OldDigest = $null; OldSize = $null; ExportDigest = $m.digest; Size = [long]($m.size ?? 0) })
    }
    $countLine = ('{0} already present, {1} custom skipped' -f ($same + $extra.Count), $custom)
    if (-not $toPull.Count) {
        Write-Host 'Every listed model is already present.' -ForegroundColor Green
        if ($extra.Count) { $extra | Select-Object Name, Status | Format-Table -AutoSize | Out-Host }
        Write-Host $countLine
        if ($o.PassThru) { $extra }
        return
    }
    $estimate = [long](($toPull | Measure-Object Size -Sum).Sum ?? 0)
    Write-Host ('{0} model(s) to pull, about {1} (sizes from the export).' -f $toPull.Count, (Format-Size $estimate))
    if (-not $WhatIfPreference -and -not (Test-FreeSpace -Required $estimate -ModelsDir (Get-OptionalModelsPath))) {
        $script:ExitCode = 4; return
    }
    $rows = @(Invoke-PullLoop -Items $toPull -Mode import -ExtraRows $extra -CountLine $countLine)
    if ($rows | Where-Object { $_.Status -in 'failed', 'interrupted' }) { $script:ExitCode = 1 }
    if ($o.PassThru) { $rows; $extra }
}

function Invoke-Backup {
    param($o)
    $dir = Get-RequiredModelsPath -ExplicitPath $o.ModelsPath
    if (-not $dir) { $script:ExitCode = 4; return }
    $index = @(Get-ManifestIndex -ModelsDir $dir)
    $sel = Select-Models -Items $index -Model $o.Model -Exclude $o.Exclude -NameProperty Name
    if ($null -eq $sel) { $script:ExitCode = 4; return }
    $dest = [IO.Path]::GetFullPath($o.Path, (Get-Location).ProviderPath)
    $rows = foreach ($e in $sel | Sort-Object Name) {
        if ($e.IsCloud) {
            Write-Host "$($e.Name): cloud model, skipped (nothing stored locally)."
            [pscustomobject]@{ Name = $e.Name; Digest = $e.Digest; Status = 'skipped (cloud)'; BlobsCopied = 0; BytesCopied = 0L }
            continue
        }
        if (-not $script:Cmdlet.ShouldProcess($e.Name, "Back up to $dest")) { continue }
        Write-Host "Backing up $($e.Name)..."
        $r = Copy-ModelFiles -Entry $e -SourceDir $dir -DestDir $dest
        if ($r.Error) {
            Write-Warning "$($e.Name): $($r.Error)"
            [pscustomobject]@{ Name = $e.Name; Digest = $e.Digest; Status = 'failed'; BlobsCopied = $r.BlobsCopied; BytesCopied = $r.BytesCopied }
            continue
        }
        # Manifest last, so a partial backup never looks complete.
        $manDest = Join-Path (Join-Path $dest 'manifests') $e.RelPath
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $manDest) -Force
        Copy-Item -LiteralPath $e.Path -Destination $manDest -Force
        [pscustomobject]@{ Name = $e.Name; Digest = $e.Digest; Status = ($r.BlobsCopied ? 'copied' : 'up to date'); BlobsCopied = $r.BlobsCopied; BytesCopied = $r.BytesCopied }
    }
    if ($rows | Where-Object Status -EQ 'failed') { $script:ExitCode = 1 }
    if ($o.PassThru) { $rows; return }
    if ($rows) {
        $rows | Select-Object Name, @{ n = 'Digest'; e = { Get-ShortDigest $_.Digest } }, Status, BlobsCopied,
        @{ n = 'Copied'; e = { Format-Size $_.BytesCopied } } | Format-Table -AutoSize | Out-Host
    }
}

function Invoke-Restore {
    param($o)
    $src = [IO.Path]::GetFullPath($o.Path, (Get-Location).ProviderPath)
    if (-not (Test-ModelsDirectory $src)) {
        Write-Fail "$src is not a backup folder (it needs 'manifests' and 'blobs' folders with at least one manifest)."
        $script:ExitCode = 4; return
    }
    $dir = Get-RequiredModelsPath -ExplicitPath $o.ModelsPath -AllowEmpty
    if (-not $dir) { $script:ExitCode = 4; return }
    $index = @(Get-ManifestIndex -ModelsDir $src)
    $sel = Select-Models -Items $index -Model $o.Model -Exclude $o.Exclude -NameProperty Name
    if ($null -eq $sel) { $script:ExitCode = 4; return }
    if ($o.AsTag) {
        if ($sel.Count -ne 1) {
            Write-Fail "-AsTag needs exactly one matching model; $($sel.Count) match."
            $script:ExitCode = 4; return
        }
        if ($o.AsTag -notmatch '^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$') {
            Write-Fail "Invalid tag '$($o.AsTag)'."
            $script:ExitCode = 4; return
        }
    }
    $manRoot = Join-Path $dir 'manifests'
    $rows = foreach ($e in $sel | Sort-Object Name) {
        $tag = $o.AsTag ? $o.AsTag : $e.Tag
        $targetName = ConvertTo-ModelName -Registry $e.Registry -Namespace $e.Namespace -Name $e.Model -Tag $tag
        $targetMan = Join-Path $manRoot (Join-Path (Split-Path -Parent $e.RelPath) $tag)
        $status = 'restored'
        if (Test-Path -LiteralPath $targetMan) {
            $existing = Get-Sha256Hex ([IO.File]::ReadAllBytes($targetMan))
            if ($existing -eq $e.Digest) {
                $status = 'up to date'
            }
            else {
                $q = "Replace $targetName ($(Get-ShortDigest $existing)) with backed-up version ($(Get-ShortDigest $e.Digest))?"
                if (-not (Confirm-Destructive -Target $targetName -Operation 'Replace with backed-up version' -Query $q)) {
                    [pscustomobject]@{ Name = $targetName; Digest = $e.Digest; Status = ($WhatIfPreference ? 'would replace' : 'skipped'); BlobsCopied = 0; BytesCopied = 0L }
                    continue
                }
                $status = 'replaced'
            }
        }
        elseif (-not $script:Cmdlet.ShouldProcess($targetName, "Restore from $src")) { continue }

        $r = Copy-ModelFiles -Entry $e -SourceDir $src -DestDir $dir -SkipExistingAnySize -CleanupOnError
        if ($r.Error) {
            Write-Warning "${targetName}: $($r.Error); not restored."
            [pscustomobject]@{ Name = $targetName; Digest = $e.Digest; Status = 'failed'; BlobsCopied = $r.BlobsCopied; BytesCopied = $r.BytesCopied }
            continue
        }
        if ($status -ne 'up to date') {
            $null = New-Item -ItemType Directory -Path (Split-Path -Parent $targetMan) -Force
            Copy-Item -LiteralPath $e.Path -Destination $targetMan -Force
        }
        [pscustomobject]@{ Name = $targetName; Digest = $e.Digest; Status = $status; BlobsCopied = $r.BlobsCopied; BytesCopied = $r.BytesCopied }
    }
    if ($rows | Where-Object Status -EQ 'failed') { $script:ExitCode = 1 }

    # The server reads manifests on each /api/tags request, so no restart is needed. Confirm when possible.
    $restored = @($rows | Where-Object { $_.Status -in 'restored', 'replaced' })
    $serverDir = (Resolve-ModelsPath -IsLocal $true).Path
    $sameDir = $serverDir -and ([IO.Path]::GetFullPath($serverDir).TrimEnd('\') -ieq [IO.Path]::GetFullPath($dir).TrimEnd('\'))
    if ($restored.Count -and $sameDir -and (Test-OllamaServer)) {
        $names = @(Get-OllamaModels | ForEach-Object name)
        $missing = @($restored | Where-Object { $_.Name -notin $names })
        if ($missing.Count) { Write-Warning ("Not yet visible to the server: {0}. Restart Ollama." -f (($missing.Name) -join ', ')) }
    }
    if ($o.PassThru) { $rows; return }
    if ($rows) {
        $rows | Select-Object Name, @{ n = 'Digest'; e = { Get-ShortDigest $_.Digest } }, Status, BlobsCopied,
        @{ n = 'Copied'; e = { Format-Size $_.BytesCopied } } | Format-Table -AutoSize | Out-Host
    }
}

function ConvertTo-SemVer {
    # 'v0.35.0-rc1' -> Core [version] 0.35.0, Pre 'rc1'. $null when unparsable.
    param([string]$Text)
    if ($Text -notmatch '(\d+)\.(\d+)(?:\.(\d+))?(?:-([0-9A-Za-z.-]+))?') { return $null }
    [pscustomobject]@{
        Core = [version]::new([int]$Matches[1], [int]$Matches[2], [int]($Matches[3] ?? 0))
        Pre  = $Matches[4]
        Text = $Matches[0]
    }
}

function Compare-SemVer {
    # -1 / 0 / 1. A pre-release is older than the matching release.
    param([string]$A, [string]$B)
    $x = ConvertTo-SemVer $A; $y = ConvertTo-SemVer $B
    $c = $x.Core.CompareTo($y.Core)
    if ($c -ne 0) { return [Math]::Sign($c) }
    if ($x.Pre -eq $y.Pre) { return 0 }
    if (-not $x.Pre) { return 1 }
    if (-not $y.Pre) { return -1 }
    [Math]::Sign([string]::CompareOrdinal($x.Pre, $y.Pre))
}

function Invoke-Doctor {
    param($o)
    $checks = [Collections.Generic.List[object]]::new()
    function Add-Check($check, $result, $detail) { $checks.Add([pscustomobject]@{ Check = $check; Result = $result; Detail = $detail }) }

    # Server
    $serverVersion = $null
    try {
        $serverVersion = (Invoke-OllamaApi -Endpoint '/api/version' -TimeoutSec 5).version
        Add-Check 'Server' 'OK' "reachable at $($script:Server.Url), version $serverVersion"
    }
    catch { Add-Check 'Server' 'FAIL' "not reachable at $($script:Server.Url)" }

    # CLI
    $cliVersion = $null
    $exe = Get-OllamaExe
    if (-not $exe) { Add-Check 'CLI' 'WARN' 'ollama is not on PATH (pulls use the API instead)' }
    else {
        try {
            $out = (Invoke-OllamaCli -Arguments '--version').Output
            $cliVersion = if ($out -match 'client version is\s+(\S+)') { $Matches[1] } elseif ($out -match 'version is\s+(\S+)') { $Matches[1] } else { $null }
            if (-not $cliVersion) { Add-Check 'CLI' 'WARN' "$exe; could not read version" }
            elseif ($serverVersion -and $cliVersion -ne $serverVersion) { Add-Check 'CLI' 'WARN' "$exe, version $cliVersion differs from server $serverVersion" }
            else { Add-Check 'CLI' 'OK' "$exe, version $cliVersion" }
        }
        catch { Add-Check 'CLI' 'WARN' "$exe; $_" }
    }

    # Latest release
    $installed = $serverVersion ?? $cliVersion
    try {
        $rel = Invoke-RestMethod -Uri 'https://api.github.com/repos/ollama/ollama/releases/latest' -TimeoutSec 10 -ErrorAction Stop -Verbose:$false
        $latest = $rel.tag_name -replace '^v', ''
        if (-not $installed) { Add-Check 'Latest release' 'OK' "latest is $latest" }
        elseif ((Compare-SemVer $installed $latest) -lt 0) { Add-Check 'Latest release' 'WARN' "update available: $installed -> $latest" }
        else { Add-Check 'Latest release' 'OK' "$installed is current (latest $latest)" }
    }
    catch { Add-Check 'Latest release' 'WARN' "could not check latest release ($($_.Exception.Message))" }

    # Configuration
    $envRows = [Collections.Generic.List[object]]::new()
    foreach ($scope in 'Process', 'User', 'Machine') {
        $vars = [Environment]::GetEnvironmentVariables($scope)
        foreach ($k in $vars.Keys | Where-Object { $_ -like 'OLLAMA_*' } | Sort-Object) {
            $envRows.Add([pscustomobject]@{ Scope = $scope; Name = $k; Value = $vars[$k] })
        }
    }
    $serverCfg = $null
    if ($script:Server.IsLocal) {
        $serverCfg = Get-ServerConfig
        if ($serverCfg) {
            $diff = @(foreach ($e in $envRows) {
                    if ($serverCfg.Settings.Contains($e.Name) -and $serverCfg.Settings[$e.Name] -ne $e.Value) { $e.Name }
                }) | Select-Object -Unique
            Add-Check 'Server config' ($diff ? 'WARN' : 'OK') ("from server.log (started $($serverCfg.Time))" + ($diff ? "; environment differs for $($diff -join ', ')" : ''))
        }
        else { Add-Check 'Server config' 'WARN' 'no "server config" line in server.log' }
    }
    else { Add-Check 'Server config' 'WARN' 'remote host: only the client-side environment (this machine) is shown, not the server''s' }

    # Models directory
    if ($script:Server.IsLocal) {
        $mp = Resolve-ModelsPath -IsLocal $true
        if ($mp.Path) {
            $free = Get-FreeSpace $mp.Path
            Add-Check 'Models directory' 'OK' ('{0} (from {1}), {2} free' -f $mp.Path, $mp.Source, (Format-Size $free))
        }
        else { Add-Check 'Models directory' 'FAIL' ('not found; tried ' + ($mp.Tried -join '; ')) }
    }
    else { Add-Check 'Models directory' 'WARN' 'remote host; not checked' }

    # Running models
    if ($serverVersion) {
        try {
            $loaded = @(Get-LoadedModels)
            if (-not $loaded.Count) { Add-Check 'Running models' 'OK' 'none loaded' }
            else {
                $parts = foreach ($m in $loaded) {
                    $gpu = if ($m.size) { [int](100 * $m.size_vram / $m.size) } else { 0 }
                    $split = if ($gpu -ge 100) { '100% GPU' } elseif ($gpu -le 0) { '100% CPU' } else { "$(100 - $gpu)%/$gpu% CPU/GPU" }
                    "$($m.name) ($(Format-Size ([long]$m.size)), $split)"
                }
                Add-Check 'Running models' 'OK' ($parts -join '; ')
            }
        }
        catch { Add-Check 'Running models' 'WARN' "$_" }
    }

    # Registry
    $reg = Invoke-RegistryRequest -Uri "https://$($script:DefaultRegistry)/v2/library/llama3.2/manifests/latest" -Method HEAD -TimeoutSec $script:RegistryTimeoutSec
    if ($reg.StatusCode -eq 200) { Add-Check 'Registry' 'OK' "$($script:DefaultRegistry) reachable" }
    else { Add-Check 'Registry' 'FAIL' ("$($script:DefaultRegistry): " + ($reg.Error ?? "HTTP $($reg.StatusCode)")) }

    if ($checks | Where-Object Result -EQ 'FAIL') { $script:ExitCode = 1 }
    if ($o.PassThru) { $checks; return }

    foreach ($c in $checks) {
        $color = switch ($c.Result) { 'OK' { 'Green' } 'WARN' { 'Yellow' } default { 'Red' } }
        Write-Host ('{0,-5}' -f $c.Result) -ForegroundColor $color -NoNewline
        Write-Host (' {0,-17} {1}' -f $c.Check, $c.Detail)
    }
    if ($serverCfg) {
        Write-Host "`nServer settings (non-empty OLLAMA_*, from $($serverCfg.LogPath)):"
        $serverCfg.Settings.GetEnumerator() | Where-Object { $_.Key -like 'OLLAMA_*' -and $_.Value -ne '' } |
            Select-Object @{ n = 'Name'; e = { $_.Key } }, Value | Format-Table -AutoSize -Wrap | Out-Host
    }
    Write-Host ($script:Server.IsLocal ? 'OLLAMA_* environment variables:' : "Client-side environment (this machine), not the server's:")
    if ($envRows.Count) {
        $envRows | Select-Object Scope, Name, Value, @{ n = 'Note'; e = {
                if ($serverCfg -and $serverCfg.Settings.Contains($_.Name) -and $serverCfg.Settings[$_.Name] -ne $_.Value) { "server uses '$($serverCfg.Settings[$_.Name])'" }
            }
        } | Format-Table -AutoSize -Wrap | Out-Host
    }
    else { Write-Host '  (none set)' }
}

function Invoke-Bench {
    param($o)
    $models = @(Get-OllamaModels)
    $sel = Select-Models -Items $models -Model $o.Model -Exclude $o.Exclude
    if ($null -eq $sel) { $script:ExitCode = 4; return }
    $targets = foreach ($m in $sel) {
        $caps = @($m.capabilities)
        if ($m.IsCloud) { Write-Host "$($m.name): cloud model, skipped."; continue }
        if ($caps -contains 'embedding' -and $caps -notcontains 'completion') { Write-Host "$($m.name): embedding model, skipped."; continue }
        $m
    }
    $targets = @($targets)
    if (-not $targets.Count) { Write-Host 'No models to benchmark.'; return }
    $prompt = $o.Prompt ? $o.Prompt : $script:DefaultBenchPrompt
    $failed = $false

    $rows = foreach ($m in $targets) {
        if (-not $script:Cmdlet.ShouldProcess($m.name, 'Benchmark')) { continue }
        $loadedBefore = @(Get-LoadedModels)
        $wasLoaded = [bool]($loadedBefore | Where-Object name -EQ $m.name)
        $others = @($loadedBefore | Where-Object name -NE $m.name)
        if ($others.Count) { Write-Warning ("Other loaded models share VRAM and can distort results: {0}" -f (($others.name) -join ', ')) }
        $body = @{
            model   = $m.name
            prompt  = $prompt
            stream  = $false
            options = @{ num_predict = 256; temperature = 0; seed = 42 }
        }
        if (@($m.capabilities) -contains 'thinking') { $body.think = $false }
        Write-Host "Benchmarking $($m.name)..." -ForegroundColor Cyan
        try {
            $warm = Invoke-OllamaApi -Endpoint '/api/generate' -Method POST -Body $body -TimeoutSec $script:GenerateTimeoutSec
            $load = if ($wasLoaded) { $null } else { [double]$warm.load_duration / 1e9 }
            $pp = [Collections.Generic.List[double]]::new(); $gen = [Collections.Generic.List[double]]::new()
            for ($i = 1; $i -le $o.Runs; $i++) {
                Write-Progress -Activity "Benchmarking $($m.name)" -Status "run $i of $($o.Runs)" -PercentComplete (100 * ($i - 1) / $o.Runs)
                $r = Invoke-OllamaApi -Endpoint '/api/generate' -Method POST -Body $body -TimeoutSec $script:GenerateTimeoutSec
                if ($r.prompt_eval_duration -gt 0) { $pp.Add($r.prompt_eval_count / ($r.prompt_eval_duration / 1e9)) }
                if ($r.eval_duration -gt 0) { $gen.Add($r.eval_count / ($r.eval_duration / 1e9)) }
            }
            Write-Progress -Activity "Benchmarking $($m.name)" -Completed
            [pscustomobject]@{
                Name               = $m.name
                LoadSeconds        = if ($null -ne $load) { [Math]::Round($load, 2) } else { $null }
                PromptTokensPerSec = if ($pp.Count) { [Math]::Round(($pp | Measure-Object -Average).Average, 1) } else { $null }
                GenTokensPerSec    = if ($gen.Count) { [Math]::Round(($gen | Measure-Object -Average).Average, 1) } else { $null }
                Runs               = $o.Runs
            }
        }
        catch {
            $failed = $true
            Write-Warning "$($m.name): $_"
        }
        finally {
            if (-not $o.KeepLoaded -and -not $wasLoaded) {
                try { Stop-OllamaModel -Name $m.name } catch { Write-Warning "Could not unload $($m.name): $_" }
            }
        }
    }
    if ($failed) { $script:ExitCode = 1 }
    if ($o.PassThru) { $rows; return }
    if ($rows) {
        $rows | Select-Object Name,
        @{ n = 'Load'; e = { $null -eq $_.LoadSeconds ? '(already loaded)' : ('{0:N1} s' -f $_.LoadSeconds) } },
        @{ n = 'Prompt tok/s'; e = { '{0:N1}' -f $_.PromptTokensPerSec } },
        @{ n = 'Gen tok/s'; e = { '{0:N1}' -f $_.GenTokensPerSec } }, Runs | Format-Table -AutoSize | Out-Host
    }
}

#endregion

#region Main

function Show-Usage {
    $name = Split-Path -Leaf $PSCommandPath
    @"
Manage a local Ollama install: models, running instances, disk usage and backups.

USAGE
    $name <action> [-Model <patterns>] [-Exclude <patterns>] [-OllamaHost <url>] [options]

ACTIONS
    update       Pull only models that changed on the registry (skips cloud models)
    outdated     List models with a newer version on the registry  [-All]
    list         Rich model inventory  [-Sort Name|Size|Modified] [-Capability ...] [-Deep]
    stop         Unload all running models (frees VRAM)
    usage        Real disk usage, shared blobs, orphans  [-ModelsPath]
    prune        Delete models or orphaned blobs  [-OlderThanDays n] [-Duplicates] [-Orphans]
    export-list  Write the model list to JSON  -Path <file>
    import-list  Pull models missing from an exported list  -Path <file>
    backup       Copy models to a folder  -Model <patterns> -Path <dir>
    restore      Copy backed-up models back  -Path <dir> [-AsTag <tag>]
    doctor       Health report
    bench        Tokens/s benchmark  [-Prompt <text>] [-Runs n] [-KeepLoaded]

COMMON
    -WhatIf / -Confirm   preview or confirm changes      -PassThru   emit objects
    -Verbose             log HTTP requests and CLI calls
    -Help, -h, --help    show this summary

EXAMPLES
    $name outdated
    $name update -Model qwen* -Exclude *:2b
    $name prune -OlderThanDays 120 -WhatIf

Full help: Get-Help .\$name -Full
"@ | Write-Host
}

# Dot-sourcing (tests) only loads the functions.
if ($MyInvocation.InvocationName -eq '.') { return }

$helpTokens = @('help', '-help', '--help', '-h', '--h', '/?', '/h', '/help', '-?')
if (-not $Action -or $Help -or $Action -in $helpTokens -or ($Rest | Where-Object { $_ -in $helpTokens })) {
    Show-Usage
    exit 0
}
if ($Rest) {
    Write-Fail "Unknown argument(s): $($Rest -join ' '). Run with -Help for usage."
    exit 4
}
if ($Action -notin $script:ActionParameters.Keys) {
    Write-Fail "Unknown action '$Action'. Valid actions: $($script:ActionParameters.Keys -join ', ')."
    exit 4
}

$paramError = Test-ActionParameters -Action $Action -BoundParameters $PSBoundParameters
if ($paramError) {
    Write-Fail $paramError
    exit 4
}

try {
    $script:Server = Resolve-OllamaHost ($OllamaHost ? $OllamaHost : $env:OLLAMA_HOST)
}
catch {
    Write-Fail $_.Exception.Message
    exit 4
}
Write-Verbose "Ollama host: $($script:Server.Url) ($($script:Server.IsLocal ? 'local' : 'remote'))"

$script:Cmdlet = $PSCmdlet
$script:ExitCode = 0
$script:YesToAll = $false
$script:ConfirmMode = if (-not $PSBoundParameters.ContainsKey('Confirm')) { 'default' } elseif ($PSBoundParameters['Confirm']) { 'on' } else { 'off' }

$needsServer = $Action -notin $script:OfflineActions -and -not ($Action -eq 'prune' -and $Orphans -and -not ($OlderThanDays -or $Duplicates -or $Model))
if ($needsServer -and -not (Test-OllamaServer)) {
    Write-Fail "Ollama server not reachable at $($script:Server.Url) — is it running?"
    exit 2
}

$options = [pscustomobject]@{
    Model         = $Model
    Exclude       = $Exclude
    ModelsPath    = $ModelsPath
    Path          = $Path
    All           = $All.IsPresent
    Deep          = $Deep.IsPresent
    Sort          = $Sort
    Capability    = $Capability
    OlderThanDays = $OlderThanDays
    Duplicates    = $Duplicates.IsPresent
    Orphans       = $Orphans.IsPresent
    AsTag         = $AsTag
    Prompt        = $Prompt
    Runs          = $Runs
    KeepLoaded    = $KeepLoaded.IsPresent
    PassThru      = $PassThru.IsPresent
}

try {
    switch ($Action) {
        'update' { Invoke-Update $options }
        'outdated' { Invoke-Outdated $options }
        'list' { Invoke-List $options }
        'stop' { Invoke-Stop $options }
        'usage' { Invoke-Usage $options }
        'prune' { Invoke-Prune $options }
        'export-list' { Invoke-ExportList $options }
        'import-list' { Invoke-ImportList $options }
        'backup' { Invoke-Backup $options }
        'restore' { Invoke-Restore $options }
        'doctor' { Invoke-Doctor $options }
        'bench' { Invoke-Bench $options }
    }
}
catch {
    Write-Fail "Error: $($_.Exception.Message)"
    if ($_.InvocationInfo) { Write-Verbose $_.InvocationInfo.PositionMessage }
    exit (Get-ExitCode @($script:ExitCode, 1))
}
exit $script:ExitCode

#endregion
