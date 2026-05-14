param(
    [string]$BaseUrl = "http://127.0.0.1:8011",
    [string]$Model = "perkunas-v2.9",
    [int]$MaxTokens = 120,
    [string]$OutDir = "reports",
    [switch]$SkipChat,
    [switch]$SkipCompletions
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepeatStats {
    param([string]$Text)

    $words = [regex]::Matches($Text.ToLowerInvariant(), "[a-z']+") | ForEach-Object { $_.Value }
    $tokenCount = @($words).Count
    if ($tokenCount -eq 0) {
        return [ordered]@{
            tokens = 0
            unique_tokens = 0
            unique_ratio = 0
            repeated_bigrams = 0
            repeated_trigrams = 0
        }
    }

    $unique = @($words | Sort-Object -Unique).Count
    $bigrams = @()
    $trigrams = @()

    for ($i = 0; $i -lt $tokenCount - 1; $i++) {
        $bigrams += "$($words[$i]) $($words[$i + 1])"
    }
    for ($i = 0; $i -lt $tokenCount - 2; $i++) {
        $trigrams += "$($words[$i]) $($words[$i + 1]) $($words[$i + 2])"
    }

    $repeatedBigrams = @($bigrams | Group-Object | Where-Object { $_.Count -gt 1 }).Count
    $repeatedTrigrams = @($trigrams | Group-Object | Where-Object { $_.Count -gt 1 }).Count

    [ordered]@{
        tokens = $tokenCount
        unique_tokens = $unique
        unique_ratio = [math]::Round($unique / [math]::Max($tokenCount, 1), 3)
        repeated_bigrams = $repeatedBigrams
        repeated_trigrams = $repeatedTrigrams
    }
}

function Invoke-JsonPost {
    param(
        [string]$Url,
        [hashtable]$Body
    )

    $json = $Body | ConvertTo-Json -Depth 12
    Invoke-RestMethod $Url -Method Post -ContentType "application/json" -Body $json
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$jsonlPath = Join-Path $OutDir "perkunas_vllm_prompt_tests_$stamp.jsonl"
$mdPath = Join-Path $OutDir "perkunas_vllm_prompt_tests_$stamp.md"

$completionTests = @(
    @{
        id = "c01_bila_ball"
        prompt = "Once upon a time, there was a little dog named Bila. Bila lost her red ball. She looked under"
        temperature = 0.75
        top_p = 0.90
        seed = 11
    },
    @{
        id = "c02_name_lock"
        prompt = "A puppy named Bayla lived with a boy named Tom. Bayla wanted to find"
        temperature = 0.80
        top_p = 0.95
        seed = 12
    },
    @{
        id = "c03_object_continuity"
        prompt = "Mira had a yellow kite. The wind took the kite into a tree. Mira felt"
        temperature = 0.70
        top_p = 0.90
        seed = 13
    },
    @{
        id = "c04_dialogue"
        prompt = "Lina saw a small frog by the pond. The frog said,"
        temperature = 0.85
        top_p = 0.95
        seed = 14
    },
    @{
        id = "c05_moral"
        prompt = "Ben had one cookie and two friends. He wanted the cookie, but then he"
        temperature = 0.75
        top_p = 0.90
        seed = 15
    },
    @{
        id = "c06_low_temp_repeat_probe"
        prompt = "Once upon a time, Lily went to the park. She saw"
        temperature = 0.35
        top_p = 0.90
        seed = 16
    },
    @{
        id = "c07_high_temp_coherence_probe"
        prompt = "Nono was a tiny robot who was afraid of rain. His friend Pip opened"
        temperature = 0.95
        top_p = 0.95
        seed = 17
    }
)

$chatTests = @(
    @{
        id = "ch01_bila_ball"
        system = "Write a simple TinyStories-style story for young children. Use short sentences. Keep the character names from the user."
        user = "Write a story about a dog named Bila who lost her red ball."
        temperature = 0.80
        top_p = 0.95
        seed = 21
    },
    @{
        id = "ch02_no_lily_probe"
        system = "Write a simple TinyStories-style story. Do not introduce Lily unless the user asks for Lily."
        user = "Write about a puppy named Bayla and a boy named Tom."
        temperature = 0.80
        top_p = 0.95
        seed = 22
    },
    @{
        id = "ch03_short_follow_instruction"
        system = "Write for very young children. Use five short sentences."
        user = "Tell me about a duck named Didi who finds a blue hat."
        temperature = 0.70
        top_p = 0.90
        seed = 23
    }
)

$results = @()

if (-not $SkipCompletions) {
    foreach ($test in $completionTests) {
        $body = @{
            model = $Model
            prompt = $test.prompt
            max_tokens = $MaxTokens
            temperature = $test.temperature
            top_p = $test.top_p
            seed = $test.seed
        }

        try {
            $response = Invoke-JsonPost "$BaseUrl/v1/completions" $body
            $text = [string]$response.choices[0].text
            $result = [ordered]@{
                id = $test.id
                endpoint = "completions"
                prompt = $test.prompt
                temperature = $test.temperature
                top_p = $test.top_p
                seed = $test.seed
                text = $text
                stats = Get-RepeatStats $text
                usage = $response.usage
            }
        }
        catch {
            $result = [ordered]@{
                id = $test.id
                endpoint = "completions"
                prompt = $test.prompt
                error = $_.Exception.Message
            }
        }
        $results += [pscustomobject]$result
        ($result | ConvertTo-Json -Depth 20 -Compress) | Add-Content -Path $jsonlPath -Encoding UTF8
    }
}

if (-not $SkipChat) {
    foreach ($test in $chatTests) {
        $body = @{
            model = $Model
            messages = @(
                @{ role = "system"; content = $test.system },
                @{ role = "user"; content = $test.user }
            )
            max_tokens = $MaxTokens
            temperature = $test.temperature
            top_p = $test.top_p
            seed = $test.seed
        }

        try {
            $response = Invoke-JsonPost "$BaseUrl/v1/chat/completions" $body
            $text = [string]$response.choices[0].message.content
            $result = [ordered]@{
                id = $test.id
                endpoint = "chat"
                system = $test.system
                user = $test.user
                temperature = $test.temperature
                top_p = $test.top_p
                seed = $test.seed
                text = $text
                stats = Get-RepeatStats $text
                usage = $response.usage
            }
        }
        catch {
            $message = if ($_.ErrorDetails.Message) { $_.ErrorDetails.Message } else { $_.Exception.Message }
            $result = [ordered]@{
                id = $test.id
                endpoint = "chat"
                system = $test.system
                user = $test.user
                error = $message
            }
        }
        $results += [pscustomobject]$result
        ($result | ConvertTo-Json -Depth 20 -Compress) | Add-Content -Path $jsonlPath -Encoding UTF8
    }
}

$lines = @()
$lines += "# Perkunas vLLM Prompt Tests"
$lines += ""
$lines += "- Base URL: ``$BaseUrl``"
$lines += "- Model: ``$Model``"
$lines += "- Max tokens: ``$MaxTokens``"
$lines += "- JSONL: ``$jsonlPath``"
$lines += ""

foreach ($result in $results) {
    $lines += "## $($result.id) [$($result.endpoint)]"
    if ($result.PSObject.Properties.Name -contains "error") {
        $lines += ""
        $lines += "**Error:**"
        $lines += ""
        $lines += '```text'
        $lines += $result.error
        $lines += '```'
        $lines += ""
        continue
    }
    $lines += ""
    if ($result.endpoint -eq "chat") {
        $lines += "**System:** $($result.system)"
        $lines += ""
        $lines += "**User:** $($result.user)"
    } else {
        $lines += "**Prompt:** $($result.prompt)"
    }
    $lines += ""
    $lines += "**Settings:** temp=$($result.temperature), top_p=$($result.top_p), seed=$($result.seed)"
    $lines += ""
    $lines += "**Stats:** tokens=$($result.stats.tokens), unique_ratio=$($result.stats.unique_ratio), repeated_bigrams=$($result.stats.repeated_bigrams), repeated_trigrams=$($result.stats.repeated_trigrams)"
    $lines += ""
    $lines += '```text'
    $lines += $result.text.Trim()
    $lines += '```'
    $lines += ""
}

$lines | Set-Content -Path $mdPath -Encoding UTF8

Write-Host "Wrote $jsonlPath"
Write-Host "Wrote $mdPath"
$results | Select-Object id, endpoint, @{Name="tokens";Expression={$_.stats.tokens}}, @{Name="unique_ratio";Expression={$_.stats.unique_ratio}}, @{Name="repeated_trigrams";Expression={$_.stats.repeated_trigrams}}, @{Name="error";Expression={$_.error}} | Format-Table -AutoSize
