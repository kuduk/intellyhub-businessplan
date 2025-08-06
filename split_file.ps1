$content = Get-Content "businessplan.tex"
$lineCount = 80
$fileNum = 1

for ($i = 0; $i -lt $content.Length; $i += $lineCount) {
    $end = [Math]::Min($i + $lineCount - 1, $content.Length - 1)
    $fileName = "ch/part_{0:D2}.tex" -f $fileNum
    $content[$i..$end] | Out-File -FilePath $fileName -Encoding UTF8
    Write-Host "Created $fileName with lines $($i+1) to $($end+1)"
    $fileNum++
}

Write-Host "Split complete. Created $($fileNum-1) files."
