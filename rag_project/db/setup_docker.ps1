# PowerShell script to deploy and healthcheck the pgvector PostgreSQL container

$ContainerName = "rag_postgres"
$Image = "pgvector/pgvector:pg16"
$Port = 15432
$DbUser = "postgres"
$DbPass = "postgres"
$DbName = "rag_db"
$VolumeName = "4d43465c81b762e871c53e98cbfaffbdb9d7193b3a3d04c19207308de9505e81"

Write-Host "=== Setting up PostgreSQL with pgvector container ($ContainerName) on port $Port ==="

# Check if container exists
$existing = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}"

if ($existing -eq $ContainerName) {
    Write-Host "Found existing container $ContainerName. Checking state..."
    $running = docker ps --filter "name=$ContainerName" --format "{{.Names}}"
    if ($running -ne $ContainerName) {
        Write-Host "Starting container $ContainerName..."
        docker start $ContainerName
    } else {
        Write-Host "Container $ContainerName is already running."
    }
} else {
    Write-Host "Running new container $ContainerName on port $Port..."
    docker run -d `
        --name $ContainerName `
        -p ${Port}:5432 `
        -e POSTGRES_USER=$DbUser `
        -e POSTGRES_PASSWORD=$DbPass `
        -e POSTGRES_DB=$DbName `
        -v ${VolumeName}:/var/lib/postgresql/data `
        --restart unless-stopped `
        $Image
}


Write-Host "Waiting for PostgreSQL to be ready..."
$retries = 30
while ($retries -gt 0) {
    $ready = docker exec $ContainerName pg_isready -U $DbUser -d $DbName 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PostgreSQL is ready and accepting connections!"
        break
    }
    Start-Sleep -Seconds 1
    $retries--
}

if ($retries -eq 0) {
    Write-Error "PostgreSQL failed to become ready within timeout."
    exit 1
}

Write-Host "PostgreSQL setup complete on localhost:$Port."
