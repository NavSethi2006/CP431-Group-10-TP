#!/bin/bash
#SBATCH --job-name=julia_bench
#SBATCH --output=julia_output_%j.txt
#SBATCH --error=julia_error_%j.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:45:00
#SBATCH --mem-per-cpu=1G

# CP431 Term Project
# Group 10
# Benchmark job script

echo "CP431 Term Project"
echo "MPI Julia Set Benchmark"
echo "Group 10"
echo ""
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Total tasks available: $SLURM_NTASKS"
echo ""

echo "Loading modules..."
module load StdEnv/2020
module load openmpi
echo ""
module list
echo ""

echo "Compiling test.c..."
mpicc -Wall -O2 -lm -o julia_mpi test.c

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

process_counts=(1 20 40 80)
iteration_count=255
julia_powers=(2)
constants=("0.35491 0.355")
width=100000
height=100000
chunk_sizes=(1024 2048 4096 8192)

echo "Benchmark plan:"
echo "  process counts: ${process_counts[*]}"
echo "  iteration count: $iteration_count"
echo "  julia powers: ${julia_powers[*]}"
echo "  constants: ${constants[*]}"
echo "  width=$width height=$height"
echo "  chunk sizes: ${chunk_sizes[*]}"
echo ""

for np in "${process_counts[@]}"
do
    for power in "${julia_powers[@]}"
    do
        for constant in "${constants[@]}"
        do
            for chunk_size in "${chunk_sizes[@]}"
            do
                read -r c_re c_im <<< "$constant"
                safe_cre=$(echo "$c_re" | sed 's/-/m/g; s/\./p/g')
                safe_cim=$(echo "$c_im" | sed 's/-/m/g; s/\./p/g')
                outdir="run_np${np}_it${iteration_count}_p${power}_cre${safe_cre}_cim${safe_cim}_chunk${chunk_size}"

                echo "============================================================"
                echo "Running benchmark: np=$np iterations=$iteration_count power=$power c=($c_re,$c_im) chunk_size=$chunk_size"
                echo "Output directory: $outdir"
                echo "Started at: $(date)"

                rm -rf "$outdir"
                srun --ntasks=$np ./julia_mpi "$outdir" "$iteration_count" "$power" "$c_re" "$c_im" "$width" "$height" "$chunk_size"

                if [ $? -ne 0 ]; then
                    echo "Run FAILED for $outdir"
                else
                    echo "Run completed for $outdir"
                    if [ -f "$outdir/benchmark.txt" ]; then
                        echo "Benchmark summary for $outdir:"
                        grep -E 'MPI_PROCESSES|WIDTH|HEIGHT|CHUNK_SIZE|MAX_ITERATIONS|JULIA_POWER|C_RE|C_IM|WALL_SECONDS|TOTAL_COMPUTE_SECONDS|TOTAL_IO_SECONDS|TOTAL_SETUP_SECONDS|ESTIMATED_IDLE_SECONDS|TOTAL_CHUNKS|TOTAL_BYTES_WRITTEN' "$outdir/benchmark.txt"
                    fi
                fi

                echo "Finished at: $(date)"
                echo ""
            done
        done
    done
done

echo "All benchmark runs completed!"
echo "Job finished at: $(date)"
