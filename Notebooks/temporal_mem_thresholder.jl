using BenchmarkTools
using Random
using SparseArrays
using StatsBase
using Test



struct TMThreshold{T}
    segments::Array{SparseMatrixCSC{T, Int64}, 1}
    θ::Array{T, 2}
    cells_per_col::Int
    num_cells::Int
    small_decr::T
    small_incr::T
    very_small_incr::T
end

function TMThreshold(
    num_cells, 
    cells_per_col, 
    segments_per_cell, 
    synapses_per_segment; 
    initial_threshold=4., 
    small_decr=0.1, 
    small_incr=0.1, 
    very_small_incr=0.01
)
    p = synapses_per_segment / num_cells
    segments = [sprand(num_cells, num_cells, p) for i in 1:segments_per_cell]
    θ = initial_threshold .* ones(num_cells, segments_per_cell)
    return TMThreshold(segments, θ, cells_per_col, num_cells, small_decr, small_incr, very_small_incr)
end

"""Specifies length of 1 for itereation api"""
Base.length(tmt::TMThreshold) = 1
Base.iterate(tmt::TMThreshold) = (tmt, nothing)
Base.iterate(tmt::TMThreshold, x) = nothing
Base.show(io::IO, tmt::TMThreshold) = print(io, "TMThreshold")

project(tmt::TMThreshold, active_col_idx::Array{Int, 1}) = project(tmt, active_cols_to_sparray(tmt, active_col_idx))
project(tmt::TMThreshold, x::SparseVector) = map(A -> Array(A * x), tmt.segments)

function predict(tmt::TMThreshold, active_col_idx::Array)
    segment_activations = hcat(project(tmt, active_col_idx)...)
    segs_above_thresh = segment_activations .> tmt.θ
    active_segs_per_cell = sum(segs_above_thresh, dims=2)
    return active_segs_per_cell .> 1
end

function update!(tmt::TMThreshold, segment_activations, active_col_idxs)
    
    # Make active column array
    x = active_cols_to_sparray(tmt, active_col_idxs)
    # Make unpredicted set for false negative updates
    unpredicted = Set(active_col_idxs)
    
    # Iterate over synapse activations
    # Update true positives and false positives
    map(enumerate(segment_activations)) do (j, sj)
        θj = @view tmt.θ[:, j]
        aj = SparseVector(sj .> θj)
        update_true_positive!(j, aj, x, tmt, active_col_idxs, unpredicted)
        update_false_pos!(j, aj, x, tmt)
    end
    
    # False negative update
    decrease_threshold!.(tmt, find_max_segments(tmt, segment_activations, unpredicted))
    very_small_increase_threshold!(tmt)
    return nothing
end
        

"""
    column_idx_to_cell_idxs(tmt::TMThreshold, i::Int) -> UnitRange

Each "column" contains `tmt.cells_per_col` cells. These are stored
contiguously inside `tmt`. So each column index maps to the range
of cells that belong to that column.

# Examples

# If there are 3 columns, 10 cells per column

    column_idx_to_cell_idxs(tmt, 2) yields 11:20

    column_idx_to_cell_idxs(tmt, 3) yields 21:30
"""
function column_idx_to_cell_idxs(tmt::TMThreshold, i::Int)
        idxs = (1 + (i - 1) * tmt.cells_per_col):(i * tmt.cells_per_col)
    return idxs
end


"""
Sends active column indexes to a sparse array with `tmt.cells_per_col` active cells 
per active column.
"""
function active_cols_to_sparray(tmt::TMThreshold, active_col_idxs, T::Type=Bool)
    # Collect lists of all cell indexes from each active column
    active_cell_ranges = map(i -> collect(column_idx_to_cell_idxs(tmt, i)), active_col_idxs)
    # Makes ranges into an array of indexes
    active_cell_idxs = vcat(active_cell_ranges...)
    return SparseVector(tmt.num_cells, active_cell_idxs, ones(T, length(active_cell_idxs)))
end


"""
    update_true_pos!(
        j::Int,
        aj::SparseVector,
        x::SparseVector,
        tmt::TMThreshold,
        active_cols,
        unpredicted::Set
    )

Lowers the activation threshold for cells that predicted the input correctly.

# Arguments
- `j` Column index of the threshold matrix stored in `tmt.θ`.
- `aj::SparseVector` The binary vector representing which segments were above 
    the threshold in a given column.
- `x::SparseVector` The external input. All cells active in neuron "columns"
    corresponding to the encoding.
- `tmt::TMThreshold` The temporal memory threshold struct containing parameters.
- `active_cols` An array of integers containing active column indexes
- `unpredicted::Set` Tracks which active columns that have not been predicted by the
    at this point of the update loop
"""
function update_true_positive!(
    j,
    aj::SparseVector,
    x::SparseVector,
    tmt::TMThreshold,
    active_cols,
    unpredicted::Set
)
    for i in first(findnz(aj .&& x))
        decrease_threshold!(tmt, (i, j))
        # Find cell column index
        col_idx = (i ÷ tmt.cells_per_col) + 1
        # Check if this column has been predicted
        # Remove it if it has not
        (col_idx in unpredicted) && delete!(unpredicted, col_idx)
    end
end   


"""
    update_false_pos!(
        j,
        aj::SparseVector,
        x::SparseVector,
        tmt::TMThreshold,

    )

Increases the activation threshold for cells in a predictive state
without a corresponding active column.

# Arguments
- `j` Column index of the threshold matrix stored in `tmt.θ`.
- `aj::SparseVector` The binary vector representing which segments were above 
    the threshold in a given column.
- `x::SparseVector` The external input. All cells active in neuron "columns"
    corresponding to the encoding.
- `tmt::TMThreshold` The temporal memory threshold struct containing parameters.
"""
function update_false_pos!(
    j, 
    aj::SparseVector,
    x::SparseVector,
    tmt::TMThreshold,
)
    error = aj - x
    for (i, v) in zip(findnz(error)...)
        v == 1 && increase_threshold!(tmt, (i, j))
    end
end


"""
    find_max_segments(tmt::TMThreshold, segment_activations, unpredicted::Set)

# Arguments

- `tmt::TMThreshold` The temporal memory threshold struct containing parameters.
- `segment_activations` An array of arrays containing segment activations. Output of
    [`project(tmt, x)`](@ref)
- `unpredicted::Set` Tracks which active columns that have not been predicted by the
    at this point of the update loop
"""
function find_max_segments(tmt::TMThreshold, segment_activations, unpredicted::Set)
    # Translate encoding columns to cell indexes
    unpredicted_col_cells = column_idx_to_cell_idxs.(tmt::TMThreshold, unpredicted)
    # Prep for finding the biggest cell activation in the column
    n = length(unpredicted_col_cells)
    argmaxes = [(0, 0) for i in 1:n]
    maxes = [-Inf for i in 1:n]
    # Find the biggest segment activation in each unpredicted column
    for (j, a) in enumerate(segment_activations)
        for (k, range) in enumerate(unpredicted_col_cells)
            i = range[argmax(a[range])]
            if a[i] > maxes[k]
                maxes[k] = a[i]
                argmaxes[k] = (i, j)
            end
        end
    end
    
    return argmaxes
end


"""
    decrease_threshold!(tmt::TMThreshold, idxs::Tuple{Int, Int})

Decreases threshold at the given index.

# Arguments
- `tmt::TMThreshold` The temporal memory threshold struct containing parameters.

"""
function decrease_threshold!(tmt::TMThreshold, idx::Tuple{Int, Int})
    tmt.θ[idx...] -= tmt.small_decr
end
    
"""
    increase_threshold!(tmt::TMThreshold, idxs::Tuple{Int, Int})

Increases threshold at the given index.

# Arguments
- `tmt::TMThreshold` The temporal memory threshold struct containing parameters.

"""
function increase_threshold!(tmt::TMThreshold, idx::Tuple{Int, Int})
    tmt.θ[idx...] += tmt.small_incr
end

"""
    small_increase_threshold!(tmt::TMThreshold, idxs::Tuple{Int, Int})

Applied a very small threshold increase.

# Arguments
- `tmt::TMThreshold` The temporal memory threshold struct containing parameters.

"""
function very_small_increase_threshold!(tmt::TMThreshold)
    tmt.θ .+= tmt.very_small_incr
end

function random_active_cell_idxs(num_cells, num_active_in_encoding)
    return randperm(num_cells)[1:num_active_in_encoding]
end

function random_input_array(num_cells, num_active_in_encoding)
    return SparseVector(num_cells, 
                        random_active_cell_idxs(num_cells, num_active_in_encoding),
                        ones(num_active_in_encoding))
end
########### TESTS ##############

function run_tests()
    @testset "TMThreshold" begin
        # Parameters

        # Cells and cols
        num_cols = 64
        cells_per_col = 8
        num_cells = num_cols * cells_per_col

        # Segments and synapses
        segments_per_cell = 8
        synapses_per_segment = 5
        segment_sparsity = synapses_per_segment / num_cells;

        # Initialize
        tmt = TMThreshold(
            num_cells, 
            cells_per_col, 
            segments_per_cell, 
            synapses_per_segment
        )

        # Test array sizes
        @test all(first.(size.(tmt.segments)) .== num_cells)
        @test all(last.(size.(tmt.segments)) .== num_cells)
        @test all(size(tmt.θ) .== (num_cells, segments_per_cell))

        # Test sparsity
        synapse_counts = mean(map(x -> length(first(findnz(x))), tmt.segments))
        @test  abs(synapse_counts/num_cells^2  - segment_sparsity) < 0.001

        # Test SDR expander function
        @test column_idx_to_cell_idxs(tmt, 2) == 9:16
        @test column_idx_to_cell_idxs(tmt, 7) == 49:56

        test_sm_incr = 5.
        test_sm_decr = 10.
        test_very_sm_incr = 1.

        # Test the update
        tmt = TMThreshold(
            num_cells, 
            cells_per_col, 
            segments_per_cell, 
            synapses_per_segment;
            initial_threshold = 0.0,
            small_incr = test_sm_incr,
            small_decr = test_sm_decr,
            very_small_incr = test_very_sm_incr
        )

        # Set all segments below the threshold
        segment_activation = [rand(num_cells) .- 2 for i in 1:segments_per_cell]
        # Set one segment above thresh (a true positive)
        segment_activation[1][1] = 1 
        # Make a maximum false negative
        segment_activation[1][cells_per_col + 1] = -1.
        # Make a false positive
        segment_activation[1][end] = 1

        active_cols = [1, 2]

        # Test finding segment maximums"""

        @test all(find_max_segments(tmt, segment_activation, Set(2)) == [(cells_per_col + 1, 1)])

        # Test that the true positive update removes from the unpredicted set

        unpredicted = Set(active_cols)
        x = active_cols_to_sparray(tmt, active_cols)
        on = SparseVector(segment_activation[1] .> tmt.θ[:, 1])
        update_true_positive!(1, on, x, tmt, active_cols, unpredicted)
        @test length(unpredicted) == 1
        @test 2 in unpredicted

        # Refresh TMThreshold
        tmt = TMThreshold(
            num_cells, 
            cells_per_col, 
            segments_per_cell, 
            synapses_per_segment;
            initial_threshold = 0.0,
            small_incr = test_sm_incr,
            small_decr = test_sm_decr,
            very_small_incr = test_very_sm_incr
        )

        @testset "update!" begin
            update!(tmt, segment_activation, active_cols)

            seg_per_col = cells_per_col * segments_per_cell
            # Test that there was only one true-positive-update
            @test tmt.θ[1, 1] == -test_sm_decr + test_very_sm_incr
            @test sum(tmt.θ[column_idx_to_cell_idxs(tmt, 1), :]) == test_very_sm_incr * seg_per_col - test_sm_decr

            # Check that only one cell was decreased in the false negative zone
            @test sum(tmt.θ[column_idx_to_cell_idxs(tmt, 2), :]) == seg_per_col - test_sm_decr
            @test tmt.θ[cells_per_col + 1, 1] == -9

            # Check that there was only one-false-positive-update
            @test sum(tmt.θ[column_idx_to_cell_idxs(tmt, 2), :]) == test_very_sm_incr * seg_per_col - test_sm_decr
            @test tmt.θ[end, 1] == test_very_sm_incr + test_sm_incr

            # Test that the sum of all thresholds is correct
            @test sum(tmt.θ) == test_very_sm_incr * seg_per_col * num_cols - 2*test_sm_decr + test_sm_incr
        end
    end
end

#### Benchmark ####

function benchmark_tmt()
    num_cols = 1024
    cells_per_col = 32
    num_cells = num_cols * cells_per_col

    # Segments and synapses
    segments_per_cell = 64
    synapses_per_segment = 50
    segment_sparsity = synapses_per_segment / num_cells;

    # Initialize
    tmt = TMThreshold(
        num_cells, 
        cells_per_col, 
        segments_per_cell, 
        synapses_per_segment
    )

    enc_len = 32
    active_cols = randperm(num_cols)[1:enc_len]
    x = active_cols_to_sparray(tmt, active_cols)
    segment_activations = project(tmt, x)

    td = @timed update!(tmt, segment_activations, active_cols)
    println(td)

    tmt = nothing
    segment_activations = nothing
end

run_tests()
benchmark_tmt()