# Julia script that implements the Nupic HTM algorithm
# as outlined here: 
# https://numenta.com/assets/pdf/temporal-memory-algorithm/Temporal-Memory-Algorithm-Details.pdf
# 
# Author: DJ Passey
# Email: djpassey@unc.edu
#
# Notes:
# An effort was made to align as closely as possible with pseudocode
# from the document linked above. The portion of the algorithm concerned
# with segment initialization was not documented in the document above
# and so the necessary functionality was implemented with guidance from
# the HTM forum, specifically the comment here:
# https://discourse.numenta.org/t/htm-learning-algorithm-temporal-memory/8884/14
# The comment points out that the algorithm starts out with no segments.
# Therefore segment adding must occur in the `burst_column!()` function.
# We use the simple method of initializing a fixed number of segment synapses 
# randomly with a fixed weight. Presynaptic cells are drawn uniformly without
# replacement from the set of all cells.

using Random
const PERMANENCE_TYPE = Float64

#------------------------------------------------------------------------------
# Basic Types 
#------------------------------------------------------------------------------

mutable struct Column{T}
    cells::Set{T}
end

mutable struct Cell{T}
    segments::Set{T}
end

mutable struct Synapse{T}
    permanence::T
    presynaptic_cell::Cell
end

mutable struct Segment{T}
    cell::Cell{Segment{T}}
    synapses::Set{Synapse{T}}
end

"""Struct for storing data at each time step. Used for internal
syntatic sugar:

Makes `tm.active_cells[t]` eqivalent to `get!(tm.active_cells, t, Set{Cells}())`
"""
mutable struct TempMemStore{T}
    data::Dict{Int, T}
end

#------------------------------------------------------------------------------
# Basic Type Constructors 
#------------------------------------------------------------------------------

Cell() = Cell(Set{Segment{PERMANENCE_TYPE}}())
Segment(c::Cell{Segment{T}}) where T = Segment(c, Set{Synapse{T}}())
Segment() = Segment(Cell())

function Cell(num_segments::Int)
    c = Cell()
    for i in 1:num_segments
        push!(c.segments, Segment(c))
    end
    return c
end

function Column(cells_per_col)
    return Column(Set([Cell() for i in 1:cells_per_col]))
end

#------------------------------------------------------------------------------
# Basic Type Functionality
#------------------------------------------------------------------------------

# Syntatic sugar for TempMemStore
TempMemStore{T}() where T = TempMemStore(Dict{Int, T}())
Base.getindex(tms::TempMemStore{T}, i::Int) where T = get!(tms.data, i, T())
Base.setindex!(tms::TempMemStore, x, i::Int) = setindex!(tms.data, x, i)

# Pretty printing
for T in [:Column, :Cell, :Segment, :Synapse, :TempMemStore]
    @eval function Base.show(io::IO, x::$T)
        print(io, $T, "[") 
        map(name -> print(io, " .", name), fieldnames($T))
        print(io, " ]")
    end
end

#------------------------------------------------------------------------------
# Temporal Memory and Parameter Structs
#------------------------------------------------------------------------------

mutable struct TempMemParams{T}
    ## Constants

    # Number of columns in the model
    num_cols::Int
    # Number of cells in each column
    cells_per_col::Int
    
    ## Keyword Args

    # The minimum number of active synapses required for a 
    # segment to be considered active
    activation_threshold::Int
    # Permanence given to all new synapses
    initial_permanence::T
    # The minimum permanence for a synapse to be considered connected
    connected_permanence::T
    # The minimum number of active synapses required for a 
    # segment to be considered "learning"
    learning_threshold::Int
    # Determines if permanence updates and synapse reconnection take place
    learning_enabled::Bool
    # Increment by which permanence is increased
    permanence_increment::T
    # Increment by which permanence is decreased
    permanence_decrement::T
    # Increment by which permanence is decreased for false positives
    predicted_decrement::T
    # How many random synapses appear on a new segment
    synapse_sample_size::Int
    # How many segments are added to each cell during model initialization
    initial_segments_per_cell::Int
end

mutable struct TempMem{T}
    # Parameters
    ps::TempMemParams{T}
    
    # Current timestep
    t::Int

    columns::Array{Column{Cell{Segment{T}}}, 1}
    cells::Array{Cell{Segment{T}}, 1}
    segments::Array{Segment{T}, 1}

    # Active columns are an exogenous variable and are determined
    # based on the encoded data that is passed to the algorithm.
    active_columns::TempMemStore{Set{Column{Cell{Segment{T}}}}}

    # Active cells are the cells with active segments from the previous timestep
    # that belong to the active columns (external input)
    active_cells::TempMemStore{Set{Cell{Segment{T}}}}

    # Winner cells are a subset of active cells, the only difference being
    # that when an entire column is activated, only one cell from that column,
    # the closest cell to activating, is added to the set of winner cells.
    # When new synapses are added, they take presynaptic cells from the
    # set of winner cells from the previous timestep (not the active cells)
    # This makes me wonder if we need the active cell variable at all.
    # There is also some weirdness about which synapse permanences are updated.
    # This is determined 
    winner_cells::TempMemStore{Set{Cell{Segment{T}}}}

    # Matching segments are determined from the previous timestep 
    # `active_cells`. This contains all segments that have more than
    # `TempMem.ps.learning_threshold` synapses with active presynapsic cells. 
    matching_segments::TempMemStore{Set{Segment{T}}}

    # Active segments are a subset of `matching_segments` and contain
    # all segments that have more than
    # `TempMem.ps.activation_threshold` synapses with active presynapsic cells.
    active_segments::TempMemStore{Set{Segment{T}}}
    
    # This variable maps each timestep to a dictionary that associates each
    # segment with the number of potential synapses on that segment 
    # (synapses with permanences above or below the permanence threshold) 
    # that have active presynaptic cells
    num_active_potential_synapses::TempMemStore{Dict{Segment{T}, Int}}
end

#------------------------------------------------------------------------------
# Temporal Memory Object Constructor
#------------------------------------------------------------------------------

function TempMem(
    num_cols::Int,
    cells_per_col::Int;

    # Optional args
    activation_threshold=15,
    initial_permanence=0.3,
    connected_permanence=0.5,
    learning_threshold=12,
    learning_enabled=true,
    permanence_increment=0.05,
    permanence_decrement=0.001,
    predicted_decrement=0.05,
    synapse_sample_size=50,
    initial_segments_per_cell=0
)
    T = PERMANENCE_TYPE
    # Initialize parameter class
    ps = TempMemParams(num_cols, cells_per_col, activation_threshold, 
        initial_permanence, connected_permanence, learning_threshold,
        learning_enabled, permanence_increment, permanence_decrement,
        predicted_decrement, synapse_sample_size, initial_segments_per_cell)
    
    columns = [Column(cells_per_col) for i in 1:num_cols]
    cells = [cell for col in columns for cell in col.cells]
    segments = Segment{PERMANENCE_TYPE}[]

    # Initialize variables to store past states
    active_columns = TempMemStore{Set{Column{Cell{Segment{T}}}}}()
    active_cells = TempMemStore{Set{Cell{Segment{T}}}}()
    winner_cells = TempMemStore{Set{Cell{Segment{T}}}}()
    active_segments = TempMemStore{Set{Segment{T}}}()
    matching_segments = TempMemStore{Set{Segment{T}}}()
    num_active_potential_synapses = TempMemStore{Dict{Segment{T}, Int}}()
    
    tm =  TempMem(    
        ps,
        0,
        columns,
        cells,
        segments,
        active_columns,
        active_cells, 
        winner_cells,
        active_segments,
        matching_segments,
        num_active_potential_synapses,
    )

    # Add segments
    for i in 1:initial_segments_per_cell
        for cell in cells 
            grow_new_segment!(tm, cell)
        end
    end

    return tm
end

#------------------------------------------------------------------------------
# Temporal Memory Struct Basic Functionality
#------------------------------------------------------------------------------

# Make it a one item iterable so that f.(tm, iterable) 
# turns into map(x -> f(tm, x), iterable)
Base.iterate(tm::TempMem) = (tm, nothing)
Base.iterate(tm::TempMem, i) = nothing

"""
    predicted_cells(tm, t)

Returns an Array{Int, 1} containing the indexes of active cells at timestep
`t`.
"""
predicted_cells(tm, t) = [findfirst([seg.cell == c for c in tm.cells])
    for seg in tm.active_segments[t]
]
predicted_cells(tm) = predicted_cells(tm, tm.t)

"""
    predicted_columns(tm, t)

Returns an Array{Int} containing the indexes of the predicted columns.
"""
predicted_columns(tm, t) = [
    cell ÷ tm.ps.cells_per_col + 1 for cell in predicted_cells(tm, t)
]
predicted_columns(tm) = predicted_columns(tm, tm.t)

"""
    grow_new_segment!(tm, cell)

Creates a segment and adds it to `cell`.
"""
function grow_new_segment!(tm, cell)

    seg = Segment(cell)
    # Select random presynaptic cells
    pre_syn_cell_idxs = randperm(length(tm.cells))[1:tm.ps.synapse_sample_size]
    # Create and add synapses
    syns = map(c -> Synapse(tm.ps.initial_permanence, c), 
               tm.cells[pre_syn_cell_idxs])
    map(s -> push!(seg.synapses, s), syns)

    # Add segment to model
    push!(tm.segments, seg)
    push!(cell.segments, seg)
    return seg
end

function Base.show(io::IO, tm::TempMem)
    print(io, "TempMem\n\t$(tm.ps.num_cols) Columns")
end

num_synapses(s::Segment) = length(s.synapses)
num_synapses(c::Cell) = sum(num_synapses.(c.segments))
num_synapses(c::Column) = sum(num_synapses.(c.cells))
num_synapses(tm::TempMem, col_idxs) = sum([num_synapses(tm.columns[i]) for i in col_idxs])

function num_synapses_from(tm, col_idxs)
    all_cells = union([tm.columns[i].cells for i in col_idxs]...)
    tot = 0
    for seg in tm.segments
        for syn in seg.synapses
            if syn.presynaptic_cell in all_cells
                tot += 1
            end
        end
    end
    return tot
end
#------------------------------------------------------------------------------
# Temporal Memory Algorithm 
#------------------------------------------------------------------------------

"""
    update!(tm, active_col_idxs)

The main entry point into the temporal memory algorithm. Provide the columns that
you want to be active next and the algorithm will try to learn them.
"""
function update!(tm::TempMem{T}, active_col_idxs::Array{Int, 1}) where T
    tm.t += 1
    tm.active_columns[tm.t] = Set([tm.columns[i] for i in active_col_idxs])
    evaluate_active_cols_against_predictions!(tm, tm.t)
    activate_segments!(tm, tm.t)
     return nothing
end


"""
    evaluate_active_cols_against_predictions!(tm::TempMem, t::Int)

Loops over the active columns for time `t`. Checks for matching predictions and
and performs the correct learning update. There are four possibilities

#### True Positive:
This occurs when a column is active, and one of its cells is in a predictive
state. When this happens, `activate_predicted_column!()` is called. This
function strengthens connections to previous winner cells, weakens 
connections to non winners, and forms new synapses to previous winner
cells on all currently active segments. New synapses are not formed if
the numer of active potential synapses exceeds `tm.ps.synapse_sample_size`

#### False Negative:
This occurs when a column is active but none of its cells are in a 
predictive state. When this even occurs, `burst_column!()` is called.
All cells in the column are activated. The best matching segment in the 
column is located and its cell is added to the set of winner cells. If
there are no "matching" segments, a new segment is formed. This segment alone
(either the matching or new segment) gains new synapses to previous
winner cells.

#### False Positive:
This event occurs when a column contains predictive cells but it is not 
an active column. This event is handled with `punish_predicted_column!()`.
This function loops through all "matching segments" (segments that activated 
or are close to activating) and reduces the permanence of synapses that
connect to previously "active" cells. For some reason, the algorithm 
makes a distiction between winner cells and active cells and it shows up
here. Also, it is interesting that no synapse reconnection occurs here.
This operation also has it's own hyper parameter: `tm.ps.predicted_decrement`.
This parameter is distinct from `tm.ps.permanence_decrement` and
`tm.ps.permanence_increment` that are used in the two cases above.

#### True Negative:
If a column had no active segments and it was not an active column, then
nothing happens unless some of the segments were "matching" which means that
they were close to activating. In this case `punish_predicted_column!()`
is called as in the case above. This function decreases the permanence
of synapses that were connected to previously "active" cells.
"""
function evaluate_active_cols_against_predictions!(tm::TempMem, t::Int)
    for column in tm.columns
        if column in tm.active_columns[t]
            # If the current column is an active column
            # Grab all segments that are active from the current column
            active_segs = segments_for_column(column, tm.active_segments[t-1])
            # Check if it was predicted
            if length(active_segs) > 0
                # Acivate cells corresponding to active segments
                activate_predicted_column!(tm, active_segs, t)
            else
                matching_segs = segments_for_column(column, tm.matching_segments[t-1])
                # Activate all column cells, find cell closest to activating
                burst_column!(tm, column, matching_segs, t)
            end
        else
            matching_segs = segments_for_column(column, tm.matching_segments[t-1])
            if length(matching_segs) > 0
                # If the column is not active, but it was close to predicting, punish it
                punish_predicted_column!(tm, matching_segs, t)
            end
        end
    end
end

"""
    activate_predicted_column!(tm, column, active_segs, t)

This function is applied when an active column was predicted correctly.

It adds all cells with active segments to `tm.active_cells[t]` and
to `tm.winner_cells[t]`. It also increases the permanence of synapses
that connect to `tm.active_cells[t-1]` and creates new synapses
to the `tm.winner_cells[t-1]`.
"""
function activate_predicted_column!(tm, active_segs, t)
    for segment in active_segs
        push!(tm.active_cells[t], segment.cell)
        push!(tm.winner_cells[t], segment.cell)
        
        if tm.ps.learning_enabled
            for synapse in segment.synapses
                if synapse.presynaptic_cell in tm.active_cells[t-1]
                    synapse.permanence += tm.ps.permanence_increment
                else
                    synapse.permanence -= tm.ps.permanence_decrement
                end
            end
            new_synapse_count = tm.ps.synapse_sample_size - get!(tm.num_active_potential_synapses[t-1], segment, 0)
            grow_synapses!(tm, segment, new_synapse_count, t)
        end
    end
end


"""
    burst_column!(tm::TempMem, column, active_segs, t)

Activates all cells in `column`. If the column has active segments
the segment that has the most active potential synapses from the previous
timestep is selected. Updates weights and reconnects synapses to previously 
active cells.

If the column has no active segments, the function creates a new one and 
populates it with synapses.
"""
function burst_column!(tm, column, matching_segs, t)
    # Add cells in column to the set of active cells
    for cell in column.cells
        push!(tm.active_cells[t], cell)
    end
    
    if length(matching_segs) > 0
        # If there were active segments in this column, find the best match
        # to the input
        learning_segment = best_matching_segment(tm, matching_segs, t)
        winner_cell = learning_segment.cell
    else
        # Otherwise, retrive the least used cell
        winner_cell = least_used_cell(column)
        if tm.ps.learning_enabled
            learning_segment = grow_new_segment!(tm, winner_cell)
        end
    end
    
    # Store cells to be updated
    push!(tm.winner_cells[t], winner_cell)
    
    # Modulate synapse weights in `learning_segment` segment
    if tm.ps.learning_enabled
        for synapse in learning_segment.synapses
            if synapse.presynaptic_cell in tm.active_cells[t-1]
                synapse.permanence += tm.ps.permanence_increment
            else
                synapse.permanence -= tm.ps.permanence_decrement
            end
        end
        
        new_synapse_count = tm.ps.synapse_sample_size - get!(
            tm.num_active_potential_synapses[t-1], learning_segment, 0)
        grow_synapses!(tm, learning_segment, new_synapse_count, t)
    end
end


"""
    punish_predicted_column!(tm, active_segs, t)

Decreases the permanence values of all synapses connected to a
cell that was active in the previous timestep. Does so for every
segment in `active_segs`. 

Note: The name of this function corresponds to the Numenta pseudo code
but it is not concerned with columns.  The pseudocode makes
multiple calls to `segments_for_column()` inside functions
but it only needs to make one call and pass along the results.
I rearranged the code slightly to make this minor optimization.
"""
function punish_predicted_column!(tm, matching_segs, t)
    if tm.ps.learning_enabled
        for segment in matching_segs
            for synapse in segment.synapses
                if synapse.presynaptic_cell in tm.active_cells[t-1]
                    synapse.permanence -= tm.ps.predicted_decrement
                end
            end
        end
    end
end


"""
    activate_segments!(tm::TempMem, t::Int)

Counts and stores active synapses per segment.
"""
function activate_segments!(tm::TempMem, t::Int)
    # Loop over all segments
    for segment in tm.segments
        num_active_connected = 0
        num_active_potential = 0
        # Count the number of synapses with active presynaptic cells
        for synapse in segment.synapses
            if synapse.presynaptic_cell in tm.active_cells[t]
                if synapse.permanence >= tm.ps.connected_permanence
                    num_active_connected += 1
                end
                # Count potential synapses (Those below permanence threshold)
                if synapse.permanence >= 0
                        num_active_potential += 1
                end
            end
        end
        # Add segments to collection of active or matching
        if num_active_connected >= tm.ps.activation_threshold
            push!(tm.active_segments[t], segment)
        end
        if num_active_potential >= tm.ps.learning_threshold
            push!(tm.matching_segments[t], segment)
        end
        # Store active potential synapses per segment
        tm.num_active_potential_synapses[t][segment] = num_active_potential
    end
end

"""
    least_used_cell(column)

Finds the cells in a column with the fewest_segments. Breaks ties
randomly.
"""
function least_used_cell(column)
    fewest_segments = Inf
    for cell in column.cells
        fewest_segments = min(fewest_segments, length(cell.segments))
    end
    least_used_cells = Cell[]
    for cell in column.cells
        if length(cell.segments) == fewest_segments
            push!(least_used_cells, cell)
        end
    end
    
    return rand(least_used_cells)
end


"""
    best_matching_segment(tm, segments, t)

Locates the segment in `segments' with the most active potential synapses from
time t-1.
"""
function best_matching_segment(tm, segments, t)
    best_matching_segment = Segment()
    best_score = -1
    for seg in segments
        score = get!(tm.num_active_potential_synapses[t-1], seg, 0)
        if score > best_score
            best_matching_segment = seg
            best_matching_segment::Segment
            best_score = score
        end
    end
    
    return best_matching_segment
end

"""
    grow_synapses!(tm, segment, new_synapse_count, t)

Creates new synapses from the previous winner cells (`tm.winner_cells[t-1]`)
onto `segment`. Does not create duplicate synapses.

Attempts to add each winner cell until either `new_synapse_count` synpases are
successfully added or the list of winner cells is exhausted.

Returns the number of new synapses.
"""
function grow_synapses!(tm, segment, num_new_synapses, t)

    new_synapse_count = 0
    candidates = Set([cell for cell in tm.winner_cells[t-1]])
    existing_presyn_cells = Set([s.presynaptic_cell for s in segment.synapses])
    while (length(candidates) > 0) && (new_synapse_count <  num_new_synapses)
        presynaptic_cell = rand(candidates)
        pop!(candidates, presynaptic_cell)

        if !(presynaptic_cell in existing_presyn_cells)
            new_synapse = create_new_synapse!(
                segment, tm.ps.initial_permanence, presynaptic_cell,  safe=false)
            new_synapse_count += 1
        end
    end
    return new_synapse_count
end


"""
    create_new_synapse!(segment, permanence, presynaptic_cell; safe=true)

Creates a new synapse from `presynaptic_cell` to `segment` with permanence
`permanence`. The variable `safe` controls if the function checks if the
synapse from `presynaptic_cell` exists before adding one. Set `safe=false`
to skip this check.
"""
function create_new_synapse!(segment, permanence, presynaptic_cell; safe=true)
    # Check for existing synapse if safe
    if safe
        cells = [syn.presynaptic_cell for syn in segment.synapses]
        if presynaptic_cell in cells
            throw(ArgumentError("Synapse from given"*
                                " presynaptic cell already exists"))
        end
    end
    # Add the synapse
    syn = Synapse(permanence, presynaptic_cell)
    push!(segment.synapses, syn)
end

"""
    random_initial!(tm::TempMem, num_active::Int)

Assigns a list of `num_active` columns to the TempMem.active_columns
dictionary at key 0.
"""
function random_initial!(tm::TempMem, num_active::Int)
    if num_active > tm.ps.num_cols
        throw(ArgumentError("Number of columns to activate" *
                                        " (received $num_active)" * 
                                        "must be less than total number " * 
                                        "of columns ($(tm.ps.num_cols))"))
    end
    active_idxs = randperm(tm.ps.num_cols)[1:num_active]
    # Dictionary assignment for time t=0
    tm.active_columns[0] = Set(tm.columns[active_idxs])
end

"""Returns segments beloning only to cells in `column`"""
segments_for_column(column, segments) = filter(s -> s.cell in column.cells, segments)