# Tests for jupic.jl
using Test
include(string(@__DIR__, "/jupic.jl"))

@info "Begin Testing"

num_cols = 10
cells_per_col = 5
init_segs = 4
tm = TempMem(num_cols, cells_per_col, initial_segments_per_cell=init_segs)

@testset "Constructors" begin
    num_cols = 10
    cells_per_col = 5
    init_segs = 4
    tm = TempMem(num_cols, cells_per_col, initial_segments_per_cell=init_segs)
    
    @test length(tm.columns) == num_cols
    
    cells = [c for c in tm.columns[1].cells]
    @test length(cells) == cells_per_col
    
    segs = [s for s in cells[1].segments]
    @test length(segs) == init_segs
    @test segs[1].cell == cells[1]

    syns = [s for s in segs[1].synapses]
    @test length(syns) == tm.ps.synapse_sample_size
    @test tm.ps.activation_threshold == 15
    @test tm.ps.initial_segments_per_cell == 4
end

@testset "Temporal Memory Algorithm Core" begin

    @testset "update!()" begin
        num_cols = 6
        cells_per_col = 2
        init_segs = 1

        tm = TempMem(
            num_cols, 
            cells_per_col,
            activation_threshold=2,
            learning_threshold=1,
            initial_segments_per_cell=1,
            synapse_sample_size=0,
            initial_permanence=1.0,
            learning_enabled=true
        )
        tm.ps.synapse_sample_size = 2
        # Make sure that each cell has one segment and each segment
        # has zero synapses
        for cell in tm.cells
            @test length(cell.segments) == 1
            for seg in cell.segments
                @test length(seg.synapses) == 0
            end
        end

        # One previous winner cell
        push!(tm.winner_cells[tm.t], tm.cells[end])
        # Assign previous active and matching segments. These were chosen
        # so that every piece of the control flow is explored
        matching_segs = tm.segments[[1, 7, 9, 11]]
        active_segs = matching_segs[[1, 3]]
        tm.active_segments[tm.t] = Set(active_segs)
        tm.matching_segments[tm.t] = Set(matching_segs)
        
        
        # Add a synapse from cell 4 to cell 11
        push!(matching_segs[end].synapses, Synapse(1.0, tm.cells[4]))
        # Add a segment to cell 3
        seg = Segment()
        push!(tm.cells[3].segments, seg)
        push!(tm.segments, seg)

        active_cols = [1, 2, 6]
        # Call function
        update!(tm, active_cols)
        # Run tests
        @test length(symdiff(tm.active_cells[tm.t], tm.cells[[1, 3, 4, 11, 12]])) == 0
        @test length(symdiff(tm.winner_cells[tm.t], tm.cells[[1, 4, 11]])) == 0
        if length(tm.active_segments[tm.t]) == 1
            @test length(symdiff(tm.active_segments[tm.t], [tm.segments[11]])) == 0
        else
            @test length(symdiff(tm.active_segments[tm.t], tm.segments[[11, end]])) == 0
        end
        @test length(symdiff(tm.matching_segments[tm.t], tm.segments[[1, 11, end]])) == 0

    end

    @testset "evaluate_active_cols_against_predictions!()" begin
        num_cols = 4
        cells_per_col = 3
        init_segs = 1
        init_syn = 2
        init_perm = 1.0
        tm = TempMem(
            num_cols, 
            cells_per_col, 
            initial_segments_per_cell=init_segs,
            synapse_sample_size=init_syn,
            initial_permanence=init_perm,
            learning_enabled=true
        )
        t = 5
        true_pos_col = tm.columns[1]
        false_neg_col = tm.columns[2]
        push!(tm.active_columns[t], true_pos_col)
        push!(tm.active_columns[t], false_neg_col)
        # Activate a segment in the true positive column
        true_pos_seg = tm.segments[1]
        @test true_pos_seg.cell in true_pos_col.cells
        push!(tm.active_segments[t-1], true_pos_seg)
        # Don't activate any segments in false_neg_col 
        # This column should gain an additional segment
        
        # Activate a segment in an inactive column (False positive)
        false_pos_seg = tm.segments[end]
        @test false_pos_seg.cell in tm.columns[end].cells
        push!(tm.active_segments[t-1], false_pos_seg)
        push!(tm.matching_segments[t-1], false_pos_seg)
        # Don't activate anything in column 3 (True negative)
        # but create a matching segment.
        true_neg_matching_seg = [
            seg for c in tm.columns[3].cells for seg in c.segments][1]
        push!(tm.matching_segments[t-1], true_neg_matching_seg)
        # Add one cell to the previous active cells and winners
        true_neg_matching_seg_active_cell = [
            syn.presynaptic_cell for syn in true_neg_matching_seg.synapses][1]
        push!(tm.active_cells[t-1], true_neg_matching_seg_active_cell)
        push!(tm.winner_cells[t-1], true_neg_matching_seg_active_cell)

    
        # Add cells to the previous active and previous winners.
        # (One cell from each active segment)
        pre_syn_cell1 = [syn.presynaptic_cell for syn in true_pos_seg.synapses][1]
        pre_syn_cell2 = [syn.presynaptic_cell for syn in false_pos_seg.synapses][1]
        push!(tm.active_cells[t-1], pre_syn_cell1)
        push!(tm.active_cells[t-1], pre_syn_cell2)
        push!(tm.winner_cells[t-1], pre_syn_cell1)
        push!(tm.winner_cells[t-1], pre_syn_cell2)
        tm.num_active_potential_synapses[t-1][true_pos_seg] = 1
        tm.num_active_potential_synapses[t-1][false_pos_seg] = 1

        # Call function and test
        evaluate_active_cols_against_predictions!(tm, t)

        @test length(tm.active_cells[t]) == 1 + cells_per_col
        @test length(tm.winner_cells[t]) == length(tm.active_columns[t])
        
        # True positive column
        # Has one active segment. This segment should have synapses updated
        # And have one new synapse added unless synapses to both active cells
        # already existed.
        if length(true_pos_seg.synapses) == 2
            pre_syn_cells = [syn.presynaptic_cell for syn in true_pos_seg.synapses]
            @test pre_syn_cell2 in pre_syn_cells
            @test true_neg_matching_seg_active_cell in pre_syn_cells
        else
            @test length(true_pos_seg.synapses) == 3
        end

        # Check that no new segments were added
        @test all([length(cell.segments) for cell in true_pos_col.cells] .== init_segs)
        for syn in true_pos_seg.synapses
            if syn.presynaptic_cell == pre_syn_cell1
                syn.permanence == init_perm + tm.ps.permanence_increment
            elseif syn.presynaptic_cell == pre_syn_cell2
                syn.permanence == init_perm
            else
                syn.permanence == init_perm - tm.ps.permanence_decrement
            end
        end

        # False Negative column
        # Since there are no matching segments, this column should grow
        # a new segment
        @test any(
            [length(cell.segments) 
            for cell in false_neg_col.cells] .== init_segs + 1
        )

        # True Negative column
        # All segments in this column should remain the same
        # unless they were matching segments, and then they
        # should have permanences decreased by tm.ps.predicted_decrement
        for cell in tm.columns[3].cells
            for seg in cell.segments
                if seg in tm.matching_segments[t-1]
                    for syn in seg.synapses
                        if syn.presynaptic_cell in tm.active_cells[t-1]
                            @test syn.permanence == init_perm - tm.ps.predicted_decrement
                        else
                            @test syn.permanence == init_perm
                        end
                    end
                else
                    for syn in seg.synapses
                        @test syn.permanence == init_perm
                    end
                end
            end
        end

        # False Positive column
        for syn in true_neg_matching_seg.synapses
            if syn.presynaptic_cell in [pre_syn_cell1, 
                                        pre_syn_cell2, 
                                        true_neg_matching_seg_active_cell]
                @test syn.permanence == init_perm - tm.ps.predicted_decrement
            else
                @test syn.permanence == init_perm
            end
        end


    end

    @testset "activate_predicted_column!()" begin
        num_cols = 10
        cells_per_col = 10
        init_segs = 4
        init_syn = 2
        init_perm = 1.0
        tm = TempMem(
            num_cols, 
            cells_per_col, 
            initial_segments_per_cell=init_segs,
            synapse_sample_size=init_syn,
            initial_permanence=init_perm,
            learning_enabled=true
        )

        # We want one synapse to active_cells[t-1] and one not
        # We want a synapse to get added. So we need a cell in winner_cells[t-1]
        # Let's do two winner cells, one who already has a connection.
        t = 3
        seg = tm.segments[1]
        active_cell, inactive_cell = [s.presynaptic_cell for s in seg.synapses]
        new_synapse_cell = Cell()
        push!(tm.active_cells[t-1], active_cell)
        push!(tm.winner_cells[t-1], active_cell)
        push!(tm.winner_cells[t-1], new_synapse_cell)
        tm.num_active_potential_synapses[t-1][seg] = 1

        activate_predicted_column!(tm, [seg], t)
        @test length(seg.synapses) == init_syn + 1
        for syn in seg.synapses
            if syn.presynaptic_cell == active_cell
                @test syn.permanence == init_perm + tm.ps.permanence_increment
            elseif syn.presynaptic_cell == inactive_cell
                @test syn.permanence == init_perm - tm.ps.permanence_decrement
            else
                @test syn.permanence == init_perm
            end
        end

    end

    @testset "burst_column!()" begin
        num_cols = 10
        cells_per_col = 10
        init_segs = 4
        init_syn = 2
        init_perm = 1.0
        tm = TempMem(
            num_cols, 
            cells_per_col, 
            initial_segments_per_cell=init_segs,
            synapse_sample_size=init_syn,
            initial_permanence=init_perm,
            learning_enabled=true
        )

        t = 3
        col = tm.columns[1]
        # Target for least used cell, remove one segment
        cell = [c for c in col.cells][1]
        seg = pop!(cell.segments)
        # No active segments
        active_segs = []
        # No active cells
        init_num_segs = length(tm.segments)
        # Call function
        burst_column!(tm, col, active_segs, t)
        # Test
        num_syns = [length(seg.synapses) for cell in col.cells for seg in cell.segments]
        # Check that no new synapses were formed in the column
        @test all(num_syns .== init_syn)
        # Check that best matching cell was added to winners
        @test cell in tm.winner_cells[t]
        @test length(cell.segments) == init_segs
        @test length(tm.segments) == init_num_segs + 1
        seg = tm.segments[end]
        for syn in seg.synapses
            @test syn.permanence == (init_perm - tm.ps.permanence_decrement)
        end

        #--- Next Test Scenario ---#

        # Clear TempMem state
        delete!(tm.winner_cells.data, t)
        delete!(tm.winner_cells.data, t-1)
        delete!(tm.active_cells.data, t)

        # Activate one of the presynaptic cells in the new segment
        pre_syn_cell = [syn.presynaptic_cell for syn in seg.synapses][1]       
        push!(tm.winner_cells[t-1], pre_syn_cell)
        push!(tm.active_cells[t-1], pre_syn_cell)
        # Activate one random cell
        rand_cell = rand(tm.cells)
        push!(tm.winner_cells[t-1], rand_cell)  
        push!(tm.active_cells[t-1], rand_cell)  

        # Call function and test
        burst_column!(tm, col, [seg], t)
        @test seg.cell in tm.winner_cells[t]
        @test length(seg.synapses) == init_syn + 1
        for syn in seg.synapses
            if syn.presynaptic_cell == rand_cell
               @test syn.permanence == init_perm
            elseif syn.presynaptic_cell == pre_syn_cell
                @test syn.permanence == (init_perm - tm.ps.permanence_decrement + tm.ps.permanence_increment)
            else
                @test syn.permanence == (init_perm - 2 * tm.ps.permanence_decrement)
            end
        end

    end

    @testset "punish_predicted_column!" begin
        num_cols = 10
        cells_per_col = 10
        init_segs = 4
        init_syn = 2
        init_perm = 1.0
        tm = TempMem(
            num_cols, 
            cells_per_col, 
            initial_segments_per_cell=init_segs,
            synapse_sample_size=init_syn,
            learning_enabled=false
        )

        t = 2
        active_cells = tm.columns[1].cells
        tm.active_cells[t] = active_cells

        syn_cells = union(active_cells, tm.columns[2].cells)
        syns = [Synapse(init_perm, c) for c in syn_cells]
        cell = rand(tm.columns[end].cells)
        active_seg = Segment(cell, Set(syns))

        punish_predicted_column!(tm, [active_seg], t+1)
        for syn in active_seg.synapses
            @test syn.permanence == init_perm
        end

        tm = TempMem(
            num_cols, 
            cells_per_col, 
            initial_segments_per_cell=init_segs,
            synapse_sample_size=init_syn,
            learning_enabled=true
        )
        t = 2
        active_cells = tm.columns[1].cells
        tm.active_cells[t] = active_cells
        
        syn_cells = union(active_cells, tm.columns[2].cells)
        syns = [Synapse(init_perm, c) for c in syn_cells]
        cell = rand(tm.columns[end].cells)
        active_seg = Segment(cell, Set(syns))

        punish_predicted_column!(tm, [active_seg], t+1)
        for syn in active_seg.synapses
            if syn.presynaptic_cell in active_cells
                @test syn.permanence == (init_perm - tm.ps.predicted_decrement)
            else
                @test syn.permanence == init_perm
            end
        end

    end

    @testset "activate_segments!()" begin

        # Initialize TempMem
        num_cols = 100
        cells_per_col = 10
        init_segs = 4
        tm = TempMem(
            num_cols, 
            cells_per_col, 
            initial_segments_per_cell=init_segs,
            activation_threshold=cells_per_col,
            learning_threshold=cells_per_col - 2
        )

        # Create two segments that should activate
        t = 2
        active_cells = tm.columns[1].cells
        tm.active_cells[t] = active_cells
        syns = [Synapse(1.0, c) for c in active_cells]
        cell = rand(tm.columns[2].cells)
        active_seg = Segment(cell, Set(syns))
        matching_seg = Segment(cell, Set(syns[1:end-2]))

        # Add to TempMem
        append!(tm.segments, [active_seg, matching_seg])
        
        # Call function and test
        activate_segments!(tm, t)
        @test active_seg in tm.active_segments[t]
        @test active_seg in tm.matching_segments[t]
        @test !(matching_seg in tm.active_segments[t])
        @test matching_seg in tm.matching_segments[t]
        @test tm.num_active_potential_synapses[t][active_seg] == length(syns)
        @test tm.num_active_potential_synapses[t][matching_seg] == length(syns) - 2

    end
end

@testset "Algorithm Helper Functions" begin

    # Initialize TempMem
    num_cols = 10
    cells_per_col = 5
    init_segs = 4
    tm = TempMem(num_cols, cells_per_col, initial_segments_per_cell=init_segs)

    @testset "least_used_cell()" begin
        col_idx = 3
        cell_idx = 1
        col = tm.columns[col_idx]
        cells = [c for c in col.cells]
        cell = cells[cell_idx]
        seg = pop!(cell.segments)
        least_used = least_used_cell(col)
        @test least_used == cell
        # Put back to normal
        push!(cell.segments, seg)
    end

    @testset "best_matching_segments()" begin
        
        tm.num_active_potential_synapses[0]  = Dict{Segment, Int}()
        segs = tm.segments[1:3]
        for (i, s) in enumerate(segs)
            tm.num_active_potential_synapses[0][s] = i
        end

        best_seg = best_matching_segment(tm, segs, 1)
        @test best_seg == segs[3]
    end

    @testset "grow_new_segment!" begin
        cell = Cell()
        init_num_segs = length(tm.segments)
        grow_new_segment!(tm, cell)
        @test length(tm.segments) == init_num_segs + 1
        @test length(cell.segments) == 1
    end

    @testset "grow_synapses!()" begin
        seg = Segment()
        tm.winner_cells[0] = Set([c for c in tm.columns[1].cells])
        num_new_synapses = 1
        t = 1
        grow_synapses!(tm, seg, num_new_synapses, t)
        @test length(seg.synapses) == 1
        
        num_new_synapses = 4
        grow_synapses!(tm, seg, num_new_synapses, t)
        @test length(seg.synapses) == 5

        @test grow_synapses!(tm, seg, num_new_synapses, t) == 0
        @test length(seg.synapses) == 5
    
    end

    @testset "create_new_synapse!()" begin
        seg = tm.segments[1]
        num_syns = length(seg.synapses)
        pre_syn_cell = [c for c in tm.columns[end].cells][1]
        create_new_synapse!(seg, 0.5, pre_syn_cell, safe=false)
        @test length(seg.synapses) == num_syns + 1
        @test_throws ArgumentError create_new_synapse!(seg, 0.5, pre_syn_cell)
        create_new_synapse!(seg, 0.5, pre_syn_cell, safe=false)
        @test length(seg.synapses) == num_syns + 2
    end

    @testset "random_initial!()" begin
        num_active = 3
        initial_active = tm.active_columns
        random_initial!(tm, num_active)
        @test length(tm.active_columns[0]) == num_active
        tm.active_columns = initial_active
        @test_throws ArgumentError random_initial!(tm, num_cols + 1)
    end

    @testset "segments_for_column()" begin
        segs = [seg for cell in tm.columns[1].cells for seg in cell.segments]
        segs_for_matching_col = segments_for_column(tm.columns[1], segs)

        # All segments belong to column 1
        @test length(segs_for_matching_col) == length(segs)
        # None of the segments belong to column 2
        @test length(segments_for_column(tm.columns[2], segs)) == 0
    end
end

@info "Tests Complete"

