#include "optifine/codegen/program.h"

#include <stdlib.h>

#include "optifine/codegen/avr_mir.h"
#include "optifine/codegen/hir_to_mir.h"
#include "optifine/codegen/instr_buf.h"
#include "optifine/emit.h"
#include "optifine/invariant.h"

/* Every program region goes the same way: workload HIR -> MIR (hir_to_mir)
 * -> AVR selection (avr_mir) -> emission. Each region builds only the MIR
 * function it emits, so a kernel is lowered once per program. */

/* Emits a candidate into the unit and adds its cost to `cost`. */
static int emit_priced(EmitUnit *unit, const Candidate *c, FILE *out, ProgramRegionCost *cost) {
    int rc = emit_candidate(unit, c, out);
    cost->energy_nj += c->energy_nj;
    cost->cycles += c->cycles;
    return rc;
}

/* Builds the workload module with `which` function, lays it out, and
 * selects that function. */
static int select_region(const IrGraph *graph, const SramLayout *layout, const ReuseAnalysis *reuse,
                         const CostModel *cost_model, const int8_t *input, size_t input_len,
                         int use_real_candidates, unsigned which, AvrMirCode *code) {
    WorkloadMir mir;
    if (hir_to_mir(graph, layout, reuse, cost_model, input, input_len, use_real_candidates, which, &mir, true) != 0) {
        return -1;
    }
    AvrMirLayout avr;
    int rc = workload_mir_layout(&mir, layout, &avr);
    if (rc == 0) {
        uint32_t fn = which == HIR_TO_MIR_INITIALIZE ? mir.initialize : mir.infer;
        /* The initialization region is where a program starts, so it carries
         * the backend's entry prologue (clr r2). */
        rc = avr_mir_select_function(&mir.module, &avr, fn, which == HIR_TO_MIR_INITIALIZE, cost_model, code);
        avr_mir_layout_free(&avr);
        /* The region costs printed and compared against Avrora are execution
         * costs only because every workload function is one block: its
         * control flow lives inside counted-loop target regions, which
         * instrbuf_price prices by trip count. */
        if (rc == 0) OPTIFINE_INVARIANT(code->straight_line);
    }
    workload_mir_free(&mir);
    return rc;
}

static int emit_region(const AvrMirCode *code, EmitUnit *unit, FILE *out, ProgramRegionCost *cost) {
    for (size_t i = 0; i < code->count; i++) {
        if (emit_priced(unit, &code->segments[i].code, out, cost) != 0) return -1;
    }
    return 0;
}

int codegen_emit_initialization(const IrGraph *graph, const SramLayout *layout,
                                const ReuseAnalysis *reuse, const CostModel *cost_model,
                                const int8_t *demo_input, size_t demo_input_len,
                                int use_real_candidates,
                                EmitUnit *unit, FILE *out, ProgramRegionCost *out_cost) {
    out_cost->energy_nj = 0.0;
    out_cost->cycles = 0;
    AvrMirCode code;
    if (select_region(graph, layout, reuse, cost_model, demo_input, demo_input_len, use_real_candidates,
                      HIR_TO_MIR_INITIALIZE, &code) != 0) {
        fprintf(stderr, "codegen_emit_initialization: failed to lower the initialization region\n");
        return -1;
    }
    int rc = emit_region(&code, unit, out, out_cost);
    avr_mir_code_free(&code);
    return rc;
}

int codegen_emit_inference_body(const IrGraph *graph, const SramLayout *layout,
                                const ReuseAnalysis *reuse, const CostModel *cost_model,
                                const int8_t *demo_input, size_t demo_input_len,
                                int use_real_candidates,
                                EmitUnit *unit, FILE *out, ProgramRegionCost *out_cost) {
    out_cost->energy_nj = 0.0;
    out_cost->cycles = 0;
    AvrMirCode code;
    if (select_region(graph, layout, reuse, cost_model, demo_input, demo_input_len, use_real_candidates,
                      HIR_TO_MIR_INFER, &code) != 0) {
        fprintf(stderr, "codegen_emit_inference_body: failed to lower the inference body\n");
        return -1;
    }
    if (code.count > 0) {
        fprintf(out, "\n    ; ---- inference begins here ----\n");
    }
    int rc = emit_region(&code, unit, out, out_cost);
    avr_mir_code_free(&code);
    return rc;
}

int codegen_emit_program(const IrGraph *graph, const SramLayout *layout,
                          const ReuseAnalysis *reuse, const CostModel *cost_model,
                          const int8_t *demo_input, size_t demo_input_len,
                          int use_real_candidates,
                          FILE *out, ProgramCost *out_cost) {
    out_cost->prologue_energy_nj = 0.0;
    out_cost->prologue_cycles = 0;
    out_cost->body_energy_nj = 0.0;
    out_cost->body_cycles = 0;

    ProgramRegionCost initialization = {0};
    ProgramRegionCost inference_body = {0};

    EmitUnit unit;
    emit_unit_init(&unit);
    emit_program_prologue(out);
    int rc = codegen_emit_initialization(graph, layout, reuse, cost_model,
                                         demo_input, demo_input_len, use_real_candidates,
                                         &unit, out, &initialization);
    if (rc == 0) {
        rc = codegen_emit_inference_body(graph, layout, reuse, cost_model,
                                         demo_input, demo_input_len, use_real_candidates,
                                         &unit, out, &inference_body);
    }
    emit_unit_free(&unit);
    if (rc != 0) {
        return -1;
    }

    out_cost->prologue_energy_nj = initialization.energy_nj;
    out_cost->prologue_cycles = initialization.cycles;
    out_cost->body_energy_nj = inference_body.energy_nj;
    out_cost->body_cycles = inference_body.cycles;

    emit_program_epilogue(out);
    return 0;
}

/* The module's program-memory constants (the DSP twiddle table) as one
 * candidate; empty when there are none. */
static int constant_data(const IrGraph *graph, const CostModel *cost_model, Candidate *out) {
    /* Objects only: no function is built, so no layout or input is read. */
    WorkloadMir mir;
    if (hir_to_mir(graph, NULL, NULL, cost_model, NULL, 0, 0, 0, &mir, false) != 0) return -1;
    int rc = avr_mir_select_constants(&mir.module, cost_model, out);
    workload_mir_free(&mir);
    return rc;
}

int codegen_emit_constant_data(const IrGraph *graph, const CostModel *cost_model,
                               EmitUnit *unit, FILE *out) {
    Candidate data;
    if (constant_data(graph, cost_model, &data) != 0) {
        return -1;
    }
    int rc = 0;
    if (data.num_instructions > 0) {
        fprintf(out, "\n; ---- constant data (program memory, never executed) ----\n");
        rc = emit_candidate(unit, &data, out);
    }
    candidate_free(&data);
    return rc;
}

/* codegen_emit_dsp_program's body, into one assembly unit. */
static int emit_dsp_program(const IrGraph *graph, const SramLayout *layout, const ReuseAnalysis *reuse,
                            const CostModel *cost_model, const int8_t *input_bytes, size_t input_len,
                            FILE *out, DspProgramCost *out_cost, EmitUnit *unit) {
    emit_program_prologue(out);
    if (codegen_emit_initialization(graph, layout, reuse, cost_model, input_bytes, input_len, 0,
                                    unit, out, &out_cost->initialization) != 0) {
        return -1;
    }
    if (codegen_emit_inference_body(graph, layout, reuse, cost_model, input_bytes, input_len, 0,
                                    unit, out, &out_cost->body) != 0) {
        return -1;
    }

    /* The terminating break, priced (it is part of what Avrora counts), then
     * the constant data, which control flow never reaches. */
    InstrBuf end;
    instrbuf_init(&end);
    ins0(&end, "break");
    Candidate brk, data;
    if (instrbuf_price(&end, cost_model, &brk) != 0) {
        return -1;
    }
    if (constant_data(graph, cost_model, &data) != 0) {
        candidate_free(&brk);
        return -1;
    }
    fprintf(out, "\n    ; ---- program end: break, then constant data ----\n");
    int rc = emit_priced(unit, &brk, out, &out_cost->termination);
    if (rc == 0) rc = emit_candidate(unit, &data, out);
    candidate_free(&brk);
    candidate_free(&data);
    return rc;
}

int codegen_emit_dsp_program(const IrGraph *graph, const SramLayout *layout,
                             const ReuseAnalysis *reuse, const CostModel *cost_model,
                             const int8_t *input_bytes, size_t input_len,
                             FILE *out, DspProgramCost *out_cost) {
    ProgramRegionCost zero = {0};
    out_cost->initialization = zero;
    out_cost->body = zero;
    out_cost->termination = zero;

    EmitUnit unit;
    emit_unit_init(&unit);
    int rc = emit_dsp_program(graph, layout, reuse, cost_model, input_bytes, input_len, out, out_cost, &unit);
    emit_unit_free(&unit);
    return rc;
}
