#include "optifine/codegen/hir_to_mir.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "optifine/codegen/lower.h"
#include "optifine/codegen/select.h"
#include "optifine/invariant.h"

_Static_assert(SRAM_LAYOUT_BASE == AVR_MIR_SRAM_BASE && SRAM_LAYOUT_LIMIT == AVR_MIR_SRAM_LIMIT,
               "the workload layout and the generic backend must agree on usable SRAM");

#define MAX_CANDIDATES 4

static void free_candidate_payload(void *payload) {
    Candidate *c = payload;
    candidate_free(c);
    free(c);
}

static int uses_scratch(OpKind kind) {
    return kind == OP_FFT_BUTTERFLY || kind == OP_MAGNITUDE || kind == OP_PEAK_EXTRACT;
}

static int is_dsp(OpKind kind) {
    return kind == OP_WINDOW || kind == OP_BIT_REVERSE || kind == OP_FFT_BUTTERFLY || kind == OP_MAGNITUDE ||
           kind == OP_PEAK_EXTRACT;
}

/* Memory objects: one tensor per op, then scratch and the twiddle table when
 * the graph's ops use them. */
static int build_objects(const IrGraph *graph, WorkloadMir *w) {
    MirModule *m = &w->module;
    w->tensor = calloc(graph->count ? graph->count : 1, sizeof(uint32_t));
    if (!w->tensor) return -1;
    w->num_tensors = graph->count;
    int dsp = 0, fft = 0;
    for (size_t i = 0; i < graph->count; i++) {
        size_t bytes = 0;
        OPTIFINE_INVARIANT(ir_op_tensor_size(&graph->ops[i], NULL, &bytes) == 0);
        char name[32];
        snprintf(name, sizeof(name), "t%zu", i);
        w->tensor[i] = mir_add_object(m, MIR_MEM_TENSOR, bytes, name, MIR_NONE);
        dsp |= is_dsp(graph->ops[i].kind);
        fft |= graph->ops[i].kind == OP_FFT_BUTTERFLY;
    }
    if (dsp) w->scratch = mir_add_object(m, MIR_MEM_SCRATCH, DSP_SCRATCH_BYTES, "dsp_scratch", MIR_NONE);
    if (fft) {
        uint8_t table[DSP_TWIDDLE_ENTRIES * 4];
        for (int k = 0; k < DSP_TWIDDLE_ENTRIES; k++) {
            int16_t wr, wi;
            dsp_twiddle_q15(k, &wr, &wi);
            table[4 * k + 0] = (uint8_t)((uint16_t)wr & 0xFF);
            table[4 * k + 1] = (uint8_t)((uint16_t)wr >> 8);
            table[4 * k + 2] = (uint8_t)((uint16_t)wi & 0xFF);
            table[4 * k + 3] = (uint8_t)((uint16_t)wi >> 8);
        }
        w->twiddle = mir_add_object(m, MIR_MEM_CONST, sizeof(table), DSP_TWIDDLE_LABEL, MIR_NONE);
        if (w->twiddle != MIR_NONE) mir_object_set_init(m, w->twiddle, table, sizeof(table));
    }
    return m->out_of_memory ? -1 : 0;
}

/* OP_INPUT, OP_CONST and OP_OUTPUT as MIR: byte stores of the embedded data,
 * or a byte-for-byte copy from the producer's tensor. */
static int emit_data_movement(const IrGraph *graph, size_t op_id, const int8_t *input, size_t input_len,
                              WorkloadMir *w, uint32_t fn, uint32_t block) {
    MirModule *m = &w->module;
    const IrOp *op = &graph->ops[op_id];
    size_t bytes = 0;
    OPTIFINE_INVARIANT(ir_op_tensor_size(op, NULL, &bytes) == 0);
    uint32_t origin = (uint32_t)op_id;
    const uint8_t *data = NULL;
    if (op->kind == OP_INPUT) {
        if (input_len != bytes) {
            fprintf(stderr, "lower: demo input has %zu bytes, graph expects %zu\n", input_len, bytes);
            return -1;
        }
        data = (const uint8_t *)input;
    } else if (op->kind == OP_CONST) {
        /* ir_verify guarantees the initializer is exactly the tensor layout
         * reserved; more would overwrite the next tensor. */
        OPTIFINE_INVARIANT(op->data_len == bytes);
        data = op->data;
    }
    for (size_t k = 0; k < bytes; k++) {
        MirAddress dst = mir_at_object(w->tensor[op_id], (int32_t)k);
        if (data) {
            if (mir_emit_store(m, fn, block, MIR_TYPE_I8, dst, mir_imm(data[k]), origin) != 0) return -1;
        } else {
            uint32_t v = mir_new_value(m, fn, MIR_TYPE_I8);
            if (v == MIR_NONE ||
                mir_emit_load(m, fn, block, v, MIR_TYPE_I8, mir_at_object(w->tensor[op->inputs[0]], (int32_t)k),
                              origin) != 0 ||
                mir_emit_store(m, fn, block, MIR_TYPE_I8, dst, mir_value(v), origin) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

/* A kernel op as a MIR_TARGET region around its selected AVR candidate. */
static int emit_kernel(const IrGraph *graph, size_t op_id, const SramLayout *layout, const ReuseAnalysis *reuse,
                       const CostModel *cost_model, int use_real_candidates, WorkloadMir *w, uint32_t fn,
                       uint32_t block) {
    const IrOp *op = &graph->ops[op_id];
    Candidate *chosen = malloc(sizeof(Candidate));
    MirTargetCode *code = calloc(1, sizeof(MirTargetCode));
    if (!chosen || !code) {
        free(chosen);
        free(code);
        fprintf(stderr, "hir_to_mir: out of memory\n");
        return -1;
    }
    if (!use_real_candidates) {
        if (lower_op(graph, op_id, layout, reuse, cost_model, NULL, 0, chosen) != 0) {
            free(chosen);
            free(code);
            return -1;
        }
    } else {
        Candidate candidates[MAX_CANDIDATES];
        size_t count = candidates_generate(graph, op_id, layout, reuse, cost_model, NULL, 0, candidates,
                                           MAX_CANDIDATES);
        if (count == 0) {
            free(chosen);
            free(code);
            return -1;
        }
        const Candidate *best = select_min_energy(candidates, count);
        *chosen = *best;
        for (size_t i = 0; i < count; i++) {
            if (&candidates[i] != best) candidate_free(&candidates[i]);
        }
    }
    code->target = "avr";
    code->payload = chosen;
    code->free_payload = free_candidate_payload;

    size_t extra = uses_scratch(op->kind) ? 1 : 0;
    size_t twiddle = op->kind == OP_FFT_BUTTERFLY ? 1 : 0;
    code->reads = malloc((op->num_inputs + extra + twiddle) * sizeof(uint32_t));
    code->writes = malloc((1 + extra) * sizeof(uint32_t));
    if (!code->reads || !code->writes) {
        mir_target_code_free(code);
        fprintf(stderr, "hir_to_mir: out of memory\n");
        return -1;
    }
    for (size_t j = 0; j < op->num_inputs; j++) code->reads[code->num_reads++] = w->tensor[op->inputs[j]];
    if (extra) code->reads[code->num_reads++] = w->scratch;
    if (twiddle) code->reads[code->num_reads++] = w->twiddle;
    code->writes[code->num_writes++] = w->tensor[op_id];
    if (extra) code->writes[code->num_writes++] = w->scratch;
    return mir_emit_target(&w->module, fn, block, code, (uint32_t)op_id);
}

static int is_initialization(OpKind kind) {
    return kind == OP_INPUT || kind == OP_CONST;
}

int hir_to_mir(const IrGraph *graph, const SramLayout *layout, const ReuseAnalysis *reuse,
               const CostModel *cost_model, const int8_t *input, size_t input_len, int use_real_candidates,
               unsigned functions, WorkloadMir *out, bool dump_mir) {
    memset(out, 0, sizeof(*out));
    mir_module_init(&out->module);
    out->initialize = out->infer = out->scratch = out->twiddle = MIR_NONE;
    if (build_objects(graph, out) != 0) {
        fprintf(stderr, "hir_to_mir: out of memory\n");
        workload_mir_free(out);
        return -1;
    }
    MirModule *m = &out->module;
    int rc = 0;
    if (functions & HIR_TO_MIR_INITIALIZE) {
        out->initialize = mir_add_function(m, "initialize", MIR_TYPE_VOID);
        uint32_t b = out->initialize == MIR_NONE ? MIR_NONE : mir_add_block(m, out->initialize);
        rc = b == MIR_NONE ? -1 : 0;
        for (size_t i = 0; i < graph->count && rc == 0; i++) {
            if (is_initialization(graph->ops[i].kind)) {
                rc = emit_data_movement(graph, i, input, input_len, out, out->initialize, b);
            }
        }
        if (rc == 0) mir_ret(m, out->initialize, b, mir_none());
    }
    if (rc == 0 && (functions & HIR_TO_MIR_INFER)) {
        out->infer = mir_add_function(m, "infer", MIR_TYPE_VOID);
        uint32_t b = out->infer == MIR_NONE ? MIR_NONE : mir_add_block(m, out->infer);
        rc = b == MIR_NONE ? -1 : 0;
        for (size_t i = 0; i < graph->count && rc == 0; i++) {
            OpKind kind = graph->ops[i].kind;
            if (is_initialization(kind)) continue;
            rc = kind == OP_OUTPUT ? emit_data_movement(graph, i, input, input_len, out, out->infer, b)
                                   : emit_kernel(graph, i, layout, reuse, cost_model, use_real_candidates, out,
                                                 out->infer, b);
            if (rc != 0) fprintf(stderr, "hir_to_mir: failed to lower op %zu\n", i);
        }
        if (rc == 0) mir_ret(m, out->infer, b, mir_none());
        if(dump_mir) mir_dump(m, stdout);
    }

    char message[256];
    if (rc == 0 && mir_verify(m, message, sizeof(message)) != 0) {
        fprintf(stderr, "hir_to_mir: produced invalid MIR: %s\n", message);
        rc = -1;
    }
    if (rc != 0) {
        workload_mir_free(out);
        return -1;
    }
    return 0;
}

void workload_mir_free(WorkloadMir *mir) {
    mir_module_free(&mir->module);
    free(mir->tensor);
    mir->tensor = NULL;
    mir->num_tensors = 0;
}

int workload_mir_layout(const WorkloadMir *mir, const SramLayout *layout, AvrMirLayout *out) {
    if (avr_mir_layout_init(&mir->module, out) != 0) {
        fprintf(stderr, "hir_to_mir: out of memory\n");
        return -1;
    }
    for (size_t i = 0; i < mir->num_tensors; i++) out->object_addr[mir->tensor[i]] = layout->op_addr[i];
    if (mir->scratch != MIR_NONE) out->object_addr[mir->scratch] = layout->dsp_scratch_addr;
    /* Nothing else may need SRAM: the space after the tensors belongs to the
     * program wrapper (the periodic scheduler bytes). */
    uint16_t end = (uint16_t)(SRAM_LAYOUT_BASE + layout->bytes_used);
    if (avr_mir_layout_place_rest(&mir->module, out, end, end, NULL) != 0) {
        fprintf(stderr, "hir_to_mir: the workload needs SRAM beyond its fixed tensor layout\n");
        avr_mir_layout_free(out);
        return -1;
    }
    return 0;
}

int hir_lower_data_movement(const IrGraph *graph, size_t op_id, const SramLayout *layout,
                            const CostModel *cost_model, const int8_t *input, size_t input_len,
                            Candidate *out) {
    WorkloadMir w;
    memset(&w, 0, sizeof(w));
    mir_module_init(&w.module);
    w.initialize = w.infer = w.scratch = w.twiddle = MIR_NONE;
    int rc = build_objects(graph, &w);
    uint32_t fn = MIR_NONE, block = MIR_NONE;
    if (rc == 0) {
        fn = mir_add_function(&w.module, "op", MIR_TYPE_VOID);
        block = fn == MIR_NONE ? MIR_NONE : mir_add_block(&w.module, fn);
        rc = block == MIR_NONE ? -1 : emit_data_movement(graph, op_id, input, input_len, &w, fn, block);
    }
    AvrMirLayout avr;
    AvrMirCode code = {0};
    if (rc == 0) {
        mir_ret(&w.module, fn, block, mir_none());
        rc = workload_mir_layout(&w, layout, &avr);
        if (rc == 0) {
            rc = avr_mir_select_function(&w.module, &avr, fn, 0, cost_model, &code);
            avr_mir_layout_free(&avr);
        }
    }
    workload_mir_free(&w);
    if (rc != 0) return -1;
    /* One op, one origin: exactly one segment. */
    OPTIFINE_INVARIANT(code.count == 1);
    *out = code.segments[0].code;
    free(code.segments);
    return 0;
}
