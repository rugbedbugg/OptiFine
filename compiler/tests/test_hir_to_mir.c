/* The workload HIR -> MIR translation for both real workloads: the modules
 * verify, have the shape hir_to_mir.h documents (data movement as MIR loads
 * and stores, every kernel a declared target region, the twiddle table a
 * constant object), and the MIR initialize function, run in the reference
 * interpreter, leaves exactly the embedded input and constants in the
 * tensors -- MIR semantics checked with no AVR code involved.
 *
 * argv: <tiny_classifier.onnx> <cost_table.toml> */
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "mir_interp.h"
#include "optifine/codegen/hir_to_mir.h"
#include "optifine/codegen/lower.h"
#include "optifine/dsp_build.h"
#include "optifine/ingest.h"
#include "optifine/ir_verify.h"

static CostModel g_cm;

typedef struct {
    size_t stores, loads, targets, other;
} Census;

static Census census(const MirModule *m, uint32_t fn) {
    Census c = {0, 0, 0, 0};
    const MirBlock *b = &m->functions[fn].blocks[0];
    assert(m->functions[fn].num_blocks == 1 && b->term.kind == MIR_TERM_RET);
    for (size_t i = 0; i < b->count; i++) {
        switch (b->insts[i].op) {
            case MIR_STORE: c.stores++; break;
            case MIR_LOAD: c.loads++; break;
            case MIR_TARGET: c.targets++; break;
            default: c.other++; break;
        }
    }
    return c;
}

static void check(const char *what, const IrGraph *g, const int8_t *input, size_t input_len, int optimized,
                  size_t want_targets) {
    char msg[256];
    assert(ir_verify(g, msg, sizeof(msg)) == 0);
    SramLayout layout;
    ReuseAnalysis reuse;
    assert(sram_layout_build(g, &layout) == 0);
    assert(reuse_analyze(g, &reuse) == 0);
    WorkloadMir w;
    assert(hir_to_mir(g, &layout, &reuse, &g_cm, input, input_len, optimized,
                      HIR_TO_MIR_INITIALIZE | HIR_TO_MIR_INFER, &w, false) == 0);
    assert(mir_verify(&w.module, msg, sizeof(msg)) == 0);

    size_t init_bytes = 0, output_bytes = 0;
    for (size_t i = 0; i < g->count; i++) {
        size_t bytes;
        assert(ir_op_tensor_size(&g->ops[i], NULL, &bytes) == 0);
        assert(w.module.objects[w.tensor[i]].kind == MIR_MEM_TENSOR && w.module.objects[w.tensor[i]].size == bytes);
        if (g->ops[i].kind == OP_INPUT || g->ops[i].kind == OP_CONST) init_bytes += bytes;
        if (g->ops[i].kind == OP_OUTPUT) output_bytes = bytes;
    }
    Census init = census(&w.module, w.initialize), infer = census(&w.module, w.infer);
    assert(init.stores == init_bytes && init.loads == 0 && init.targets == 0 && init.other == 0);
    assert(infer.loads == output_bytes && infer.stores == output_bytes && infer.targets == want_targets &&
           infer.other == 0);

    /* Every target region declares what it touches, and only the FFT stages
     * read the program-memory table. */
    const MirBlock *b = &w.module.functions[w.infer].blocks[0];
    for (size_t i = 0; i < b->count; i++) {
        const MirInst *in = &b->insts[i];
        if (in->op != MIR_TARGET) continue;
        const IrOp *op = &g->ops[in->origin];
        assert(!strcmp(in->target->target, "avr"));
        assert(in->target->num_writes >= 1 && in->target->writes[0] == w.tensor[in->origin]);
        int reads_table = 0;
        for (size_t r = 0; r < in->target->num_reads; r++) reads_table |= in->target->reads[r] == w.twiddle;
        assert(reads_table == (op->kind == OP_FFT_BUTTERFLY));
    }

    /* MIR semantics alone: initialize leaves the input and constants. */
    MirInterp interp;
    assert(mir_interp_init(&interp, &w.module) == 0);
    int64_t none = 0;
    assert(mir_interp_call(&interp, w.initialize, NULL, &none) == 0);
    for (size_t i = 0; i < g->count; i++) {
        const uint8_t *t = interp.memory[w.tensor[i]];
        if (g->ops[i].kind == OP_INPUT) assert(memcmp(t, input, input_len) == 0);
        if (g->ops[i].kind == OP_CONST) assert(memcmp(t, g->ops[i].data, g->ops[i].data_len) == 0);
    }
    mir_interp_free(&interp);

    AvrMirLayout avr;
    assert(workload_mir_layout(&w, &layout, &avr) == 0);
    for (size_t i = 0; i < g->count; i++) assert(avr.object_addr[w.tensor[i]] == layout.op_addr[i]);
    avr_mir_layout_free(&avr);

    printf("  %s: %zu tensor objects; initialize %zu stores; infer %zu target regions + %zu-byte output copy%s\n",
           what, w.num_tensors, init.stores, infer.targets, output_bytes,
           w.twiddle != MIR_NONE ? "; twiddle table as a MIR constant" : "");
    workload_mir_free(&w);
    reuse_analysis_free(&reuse);
    sram_layout_free(&layout);
}

int main(int argc, char **argv) {
    assert(argc == 3);
    assert(cost_model_load(argv[2], &g_cm) == 0);

    IrGraph ml;
    assert(ingest_load_onnx(argv[1], &ml) == 0);
    int8_t ml_input[16];
    for (int i = 0; i < 16; i++) ml_input[i] = (int8_t)(i * 7 - 50);
    check("ML naive", &ml, ml_input, sizeof(ml_input), 0, 7);
    check("ML optimized", &ml, ml_input, sizeof(ml_input), 1, 7);

    /* A wrong-length input is refused before any code exists. */
    {
        SramLayout layout;
        ReuseAnalysis reuse;
        WorkloadMir w;
        assert(sram_layout_build(&ml, &layout) == 0 && reuse_analyze(&ml, &reuse) == 0);
        assert(hir_to_mir(&ml, &layout, &reuse, &g_cm, ml_input, 15, 0, HIR_TO_MIR_INITIALIZE, &w, false) != 0);
        reuse_analysis_free(&reuse);
        sram_layout_free(&layout);
        printf("  15-byte input for a 16-byte Input tensor: refused\n");
    }
    ir_graph_free(&ml);

    IrGraph dsp;
    assert(dsp_build_pipeline(&dsp) == 0);
    int8_t samples[DSP_FFT_SIZE * 2];
    for (size_t i = 0; i < sizeof(samples); i++) samples[i] = (int8_t)(i * 13);
    check("DSP", &dsp, samples, sizeof(samples), 0, 10);
    ir_graph_free(&dsp);

    printf("test_hir_to_mir: all tests passed\n");
    return 0;
}
