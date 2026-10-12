#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "optifine/codegen/lower.h"
#include "optifine/codegen/periodic.h"
#include "optifine/codegen/program.h"
#include "optifine/codegen/reuse_analysis.h"
#include "optifine/codegen/sram_layout.h"
#include "optifine/cost_model.h"
#include "optifine/dsp_build.h"
#include "optifine/ingest.h"
#include "optifine/ir.h"
#include "optifine/ir_verify.h"
#include "optifine/frontend/c_frontend.h"

#define STB_C_LEXER_IMPLEMENTATION
#define STB_DS_IMPLEMENTATION

#include "stb_c_lexer.h"
#include "stb_ds.h"

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s <model.onnx> --cost-table <cost_table.toml> --out <out.s> "
            "[--input <golden_input.txt>] [--optimized] "
            "[--periodic-count N --wait-policy active|powersave --timer-prescaler N]\n"
            "       %s --dsp --cost-table <cost_table.toml> --out <out.s> [--input <samples.txt>]\n"
            "            [--periodic-count N --wait-policy active|powersave --timer-prescaler N]\n"
            "  --dsp        compile the fixed 64-point DSP pipeline (Window, BitReverse, FFT x6,\n"
            "               Magnitude, PeakExtract) instead of an ONNX model. --input holds its 64\n"
            "               int16 Q15 samples (default models/dsp_demo_input.txt). No model path or\n"
            "               --optimized; the periodic flags wrap it as they wrap the ML body\n"
            "  --optimized  use codegen/candidates.c's real candidate diversity "
            "instead of the naive single-candidate baseline\n"
            "  periodic mode requires --periodic-count, --wait-policy, and "
            "--timer-prescaler together\n",
            argv0, argv0);
}

static int parse_unsigned(const char *text, unsigned long maximum, unsigned long *out_value) {
    char *end = NULL;
    unsigned long value;

    if (text == NULL || text[0] == '\0') {
        return -1;
    }
    for (const char *digit = text; *digit != '\0'; digit++) {
        if (*digit < '0' || *digit > '9') {
            return -1;
        }
    }
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > maximum) {
        return -1;
    }
    *out_value = value;
    return 0;
}

/* Reads whitespace-separated decimal integers in [min_value, max_value] from
 * `path`, with `#` starting a comment that runs to the end of the line, and
 * stores each as `width` little-endian bytes (1 for int8, 2 for int16) -- the
 * byte image OP_INPUT embeds. Anything else on a line (a non-numeric token, a
 * value out of range, a line too long for the buffer) is an error, not the
 * end of the data. The byte count is checked against the graph later.
 * Returns the buffer (caller frees) and sets *out_len, or NULL with a
 * diagnostic. */
static int8_t *read_input_values(const char *path, const char *what, long min_value, long max_value,
                                 size_t width, size_t *out_len) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "failed to open %s file: %s\n", what, path);
        return NULL;
    }
    size_t cap = 128, len = 0;
    uint8_t *buf = malloc(cap);
    char line[1024];
    unsigned line_no = 0;
    const char *error = buf ? NULL : "out of memory";
    while (!error && fgets(line, sizeof(line), f)) {
        line_no++;
        if (strchr(line, '\n') == NULL && !feof(f)) {
            error = "line too long";
            break;
        }
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        char *cursor = line;
        for (;;) {
            while (isspace((unsigned char)*cursor)) cursor++;
            if (*cursor == '\0') break;
            char *endptr;
            errno = 0;
            long v = strtol(cursor, &endptr, 10);
            if (endptr == cursor || (*endptr != '\0' && !isspace((unsigned char)*endptr))) {
                error = "not a decimal integer";
                break;
            }
            if (errno != 0 || v < min_value || v > max_value) {
                fprintf(stderr, "%s file %s:%u: value out of range [%ld,%ld]\n", what, path, line_no,
                        min_value, max_value);
                error = "";
                break;
            }
            if (len + width > cap) {
                uint8_t *grown = cap <= SIZE_MAX / 2 ? realloc(buf, cap * 2) : NULL;
                if (!grown) {
                    error = "out of memory";
                    break;
                }
                buf = grown;
                cap *= 2;
            }
            uint16_t u = (uint16_t)v; /* two's complement image of the in-range value */
            for (size_t b = 0; b < width; b++) {
                buf[len++] = (uint8_t)(u >> (8 * b));
            }
            cursor = endptr;
        }
    }
    if (!error && ferror(f)) error = "read error";
    fclose(f);
    if (error) {
        if (error[0] != '\0') fprintf(stderr, "%s file %s:%u: %s\n", what, path, line_no, error);
        free(buf);
        return NULL;
    }
    *out_len = len;
    return (int8_t *)buf;
}

/* 64 int16 Q15 samples for --dsp. */
static int8_t *read_dsp_input(const char *path, size_t *out_len) {
    return read_input_values(path, "DSP input", INT16_MIN, INT16_MAX, 2, out_len);
}

/* The ML demo input: int8 values in [-127,127], the symmetric range the
 * quantized model uses. */
static int8_t *read_demo_input(const char *path, size_t *out_len) {
    return read_input_values(path, "input", -127, 127, 1, out_len);
}

/* The periodic cost lines, shared by the ML and --dsp periodic modes;
 * sim/run_periodic_*.py parse them. */
static void print_periodic_cost(const PeriodicOptions *options, const PeriodicProgramCost *cost) {
    fprintf(stderr, "periodic policy: %s\n", options->policy == WAIT_ACTIVE ? "active" : "powersave");
    fprintf(stderr, "periodic timer prescaler: %u (Timer0 divisor)\n", options->timer_prescaler);
    fprintf(stderr, "periodic inference count: %u\n", options->inference_count);
    fprintf(stderr, "periodic initialization predicted cost: %u cycles, %.3f nJ\n",
            cost->initialization.cycles, cost->initialization.energy_nj);
    fprintf(stderr, "periodic per-inference predicted compute cost: %u cycles, %.3f nJ\n",
            cost->inference.cycles, cost->inference.energy_nj);
}

/* --dsp: the graph comes from dsp_build_pipeline, not ONNX, and is emitted
 * by codegen_emit_dsp_program. There is no ML forward-pass check to run; the
 * DSP lowering's correctness is pinned by test_dsp_pipeline's host oracle. */
static int run_dsp(const char *cost_table_path, const char *input_path, const char *out_path,
                   const PeriodicOptions *periodic) {
    CostModel cost_model;
    if (cost_model_load(cost_table_path, &cost_model) != 0) {
        fprintf(stderr, "failed to load cost table: %s\n", cost_table_path);
        return 1;
    }
    size_t input_len = 0;
    int8_t *input = read_dsp_input(input_path, &input_len);
    if (!input) {
        return 1;
    }
    IrGraph graph;
    if (dsp_build_pipeline(&graph) != 0) {
        free(input);
        return 1;
    }
    if (ir_verify_or_report(&graph, "DSP pipeline") != 0) {
        free(input);
        ir_graph_free(&graph);
        return 1;
    }
    SramLayout layout;
    if (sram_layout_build(&graph, &layout) != 0) {
        free(input);
        ir_graph_free(&graph);
        return 1;
    }
    ReuseAnalysis reuse;
    if (reuse_analyze(&graph, &reuse) != 0) {
        fprintf(stderr, "out of memory in reuse analysis\n");
        free(input);
        sram_layout_free(&layout);
        ir_graph_free(&graph);
        return 1;
    }

    int rc = 1;
    FILE *out = fopen(out_path, "w");
    if (!out) {
        fprintf(stderr, "failed to open output file: %s\n", out_path);
    } else if (periodic) {
        /* Periodic scheduling: the same DSP initialization and body inside the periodic
         * wrapper the ML path uses. */
        PeriodicProgramCost cost;
        rc = codegen_emit_periodic_program(&graph, &layout, &reuse, &cost_model, input, input_len,
                                           periodic, out, &cost);
        fclose(out);
        if (rc == 0) {
            print_periodic_cost(periodic, &cost);
        }
    } else {
        DspProgramCost cost;
        rc = codegen_emit_dsp_program(&graph, &layout, &reuse, &cost_model, input, input_len, out, &cost);
        fclose(out);
        if (rc == 0) {
            uint32_t total = cost.initialization.cycles + cost.body.cycles + cost.termination.cycles;
            fprintf(stderr, "dsp initialization (clr r2, input, window coefficients): %u cycles, %.3f nJ (predicted)\n",
                    cost.initialization.cycles, cost.initialization.energy_nj);
            fprintf(stderr, "dsp pipeline (window .. output): %u cycles, %.3f nJ (predicted)\n",
                    cost.body.cycles, cost.body.energy_nj);
            fprintf(stderr, "dsp termination (break): %u cycles, %.3f nJ (predicted)\n",
                    cost.termination.cycles, cost.termination.energy_nj);
            fprintf(stderr, "total: %u cycles, %.3f nJ (predicted, break included; compare against a real "
                            "sim/run_avrora.sh run for the actual simulated numbers)\n",
                    total, cost.initialization.energy_nj + cost.body.energy_nj + cost.termination.energy_nj);
            fprintf(stderr, "sram: %u bytes from 0x%04X (graph tensors + %d-byte DSP scratch)\n",
                    layout.bytes_used, SRAM_LAYOUT_BASE, DSP_SCRATCH_BYTES);
        }
    }
    free(input);
    reuse_analysis_free(&reuse);
    sram_layout_free(&layout);
    ir_graph_free(&graph);
    return rc == 0 ? 0 : 1;
}

int compile_c_file(const char* input_fp, const char* output_fp) {
   FILE *f = fopen(input_fp,"rb");
   char *text = (char *) malloc(1 << 20);
   int len = f ? (int) fread(text, 1, 1<<20, f) : -1;
   if (len < 0) {
      fprintf(stderr, "Error opening file\n");
      free(text);
      fclose(f);
      return 1;
   }
   fclose(f);
   printf("File:\n%s", text);
   CompilerState s = {0};
   frontend_c_init_state(&s, text, len);
   frontend_c_compile_file(&s);
   return 0;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *cost_table_path = "cost_table.toml";
    const char *out_path = "out.s";
    const char *input_path = "models/tiny_classifier_golden_input.txt";
    int use_real_candidates = 0;
    int dsp_mode = 0;
    int input_seen = 0;
    int periodic_count_seen = 0;
    int wait_policy_seen = 0;
    int timer_prescaler_seen = 0;
    PeriodicOptions periodic_options = {0};

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cost-table") == 0 && i + 1 < argc) {
            cost_table_path = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
            input_seen = 1;
        } else if (strcmp(argv[i], "--dsp") == 0) {
            dsp_mode = 1;
        } else if (strcmp(argv[i], "--optimized") == 0) {
            use_real_candidates = 1;
        } else if (strcmp(argv[i], "--periodic-count") == 0 && i + 1 < argc) {
            unsigned long value;
            if (periodic_count_seen || parse_unsigned(argv[++i], UINT8_MAX, &value) != 0) {
                usage(argv[0]);
                return 2;
            }
            periodic_options.inference_count = (uint8_t)value;
            periodic_count_seen = 1;
        } else if (strcmp(argv[i], "--wait-policy") == 0 && i + 1 < argc) {
            const char *policy = argv[++i];
            if (wait_policy_seen) {
                usage(argv[0]);
                return 2;
            }
            if (strcmp(policy, "active") == 0) {
                periodic_options.policy = WAIT_ACTIVE;
            } else if (strcmp(policy, "powersave") == 0) {
                periodic_options.policy = WAIT_POWER_SAVE;
            } else {
                usage(argv[0]);
                return 2;
            }
            wait_policy_seen = 1;
        } else if (strcmp(argv[i], "--timer-prescaler") == 0 && i + 1 < argc) {
            unsigned long value;
            if (timer_prescaler_seen || parse_unsigned(argv[++i], UINT16_MAX, &value) != 0) {
                usage(argv[0]);
                return 2;
            }
            periodic_options.timer_prescaler = (uint16_t)value;
            timer_prescaler_seen = 1;
        } else if (strcmp(argv[i], "--cc") == 0 && i + 1 < argc) {
            const char *c_source_file = argv[++i];
            return compile_c_file(c_source_file, out_path);
        } else if (!model_path) {
            model_path = argv[i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    int periodic_mode = periodic_count_seen || wait_policy_seen || timer_prescaler_seen;
    if (periodic_mode && !(periodic_count_seen && wait_policy_seen && timer_prescaler_seen)) {
        usage(argv[0]);
        return 2;
    }
    periodic_options.use_real_candidates = use_real_candidates;
    if (periodic_mode && periodic_options_validate(&periodic_options) != 0) {
        fprintf(stderr, "invalid periodic scheduling options\n");
        return 2;
    }

    if (dsp_mode) {
        /* One workload per program: the DSP and ML paths share registers on the
         * assumption that they are never lowered into the same program (see
         * registers.h), so a model path is refused rather than ignored. */
        if (model_path || use_real_candidates) {
            fprintf(stderr, "--dsp takes no model path or --optimized (the DSP path has no candidate "
                            "diversity)\n");
            usage(argv[0]);
            return 2;
        }
        return run_dsp(cost_table_path, input_seen ? input_path : "models/dsp_demo_input.txt", out_path,
                       periodic_mode ? &periodic_options : NULL);
    }

    if (!model_path) {
        usage(argv[0]);
        return 2;
    }

    CostModel cost_model;
    if (cost_model_load(cost_table_path, &cost_model) != 0) {
        fprintf(stderr, "failed to load cost table: %s\n", cost_table_path);
        return 1;
    }

    IrGraph graph;
    ir_graph_init(&graph);
    if (ingest_load_onnx(model_path, &graph) != 0) {
        fprintf(stderr, "failed to ingest model: %s\n", model_path);
        ir_graph_free(&graph);
        return 1;
    }

    /* The verification boundary: nothing below this point sees a graph that
     * breaks an operator contract (ir_verify.h). */
    if (ir_verify_or_report(&graph, model_path) != 0) {
        ir_graph_free(&graph);
        return 1;
    }

    size_t demo_input_len = 0;
    int8_t *demo_input = read_demo_input(input_path, &demo_input_len);
    if (!demo_input) {
        ir_graph_free(&graph);
        return 1;
    }

    /* Must run before any codegen: refuses to compile if this specific
     * input would overflow int8 through any Requantize stage (see
     * lower.h/lower_requantize.c for why this checks the actual input rather than a
     * worst-case bound over all possible inputs). */
    if (lower_verify_demo_forward_pass(&graph, demo_input, demo_input_len) != 0) {
        fprintf(stderr, "failed to verify demo input against the model: %s\n", input_path);
        free(demo_input);
        ir_graph_free(&graph);
        return 1;
    }

    SramLayout layout;
    if (sram_layout_build(&graph, &layout) != 0) {
        free(demo_input);
        ir_graph_free(&graph);
        return 1;
    }

    ReuseAnalysis reuse;
    /* Consulted only when --optimized routes through candidates_generate;
     * the naive path (lower_op) ignores it (reuse_analysis.h). */
    if (reuse_analyze(&graph, &reuse) != 0) {
        fprintf(stderr, "out of memory in reuse analysis\n");
        free(demo_input);
        sram_layout_free(&layout);
        ir_graph_free(&graph);
        return 1;
    }

    FILE *out = fopen(out_path, "w");
    if (!out) {
        fprintf(stderr, "failed to open output file: %s\n", out_path);
        free(demo_input);
        reuse_analysis_free(&reuse);
        sram_layout_free(&layout);
        ir_graph_free(&graph);
        return 1;
    }

    int rc;
    ProgramCost cost;
    PeriodicProgramCost periodic_cost;
    if (periodic_mode) {
        rc = codegen_emit_periodic_program(&graph, &layout, &reuse, &cost_model,
                                           demo_input, demo_input_len, &periodic_options,
                                           out, &periodic_cost);
    } else {
        rc = codegen_emit_program(&graph, &layout, &reuse, &cost_model,
                                  demo_input, demo_input_len, use_real_candidates, out, &cost);
    }
    fclose(out);

    if (rc == 0 && periodic_mode) {
        print_periodic_cost(&periodic_options, &periodic_cost);
    } else if (rc == 0) {
        fprintf(stderr, "prologue (const/input load): %u cycles, %.3f nJ (predicted)\n",
                cost.prologue_cycles, cost.prologue_energy_nj);
        fprintf(stderr, "inference body: %u cycles, %.3f nJ (predicted)\n",
                cost.body_cycles, cost.body_energy_nj);
        fprintf(stderr, "total: %u cycles, %.3f nJ (predicted; compare against a real "
                        "sim/run_avrora.sh run for the actual simulated numbers)\n",
                cost.prologue_cycles + cost.body_cycles, cost.prologue_energy_nj + cost.body_energy_nj);
    }

    free(demo_input);
    reuse_analysis_free(&reuse);
    sram_layout_free(&layout);
    ir_graph_free(&graph);
    return rc == 0 ? 0 : 1;
}
