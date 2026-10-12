#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "optifine/frontend/c_frontend.h"
#include "optifine/mir.h"
#include "stb_c_lexer.h"
#include "stb_ds.h"

void frontend_c_init_state(CompilerState* s, char* input_buf, int input_buf_len) {
    arrfree(s->da_variables);
    stb_c_lexer_init(&s->lex, input_buf, input_buf + input_buf_len,
                     (char *)malloc(0x10000), 0x10000);
    MirModule *mir = malloc(sizeof(MirModule));
    s->mir = mir;
    mir_module_init(mir);
}

// Peeking functions that take 'stb_lexer' do not modify the state as 'stb_lexer' is copied

bool frontend_c_expect_next_token(stb_lexer lex, int token_type) {
    stb_c_lexer_get_token(&lex);
    return lex.token == token_type;
}

int frontend_c_peek_next_token(stb_lexer lex) {
    stb_c_lexer_get_token(&lex);
    return lex.token;
}

void frontend_c_error_and_exit(stb_lexer lex) {
    stb_lex_location loc;
    stb_c_lexer_get_location(&lex, lex.parse_point, &loc);
    printf("\nError at %d:%d\n", loc.line_number, loc.line_offset);
    exit(1);
}

void frontend_c_parse_args(CompilerState *s) {
    printf("Parsing args: ");
    stb_c_lexer_print_token(&s->lex);
    printf(" \n");
    return;
}

void frontend_c_expect_and_get_next_token(CompilerState *s, int token_type) {
    if (!frontend_c_expect_next_token(s->lex, token_type)) frontend_c_error_and_exit(s->lex);
    stb_c_lexer_get_token(&s->lex);
    printf("Got token: ");
    stb_c_lexer_print_token(&s->lex);
    printf("\n");
}

MirOperand frontend_c_parse_expr(CompilerState *s) {
    frontend_c_expect_and_get_next_token(s, CLEX_intlit);
    int expr_res = s->lex.int_number;
    printf("Parsed expression: %d\n", expr_res);
    MirOperand ret = mir_imm(expr_res);
    return ret;
}

void frontend_c_parse_statement(CompilerState *s, uint32_t fn_id, uint32_t block_id) {
    int tok = frontend_c_peek_next_token(s->lex);
    switch (tok) {
    case CLEX_id: {
        char *type = s->lex.string;
        if (strcmp(type, "return") == 0) {
            stb_c_lexer_get_token(&s->lex);
            MirOperand expr = frontend_c_parse_expr(s);
            frontend_c_expect_and_get_next_token(s, ';');
            mir_ret(s->mir, fn_id, block_id, expr);
        }
    } break;
    }
    return;
}

void frontend_c_parse_block(CompilerState *s, uint32_t fn_id) {
    uint32_t block_id = mir_add_block(s->mir, fn_id);
    while (!frontend_c_expect_next_token(s->lex, '}')) {
        frontend_c_parse_statement(s, fn_id, block_id);
    }
}

MirType frontend_c_get_mir_type(char *type) {
  if (!strcmp(type, "void"))
    return MIR_TYPE_VOID;
  if (!strcmp(type, "char"))
      return MIR_TYPE_I8;
  if (!strcmp(type, "short"))
      return MIR_TYPE_I16;
  if (!strcmp(type, "int"))
      return MIR_TYPE_I32;
  else {
    printf("Unknown type.\n");
    exit(1);
  }
}

void frontend_c_compile_file(CompilerState* s) {
    frontend_c_expect_and_get_next_token(s, CLEX_id);
    char* type = strdup(s->lex.string);

    frontend_c_expect_and_get_next_token(s, CLEX_id);
    char* identifier = strdup(s->lex.string);

    if (!frontend_c_expect_next_token(s->lex, '(')) {
        printf("Global var with type '%s' and name '%s'\nParsing is yet not implemented.", type, identifier);
        return;
    }
    stb_c_lexer_get_token(&s->lex);
    frontend_c_parse_args(s);
    frontend_c_expect_and_get_next_token(s, ')');

    if (!frontend_c_expect_next_token(s->lex, '{')) {
        printf("Function declaration with type '%s' and name '%s'\n", type, identifier);
        return;
    }
    stb_c_lexer_get_token(&s->lex);

    uint32_t fn_id = mir_add_function(s->mir, identifier, frontend_c_get_mir_type(type));
    printf("Function definition with type '%s' and name '%s'\n", type,
           identifier);

    frontend_c_parse_block(s, fn_id);
    frontend_c_expect_and_get_next_token(s, '}');

    mir_dump(s->mir, stdout);

    mir_module_free(s->mir);
    free(type);
    free(identifier);
    return;
}
