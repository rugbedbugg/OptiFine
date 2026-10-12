#include <stdbool.h>
#include <stdint.h>

#include "optifine/mir.h"
#include "stb_c_lexer.h"
#include "stb_ds.h"

/* Pointers with 'da' prefix are dynamic arrays and SHOULD ONLY be
 * used with the builtin array methods in "stb_ds"
 */

typedef enum {
    C_TYPE_INT,
    C_TYPE_FLOAT,
    C_TYPE_VOID // Variables can only be void pointers.
} CompilerPrimitiveType;

typedef struct {
    char* identifier;
    CompilerPrimitiveType type;
    int stack_height; // Used for scoping variables. 0 means global
    bool is_pointer;
} CompilerVariable;

typedef struct {
    stb_lexer lex;
    CompilerVariable *da_variables;
    MirModule *mir;
} CompilerState;

void frontend_c_init_state(CompilerState* s, char* input_buf, int input_buf_len);
void frontend_c_compile_file(CompilerState* s);
