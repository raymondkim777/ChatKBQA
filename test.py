normed_expr = 1
entity_label_map = 1
type_label_map = 1
surface_index = 1

def execute_normed_s_expr_from_label_maps(
    normed_expr, 
    entity_label_map,
    type_label_map,
    surface_index
):
    pass

lf, answers = execute_normed_s_expr_from_label_maps(
    normed_expr,        # s-expr with placeholders
    entity_label_map,   # for oracle entity-linking 
    type_label_map,     # KB literal type map
    surface_index       # KB entity/relation mentions
)

