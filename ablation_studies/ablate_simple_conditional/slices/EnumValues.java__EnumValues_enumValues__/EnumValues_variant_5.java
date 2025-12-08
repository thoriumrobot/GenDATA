/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_variable_operation

import org.checkerframework.common.value.qual.*;

public class EnumValues {

    public static void enumValues() {
        Direction @ArrayLen(4) [] arr4 = Direction.values();
        Direction[] arr = Direction.values();
        Direction a = arr[0];
        Direction b = arr[1];
        Direction c = arr[2];
        Direction d = arr[3];
        Direction e = arr[4];
    }
}
