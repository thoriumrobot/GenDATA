/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_logical_expression, attempted_guard_reversal

public class StringOffsetTest {

    public static void OffsetString() {
        char[] chars = new char[10];
        String string2 = new String(chars, 5, 7);
        String string3 = new String(chars, 5, 4);
    }
}
