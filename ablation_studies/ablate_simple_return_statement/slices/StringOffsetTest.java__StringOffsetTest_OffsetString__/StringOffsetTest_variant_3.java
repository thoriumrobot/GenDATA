/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_string_concatenation, attempted_loop_conversion

public class StringOffsetTest {

    public static void OffsetString() {
        char[] chars = new char[10];
        String string2 = new String(chars, 5, 7);
        String string3 = new String(chars, 5, 4);
    }
}
