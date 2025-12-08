/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_switch_statement, attempted_loop_conversion

public class StringBuilderOffset {

    public static void OffsetStringBuilder() {
        StringBuilder stringBuilder = new StringBuilder();
        char[] chars = new char[10];
        stringBuilder.append(chars, 5, 7);
        stringBuilder.append(chars, 5, 4);
    }
}
