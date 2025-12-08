/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_string_concatenation

public class StringBuilderOffset {

    public static void OffsetStringBuilder() {
        StringBuilder stringBuilder = new StringBuilder();
        char[] chars = new char[10];
        stringBuilder.append(chars, 5, 7);
        stringBuilder.append(chars, 5, 4);
    }
}
