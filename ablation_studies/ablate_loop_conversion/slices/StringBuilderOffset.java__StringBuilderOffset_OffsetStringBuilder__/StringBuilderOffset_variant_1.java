/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_guard_reversal, attempted_ternary_operator

public class StringBuilderOffset {

    public static void OffsetStringBuilder() {
        StringBuilder stringBuilder = new StringBuilder();
        char[] chars = new char[10];
        stringBuilder.append(chars, 5, 7);
        stringBuilder.append(chars, 5, 4);
    }
}
