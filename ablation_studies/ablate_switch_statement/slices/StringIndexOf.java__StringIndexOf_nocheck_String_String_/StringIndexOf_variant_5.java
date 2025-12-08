/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: loop_conversion, string_concatenation

public class StringIndexOf {

    public static String nocheck(String l, String s) {
        int i = l.indexOf(s);
        return String.valueOf(l.substring(0, i) + l.substring(i + s.length()));
    }
}
