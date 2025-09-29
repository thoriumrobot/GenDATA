public class StringIndexOf {

    public static String stringLiteral(String l) {
        int i = l.indexOf("constant");
        if (i != -1) {
            return l.substring(0, i) + l.substring(i + "constant".length());
        }
        return l.substring(0, i) + l.substring(i + "constant".length());
    }
}
