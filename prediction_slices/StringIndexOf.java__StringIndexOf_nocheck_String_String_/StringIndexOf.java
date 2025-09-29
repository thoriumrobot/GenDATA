public class StringIndexOf {

    public static String nocheck(String l, String s) {
        int i = l.indexOf(s);
        return l.substring(0, i) + l.substring(i + s.length());
    }
}
