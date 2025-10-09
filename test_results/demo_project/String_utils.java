
public class StringUtils {
    public String reverse(String str) {
        if (str == null) {
            return null;
        }
        
        StringBuilder sb = new StringBuilder();
        for (int i = str.length() - 1; i >= 0; i--) {
            sb.append(str.charAt(i));
        }
        return sb.toString();
    }
    
    public boolean isPalindrome(String str) {
        if (str == null || str.length() <= 1) {
            return true;
        }
        
        String reversed = reverse(str);
        return str.equals(reversed);
    }
}